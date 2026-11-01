"""Evaluate all trained HARN ablation variants with stateful streaming inference.

This evaluator genuinely reuses the production HARN data adapter
(``train_harn_ablations.HARNDataset``), target preparation
(``_prepare_targets``), streaming batcher (``StreamingBatcher``), device
resolution, and checkpoint loader from ``train_harn_ablations.py`` -- there
is exactly one implementation of each, imported here rather than
reimplemented, so the two scripts cannot silently drift apart on data
schema or checkpoint format.

For every (asset, variant) pair it loads ``<checkpoint-dir>/<asset>/<variant>/
<checkpoint-name>.pt`` (``best_model.pt`` by default -- the same checkpoint
``train_harn_ablations.py`` itself uses for its own final test-set report)
and evaluates it on ``<data-dir>/<asset>/test.pkl``, writing per-timeframe
MAE/RMSE/sMAPE/MASE/Directional-Accuracy rows to a CSV.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

try:
    from .harn_ablations import HARN, VALID_VARIANTS
    from . import train_harn_ablations as train_mod
except ImportError:  # Support direct execution as well as ``python -m``.
    from src.ablation.harn_ablations import HARN, VALID_VARIANTS
    from src.ablation import train_harn_ablations as train_mod


ASSETS: Tuple[str, ...] = ("AAPL", "EURUSD")
VARIANTS: Tuple[str, ...] = tuple(VALID_VARIANTS)
NUM_STREAMS = 8
CHECKPOINT_NAME = "best_model.pt"
RESULT_COLUMNS = [
    "asset",
    "timeframe",
    "variant",
    "MAE",
    "RMSE",
    "sMAPE",
    "MASE",
    "Directional_Accuracy",
]


class EvaluationError(RuntimeError):
    """An error tied to one asset/variant evaluation."""


def _load_pickle(path: Path, asset: str, split: str) -> Dict[str, Any]:
    if not path.is_file():
        raise EvaluationError(f"{asset}: missing {split} dataset: {path}")
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception as exc:
        raise EvaluationError(f"{asset}: could not load {split} dataset {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise EvaluationError(f"{asset}: {split} dataset {path} is not a dictionary")
    return payload


def _load_checkpoint(path: Path, asset: str, variant: str) -> Any:
    """Thin wrapper around the shared `train_mod.load_checkpoint`.

    Only adds asset/variant context to the error message; the actual
    deserialization logic (cross-platform path handling, weights_only
    compatibility) lives in exactly one place (`train_harn_ablations.py`).
    """
    if not path.is_file():
        raise EvaluationError(f"{asset}/{variant}: missing required checkpoint {path}")
    try:
        return train_mod.load_checkpoint(path, map_location="cpu")
    except Exception as exc:
        raise EvaluationError(f"{asset}/{variant}: could not load {path}: {exc}") from exc


def _checkpoint_state(
    checkpoint: Any, path: Path, asset: str, variant: str
) -> Tuple[Dict[str, torch.Tensor], Tuple[str, ...], "train_mod.ModelConfig"]:
    """Validate a loaded checkpoint and reconstruct its exact training-time config.

    The checkpoint's own stored ``config`` (not the current code's
    ``ModelConfig`` defaults) is the source of truth for hierarchy levels
    and hyperparameters, so the reconstructed model always matches the
    saved weights -- see `train_mod.config_from_checkpoint`.
    """
    if not isinstance(checkpoint, dict):
        raise EvaluationError(f"{asset}/{variant}: checkpoint {path} is not a dictionary")

    declared_variant = checkpoint.get("variant")
    if declared_variant is not None and declared_variant != variant:
        raise EvaluationError(
            f"{asset}/{variant}: checkpoint at {path} declares variant "
            f"{declared_variant!r}, expected {variant!r} (misplaced checkpoint file?)"
        )

    try:
        config = train_mod.config_from_checkpoint(checkpoint)
    except (KeyError, ValueError, TypeError) as exc:
        raise EvaluationError(f"{asset}/{variant}: {path}: {exc}") from exc
    if config.variant != variant:
        raise EvaluationError(
            f"{asset}/{variant}: checkpoint config at {path} declares variant "
            f"{config.variant!r}, expected {variant!r}."
        )
    checkpoint_timeframes = config.timeframes

    state_dict = checkpoint.get("model_state_dict")
    if state_dict is None:
        raise EvaluationError(
            f"{asset}/{variant}: {path} is not the required wrapped checkpoint "
            "(missing 'model_state_dict')"
        )
    if not isinstance(state_dict, dict) or not state_dict:
        raise EvaluationError(f"{asset}/{variant}: model_state_dict is empty or malformed")
    if not all(isinstance(key, str) and torch.is_tensor(value) for key, value in state_dict.items()):
        raise EvaluationError(f"{asset}/{variant}: model_state_dict contains invalid entries")
    return state_dict, checkpoint_timeframes, config


def _build_dataset(
    payload: Dict[str, Any], timeframes: Tuple[str, ...], path: Path, asset: str, variant: str, split: str
) -> "train_mod.HARNDataset":
    """Construct a `HARNDataset`, translating its errors into `EvaluationError`.

    This is the *only* place payload schema/shape validation happens: it
    reuses the exact same `HARNDataset` that `train_harn_ablations.py` uses,
    so there is one data adapter for the whole pipeline, not two that can
    silently drift apart.
    """
    try:
        return train_mod.HARNDataset(payload, timeframes=timeframes)
    except (KeyError, ValueError, TypeError) as exc:
        raise EvaluationError(f"{asset}/{variant}: {split} data at {path} incompatible with checkpoint: {exc}") from exc


def _mase_scale(train_dataset: "train_mod.HARNDataset", timeframe: str, asset: str, variant: str) -> float:
    block = train_dataset.statistics[f"target_{timeframe}"]
    mean_arr = np.asarray(block["mean"], dtype=np.float64).reshape(-1)
    std_arr = np.asarray(block["std"], dtype=np.float64).reshape(-1)
    if mean_arr.size != 1 or std_arr.size != 1:
        raise EvaluationError(
            f"{asset}/{variant}/{timeframe}: target statistics must be scalar (mean/std of size 1)."
        )
    mean, std = float(mean_arr[0]), float(std_arr[0])

    target_norm = train_dataset.targets[timeframe]
    if target_norm.dim() == 2:
        target_norm = target_norm[:, -1]
    values = target_norm.double().numpy() * std + mean
    if values.shape[0] < 2:
        raise EvaluationError(
            f"{asset}/{variant}/{timeframe}: need at least 2 training targets to compute a MASE scale."
        )
    scale = float(np.mean(np.abs(np.diff(values))))
    if not np.isfinite(scale) or scale == 0.0:
        raise EvaluationError(f"{asset}/{variant}/{timeframe}: training MASE denominator is zero or invalid ({scale})")
    return scale


def _metrics(
    true_price: np.ndarray,
    pred_price: np.ndarray,
    true_delta: np.ndarray,
    pred_delta: np.ndarray,
    baseline: np.ndarray,
    mase_scale: float,
    asset: str,
    variant: str,
    timeframe: str,
) -> Dict[str, float]:
    arrays = [np.asarray(value, dtype=np.float64).reshape(-1) for value in (true_price, pred_price, true_delta, pred_delta, baseline)]
    if not arrays[0].size or any(value.size != arrays[0].size for value in arrays):
        raise EvaluationError(f"{asset}/{variant}/{timeframe}: metric arrays are empty or have mismatched shapes")
    if any(not np.isfinite(value).all() for value in arrays):
        raise EvaluationError(f"{asset}/{variant}/{timeframe}: metric arrays contain NaN or infinity")
    error = arrays[1] - arrays[0]
    denominator = np.abs(arrays[0]) + np.abs(arrays[1])
    smape = np.divide(2.0 * np.abs(error), denominator, out=np.zeros_like(error), where=denominator != 0.0)
    # Zero/zero is consistently counted as correct by sign equality; this is
    # the documented convention for directionless observations.
    actual_direction = np.sign(arrays[0] - arrays[4])
    predicted_direction = np.sign(arrays[1] - arrays[4])
    return {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "sMAPE": float(np.mean(smape) * 100.0),
        "MASE": float(np.mean(np.abs(error)) / mase_scale),
        "Directional_Accuracy": float(np.mean(actual_direction == predicted_direction) * 100.0),
    }


def _evaluate_one(
    test_payload: Dict[str, Any],
    train_payload: Dict[str, Any],
    checkpoint_root: Path,
    checkpoint_name: str,
    asset: str,
    variant: str,
    device: torch.device,
) -> list:
    label = f"{asset}/{variant}"
    checkpoint_path = checkpoint_root / asset / variant / checkpoint_name

    raw_checkpoint = _load_checkpoint(checkpoint_path, asset, variant)
    state_dict, checkpoint_timeframes, config = _checkpoint_state(raw_checkpoint, checkpoint_path, asset, variant)

    # The checkpoint's own hierarchy is the source of truth; both datasets
    # must actually contain it (HARNDataset enforces this, and also
    # cross-checks any self-declared `payload["timeframes"]`).
    test_dataset = _build_dataset(test_payload, checkpoint_timeframes, checkpoint_path, asset, variant, "test")
    train_dataset = _build_dataset(train_payload, checkpoint_timeframes, checkpoint_path, asset, variant, "train")

    model = HARN(config).to(device)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise EvaluationError(f"{label}: incompatible model_state_dict in {checkpoint_path}: {exc}") from exc

    streams = min(NUM_STREAMS, len(test_dataset))
    if streams < 1:
        raise EvaluationError(f"{label}: test dataset is empty")
    batcher = train_mod.StreamingBatcher(test_dataset, streams)
    model.set_feature_statistics(test_dataset.statistics)
    model.reset_states(streams, device)
    model.eval()

    timeframes = config.timeframes
    raw: Dict[str, Dict[str, list]] = {
        tf: {key: [] for key in ("true_price", "pred_price", "true_delta", "pred_delta", "baseline")}
        for tf in timeframes
    }
    with torch.no_grad():
        for batch in batcher:
            windows = {tf: batch[tf].to(device) for tf in timeframes}
            update_flags = {tf: batch[f"is_{tf}_update"].to(device) for tf in timeframes[1:]}
            targets = train_mod._prepare_targets(
                batch, test_dataset.statistics, device, timeframes, config.close_index, config.target_scale
            )
            output = model(windows, update_flags)
            for tf in timeframes:
                required = (f"pred_{tf}", f"baseline_{tf}")
                if any(key not in output for key in required):
                    raise EvaluationError(f"{label}/{tf}: model output is missing {required}")
                raw[tf]["true_price"].append(targets[f"absolute_{tf}"].cpu().numpy())
                raw[tf]["true_delta"].append(targets[f"delta_{tf}"].cpu().numpy())
                raw[tf]["pred_price"].append(output[f"pred_{tf}"].cpu().numpy())
                raw[tf]["pred_delta"].append(output[f"pred_delta_{tf}"].cpu().numpy())
                raw[tf]["baseline"].append(output[f"baseline_{tf}"].cpu().numpy())
            model.detach_states()

    rows: list = []
    for tf in timeframes:
        values = {key: np.concatenate(chunks).reshape(-1) for key, chunks in raw[tf].items()}
        metrics = _metrics(
            values["true_price"], values["pred_price"], values["true_delta"], values["pred_delta"],
            values["baseline"], _mase_scale(train_dataset, tf, asset, variant), asset, variant, tf,
        )
        rows.append({"asset": asset, "timeframe": tf, "variant": variant, **metrics})
    print(f"{label}: evaluated {len(timeframes)} timeframe(s) on {streams} stream(s)")
    return rows


def _parse_csv_list(raw: str, label: str) -> Tuple[str, ...]:
    items = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not items:
        raise ValueError(f"--{label} produced an empty list: {raw!r}")
    return items


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate all HARN ablations with chronological streaming state.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"),
                        help="Root directory with <data-dir>/<asset>/{train,test}.pkl.")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/HARN_Ablations"),
                        help="Root directory with <checkpoint-dir>/<asset>/<variant>/<checkpoint-name>.")
    parser.add_argument("--checkpoint-name", type=str, default=CHECKPOINT_NAME,
                        choices=["best_model.pt", "last_model.pt"],
                        help="Which checkpoint artifact to evaluate (default: best_model.pt, "
                             "the same checkpoint train_harn_ablations.py uses for its own "
                             "final test-set report).")
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/harn_ablation"))
    parser.add_argument("--assets", type=str, default=",".join(ASSETS),
                        help="Comma-separated asset names to evaluate.")
    parser.add_argument("--variants", type=str, default=",".join(VARIANTS),
                        help="Comma-separated ablation variants to evaluate.")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda", "mps"],
                        help="Override device auto-detection (default: cuda > mps > cpu).")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    device = train_mod.resolve_device(args.device)
    try:
        assets = _parse_csv_list(args.assets, "assets")
        variants = _parse_csv_list(args.variants, "variants")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    unknown_variants = [v for v in variants if v not in VARIANTS]
    if unknown_variants:
        raise SystemExit(f"Unknown variant(s) {unknown_variants}; must be a subset of {VARIANTS}.")

    print(f"Evaluating {len(assets) * len(variants)} HARN ablation experiments on {device}.")
    rows: list = []
    frame: Optional[pd.DataFrame] = None
    output_path = args.output_dir / "harn_ablation_results.csv"
    try:
        for asset in assets:
            # Load each asset's payloads once and reuse across every variant,
            # rather than re-reading from disk per-variant: guarantees every
            # variant is scored against the identical in-memory data and
            # avoids redundant I/O.
            test_payload = _load_pickle(args.data_dir / asset / "test.pkl", asset, "test")
            train_payload = _load_pickle(args.data_dir / asset / "train.pkl", asset, "train")
            for variant in variants:
                rows.extend(_evaluate_one(
                    test_payload, train_payload, args.checkpoint_dir, args.checkpoint_name,
                    asset, variant, device,
                ))

        # Cross-variant consistency check: every variant for a given asset
        # should report the same timeframe set. A mismatch means one
        # variant's checkpoint was trained on a different hierarchy than its
        # siblings -- surface that loudly instead of silently writing a
        # ragged results table.
        per_asset_variant_tfs: Dict[str, Dict[str, set]] = defaultdict(dict)
        for row in rows:
            per_asset_variant_tfs[row["asset"]].setdefault(row["variant"], set()).add(row["timeframe"])
        for asset, per_variant in per_asset_variant_tfs.items():
            tf_sets = list(per_variant.values())
            if any(s != tf_sets[0] for s in tf_sets[1:]):
                raise EvaluationError(
                    f"{asset}: inconsistent timeframe sets across variants: "
                    f"{ {v: sorted(s) for v, s in per_variant.items()} }"
                )

        frame = pd.DataFrame(rows, columns=RESULT_COLUMNS).sort_values(["asset", "variant", "timeframe"])
        args.output_dir.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_suffix(".tmp.csv")
        frame.to_csv(temporary_path, index=False)
        temporary_path.replace(output_path)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    for (asset, variant), group in frame.groupby(["asset", "variant"], sort=False):
        print(f"{asset}/{variant}: {len(group)} timeframe result(s)")
    print(f"Wrote {len(frame)} rows to {output_path.resolve()}")


if __name__ == "__main__":
    main()
