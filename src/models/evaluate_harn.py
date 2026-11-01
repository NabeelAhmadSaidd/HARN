"""Evaluate HARN checkpoints with the notebook's stateful streaming procedure.

Example
-------
    python -m src.models.evaluate_harn
    python -m src.models.evaluate_harn --asset EURUSD --seed 42
"""

from __future__ import annotations

import argparse
import pathlib
import pickle
import sys
import warnings
from dataclasses import fields
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

try:
    from .harn import HARN, ModelConfig
    from . import train_harn
except ImportError:  # Support direct execution as well as ``python -m``.
    from src.models.harn import HARN, ModelConfig
    from src.models import train_harn


ASSETS = ("AAPL", "EURUSD", "USDCHF", "XAUUSD")
SEEDS = (42, 151, 359)
NUM_STREAMS = 1
METRIC_NAMES = ("MAE", "RMSE", "sMAPE", "MASE", "Directional_Accuracy")
PER_SEED_COLUMNS = ["asset", "seed", "timeframe", *METRIC_NAMES]
AGGREGATED_COLUMNS = [
    "asset",
    "timeframe",
    *[f"{metric}_{suffix}" for metric in METRIC_NAMES for suffix in ("mean", "std")],
]


def _load_pickle(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing processed payload: {path}")
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a dictionary payload in {path}")
    return payload


def _torch_load(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {path}")

    def load() -> Any:
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")

    try:
        return load()
    except NotImplementedError as exc:
        # Permit checkpoints containing POSIX paths to be read on Windows.
        if "PosixPath" not in str(exc) or not hasattr(pathlib, "WindowsPath"):
            raise
        original_posix_path = pathlib.PosixPath
        pathlib.PosixPath = pathlib.WindowsPath  # type: ignore[assignment]
        try:
            return load()
        finally:
            pathlib.PosixPath = original_posix_path  # type: ignore[assignment]


def _extract_state_dict(checkpoint: Any, path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Expected a dictionary checkpoint in {path}")
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        metadata = {key: value for key, value in checkpoint.items() if key != "model_state_dict"}
    else:
        state_dict = checkpoint
        metadata = {}
    if not isinstance(state_dict, dict) or not state_dict or not all(
        torch.is_tensor(value) for value in state_dict.values()
    ):
        raise ValueError(
            f"Unsupported checkpoint format in {path}; expected a raw state_dict "
            "or a dictionary containing model_state_dict."
        )
    return state_dict, metadata


def _validate_payload(payload: dict[str, Any], path: Path) -> tuple[str, ...]:
    required = {"features", "targets", "is_update", "timeframes", "statistics"}
    missing = sorted(required.difference(payload))
    if missing:
        raise KeyError(f"{path} is missing required payload keys: {missing}")
    if not isinstance(payload["timeframes"], (list, tuple)) or not payload["timeframes"]:
        raise ValueError(f"{path} has an invalid non-empty 'timeframes' sequence")
    timeframes = tuple(str(tf) for tf in payload["timeframes"])
    if len(set(timeframes)) != len(timeframes):
        raise ValueError(f"{path} contains duplicate timeframes: {timeframes}")
    if not isinstance(payload["statistics"], dict):
        raise ValueError(f"{path} has an invalid 'statistics' dictionary")

    features = payload["features"]
    targets = payload["targets"]
    flags = payload["is_update"]
    for tf in timeframes:
        for name, container in (("features", features), ("targets", targets), ("is_update", flags)):
            if not isinstance(container, dict) or tf not in container:
                raise KeyError(f"{path} is missing {name}[{tf!r}]")
        if not isinstance(features[tf], np.ndarray) or features[tf].ndim != 3:
            raise ValueError(f"{path} features[{tf!r}] must have shape (N, window, features)")
        if features[tf].shape[0] == 0:
            raise ValueError(f"{path} has no samples for timeframe {tf!r}")
        if not isinstance(targets[tf], np.ndarray) or targets[tf].shape[0] != features[tf].shape[0]:
            raise ValueError(f"{path} targets[{tf!r}] length does not match features")
        if not isinstance(flags[tf], np.ndarray) or flags[tf].shape[0] != features[tf].shape[0]:
            raise ValueError(f"{path} is_update[{tf!r}] length does not match features")
        for stats_key in (tf, f"target_{tf}"):
            block = payload["statistics"].get(stats_key)
            if not isinstance(block, dict) or "mean" not in block or "std" not in block:
                raise KeyError(f"{path} statistics is missing a valid {stats_key!r} block")
    sample_count = features[timeframes[0]].shape[0]
    if any(features[tf].shape[0] != sample_count for tf in timeframes):
        raise ValueError(f"{path} feature sample counts differ across timeframes")
    return timeframes


def _validate_consistency(
    metadata: dict[str, Any], asset: str, seed: int, timeframes: tuple[str, ...], path: Path
) -> None:
    problems: list[str] = []
    checkpoint_asset = metadata.get("asset", metadata.get("symbol"))
    if checkpoint_asset is not None and str(checkpoint_asset) != asset:
        problems.append(f"checkpoint asset {checkpoint_asset!r} != {asset!r}")
    checkpoint_seed = metadata.get("seed")
    if checkpoint_seed is not None and int(checkpoint_seed) != seed:
        problems.append(f"checkpoint seed {checkpoint_seed!r} != {seed!r}")
    checkpoint_timeframes = metadata.get("timeframes")
    if checkpoint_timeframes is not None:
        checkpoint_timeframes = tuple(str(tf) for tf in checkpoint_timeframes)
        if checkpoint_timeframes != timeframes:
            problems.append(f"checkpoint timeframes {checkpoint_timeframes} != {timeframes}")
    if problems:
        raise ValueError(f"Checkpoint/data consistency failed for {path}: " + "; ".join(problems))


def _state_shape(state_dict: dict[str, torch.Tensor], key: str, path: Path) -> torch.Tensor:
    value = state_dict.get(key)
    if value is None:
        raise ValueError(f"Checkpoint {path} is missing required tensor {key!r}")
    return value


def infer_model_config_from_state_dict(
    state_dict: dict[str, torch.Tensor],
    timeframes: Sequence[str],
    metadata: dict[str, Any] | None = None,
    checkpoint_path: Path | None = None,
) -> ModelConfig:
    """Reconstruct the notebook's ModelConfig from checkpoint tensors."""
    metadata = metadata or {}
    names = tuple(str(tf) for tf in timeframes)
    default = ModelConfig(timeframes=names)
    config = ModelConfig(
        timeframes=names,
        dropout=float(metadata.get("dropout", default.dropout)),
        history_len=int(metadata.get("history_len", default.history_len)),
        close_index=int(metadata.get("close_index", default.close_index)),
    )
    prefix = str(checkpoint_path) if checkpoint_path else "checkpoint"
    input_proj = _state_shape(state_dict, "encoders.0.input_proj.weight", Path(prefix))
    config.input_dim = int(input_proj.shape[1])
    config.d_model = int(input_proj.shape[0])
    config.kernel_size = int(_state_shape(state_dict, "encoders.0.branches.0.depthwise.weight", Path(prefix)).shape[-1])

    branch_count = sum(
        key.startswith("encoders.0.branches.") and key.endswith(".depthwise.weight")
        for key in state_dict
    )
    if branch_count < 1:
        raise ValueError(f"Checkpoint {prefix} has no convolution branches")
    metadata_dilations = metadata.get("dilations")
    if metadata_dilations is not None and len(metadata_dilations) == branch_count:
        config.dilations = tuple(int(value) for value in metadata_dilations)
    elif branch_count == len(default.dilations):
        config.dilations = default.dilations
    else:
        config.dilations = tuple(2**index for index in range(branch_count))
        warnings.warn(
            "Dilation values are not recoverable from this state_dict; using "
            f"fallback {config.dilations} inferred from branch count={branch_count}.",
            RuntimeWarning,
        )

    config.mem_key_dim = int(_state_shape(state_dict, "cells.0.to_key.weight", Path(prefix)).shape[0])
    config.state_dim = int(_state_shape(state_dict, "cells.0.to_value.weight", Path(prefix)).shape[0])
    config.coupling_rank = int(_state_shape(state_dict, "resonance.q_proj.weight", Path(prefix)).shape[0])
    config.head_hidden = int(_state_shape(state_dict, "heads.0.net.0.weight", Path(prefix)).shape[0])
    if "target_scale" in state_dict:
        config.target_scale = float(state_dict["target_scale"].item())
    elif "target_scale" in metadata:
        config.target_scale = float(metadata["target_scale"])
    return config


def _resolve_checkpoint(root: Path, asset: str, seed: int, name: str) -> Path:
    candidates = [root / asset / str(seed) / name]
    if root.name.lower() == "harn":
        candidates.append(root.with_name("HARN") / asset / str(seed) / name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Missing checkpoint for asset={asset!r}, seed={seed}: "
        + " or ".join(str(candidate) for candidate in candidates)
    )


def _denormalize_targets(values: np.ndarray, stats: dict[str, Any], timeframe: str) -> np.ndarray:
    block = stats.get(f"target_{timeframe}")
    if not isinstance(block, dict):
        raise KeyError(f"Missing target statistics for {timeframe!r}")
    mean = np.asarray(block["mean"], dtype=np.float64).reshape(-1)
    std = np.asarray(block["std"], dtype=np.float64).reshape(-1)
    if mean.size != 1 or std.size != 1:
        raise ValueError(f"Target statistics for {timeframe!r} must be scalar")
    return np.asarray(values, dtype=np.float64).reshape(-1) * std[0] + mean[0]


def _mase_scale(train_payload: dict[str, Any], timeframe: str) -> float:
    values = _denormalize_targets(train_payload["targets"][timeframe], train_payload["statistics"], timeframe)
    if values.size < 2:
        raise ValueError(f"At least two training targets are required for MASE ({timeframe})")
    scale = float(np.mean(np.abs(np.diff(values))))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Training naive MASE scale for {timeframe!r} is invalid: {scale}")
    return scale


def compute_metrics(
    true_price: np.ndarray,
    pred_price: np.ndarray,
    true_delta_bps: np.ndarray,
    pred_delta_bps: np.ndarray,
    mase_scale: float,
) -> dict[str, float]:
    """Compute the requested absolute-price metrics and bps directional accuracy."""
    y_true = np.asarray(true_price, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(pred_price, dtype=np.float64).reshape(-1)
    true_delta = np.asarray(true_delta_bps, dtype=np.float64).reshape(-1)
    pred_delta = np.asarray(pred_delta_bps, dtype=np.float64).reshape(-1)
    if not y_true.size or not (y_true.size == y_pred.size == true_delta.size == pred_delta.size):
        raise ValueError("Metric inputs must be non-empty arrays with equal lengths")
    if not all(np.isfinite(values).all() for values in (y_true, y_pred, true_delta, pred_delta)):
        raise ValueError("Metric inputs contain NaN or infinite values")
    if not np.isfinite(mase_scale) or mase_scale <= 0:
        raise ValueError(f"MASE scale must be finite and positive, got {mase_scale}")
    error = y_pred - y_true
    denominator = np.abs(y_true) + np.abs(y_pred)
    smape = np.divide(2 * np.abs(error), denominator, out=np.zeros_like(error), where=denominator != 0)
    return {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "sMAPE": float(np.mean(smape) * 100),
        "MASE": float(np.mean(np.abs(error)) / mase_scale),
        "Directional_Accuracy": float(np.mean(np.sign(true_delta) == np.sign(pred_delta)) * 100),
    }


def evaluate_one(
    data_dir: Path,
    checkpoint_dir: Path,
    asset: str,
    seed: int,
    checkpoint_name: str,
    device: torch.device,
) -> list[dict[str, Any]]:
    test_path = data_dir / asset / "test.pkl"
    train_path = data_dir / asset / "train.pkl"
    test_payload = _load_pickle(test_path)
    train_payload = _load_pickle(train_path)
    timeframes = _validate_payload(test_payload, test_path)
    train_timeframes = _validate_payload(train_payload, train_path)
    if train_timeframes != timeframes:
        raise ValueError(f"Train/test timeframes differ for {asset}: {train_timeframes} != {timeframes}")

    checkpoint_path = _resolve_checkpoint(checkpoint_dir, asset, seed, checkpoint_name)
    checkpoint = _torch_load(checkpoint_path)
    state_dict, metadata = _extract_state_dict(checkpoint, checkpoint_path)
    _validate_consistency(metadata, asset, seed, timeframes, checkpoint_path)
    config = infer_model_config_from_state_dict(state_dict, timeframes, metadata, checkpoint_path)

    dataset = train_harn.HARNDataset(test_payload, timeframes=timeframes)
    batcher = train_harn.StreamingBatcher(dataset, NUM_STREAMS)
    model = HARN(config).to(device)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise ValueError(f"Could not strictly load {checkpoint_path}: {exc}") from exc
    model.eval()
    model.set_feature_statistics(test_payload["statistics"])
    model.reset_states(NUM_STREAMS, device)

    raw: dict[str, dict[str, list[np.ndarray]]] = {
        tf: {key: [] for key in ("pred_delta", "true_delta", "pred_price", "true_price")}
        for tf in timeframes
    }
    with torch.no_grad():
        for batch in batcher:
            windows = {tf: batch[tf].to(device) for tf in timeframes}
            update_flags = {
                tf: batch[f"is_{tf}_update"].to(device) for tf in timeframes[1:]
            }
            targets = train_harn._prepare_targets(
                batch,
                dataset.statistics,
                device,
                timeframes,
                config.close_index,
                config.target_scale,
            )
            output = model(windows, update_flags)
            for tf in timeframes:
                raw[tf]["pred_delta"].append(output[f"pred_delta_{tf}"].cpu().numpy().reshape(-1))
                raw[tf]["true_delta"].append(targets[f"delta_{tf}"].cpu().numpy().reshape(-1))
                raw[tf]["pred_price"].append(output[f"pred_{tf}"].cpu().numpy().reshape(-1))
                raw[tf]["true_price"].append(targets[f"absolute_{tf}"].cpu().numpy().reshape(-1))
            model.detach_states()

    rows: list[dict[str, Any]] = []
    for tf in timeframes:
        result = {key: np.concatenate(values) for key, values in raw[tf].items()}
        metrics = compute_metrics(
            result["true_price"],
            result["pred_price"],
            result["true_delta"],
            result["pred_delta"],
            _mase_scale(train_payload, tf),
        )
        rows.append({"asset": asset, "seed": seed, "timeframe": tf, **metrics})
    return rows


def aggregate(rows: Iterable[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(list(rows))
    if frame.empty:
        return pd.DataFrame(columns=AGGREGATED_COLUMNS)
    output: list[dict[str, Any]] = []
    for (asset, timeframe), group in frame.groupby(["asset", "timeframe"], sort=True):
        row: dict[str, Any] = {"asset": asset, "timeframe": timeframe}
        for metric in METRIC_NAMES:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=0))
        output.append(row)
    return pd.DataFrame(output, columns=AGGREGATED_COLUMNS)


def _selected(value: str | None, choices: tuple[str, ...]) -> tuple[str, ...]:
    return choices if value is None or value == "all" else (value,)


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if value not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"Unsupported device {value!r}; choose auto, cpu, cuda, or mps")
    return torch.device(value)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate HARN checkpoints with continuous streaming state.")
    parser.add_argument("--asset", choices=(*ASSETS, "all"), default="all")
    parser.add_argument("--seed", type=int, action="append", choices=SEEDS, default=None)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/harn"))
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/harn"))
    parser.add_argument("--checkpoint-name", default="best_model.pt")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    assets = _selected(args.asset, ASSETS)
    seeds = tuple(args.seed) if args.seed is not None else SEEDS
    device = _resolve_device(args.device)
    combinations = [(asset, seed) for asset in assets for seed in seeds]
    print(f"Evaluating {len(combinations)} HARN checkpoints on {device} with NUM_STREAMS={NUM_STREAMS}.")

    rows: list[dict[str, Any]] = []
    for asset, seed in tqdm(combinations, desc="HARN evaluations", unit="run"):
        rows.extend(evaluate_one(args.data_dir, args.checkpoint_dir, asset, seed, args.checkpoint_name, device))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_seed = pd.DataFrame(rows, columns=PER_SEED_COLUMNS).sort_values(
        ["asset", "timeframe", "seed"]
    )
    aggregated = aggregate(rows)
    per_seed_path = args.output_dir / "harn_metrics_per_seed.csv"
    aggregated_path = args.output_dir / "harn_metrics_aggregated.csv"
    per_seed.to_csv(per_seed_path, index=False)
    aggregated.to_csv(aggregated_path, index=False)
    print(f"Wrote {len(per_seed)} rows to {per_seed_path.resolve()}")
    print(f"Wrote {len(aggregated)} rows to {aggregated_path.resolve()}")


if __name__ == "__main__":
    main()
