"""Evaluate trained PatchTST and TimeXer baseline checkpoints.

The evaluator consumes the pickle payloads produced by
``src.datasets.preprocessing`` and the wrapped checkpoints produced by
``train_baselines.py``.  It evaluates the next anchor-timeframe close
(``horizon=1``) in absolute-price space after reversing the stored target
normalization.

Example
-------
    python -m src.models.baselines.evaluate_baselines
    python -m src.models.baselines.evaluate_baselines --asset EURUSD --model PatchTST
"""

from __future__ import annotations

import argparse
import pickle
import pathlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

try:
    from .train_baselines import (
        INPUT_SIZE,
        MODEL_NAMES,
        TARGET_INDEX,
        build_model,
        load_anchor_splits,
        resolve_device,
    )
except ImportError:  # Support running this file directly as well as with -m.
    from src.models.baselines.train_baselines import (
        INPUT_SIZE,
        MODEL_NAMES,
        TARGET_INDEX,
        build_model,
        load_anchor_splits,
        resolve_device,
    )


ASSETS = ("AAPL", "EURUSD", "USDCHF", "XAUUSD")
SEEDS = (42, 151, 359)
SPLIT_NAMES = ("train", "val", "test")
METRIC_NAMES = ("MAE", "RMSE", "sMAPE", "MASE", "Directional_Accuracy")


def _load_payload(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing processed split file: {path}")
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a dictionary payload in {path}")
    return payload


def _torch_load(path: Path, device: torch.device) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    def load() -> Any:
        try:
            return torch.load(path, map_location=device, weights_only=False)
        except TypeError:  # Older PyTorch versions do not accept weights_only.
            return torch.load(path, map_location=device)

    try:
        checkpoint = load()
    except NotImplementedError as exc:
        # Checkpoints created on POSIX can contain pathlib.PosixPath objects
        # in argparse metadata. Python refuses to instantiate PosixPath on
        # Windows, so temporarily map that serialized class to WindowsPath.
        if "PosixPath" not in str(exc) or not hasattr(pathlib, "WindowsPath"):
            raise
        original_posix_path = pathlib.PosixPath
        pathlib.PosixPath = pathlib.WindowsPath  # type: ignore[assignment]
        try:
            checkpoint = load()
        finally:
            pathlib.PosixPath = original_posix_path  # type: ignore[assignment]
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Expected a dictionary checkpoint in {path}")
    return checkpoint


def _stats_from_payload(payload: dict[str, Any], anchor_tf: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return feature and dedicated-target mean/std for the anchor timeframe."""
    statistics = payload.get("statistics")
    if not isinstance(statistics, dict):
        raise ValueError("Processed payload is missing its 'statistics' dictionary.")
    try:
        feature_stats = statistics[anchor_tf]
        target_stats = statistics[f"target_{anchor_tf}"]
        feature_mean = np.asarray(feature_stats["mean"], dtype=np.float64)
        feature_std = np.asarray(feature_stats["std"], dtype=np.float64)
        target_mean = np.asarray(target_stats["mean"], dtype=np.float64)
        target_std = np.asarray(target_stats["std"], dtype=np.float64)
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"Processed payload lacks feature/target statistics for anchor '{anchor_tf}'."
        ) from exc
    feature_std = np.where(feature_std == 0.0, 1.0, feature_std)
    target_std = np.where(target_std == 0.0, 1.0, target_std)
    if feature_mean.shape != (INPUT_SIZE,) or feature_std.shape != (INPUT_SIZE,):
        raise ValueError(f"Feature statistics for '{anchor_tf}' must contain {INPUT_SIZE} values.")
    if target_mean.size != 1 or target_std.size != 1:
        raise ValueError(f"Target statistics for '{anchor_tf}' must be scalar.")
    return feature_mean, feature_std, target_mean.reshape(-1), target_std.reshape(-1)


def _denormalize_target(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return values.astype(np.float64, copy=False) * std[0] + mean[0]


def _checkpoint_args(checkpoint: dict[str, Any], model_name: str, path: Path) -> SimpleNamespace:
    raw_args = checkpoint.get("args")
    if not isinstance(raw_args, dict):
        raise ValueError(f"Checkpoint {path} has no usable training 'args' metadata.")
    args = dict(raw_args)
    checkpoint_model = checkpoint.get("model", args.get("model"))
    if checkpoint_model != model_name:
        raise ValueError(
            f"Checkpoint {path} is for model {checkpoint_model!r}, requested {model_name!r}."
        )
    if args.get("horizon", 1) != 1:
        raise ValueError(f"Checkpoint {path} was trained with horizon={args.get('horizon')}; expected 1.")
    args["model"] = model_name
    args["window_size"] = checkpoint.get("window_size", args.get("window_size"))
    required = ("patch_len", "d_model", "d_ff", "e_layers", "n_heads", "dropout", "target_idx")
    missing = [key for key in required if key not in args or args[key] is None]
    if args["window_size"] is None:
        missing.append("window_size")
    if missing:
        raise ValueError(f"Checkpoint {path} is missing architecture metadata: {', '.join(missing)}")
    if args["target_idx"] != TARGET_INDEX:
        raise ValueError(
            f"Checkpoint {path} uses target_idx={args['target_idx']}; expected close index {TARGET_INDEX}."
        )
    return SimpleNamespace(**args)


def _load_model(
    checkpoint_path: Path,
    model_name: str,
    anchor_tf: str,
    window_size: int,
    device: torch.device,
) -> nn.Module:
    checkpoint = _torch_load(checkpoint_path, device)
    checkpoint_anchor = checkpoint.get("anchor_timeframe")
    if checkpoint_anchor != anchor_tf:
        raise ValueError(
            f"Checkpoint {checkpoint_path} uses anchor timeframe {checkpoint_anchor!r}; "
            f"processed data uses {anchor_tf!r}."
        )
    checkpoint_window = checkpoint.get("window_size")
    if checkpoint_window != window_size:
        raise ValueError(
            f"Checkpoint {checkpoint_path} uses window_size={checkpoint_window}; "
            f"processed data uses {window_size}."
        )
    args = _checkpoint_args(checkpoint, model_name, checkpoint_path)
    if args.window_size != window_size:
        raise ValueError(
            f"Checkpoint args in {checkpoint_path} use window_size={args.window_size}; "
            f"processed data uses {window_size}."
        )
    model = build_model(args, window_size).to(device)
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError(f"Checkpoint {checkpoint_path} has no model_state_dict.")
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        stripped = {
            key.removeprefix("module."): value for key, value in state_dict.items()
        }
        try:
            model.load_state_dict(stripped)
        except RuntimeError as exc:
            raise ValueError(f"Could not load model weights from {checkpoint_path}: {exc}") from exc
    model.eval()
    return model


def infer(model: nn.Module, features: np.ndarray, batch_size: int, device: torch.device, label: str) -> np.ndarray:
    dataset = TensorDataset(torch.from_numpy(features))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=device.type == "cuda")
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for (batch_features,) in tqdm(loader, desc=f"Inference {label}", unit="batch", leave=False):
            output = model(batch_features.to(device, non_blocking=device.type == "cuda"))
            if output.ndim != 2 or output.shape[1] != 1:
                raise RuntimeError(f"{label} returned shape {tuple(output.shape)}, expected (batch, 1).")
            predictions.append(output.detach().cpu().numpy())
    return np.concatenate(predictions, axis=0).reshape(-1)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    baselines: np.ndarray,
    mase_scale: float,
) -> dict[str, float]:
    """Compute requested metrics in absolute-price space."""
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    baselines = np.asarray(baselines, dtype=np.float64).reshape(-1)
    if not (y_true.size == y_pred.size == baselines.size) or y_true.size == 0:
        raise ValueError("Metric inputs must be non-empty arrays with equal lengths.")
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all() or not np.isfinite(baselines).all():
        raise ValueError("Metric inputs contain NaN or infinite values.")
    if not np.isfinite(mase_scale) or mase_scale <= 0.0:
        raise ValueError(f"MASE scale must be finite and positive, got {mase_scale}.")

    error = y_pred - y_true
    denominator = np.abs(y_true) + np.abs(y_pred)
    smape_terms = np.divide(2.0 * np.abs(error), denominator, out=np.zeros_like(error), where=denominator != 0.0)
    true_direction = np.sign(y_true - baselines)
    pred_direction = np.sign(y_pred - baselines)
    return {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "sMAPE": float(np.mean(smape_terms) * 100.0),
        "MASE": float(np.mean(np.abs(error)) / mase_scale),
        "Directional_Accuracy": float(np.mean(true_direction == pred_direction) * 100.0),
    }


def _mase_scale(training_targets: np.ndarray, target_mean: np.ndarray, target_std: np.ndarray) -> float:
    values = _denormalize_target(training_targets.reshape(-1), target_mean, target_std)
    if values.size < 2:
        raise ValueError("At least two training targets are required to compute MASE.")
    scale = float(np.mean(np.abs(np.diff(values))))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"Training naive scale for MASE must be positive, got {scale}.")
    return scale


def evaluate_one(
    data_dir: Path,
    checkpoint_dir: Path,
    asset: str,
    model_name: str,
    seed: int,
    batch_size: int,
    device: torch.device,
    checkpoint_name: str,
) -> dict[str, Any]:
    splits, anchor_tf, window_size = load_anchor_splits(data_dir, asset)
    train_payload = _load_payload(data_dir / asset / "train.pkl")
    payload_anchor = train_payload.get("anchor_timeframe")
    if payload_anchor != anchor_tf:
        raise ValueError(
            f"Train payload anchor timeframe {payload_anchor!r} differs from loaded splits {anchor_tf!r}."
        )
    feature_mean, feature_std, target_mean, target_std = _stats_from_payload(train_payload, anchor_tf)
    train_feature, train_target = splits["train"]
    test_features, test_targets = splits["test"]
    if train_feature.shape[1:] != test_features.shape[1:]:
        raise ValueError(f"Train/test feature shapes differ for {asset}: {train_feature.shape} vs {test_features.shape}.")

    checkpoint_path = checkpoint_dir / model_name / asset / str(seed) / checkpoint_name
    model = _load_model(checkpoint_path, model_name, anchor_tf, window_size, device)
    predictions_norm = infer(model, test_features, batch_size, device, f"{model_name}/{asset}/{seed}")
    targets_norm = test_targets.reshape(-1)
    y_true = _denormalize_target(targets_norm, target_mean, target_std)
    y_pred = _denormalize_target(predictions_norm, target_mean, target_std)
    baselines = test_features[:, -1, TARGET_INDEX].astype(np.float64) * feature_std[TARGET_INDEX] + feature_mean[TARGET_INDEX]
    metrics = compute_metrics(y_true, y_pred, baselines, _mase_scale(train_target, target_mean, target_std))
    return {"asset": asset, "model": model_name, "seed": seed, **metrics}


def _format_metric(mean: float, std: float) -> str:
    return f"{mean:.8g} ± {std:.8g}"


def aggregate(rows: Iterable[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(list(rows))
    if frame.empty:
        return pd.DataFrame(columns=["asset", "model"] + [f"{metric}_{suffix}" for metric in METRIC_NAMES for suffix in ("mean", "std")])
    grouped = frame.groupby(["asset", "model"], sort=True)
    output: list[dict[str, Any]] = []
    for (asset, model), group in grouped:
        row: dict[str, Any] = {"asset": asset, "model": model}
        for metric in METRIC_NAMES:
            mean = float(group[metric].mean())
            std = float(group[metric].std(ddof=0))
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[metric] = _format_metric(mean, std)
        output.append(row)
    columns = ["asset", "model"]
    columns += [f"{metric}_{suffix}" for metric in METRIC_NAMES for suffix in ("mean", "std")]
    columns += list(METRIC_NAMES)
    return pd.DataFrame(output)[columns]


def _selected(value: str | None, choices: tuple[str, ...]) -> tuple[str, ...]:
    return choices if value is None or value == "all" else (value,)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained PatchTST and TimeXer baseline checkpoints.")
    parser.add_argument("--asset", choices=(*ASSETS, "all"), default="all")
    parser.add_argument("--model", choices=(*MODEL_NAMES, "all"), default="all")
    parser.add_argument("--seed", type=int, action="append", choices=SEEDS, default=None, help="Repeat to select specific seeds; default evaluates all.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/Baselines"))
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/baselines"))
    parser.add_argument("--checkpoint-name", default="best_model.pt")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive.")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    device = resolve_device(args.device)
    assets = _selected(args.asset, ASSETS)
    models = _selected(args.model, MODEL_NAMES)
    seeds = tuple(args.seed) if args.seed is not None else SEEDS
    print(f"Evaluating {len(assets) * len(models) * len(seeds)} baseline checkpoints on {device}.")

    rows: list[dict[str, Any]] = []
    combinations = [(asset, model, seed) for asset in assets for model in models for seed in seeds]
    for asset, model, seed in tqdm(combinations, desc="Baseline evaluations", unit="run"):
        rows.append(evaluate_one(args.data_dir, args.checkpoint_dir, asset, model, seed, args.batch_size, device, args.checkpoint_name))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_seed_columns = ["asset", "model", "seed", *METRIC_NAMES]
    per_seed = pd.DataFrame(rows)[per_seed_columns].sort_values(["asset", "model", "seed"])
    per_seed_path = args.output_dir / "baseline_metrics_per_seed.csv"
    aggregate_path = args.output_dir / "baseline_metrics_aggregated.csv"
    per_seed.to_csv(per_seed_path, index=False)
    aggregate(rows).to_csv(aggregate_path, index=False)
    print(f"Wrote {len(per_seed)} rows to {per_seed_path.resolve()}")
    print(f"Wrote {len(aggregate(rows))} rows to {aggregate_path.resolve()}")


if __name__ == "__main__":
    main()
