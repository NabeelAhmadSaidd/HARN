#!/usr/bin/env python3
"""
End-to-end training script for the Hierarchical Associative Resonance Network
(HARN).

Data Requirements
-----------------
The preprocessing pipeline now produces per‑symbol outputs in
``data/processed/<SYMBOL>/`` (e.g. ``data/processed/EURUSD/``).  Each split
(`train.pkl`, `val.pkl`, `test.pkl`) is a dictionary with keys:

* ``features``: dict of ``(N, T_tf, 5)`` windows keyed by timeframe name
* ``targets``: dict of ``(N, 1)`` target close prices keyed by timeframe name
* ``is_update``: dict of ``(N,)`` boolean update flags for every timeframe
* ``timeframes``: ordered list of timeframe strings (finest to coarsest)
* ``statistics``: normalization stats per timeframe and per target

The hierarchy is automatically read from the payload, so ``--timeframes`` is
optional.  If omitted, the payload's ``timeframes`` field is used.

Artifacts are saved under ``checkpoints/HARN/<asset>/``.

Usage
-----

    python -m src.models.train_harn --asset EURUSD
    python -m src.models.train_harn --asset EURUSD --timeframes M15,H1,H4   # override (must match payload)
    python -m src.models.train_harn --asset AAPL
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset
from tqdm.auto import tqdm


# Import the HARN model (must be in PYTHONPATH or same directory)
try:
    from .harn import HARN, ModelConfig
except ImportError:
    print("Error: harn.py not found. Please place it in the current directory.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training configuration."""
    parser = argparse.ArgumentParser(description="Train the HARN model.")
    parser.add_argument(
        "--data-dir", type=str, default="data/processed",
        help="Root directory containing per-symbol subdirectories "
             "(e.g. data/processed/EURUSD/)."
    )
    parser.add_argument(
        "--asset", type=str, required=True,
        help="Asset/symbol to train on (must exist as a subdirectory of --data-dir)."
    )
    parser.add_argument(
        "--timeframes", type=str, default=None,
        help="Optional comma-separated hierarchy level names from finest to coarsest. "
             "If omitted, the hierarchy is read from the pickle payload. "
             "Example: 'M15,H1,H4'."
    )
    parser.add_argument(
        "--batch-size", type=int, default=128,
        help="Number of parallel contiguous streams processed per training step."
    )
    parser.add_argument(
        "--eval-streams", type=int, default=8,
        help="Number of parallel contiguous streams used for stateful validation/test."
    )
    parser.add_argument(
        "--epochs", type=int, default=150,
        help="Maximum number of training epochs."
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Initial learning rate."
    )
    parser.add_argument(
        "--weight-decay", type=float, default=1e-5,
        help="AdamW weight decay."
    )
    parser.add_argument(
        "--grad-clip", type=float, default=1.0,
        help="Gradient clipping norm."
    )
    parser.add_argument(
        "--patience", type=int, default=15,
        help="Early stopping patience (epochs without improvement)."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)."
    )
    return parser.parse_args()


def _parse_timeframes(raw: str) -> Tuple[str, ...]:
    """Parse comma-separated level names into a tuple, validating uniqueness."""
    names = tuple(tf.strip() for tf in raw.split(",") if tf.strip())
    if not names:
        raise ValueError(f"--timeframes produced no hierarchy levels: {raw!r}")
    if len(set(names)) != len(names):
        raise ValueError(f"--timeframes must be unique level names, got {names}.")
    return names


# ---------------------------------------------------------------------------
# Dataset – fully generic, reads hierarchy from the payload
# ---------------------------------------------------------------------------
class HARNDataset(Dataset):
    """PyTorch Dataset wrapping a pickled HARN data dictionary.

    The hierarchy is read from ``data_dict["timeframes"]``.  Windows,
    targets, and update flags are stored in the nested ``features``,
    ``targets``, and ``is_update`` dictionaries.

    Parameters
    ----------
    data_dict : dict
        Loaded pickle payload containing ``features``, ``targets``,
        ``is_update``, ``timeframes``, and ``statistics``.
    timeframes : sequence of str, optional
        Override the hierarchy ordering found in the payload.  Must contain
        exactly the same names.  If ``None``, the payload's order is used.

    Raises
    ------
    KeyError
        If a required key is missing.
    ValueError
        If the provided ``timeframes`` does not match the payload's hierarchy.
    """

    def __init__(
        self,
        data_dict: Dict[str, Any],
        timeframes: Optional[Sequence[str]] = None,
    ) -> None:
        # Determine hierarchy
        payload_tfs = list(data_dict["timeframes"])
        if timeframes is not None:
            provided = list(timeframes)
            if set(provided) != set(payload_tfs):
                raise ValueError(
                    f"Provided timeframes {provided} do not match payload "
                    f"timeframes {payload_tfs}."
                )
            self.timeframes: Tuple[str, ...] = tuple(provided)
        else:
            self.timeframes: Tuple[str, ...] = tuple(payload_tfs)

        self.statistics = data_dict["statistics"]

        # Windows are in data_dict["features"]
        features = data_dict["features"]
        missing_windows = [tf for tf in self.timeframes if tf not in features]
        if missing_windows:
            raise KeyError(
                f"Payload is missing input windows for level(s): {missing_windows}"
            )
        self.windows: Dict[str, torch.Tensor] = {
            tf: torch.as_tensor(features[tf], dtype=torch.float32)
            for tf in self.timeframes
        }

        # Targets are in data_dict["targets"]
        targets = data_dict["targets"]
        missing_targets = [tf for tf in self.timeframes if tf not in targets]
        if missing_targets:
            raise KeyError(
                f"Payload is missing target(s) for level(s): {missing_targets}"
            )
        self.targets: Dict[str, torch.Tensor] = {
            tf: torch.as_tensor(targets[tf], dtype=torch.float32)
            for tf in self.timeframes
        }

        # Update flags exist for every timeframe (anchor included)
        update_flags = data_dict["is_update"]
        self.update_flags: Dict[str, torch.Tensor] = {}
        for tf in self.timeframes:
            if tf not in update_flags:
                raise KeyError(f"Payload is missing update flag for level '{tf}'.")
            self.update_flags[tf] = torch.as_tensor(update_flags[tf], dtype=torch.bool)

    def __len__(self) -> int:
        return len(self.windows[self.timeframes[0]])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return a dict with window, target, and update flag tensors for all levels.

        The model expects batch keys like ``tf`` (window) and
        ``f"is_{tf}_update"`` (update flag) for every level.
        """
        item: Dict[str, torch.Tensor] = {}
        for tf in self.timeframes:
            item[tf] = self.windows[tf][idx]
            item[f"target_{tf}"] = self.targets[tf][idx]
        for tf in self.timeframes[1:]:   # base level's flag is not used by model
            item[f"is_{tf}_update"] = self.update_flags[tf][idx]
        return item


# ---------------------------------------------------------------------------
# Target and data helpers
# ---------------------------------------------------------------------------
def _unpack_stats(
    stats_blob: Dict[str, Any], key: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract (mean, std) arrays from a statistics block, replacing zero std with 1."""
    block = stats_blob[key]
    mean = np.asarray(block["mean"], dtype=np.float32)
    std = np.asarray(block["std"], dtype=np.float32)
    std = np.where(std == 0.0, 1.0, std)
    return mean, std


def _close_from_normalized(
    values: torch.Tensor,
    stats: Tuple[np.ndarray, np.ndarray],
    device: torch.device,
    close_index: int,
) -> torch.Tensor:
    """Denormalize a close price using OHLCV window statistics."""
    mean, std = stats
    mean_t = torch.as_tensor(mean, dtype=torch.float32, device=device)
    std_t = torch.as_tensor(std, dtype=torch.float32, device=device)
    return values * std_t[close_index] + mean_t[close_index]


def _target_close_from_normalized(
    values: torch.Tensor,
    stats: Tuple[np.ndarray, np.ndarray],
    device: torch.device,
) -> torch.Tensor:
    """Denormalize a target close price using its dedicated statistics.

    In preprocessing, each target is normalized with its own scalar
    ``target_<level>`` statistics block, independent of the input-window
    OHLCV statistics. This function reverses that transformation.
    """
    mean, std = stats
    mean_t = torch.as_tensor(mean, dtype=torch.float32, device=device).view(-1)[0]
    std_t = torch.as_tensor(std, dtype=torch.float32, device=device).view(-1)[0]
    return values * std_t + mean_t


def _delta_target_bps(
    target_close: torch.Tensor, baseline_close: torch.Tensor, target_scale: float
) -> torch.Tensor:
    """Compute stationary basis-point change from baseline to target."""
    eps = torch.finfo(target_close.dtype).eps
    return ((target_close / torch.clamp(baseline_close, min=eps)) - 1.0) * target_scale


def _prepare_targets(
    batch: Dict[str, torch.Tensor],
    stats: Dict[str, Dict[str, Any]],
    device: torch.device,
    timeframes: Sequence[str],
    close_index: int,
    target_scale: float,
) -> Dict[str, torch.Tensor]:
    """Build stationary bps-delta and absolute-price targets for every level.

    For each hierarchy level:
    1. Extract the future target close (last element of the target window).
    2. Extract the baseline close (last bar of the input window).
    3. Denormalize both values to absolute price space using the appropriate
       statistics (window stats for the baseline, dedicated target stats for
       the target).
    4. Compute the delta in basis points.
    Returns tensors of shape ``(batch, 1)``.
    """
    feature_stats = {tf: _unpack_stats(stats, tf) for tf in timeframes}
    target_stats = {tf: _unpack_stats(stats, f"target_{tf}") for tf in timeframes}

    targets: Dict[str, torch.Tensor] = {}
    for tf in timeframes:
        target_close_norm = batch[f"target_{tf}"].to(device)

        # The target may be a full future window; take the last timestep.
        if target_close_norm.ndim == 2:
            target_close_norm = target_close_norm[:, -1]
        elif target_close_norm.ndim != 1:
            raise RuntimeError(
                f"Unexpected target shape {target_close_norm.shape} for '{tf}'"
            )

        baseline_norm = batch[tf][:, -1, close_index].to(device)  # (batch,)

        target_close = _target_close_from_normalized(
            target_close_norm, target_stats[tf], device
        )
        baseline_close = _close_from_normalized(
            baseline_norm, feature_stats[tf], device, close_index
        )

        delta = _delta_target_bps(target_close, baseline_close, target_scale)
        targets[f"delta_{tf}"] = delta.unsqueeze(-1)
        targets[f"absolute_{tf}"] = target_close.unsqueeze(-1)

    return targets


def _model_inputs_from_batch(
    batch: Dict[str, torch.Tensor],
    timeframes: Sequence[str],
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Split a raw batch into ``(windows, update_flags)`` for ``HARN.forward``."""
    windows = {tf: batch[tf].to(device) for tf in timeframes}
    update_flags = {tf: batch[f"is_{tf}_update"].to(device) for tf in timeframes[1:]}
    return windows, update_flags


# ---------------------------------------------------------------------------
# Streaming batcher
# ---------------------------------------------------------------------------
class StreamingBatcher:
    """Yields per-step batches that preserve temporal continuity.

    HARN maintains persistent memory across calls. For the model to
    experience realistic streaming dynamics, each position in the batch must
    correspond to a single contiguous time series advancing one step at a
    time. This batcher splits the chronological dataset into ``num_streams``
    non‑overlapping segments and, at step ``t``, returns row
    ``segment_start[s] + t`` for every stream ``s``. Thus the memory state
    carried for batch position ``s`` remains temporally consistent across
    the whole epoch.
    """

    def __init__(self, dataset: HARNDataset, num_streams: int) -> None:
        n = len(dataset)
        if num_streams < 1:
            raise ValueError("num_streams must be >= 1")
        seg_len = n // num_streams
        if seg_len < 1:
            raise ValueError(
                f"Dataset has only {n} samples, too few for {num_streams} streams."
            )
        self.dataset = dataset
        self.timeframes = dataset.timeframes
        self.num_streams = num_streams
        self.seg_len = seg_len
        self.starts = torch.arange(num_streams) * seg_len

    def __len__(self) -> int:
        return self.seg_len

    def __iter__(self):
        d = self.dataset
        for t in range(self.seg_len):
            idx = self.starts + t
            batch: Dict[str, torch.Tensor] = {}
            for tf in d.timeframes:
                batch[tf] = d.windows[tf][idx]
                batch[f"target_{tf}"] = d.targets[tf][idx]
            for tf in d.timeframes[1:]:
                batch[f"is_{tf}_update"] = d.update_flags[tf][idx]
            yield batch


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(
    data_root: str, asset: str, timeframes: Optional[Sequence[str]] = None
) -> Dict[str, HARNDataset]:
    """Load train/val/test datasets for a specific asset.

    Parameters
    ----------
    data_root : str
        Root directory containing per-symbol subdirectories (e.g.
        ``data/processed``).
    asset : str
        Symbol subdirectory name (e.g. ``EURUSD``).
    timeframes : sequence of str, optional
        If provided, must match the timeframe names found in the payload.
        If ``None``, the hierarchy is read from the payload.

    Returns
    -------
    dict
        Mapping ``{"train": ..., "val": ..., "test": ...}`` to HARNDataset
        instances.
    """
    data_dir = Path(data_root) / asset
    required = ["train.pkl", "val.pkl", "test.pkl"]
    for f in required:
        if not (data_dir / f).exists():
            raise FileNotFoundError(f"Missing {f} in {data_dir}")

    datasets = {}
    for split in ("train", "val", "test"):
        with open(data_dir / f"{split}.pkl", "rb") as fh:
            data = pickle.load(fh)
        # Pass user‑provided timeframes (or None to infer from payload)
        datasets[split] = HARNDataset(data, timeframes)
    return datasets


# ---------------------------------------------------------------------------
# Training & evaluation helpers
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(
    model: HARN,
    dataset: HARNDataset,
    device: torch.device,
    close_index: int,
    target_scale: float,
    num_streams: int = 8,
) -> Dict[str, float]:
    """Stateful evaluation using StreamingBatcher for temporal continuity.

    Runs the model in streaming mode over the full dataset, accumulating
    persistent memory state exactly as during online inference. Returns the
    average overall MSE and per-level MSE across all steps.
    """
    model.eval()
    model.set_feature_statistics(dataset.statistics)
    timeframes = dataset.timeframes

    streams = min(num_streams, len(dataset))
    batcher = StreamingBatcher(dataset, streams)
    model.reset_states(streams, device)

    total_mse = 0.0
    per_level_mse: Dict[str, float] = {tf: 0.0 for tf in timeframes}
    num_steps = len(batcher)

    for batch in batcher:
        windows, update_flags = _model_inputs_from_batch(batch, timeframes, device)
        targets = _prepare_targets(
            batch, dataset.statistics, device, timeframes, close_index, target_scale
        )

        preds = model(windows, update_flags)
        level_losses = {
            tf: F.mse_loss(preds[f"pred_delta_{tf}"], targets[f"delta_{tf}"])
            for tf in timeframes
        }
        loss = sum(level_losses.values())

        total_mse += loss.item()
        for tf in timeframes:
            per_level_mse[tf] += level_losses[tf].item()

    result: Dict[str, float] = {"loss": total_mse / num_steps}
    for tf in timeframes:
        result[tf] = per_level_mse[tf] / num_steps
    return result


def train_one_epoch(
    model: HARN,
    batcher: StreamingBatcher,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
    scaler: Optional[torch.cuda.amp.GradScaler],
    epoch: int,
    total_epochs: int,
    statistics: Dict[str, Dict[str, Any]],
    close_index: int,
    target_scale: float,
) -> Dict[str, float]:
    """Run one training epoch with truncated BPTT.

    Resets and detaches persistent states appropriately, computes per-level
    MSE losses, and performs gradient clipping. Returns average losses
    (overall and per-level) and the current learning rate.
    """
    model.train()
    model.reset_states(batcher.num_streams, device)
    model.set_feature_statistics(statistics)
    timeframes = batcher.timeframes

    total_loss = 0.0
    per_level_loss: Dict[str, float] = {tf: 0.0 for tf in timeframes}
    num_batches = len(batcher)
    start_time = time.time()

    pbar = tqdm(
        batcher,
        desc=f"Epoch {epoch:3d}/{total_epochs}",
        unit="step",
        leave=False,
        total=num_batches,
    )
    for batch_idx, batch in enumerate(pbar):
        windows, update_flags = _model_inputs_from_batch(batch, timeframes, device)
        targets = _prepare_targets(
            batch, statistics, device, timeframes, close_index, target_scale
        )

        optimizer.zero_grad()

        with torch.cuda.amp.autocast() if scaler else torch.enable_grad():
            preds = model(windows, update_flags)
            level_losses = {
                tf: F.mse_loss(preds[f"pred_delta_{tf}"], targets[f"delta_{tf}"])
                for tf in timeframes
            }
            loss = sum(level_losses.values())

        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        # Truncated BPTT: detach persistent states after each backward to
        # prevent backpropagation through the full sequence.
        model.detach_states()

        total_loss += loss.item()
        for tf in timeframes:
            per_level_loss[tf] += level_losses[tf].item()

        elapsed = time.time() - start_time
        samples_per_sec = (batch_idx + 1) * batcher.num_streams / max(elapsed, 1e-6)
        postfix = {
            "loss": f"{loss.item():.4f}",
            "avg_loss": f"{total_loss / (batch_idx + 1):.4f}",
        }
        postfix.update({tf: f"{level_losses[tf].item():.4f}" for tf in timeframes})
        postfix["lr"] = f"{optimizer.param_groups[0]['lr']:.2e}"
        postfix["samples/s"] = f"{samples_per_sec:.1f}"
        pbar.set_postfix(postfix)

    result: Dict[str, float] = {"loss": total_loss / num_batches}
    for tf in timeframes:
        result[tf] = per_level_loss[tf] / num_batches
    result["lr"] = optimizer.param_groups[0]["lr"]
    return result


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------
def main() -> None:
    """Main entry point: parse arguments, set up data, model, and run training loop.

    Saves the best model (by validation loss) and the last checkpoint, along
    with a CSV training history.
    """
    args = parse_args()

    # Optional timeframe override from command line
    user_timeframes = None
    if args.timeframes is not None:
        user_timeframes = _parse_timeframes(args.timeframes)

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load data – hierarchy is now read from the payload (unless overridden)
    datasets = load_data(args.data_dir, args.asset, user_timeframes)
    train_set = datasets["train"]
    val_set = datasets["val"]
    test_set = datasets["test"]

    # The actual timeframes (either from payload or validated override)
    timeframes = train_set.timeframes
    print(f"Asset: {args.asset}")
    print(f"Hierarchy levels ({len(timeframes)}): {timeframes}")

    # Structured checkpoint directory
    checkpoint_dir = Path("checkpoints") / "HARN" / args.asset
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # StreamingBatcher preserves temporal continuity across steps.
    train_batcher = StreamingBatcher(train_set, args.batch_size)

    # Model configuration – hierarchy defined solely by the timeframes
    config = ModelConfig(timeframes=timeframes)
    model = HARN(config).to(device)
    print(
        f"Model parameters: "
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    # Optimizer & scheduler
    optimizer = AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Mixed precision
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    # History tracking
    history: Dict[str, list] = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "lr": [],
    }
    for tf in timeframes:
        history[f"{tf}_mse"] = []

    best_model_path = checkpoint_dir / "best_model.pt"
    last_model_path = checkpoint_dir / "last_model.pt"
    history_path = checkpoint_dir / "training_history.csv"

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        # Train one epoch
        train_metrics = train_one_epoch(
            model,
            train_batcher,
            optimizer,
            device,
            grad_clip=args.grad_clip,
            scaler=scaler,
            epoch=epoch,
            total_epochs=args.epochs,
            statistics=train_set.statistics,
            close_index=config.close_index,
            target_scale=config.target_scale,
        )

        # Evaluate on validation set
        val_metrics = evaluate(
            model,
            val_set,
            device,
            close_index=config.close_index,
            target_scale=config.target_scale,
            num_streams=args.eval_streams,
        )

        scheduler.step()

        # Store history
        history["epoch"].append(epoch)
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["lr"].append(train_metrics["lr"])
        for tf in timeframes:
            history[f"{tf}_mse"].append(val_metrics[tf])

        # Console summary
        print(f"\nEpoch {epoch:3d}/{args.epochs}")
        print("-" * 42)
        print(f"  Train Loss : {train_metrics['loss']:.4f}")
        print(f"  Val Loss   : {val_metrics['loss']:.4f}")
        for tf in timeframes:
            label = tf.upper().ljust(10)
            print(f"  {label} MSE: {val_metrics[tf]:.4f}")
        print(f"  LR         : {train_metrics['lr']:.2e}")
        print(f"  Best Val   : {best_val_loss:.4f} (epoch {best_epoch})")
        print(f"  Time       : {time.strftime('%H:%M:%S')}")

        # Early stopping & checkpoint saving
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1

        # Always save the most recent checkpoint (for resuming)
        torch.save(
            {
                "epoch": epoch,
                "timeframes": timeframes,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
            },
            last_model_path,
        )

        if patience_counter >= args.patience:
            print(
                f"\nEarly stopping triggered after {args.patience} epochs "
                f"without improvement."
            )
            break

    # Save history
    pd.DataFrame(history).to_csv(history_path, index=False)

    # Final test evaluation using the best model
    print("\nLoading best model for test evaluation...")
    model.load_state_dict(torch.load(best_model_path))
    test_metrics = evaluate(
        model,
        test_set,
        device,
        close_index=config.close_index,
        target_scale=config.target_scale,
        num_streams=args.eval_streams,
    )
    print(f"\nTest Results (best model, epoch {best_epoch}):")
    print(f"  Test Loss  : {test_metrics['loss']:.4f}")
    for tf in timeframes:
        label = tf.upper().ljust(10)
        print(f"  {label} MSE: {test_metrics[tf]:.4f}")

    print(f"\nBest model saved to:\n{best_model_path.resolve()}")
    print(f"Last checkpoint saved to:\n{last_model_path.resolve()}")
    print(f"Training history saved to:\n{history_path.resolve()}")


if __name__ == "__main__":
    main()