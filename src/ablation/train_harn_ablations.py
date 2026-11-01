#!/usr/bin/env python3
"""
End-to-end training script for component-wise ablation experiments on the
Hierarchical Associative Resonance Network (HARN).

This is the ablation-aware counterpart of ``train_harn.py``: it trains the
exact same architecture, dataset pipeline, preprocessing, hyperparameters,
optimizer, scheduler, and random seed, with the only variable being which
HARN component (if any) is ablated via ``--variant``:

    A0  Full HARN (unchanged).
    A1  Surprise-Gated Associative Memory removed.
    A2  Cross-Scale Resonance removed.
    A3  Bottom-Up Evidence Aggregation removed.

See ``harn_ablations.py`` for the component-level implementation of each
variant.

Data Requirements
-----------------
Preprocessed files ``train.pkl``, ``val.pkl``, ``test.pkl`` under
``<data-dir>/<asset>/`` (e.g. ``data/processed/EURUSD/train.pkl``), each a
pickled dict containing, for every hierarchy level ``tf`` in
``--timeframes``:

* ``data[tf]``: (N, T_tf, input_dim) input windows
* ``data[f"target_{tf}"]``: (N,) or (N, K) future close price(s)
* ``data[f"is_{tf}_update"]``: (N,) boolean update flags for every level
  except the base (finest) one, which always updates
* ``data["statistics"]``: dict with a ``{"mean": [...], "std": [...]}``
  block for every ``tf`` (window statistics) and every ``target_{tf}``
  (target statistics)
* ``data["timeframes"]`` (optional but recommended): the ordered level
  names this payload was built for. When present, it is cross-checked
  against ``--timeframes`` so a mismatched/misplaced data file is rejected
  immediately instead of silently reordering or subsetting levels.

This exact schema is also what ``evaluate_harn_ablations.py`` expects: both
scripts share the same ``HARNDataset`` loader (see below), so there is only
one data adapter to keep correct.

The hierarchy is defined entirely by the comma-separated ``--timeframes``
list (e.g. ``m15,h1,h4`` or ``m1,m5,m15,h1,h4,d1``). Every stage of the
script — dataset loading, streaming batching, target preparation,
evaluation, training, logging and checkpointing — adapts dynamically to the
number of configured levels, and is identical regardless of ``--variant``.

Artifacts (``best_model.pt``, ``last_model.pt``, training history, and a
text log) are saved under ``checkpoints/HARN_Ablations/<asset>/<variant>/``,
e.g.::

    checkpoints/
    └── HARN_Ablations/
        └── EURUSD/
            ├── A0/
            ├── A1/
            ├── A2/
            └── A3/

Both checkpoint files are saved as the *same* wrapped dict shape
(``epoch``, ``timeframes``, ``variant``, ``config``, ``model_state_dict``,
``best_val_loss``, ...; ``last_model.pt`` additionally carries optimizer/
scheduler state for resuming). Storing the full training-time ``ModelConfig``
inside the checkpoint means evaluation always rebuilds the *exact*
architecture that produced the weights, even if this module's defaults
change later — see ``config_from_checkpoint``.

Usage
-----
    python -m src.ablation.train_harn_ablations --variant A0
    python -m src.ablation.train_harn_ablations --variant A1
    python -m src.ablation.train_harn_ablations --variant A2
    python -m src.ablation.train_harn_ablations --variant A3
    python -m src.ablation.train_harn_ablations --variant A1 --timeframes m15,h1,h4
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import pathlib
import pickle
import random
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

try:
    from ..datasets.timeframes import canonical_key_map, resolve_timeframes
except ImportError:
    from src.datasets.timeframes import canonical_key_map, resolve_timeframes

# Import the ablation-aware HARN model (must be in PYTHONPATH or same directory)
try:
    from .harn_ablations import HARN, ModelConfig, VALID_VARIANTS, ABLATION_DESCRIPTIONS
except ImportError:
    try:
        from src.ablation.harn_ablations import HARN, ModelConfig, VALID_VARIANTS, ABLATION_DESCRIPTIONS
    except ImportError:
        print("Error: harn_ablations.py not found. Please place it in the current directory "
              "or on PYTHONPATH.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training configuration."""
    parser = argparse.ArgumentParser(description="Train HARN component-wise ablation variants.")
    parser.add_argument("--variant", type=str, default="A0", choices=VALID_VARIANTS,
                        help="Ablation variant to train: "
                             "A0=full model, A1=no associative memory, "
                             "A2=no cross-scale resonance, A3=no bottom-up evidence "
                             "aggregation. Checkpoints are saved to "
                             "<checkpoint-dir>/<asset>/<variant>/.")
    parser.add_argument("--data-dir", type=str, default="data/processed",
                        help="Root directory containing per-asset subdirectories, each with "
                             "train.pkl/val.pkl/test.pkl, i.e. <data-dir>/<asset>/{split}.pkl.")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/HARN_Ablations",
                        help="Root checkpoint directory. Artifacts are written to "
                             "<checkpoint-dir>/<asset>/<variant>/.")
    parser.add_argument("--timeframes", type=str, default="m15,h1,h4",
                        help="Comma-separated, ordered list of hierarchy levels from "
                             "finest (updates every step) to coarsest, e.g. 'm15,h1,h4' "
                             "or 'm1,m5,m15,h1,h4,d1'. Defines the entire hierarchy.")
    parser.add_argument("--asset", type=str, default="EURUSD",
                        help="Asset name. Selects both the data subdirectory "
                             "(<data-dir>/<asset>/) and the checkpoint subdirectory "
                             "(<checkpoint-dir>/<asset>/<variant>/).")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Number of parallel contiguous streams processed per "
                             "training step.")
    parser.add_argument("--eval-streams", type=int, default=8,
                        help="Number of parallel contiguous streams used for "
                             "stateful validation/test evaluation.")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Maximum number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Initial learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-5,
                        help="AdamW weight decay.")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Gradient clipping norm.")
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience (epochs without improvement).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda", "mps"],
                        help="Device to use. Default: auto-detect (cuda > mps > cpu).")
    return parser.parse_args()


def _parse_timeframes(raw: str) -> Tuple[str, ...]:
    """Parse comma-separated level names into a tuple, validating uniqueness."""
    names = tuple(tf.strip() for tf in raw.split(",") if tf.strip())
    return resolve_timeframes(names, context="--timeframes")


# ---------------------------------------------------------------------------
# Reproducibility & device resolution (shared with evaluate_harn_ablations.py)
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Seed every source of randomness this pipeline touches.

    Also pins cuDNN to deterministic algorithms. This can reduce GPU
    throughput, but the training script's whole point is reproducible
    ablation comparisons, so determinism is prioritized over raw speed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(preferred: Optional[str] = None) -> torch.device:
    """Resolve the compute device, honoring an explicit override.

    Auto-detection order (when ``preferred`` is None): CUDA, then Apple
    Silicon MPS, then CPU.
    """
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_grad_scaler(device: torch.device) -> Optional["torch.cuda.amp.GradScaler"]:
    """Construct a CUDA gradient scaler for mixed precision, or None off-CUDA."""
    if device.type != "cuda":
        return None
    try:
        return torch.amp.GradScaler("cuda")
    except (AttributeError, TypeError):  # pragma: no cover - older torch
        return torch.cuda.amp.GradScaler()


def autocast_context(device: torch.device, enabled: bool):
    """Return an autocast context manager, or a no-op off-CUDA/when disabled."""
    if not enabled:
        return contextlib.nullcontext()
    try:
        return torch.autocast(device_type=device.type, enabled=True)
    except (AttributeError, TypeError):  # pragma: no cover - older torch
        if device.type == "cuda":
            return torch.cuda.amp.autocast(enabled=True)
        return contextlib.nullcontext()


# ---------------------------------------------------------------------------
# Checkpoint I/O (shared with evaluate_harn_ablations.py)
# ---------------------------------------------------------------------------
def load_checkpoint(path: Path, map_location: Any = "cpu") -> Any:
    """Deserialize a checkpoint file robustly across torch versions/OSes.

    This function ONLY concerns getting bytes off disk into a Python object;
    it never inspects or filters checkpoint *contents*. Any mismatch between
    the checkpoint's state dict and a model's architecture must still surface
    as a hard error at ``load_state_dict(..., strict=True)`` time -- nothing
    here hides that.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    def _load() -> Any:
        try:
            return torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:  # Older torch releases do not accept `weights_only`.
            return torch.load(path, map_location=map_location)

    try:
        return _load()
    except NotImplementedError as exc:
        # Permit checkpoints containing POSIX paths (e.g. from an argparse
        # Namespace saved for provenance) to be unpickled on Windows.
        if "PosixPath" not in str(exc) or not hasattr(pathlib, "WindowsPath"):
            raise
        original_posix_path = pathlib.PosixPath
        pathlib.PosixPath = pathlib.WindowsPath  # type: ignore[assignment]
        try:
            return _load()
        finally:
            pathlib.PosixPath = original_posix_path  # type: ignore[assignment]


def build_checkpoint_payload(
    *,
    epoch: int,
    model: "HARN",
    config: ModelConfig,
    asset: str,
    seed: int,
    best_val_loss: float,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    patience_counter: Optional[int] = None,
) -> Dict[str, Any]:
    """Build the checkpoint dict, in one consistent shape for every artifact.

    Both ``best_model.pt`` and ``last_model.pt`` use this same wrapped shape
    (``timeframes``/``variant``/``config``/``model_state_dict``/...), the
    only difference being that resumable optimizer/scheduler state is only
    attached when explicitly provided (for ``last_model.pt``). Persisting the
    full ``ModelConfig`` means the exact training-time architecture can
    always be reconstructed later via `config_from_checkpoint`, independent
    of whatever `ModelConfig`'s defaults happen to be in the code at
    evaluation time.
    """
    payload: Dict[str, Any] = {
        "epoch": epoch,
        "timeframes": list(config.timeframes),
        "variant": config.variant,
        "config": dataclasses.asdict(config),
        "model_state_dict": model.state_dict(),
        "best_val_loss": best_val_loss,
        "asset": asset,
        "seed": seed,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if patience_counter is not None:
        payload["patience_counter"] = patience_counter
    return payload


def config_from_checkpoint(checkpoint: Dict[str, Any]) -> ModelConfig:
    """Reconstruct the exact `ModelConfig` used to produce a checkpoint.

    Rebuilding from the config dict stored *inside* the checkpoint (rather
    than re-deriving defaults from whatever `ModelConfig` looks like in the
    currently-installed `harn_ablations.py`) guarantees the reconstructed
    architecture matches the saved weights even if defaults drift later.
    """
    raw = checkpoint.get("config")
    if not isinstance(raw, dict):
        raise KeyError(
            "Checkpoint is missing its 'config' payload -- cannot safely "
            "reconstruct the exact training-time ModelConfig. Re-train with "
            "the current train_harn_ablations.py to produce a compatible checkpoint."
        )
    raw = dict(raw)
    raw["timeframes"] = tuple(raw["timeframes"])
    raw["dilations"] = tuple(raw["dilations"])
    return ModelConfig(**raw)


# ---------------------------------------------------------------------------
# Dataset – fully generic over the configured hierarchy levels
# ---------------------------------------------------------------------------
class HARNDataset(Dataset):
    """PyTorch Dataset wrapping a pickled HARN data dictionary.

    This is the single, canonical data adapter for the whole pipeline:
    ``train_harn_ablations.py`` and ``evaluate_harn_ablations.py`` both
    construct datasets exclusively through this class, so there is exactly
    one place that defines (and validates) what a "HARN data payload" is.

    Parameters
    ----------
    data_dict : dict
        Loaded pickle payload containing per-level windows, targets, update
        flags, and feature statistics (see the module docstring for the
        exact schema).
    timeframes : sequence of str
        Ordered level names from finest to coarsest. Must match the keys in
        ``data_dict``.

    Raises
    ------
    KeyError
        If required window, target, update-flag, or statistics entries are
        missing.
    ValueError
        If present arrays have inconsistent shapes/sample counts, or if the
        payload declares a ``"timeframes"`` list that disagrees with
        ``timeframes``.
    TypeError
        If ``data_dict["statistics"]`` is not a dict.
    """

    def __init__(self, data_dict: Dict[str, Any], timeframes: Sequence[str]) -> None:
        # Resolve aliases against the payload's ordered declaration. This
        # changes names only; it never copies, sorts, reorders, or subsets data.
        declared = data_dict.get("timeframes")
        self.timeframes = resolve_timeframes(
            timeframes, declared, context="data payload timeframes"
        ) if declared is not None else resolve_timeframes(timeframes, context="timeframes")

        if "statistics" not in data_dict:
            raise KeyError("Data payload is missing required key 'statistics'.")
        self.statistics: Dict[str, Any] = data_dict["statistics"]
        if not isinstance(self.statistics, dict):
            raise TypeError("Data payload 'statistics' must be a dictionary.")

        features = canonical_key_map(data_dict.get("features"), label="Data payload 'features'")
        targets = canonical_key_map(data_dict.get("targets"), label="Data payload 'targets'")
        update_flags = canonical_key_map(data_dict.get("is_update"), label="Data payload 'is_update'")

        missing_windows = [tf for tf in self.timeframes if tf not in features]
        if missing_windows:
            raise KeyError(f"Data payload is missing input window(s) for level(s): {missing_windows}")

        n_samples: Optional[int] = None
        self.windows: Dict[str, torch.Tensor] = {}
        for tf in self.timeframes:
            arr = torch.as_tensor(np.asarray(features[tf]), dtype=torch.float32)
            if arr.dim() != 3:
                raise ValueError(
                    f"windows['{tf}'] must be rank-3 (N, T, input_dim); got shape {tuple(arr.shape)}."
                )
            if arr.size(1) < 1:
                raise ValueError(f"windows['{tf}'] has a zero-length time window.")
            if n_samples is None:
                n_samples = arr.size(0)
            elif arr.size(0) != n_samples:
                raise ValueError(
                    f"windows['{tf}'] has {arr.size(0)} samples, expected {n_samples} "
                    "(sample count must match across every hierarchy level)."
                )
            self.windows[tf] = arr
        if not n_samples:
            raise ValueError("Data payload contains zero samples.")

        missing_targets = [tf for tf in self.timeframes if tf not in targets]
        if missing_targets:
            raise KeyError(f"Data payload is missing target(s) for level(s): {missing_targets}")
        self.targets: Dict[str, torch.Tensor] = {}
        for tf in self.timeframes:
            arr = torch.as_tensor(np.asarray(targets[tf]), dtype=torch.float32)
            if arr.dim() not in (1, 2):
                raise ValueError(f"target_{tf} must be rank 1 or 2; got shape {tuple(arr.shape)}.")
            if arr.size(0) != n_samples:
                raise ValueError(f"target_{tf} has {arr.size(0)} samples, expected {n_samples}.")
            self.targets[tf] = arr

        # Every level above the finest is event-driven and requires explicit
        # update flags; the base level always updates.
        self.update_flags: Dict[str, torch.Tensor] = {}
        for tf in self.timeframes[1:]:
            if tf not in update_flags:
                raise KeyError(f"Data payload is missing update flag for non-base level '{tf}'.")
            arr = torch.as_tensor(np.asarray(update_flags[tf]), dtype=torch.bool)
            if arr.dim() != 1 or arr.size(0) != n_samples:
                raise ValueError(
                    f"'{tf}' must have shape ({n_samples},); got {tuple(arr.shape)}."
                )
            self.update_flags[tf] = arr

        stat_keys = {key.upper(): value for key, value in self.statistics.items()}
        for tf in self.timeframes:
            for stats_key in (tf, f"target_{tf}"):
                block = stat_keys.get(stats_key.upper())
                if not isinstance(block, dict) or "mean" not in block or "std" not in block:
                    raise KeyError(
                        f"Data payload 'statistics' is missing '{stats_key}' "
                        "(needs a dict with 'mean' and 'std')."
                    )

        self.statistics = {
            key: value for key, value in self.statistics.items()
        }
        self._n_samples: int = n_samples

    def __len__(self) -> int:
        return self._n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return a dict with window, target, and update flag tensors for all levels."""
        item: Dict[str, torch.Tensor] = {}
        for tf in self.timeframes:
            item[tf] = self.windows[tf][idx]
            item[f"target_{tf}"] = self.targets[tf][idx]
        for tf in self.timeframes[1:]:
            item[f"is_{tf}_update"] = self.update_flags[tf][idx]
        return item


# ---------------------------------------------------------------------------
# Target and data helpers
# ---------------------------------------------------------------------------
def _unpack_stats(stats_blob: Dict[str, Any], key: str) -> Tuple[np.ndarray, np.ndarray]:
    """Extract (mean, std) arrays from a statistics block, replacing zero std with 1."""
    block = stats_blob[key]
    mean = np.asarray(block["mean"], dtype=np.float32)
    std = np.asarray(block["std"], dtype=np.float32)
    std = np.where(std == 0.0, 1.0, std)
    return mean, std


def _denormalize(values: np.ndarray, stats: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Reverse normalization: ``values * std + mean``."""
    mean, std = stats
    return values * std + mean


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
            raise RuntimeError(f"Unexpected target shape {target_close_norm.shape} for '{tf}'")

        baseline_norm = batch[tf][:, -1, close_index].to(device)  # (batch,)

        target_close = _target_close_from_normalized(target_close_norm, target_stats[tf], device)
        baseline_close = _close_from_normalized(baseline_norm, feature_stats[tf], device, close_index)

        delta = _delta_target_bps(target_close, baseline_close, target_scale)
        targets[f"delta_{tf}"] = delta.unsqueeze(-1)
        targets[f"absolute_{tf}"] = target_close.unsqueeze(-1)

    return targets


def _load_payload_stats(dataset: HARNDataset) -> Dict[str, Dict[str, Any]]:
    """Return the statistics dict stored in the dataset."""
    return dataset.statistics


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
def load_data(data_dir: str, asset: str, timeframes: Sequence[str]) -> Dict[str, HARNDataset]:
    """Load train/val/test datasets from pickle files for one asset.

    Parameters
    ----------
    data_dir : str
        Root directory containing per-asset subdirectories.
    asset : str
        Asset name; data is read from ``<data_dir>/<asset>/{split}.pkl``.
        This is the same layout ``evaluate_harn_ablations.py`` expects.
    timeframes : sequence of str
        Hierarchy level names.

    Returns
    -------
    dict
        Mapping ``{"train": ..., "val": ..., "test": ...}`` to HARNDataset
        instances.
    """
    data_path = Path(data_dir) / asset
    required = ["train.pkl", "val.pkl", "test.pkl"]
    missing = [f for f in required if not (data_path / f).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing} in {data_path}. Expected layout: "
            f"{data_dir}/<asset>/{{train,val,test}}.pkl."
        )

    datasets: Dict[str, HARNDataset] = {}
    for split in ("train", "val", "test"):
        path = data_path / f"{split}.pkl"
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        if not isinstance(data, dict):
            raise TypeError(f"{path} does not contain a dictionary payload.")
        try:
            datasets[split] = HARNDataset(data, timeframes)
        except (KeyError, ValueError, TypeError) as exc:
            raise type(exc)(f"{path}: {exc}") from exc
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
        targets = _prepare_targets(batch, dataset.statistics, device, timeframes, close_index, target_scale)

        preds = model(windows, update_flags)
        level_losses = {
            tf: F.mse_loss(preds[f"pred_delta_{tf}"], targets[f"delta_{tf}"]) for tf in timeframes
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

    pbar = tqdm(batcher, desc=f"Epoch {epoch:3d}/{total_epochs}", unit="step", leave=False, total=num_batches)
    for batch_idx, batch in enumerate(pbar):
        windows, update_flags = _model_inputs_from_batch(batch, timeframes, device)
        targets = _prepare_targets(batch, statistics, device, timeframes, close_index, target_scale)

        optimizer.zero_grad(set_to_none=True)

        with autocast_context(device, enabled=scaler is not None):
            preds = model(windows, update_flags)
            level_losses = {
                tf: F.mse_loss(preds[f"pred_delta_{tf}"], targets[f"delta_{tf}"]) for tf in timeframes
            }
            loss = sum(level_losses.values())

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
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
    timeframes = _parse_timeframes(args.timeframes)

    # Reproducibility: one call seeds python/numpy/torch/cuda and pins cuDNN
    # to deterministic algorithms.
    set_seed(args.seed)

    device = resolve_device(args.device)
    print(f"Using device: {device}")
    print(f"Variant: {args.variant} — {ABLATION_DESCRIPTIONS[args.variant]}")
    print(f"Hierarchy levels ({len(timeframes)}): {timeframes}")

    # Structured checkpoint directory: <checkpoint-dir>/<asset>/<variant>/.
    # This is the exact layout evaluate_harn_ablations.py reads from.
    checkpoint_dir = Path(args.checkpoint_dir) / args.asset / args.variant
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Load data from <data-dir>/<asset>/{split}.pkl.
    datasets = load_data(args.data_dir, args.asset, timeframes)
    train_set = datasets["train"]
    val_set = datasets["val"]
    test_set = datasets["test"]

    # StreamingBatcher preserves temporal continuity across steps.
    train_batcher = StreamingBatcher(train_set, args.batch_size)

    # Model configuration – the hierarchy is defined solely by `timeframes`,
    # and the ablation architecture is selected solely by `--variant`. This
    # is the one line that actually makes `--variant` do anything: every
    # downstream call (train_one_epoch/evaluate/checkpointing) is variant-
    # agnostic and just operates on whatever `model`/`config` it is given.
    config = ModelConfig(timeframes=timeframes, variant=args.variant)
    model = HARN(config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Optimizer & scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Mixed precision (CUDA only; no-op elsewhere).
    scaler = make_grad_scaler(device)

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
            model, train_batcher, optimizer, device,
            grad_clip=args.grad_clip, scaler=scaler,
            epoch=epoch, total_epochs=args.epochs,
            statistics=train_set.statistics,
            close_index=config.close_index,
            target_scale=config.target_scale,
        )

        # Evaluate on validation set
        val_metrics = evaluate(
            model, val_set, device,
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
            torch.save(
                build_checkpoint_payload(
                    epoch=epoch, model=model, config=config, asset=args.asset,
                    seed=args.seed, best_val_loss=best_val_loss,
                ),
                best_model_path,
            )
        else:
            patience_counter += 1

        # Always save the most recent checkpoint (for resuming), in the same
        # wrapped shape as best_model.pt plus resumable optimizer/scheduler
        # state.
        torch.save(
            build_checkpoint_payload(
                epoch=epoch, model=model, config=config, asset=args.asset,
                seed=args.seed, best_val_loss=best_val_loss,
                optimizer=optimizer, scheduler=scheduler,
                patience_counter=patience_counter,
            ),
            last_model_path,
        )

        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered after {args.patience} epochs without improvement.")
            break

    # Save history
    pd.DataFrame(history).to_csv(history_path, index=False)

    # Final test evaluation using the best model. Reload through the same
    # robust `load_checkpoint` + strict `load_state_dict` path evaluation
    # uses, so a corrupted/incompatible checkpoint fails loudly here too
    # rather than only in the standalone evaluation script.
    print("\nLoading best model for test evaluation...")
    best_checkpoint = load_checkpoint(best_model_path, map_location=device)
    checkpoint_timeframes = best_checkpoint.get("timeframes")
    if checkpoint_timeframes:
        try:
            resolve_timeframes(
                timeframes, checkpoint_timeframes, context="checkpoint timeframes"
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Best checkpoint at {best_model_path} has incompatible timeframes: {exc}"
            ) from exc
    try:
        model.load_state_dict(best_checkpoint["model_state_dict"], strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Best checkpoint at {best_model_path} is incompatible with the "
            f"current HARN architecture (variant={args.variant}): {exc}"
        ) from exc
    test_metrics = evaluate(
        model, test_set, device,
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