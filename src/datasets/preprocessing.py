"""
Generic multi-symbol, multi-timeframe event-driven preprocessing pipeline.

This module loads OHLCV CSV files for any symbol, discovers available
timeframes, constructs chronologically aligned multi-timeframe sequences driven
by the fastest (anchor) timeframe, normalises features using training-set
statistics, and persists train/validation/test splits as pickle and CSV files.

Usage
-----
    python -m src.datasets.preprocessing --symbol USDCHF
    python -m src.datasets.preprocessing --all
"""

from __future__ import annotations

import argparse
import pickle
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .timeframes import canonicalize_timeframe, timeframe_to_minutes

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FEATURE_COLUMNS: list[str] = ["open", "high", "low", "close", "volume"]
RAW_COLUMNS: tuple[str, ...] = ("datetime", *FEATURE_COLUMNS)

# ---------------------------------------------------------------------------
# Helpers: timeframe discovery & ordering
# ---------------------------------------------------------------------------
def parse_timeframe_from_filename(filename: str, symbol: str) -> str | None:
    """Extract timeframe from a filename like EURUSD_H1.csv."""
    if not filename.startswith(f"{symbol}_") or not filename.endswith(".csv"):
        return None
    # Strip symbol prefix and ".csv" suffix
    middle = filename[len(symbol) + 1 : -4]
    if not middle:
        return None
    try:
        return canonicalize_timeframe(middle)
    except ValueError:
        return middle


def discover_symbols(raw_root: Path) -> list[str]:
    """Return sorted list of symbol directory names under *raw_root*."""
    if not raw_root.is_dir():
        return []
    symbols = [
        d.name
        for d in raw_root.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ]
    return sorted(symbols)


def discover_timeframes(symbol_dir: Path, symbol: str) -> dict[str, Path]:
    """Find all ``{symbol}_{timeframe}.csv`` files in *symbol_dir*.

    Returns a mapping ``{timeframe: path}`` sorted by timeframe speed (fastest
    first). Duplicate timeframe files raise an error.
    """
    tfs: dict[str, Path] = {}
    for fpath in symbol_dir.iterdir():
        if not fpath.is_file() or fpath.suffix != ".csv":
            continue
        tf = parse_timeframe_from_filename(fpath.name, symbol)
        if tf is None:
            continue
        if tf in tfs:
            raise ValueError(
                f"Duplicate timeframe '{tf}' found in {symbol_dir}: "
                f"'{tfs[tf].name}' and '{fpath.name}'"
            )
        try:
            timeframe_to_minutes(tf)  # validate
        except ValueError:
            print(f"Warning: ignoring file with unsupported timeframe '{tf}': {fpath}", file=sys.stderr)
            continue
        tfs[tf] = fpath
    # Sort by minutes ascending (fastest first)
    return {tf: tfs[tf] for tf in sorted(tfs, key=timeframe_to_minutes)}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class PreprocessingConfig:
    """Generic configuration for the multi-symbol preprocessing pipeline.

    Attributes
    ----------
    raw_dir: Path
        Root directory containing per-symbol raw CSV subdirectories.
    processed_dir: Path
        Root directory where processed per-symbol outputs will be written.
    train_ratio, val_ratio, test_ratio: float
        Chronological split proportions (must sum to 1.0).
    default_window: int
        Window size for any timeframe not explicitly listed in *windows*.
    windows: dict[str, int]
        Mapping from timeframe string to window length (number of bars).
    datetime_format: str
        Expected datetime format in the CSVs (not currently used, kept for future).
    """

    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    default_window: int = 32
    windows: dict[str, int] = field(default_factory=dict)
    datetime_format: str = "%Y-%m-%d %H:%M:%S"

    def __post_init__(self) -> None:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if not np.isclose(total, 1.0):
            raise ValueError(f"Split ratios must sum to 1.0, got {total:.6f}.")
        canonical_windows: dict[str, int] = {}
        for raw_tf, w in self.windows.items():
            tf = canonicalize_timeframe(raw_tf)
            if tf in canonical_windows:
                raise ValueError(f"Duplicate window settings for timeframe '{tf}'.")
            if w <= 0:
                raise ValueError(f"Window for '{tf}' must be positive, got {w}.")
            canonical_windows[tf] = w
        self.windows = canonical_windows
        if self.default_window <= 0:
            raise ValueError("default_window must be positive.")

    def window_for(self, timeframe: str) -> int:
        """Return the window length for *timeframe*, falling back to default."""
        return self.windows.get(timeframe, self.default_window)

    def to_serializable_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["raw_dir"] = str(self.raw_dir)
        data["processed_dir"] = str(self.processed_dir)
        return data


# ---------------------------------------------------------------------------
# Data loading & cleaning (preserved from original)
# ---------------------------------------------------------------------------
def load_ohlcv_csv(path: Path) -> pd.DataFrame:
    """Load a raw OHLCV CSV and assign canonical column names."""
    frame = pd.read_csv(path, header=None, names=RAW_COLUMNS)
    return clean_ohlcv_frame(frame)


def clean_ohlcv_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Parse datetime, sort, drop duplicates/invalid, reset index."""
    cleaned = frame.copy()
    cleaned["datetime"] = pd.to_datetime(cleaned["datetime"], errors="coerce")
    for col in FEATURE_COLUMNS:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
    cleaned = cleaned.dropna(subset=["datetime", *FEATURE_COLUMNS])
    cleaned = cleaned.drop_duplicates(subset=["datetime"], keep="last")
    cleaned = cleaned.sort_values("datetime").reset_index(drop=True)
    return cleaned


# ---------------------------------------------------------------------------
# Sample specification (generic)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SampleSpec:
    """Generic event-driven multi-timeframe sample descriptor.

    The anchor is the completion of a bar in the fastest (anchor) timeframe.
    All windows end at the latest completed bar for that timeframe **≤ anchor_time**.
    Targets are the *next* completed bar for each timeframe.
    """

    anchor_time: pd.Timestamp
    anchor_idx: int  # index in anchor dataframe
    # For each timeframe: (start_idx, end_idx) inclusive
    window_slices: dict[str, tuple[int, int]]
    # For each timeframe: index of target candle
    target_indices: dict[str, int]
    # For each timeframe: True if a new bar of this timeframe closed exactly at anchor_time
    is_update: dict[str, bool]


def build_event_driven_sample_specs(
    frames: dict[str, pd.DataFrame],
    config: PreprocessingConfig,
    anchor_tf: str,
    sorted_timeframes: list[str],
) -> list[SampleSpec]:
    """Generate sample specifications driven by each completed anchor candle.

    For every anchor bar close (except the last one) we check that every
    timeframe has enough history and a valid next candle.
    """
    anchor_df = frames[anchor_tf]
    anchor_times = anchor_df["datetime"].to_numpy(dtype="datetime64[ns]")
    anchor_win = config.window_for(anchor_tf)
    n_anchor = len(anchor_df)

    # Pre-extract datetime arrays and window lengths for all timeframes
    tf_info: dict[str, dict[str, Any]] = {}
    for tf in sorted_timeframes:
        df = frames[tf]
        tf_info[tf] = {
            "times": df["datetime"].to_numpy(dtype="datetime64[ns]"),
            "window": config.window_for(tf),
            "len": len(df),
        }

    specs: list[SampleSpec] = []
    # Anchor indices: need at least anchor_win history and a next candle
    for i in range(anchor_win - 1, n_anchor - 1):
        anchor_dt = anchor_times[i]

        # Check all timeframes
        window_slices: dict[str, tuple[int, int]] = {}
        target_indices: dict[str, int] = {}
        is_update: dict[str, bool] = {}

        valid = True
        for tf in sorted_timeframes:
            info = tf_info[tf]
            times = info["times"]
            win_len = info["window"]
            n_bars = info["len"]

            # Latest completed bar <= anchor_dt
            end_idx = int(np.searchsorted(times, anchor_dt, side="right") - 1)
            if end_idx < win_len - 1:  # not enough history
                valid = False
                break
            next_idx = end_idx + 1
            if next_idx >= n_bars:     # no target candle
                valid = False
                break

            window_slices[tf] = (end_idx - win_len + 1, end_idx)
            target_indices[tf] = next_idx
            is_update[tf] = bool(times[end_idx] == anchor_dt)

        if not valid:
            continue

        specs.append(
            SampleSpec(
                anchor_time=pd.Timestamp(anchor_dt),
                anchor_idx=i,
                window_slices=window_slices,
                target_indices=target_indices,
                is_update=is_update,
            )
        )
    return specs


# ---------------------------------------------------------------------------
# Statistics (training split only)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FeatureStats:
    """Mean and std for z-score normalisation."""

    mean: np.ndarray
    std: np.ndarray

    def normalize(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std

    def to_serializable_dict(self) -> dict[str, list[float]]:
        return {
            "mean": self.mean.astype(np.float32).tolist(),
            "std": self.std.astype(np.float32).tolist(),
        }


def _compute_training_statistics(
    train_specs: Sequence[SampleSpec],
    frames: dict[str, pd.DataFrame],
    config: PreprocessingConfig,
    sorted_timeframes: list[str],
) -> dict[str, FeatureStats]:
    """Compute normalisation stats from training samples only.

    Features are pooled across all training samples (sample & time axes).
    Targets are close prices, pooled per timeframe.
    """
    # Accumulators: one list per timeframe for features, one for target
    feat_pool: dict[str, list[np.ndarray]] = {tf: [] for tf in sorted_timeframes}
    target_pool: dict[str, list[float]] = {tf: [] for tf in sorted_timeframes}

    close_col_idx = FEATURE_COLUMNS.index("close")

    for spec in train_specs:
        for tf in sorted_timeframes:
            df = frames[tf]
            start, end = spec.window_slices[tf]
            # Extract raw features
            window = df.iloc[start : end + 1][FEATURE_COLUMNS].to_numpy(dtype=np.float32)
            feat_pool[tf].append(window)

            # Target close
            target_idx = spec.target_indices[tf]
            target_close = df.iloc[target_idx]["close"]
            target_pool[tf].append(float(target_close))

    def _stats_from_pooled(arr: np.ndarray) -> FeatureStats:
        mean = arr.mean(axis=0, dtype=np.float32)
        std = arr.std(axis=0, ddof=0, dtype=np.float32)
        std = np.where(std == 0.0, 1.0, std)
        return FeatureStats(mean=mean, std=std)

    stats: dict[str, FeatureStats] = {}
    # Feature stats per timeframe
    for tf in sorted_timeframes:
        if feat_pool[tf]:
            all_feat = np.concatenate(feat_pool[tf], axis=0)
            stats[tf] = _stats_from_pooled(all_feat)
        else:
            # Edge case: no training samples -> create dummy stats (won't be used)
            stats[tf] = FeatureStats(
                mean=np.zeros(len(FEATURE_COLUMNS), dtype=np.float32),
                std=np.ones(len(FEATURE_COLUMNS), dtype=np.float32),
            )

    # Target stats per timeframe
    for tf in sorted_timeframes:
        if target_pool[tf]:
            arr = np.array(target_pool[tf], dtype=np.float32).reshape(-1, 1)
            stats[f"target_{tf}"] = _stats_from_pooled(arr)
        else:
            stats[f"target_{tf}"] = FeatureStats(
                mean=np.zeros(1, dtype=np.float32),
                std=np.ones(1, dtype=np.float32),
            )

    return stats


# ---------------------------------------------------------------------------
# Chronological splitting (preserved)
# ---------------------------------------------------------------------------
def _split_counts(total_samples: int, config: PreprocessingConfig) -> tuple[int, int, int]:
    train_count = int(total_samples * config.train_ratio)
    val_count = int(total_samples * config.val_ratio)
    test_count = total_samples - train_count - val_count
    return train_count, val_count, test_count


# ---------------------------------------------------------------------------
# Generic payload construction
# ---------------------------------------------------------------------------
def _build_split_payload(
    frames: dict[str, pd.DataFrame],
    stats: dict[str, FeatureStats],
    split_specs: Sequence[SampleSpec],
    split_name: str,
    config: PreprocessingConfig,
    symbol: str,
    anchor_tf: str,
    sorted_timeframes: list[str],
) -> dict[str, Any]:
    """Build the multi-timeframe pickle payload for a single split."""
    n_samples = len(split_specs)
    if n_samples == 0:
        # Return empty payload structure
        features = {tf: np.empty((0, config.window_for(tf), len(FEATURE_COLUMNS)), dtype=np.float32)
                    for tf in sorted_timeframes}
        targets = {tf: np.empty((0, 1), dtype=np.float32) for tf in sorted_timeframes}
        anchor_times = np.array([], dtype=object)
        target_times = {tf: np.array([], dtype=object) for tf in sorted_timeframes}
        is_update = {tf: np.array([], dtype=bool) for tf in sorted_timeframes}
        return {
            "symbol": symbol,
            "timeframes": sorted_timeframes,
            "anchor_timeframe": anchor_tf,
            "features": features,
            "targets": targets,
            "anchor_time": anchor_times,
            "target_times": target_times,
            "is_update": is_update,
            "feature_names": FEATURE_COLUMNS,
            "statistics": {k: v.to_serializable_dict() for k, v in stats.items()},
            "config": config.to_serializable_dict(),
            "split": split_name,
        }

    # Raw feature arrays for each timeframe
    feat_raw = {tf: frames[tf][FEATURE_COLUMNS].to_numpy(dtype=np.float32) for tf in sorted_timeframes}
    close_idx = FEATURE_COLUMNS.index("close")

    # Normalisation functions
    norm_feat = {tf: stats[tf].normalize for tf in sorted_timeframes}
    norm_target = {tf: stats[f"target_{tf}"].normalize for tf in sorted_timeframes}

    # Accumulators
    features_acc = {tf: [] for tf in sorted_timeframes}
    targets_acc = {tf: [] for tf in sorted_timeframes}
    anchor_times_list: list[str] = []
    target_times_acc = {tf: [] for tf in sorted_timeframes}
    is_update_acc = {tf: [] for tf in sorted_timeframes}

    for spec in split_specs:
        anchor_times_list.append(spec.anchor_time.isoformat())
        for tf in sorted_timeframes:
            start, end = spec.window_slices[tf]
            win = feat_raw[tf][start : end + 1]
            features_acc[tf].append(norm_feat[tf](win).astype(np.float32))

            target_idx = spec.target_indices[tf]
            target_close = feat_raw[tf][target_idx, close_idx]
            # Normalize target (needs a 2D array)
            norm_close = norm_target[tf](np.array([[target_close]], dtype=np.float32))[0, 0]
            targets_acc[tf].append(norm_close)

            target_time = frames[tf].iloc[target_idx]["datetime"].isoformat()
            target_times_acc[tf].append(target_time)

            is_update_acc[tf].append(spec.is_update[tf])

    # Stack into arrays
    features = {tf: np.stack(features_acc[tf], axis=0) for tf in sorted_timeframes}
    targets = {tf: np.array(targets_acc[tf], dtype=np.float32).reshape(-1, 1) for tf in sorted_timeframes}
    anchor_times = np.array(anchor_times_list, dtype=object)
    target_times = {tf: np.array(target_times_acc[tf], dtype=object) for tf in sorted_timeframes}
    is_update = {tf: np.array(is_update_acc[tf], dtype=bool) for tf in sorted_timeframes}

    return {
        "symbol": symbol,
        "timeframes": sorted_timeframes,
        "anchor_timeframe": anchor_tf,
        "features": features,
        "targets": targets,
        "anchor_time": anchor_times,
        "target_times": target_times,
        "is_update": is_update,
        "feature_names": FEATURE_COLUMNS,
        "statistics": {k: v.to_serializable_dict() for k, v in stats.items()},
        "config": config.to_serializable_dict(),
        "split": split_name,
    }


# ---------------------------------------------------------------------------
# Generic CSV export
# ---------------------------------------------------------------------------
def _build_csv_columns(
    sorted_timeframes: list[str],
    config: PreprocessingConfig,
) -> list[str]:
    """Dynamically generate flat CSV column names for the given timeframes.

    Feature order: for each timeframe (in the given order), bars from oldest to
    anchor (offset -(window-1) ... 0), each with open/high/low/close/volume.
    Then metadata columns.
    """
    columns: list[str] = []
    for tf in sorted_timeframes:
        win = config.window_for(tf)
        for i in range(win):
            offset = i - (win - 1)  # oldest: -(win-1), anchor: 0
            for feat in FEATURE_COLUMNS:
                columns.append(f"{tf}_{feat}_t{offset}")

    # Metadata
    columns.append("anchor_time")
    for tf in sorted_timeframes:
        columns.append(f"target_time_{tf}")
    for tf in sorted_timeframes:
        columns.append(f"target_{tf}")
    for tf in sorted_timeframes:
        columns.append(f"is_{tf}_update")

    return columns


def _payload_to_dataframe(
    payload: dict[str, Any],
    config: PreprocessingConfig,
    sorted_timeframes: list[str],
) -> pd.DataFrame:
    """Convert a payload into a flat CSV-friendly DataFrame."""
    n_samples = payload["anchor_time"].shape[0]
    if n_samples == 0:
        return pd.DataFrame(columns=_build_csv_columns(sorted_timeframes, config))

    # Stack features from all timeframes (already flattened per sample)
    feat_blocks = []
    for tf in sorted_timeframes:
        win_len = config.window_for(tf)
        # payload["features"][tf] shape: (N, win_len, n_features)
        flat = payload["features"][tf].reshape(n_samples, -1)  # (N, win_len * n_features)
        feat_blocks.append(flat)
    features = np.hstack(feat_blocks)

    metadata: dict[str, Any] = {"anchor_time": payload["anchor_time"]}
    for tf in sorted_timeframes:
        metadata[f"target_time_{tf}"] = payload["target_times"][tf]
    for tf in sorted_timeframes:
        metadata[f"target_{tf}"] = payload["targets"][tf].flatten()
    for tf in sorted_timeframes:
        metadata[f"is_{tf}_update"] = payload["is_update"][tf]

    col_names = _build_csv_columns(sorted_timeframes, config)
    # The first part of col_names matches features
    df = pd.DataFrame(features, columns=col_names[: features.shape[1]])
    for col_name, col_data in metadata.items():
        df[col_name] = col_data
    return df


# ---------------------------------------------------------------------------
# Per-symbol preprocessing
# ---------------------------------------------------------------------------
def preprocess_symbol(
    symbol: str,
    config: PreprocessingConfig,
) -> dict[str, Path]:
    """Run the full preprocessing pipeline for a single symbol.

    Parameters
    ----------
    symbol : str
        Name of the symbol (must correspond to a directory in ``config.raw_dir``).
    config : PreprocessingConfig
        Configuration.

    Returns
    -------
    dict[str, Path]
        Mapping from split name (train/val/test) to the saved pickle file path.
    """
    symbol_raw_dir = config.raw_dir / symbol
    if not symbol_raw_dir.is_dir():
        raise ValueError(f"Symbol directory not found: {symbol_raw_dir}")

    # 1. Discover timeframes
    tf_paths = discover_timeframes(symbol_raw_dir, symbol)
    if len(tf_paths) < 2:
        raise ValueError(
            f"Symbol '{symbol}' requires at least 2 timeframes, found {len(tf_paths)}: "
            f"{list(tf_paths.keys())}"
        )

    # Sort timeframes fastest-first
    sorted_tfs = list(tf_paths.keys())  # already sorted by minutes
    anchor_tf = sorted_tfs[0]  # fastest

    # 2. Load and clean all frames
    frames: dict[str, pd.DataFrame] = {}
    for tf, path in tf_paths.items():
        df = load_ohlcv_csv(path)
        if df.empty:
            raise ValueError(f"Timeframe '{tf}' for symbol '{symbol}' is empty after cleaning.")
        frames[tf] = df

    # 3. Build event-driven sample specs
    specs = build_event_driven_sample_specs(frames, config, anchor_tf, sorted_tfs)
    if not specs:
        raise ValueError(f"No valid samples could be generated for symbol '{symbol}'.")

    total = len(specs)
    print(f"[{symbol}]")
    print(f"  Timeframes: {', '.join(sorted_tfs)}")
    print(f"  Anchor: {anchor_tf}")
    print(f"  Samples: {total}")

    # 4. Chronological split
    train_cnt, val_cnt, test_cnt = _split_counts(total, config)
    if train_cnt == 0:
        raise ValueError("Training split is empty after chronological splitting.")
    train_specs = specs[:train_cnt]
    val_specs = specs[train_cnt : train_cnt + val_cnt]
    test_specs = specs[train_cnt + val_cnt :]

    print(f"  Train: {train_cnt}")
    print(f"  Val: {val_cnt}")
    print(f"  Test: {test_cnt}")

    # 5. Compute training statistics
    stats = _compute_training_statistics(train_specs, frames, config, sorted_tfs)

    # 6. Build payloads
    split_specs_map = {
        "train": train_specs,
        "val": val_specs,
        "test": test_specs,
    }
    payloads = {}
    for split_name, specs_list in split_specs_map.items():
        payloads[split_name] = _build_split_payload(
            frames, stats, specs_list, split_name, config, symbol, anchor_tf, sorted_tfs
        )

    # 7. Save outputs
    symbol_out_dir = config.processed_dir / symbol
    symbol_out_dir.mkdir(parents=True, exist_ok=True)

    output_paths: dict[str, Path] = {}
    for split_name, payload in payloads.items():
        # Pickle
        pkl_path = symbol_out_dir / f"{split_name}.pkl"
        with pkl_path.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        output_paths[split_name] = pkl_path

        # CSV
        csv_path = symbol_out_dir / f"{split_name}.csv"
        df = _payload_to_dataframe(payload, config, sorted_tfs)
        df.to_csv(csv_path, index=False)

    # Print saved paths
    for split_name in ("train", "val", "test"):
        pkl = symbol_out_dir / f"{split_name}.pkl"
        csv = symbol_out_dir / f"{split_name}.csv"
        print(f"    {pkl}")
        print(f"    {csv}")
    print()

    return output_paths


def preprocess_all_symbols(config: PreprocessingConfig) -> None:
    """Discover and preprocess all symbols in ``config.raw_dir``."""
    symbols = discover_symbols(config.raw_dir)
    if not symbols:
        print("No symbol directories found under", config.raw_dir)
        return

    success: list[str] = []
    failed: list[tuple[str, str]] = []

    for sym in symbols:
        try:
            preprocess_symbol(sym, config)
            success.append(sym)
        except Exception as e:
            failed.append((sym, str(e)))
            print(f"ERROR processing {sym}: {e}", file=sys.stderr)

    # Summary
    print("=" * 50)
    print("PREPROCESSING SUMMARY")
    print("=" * 50)
    if success:
        print("SUCCESS:")
        for s in success:
            print(f"  {s}")
    if failed:
        print("\nFAILED:")
        for s, reason in failed:
            print(f"  {s}: {reason}")
    print(f"\nProcessed: {len(success)}")
    print(f"Failed: {len(failed)}")
    print("=" * 50)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_window_arg(window_str: str) -> tuple[str, int]:
    """Parse a ``--window`` argument of the form ``TIMEFRAME=WINDOW``."""
    if "=" not in window_str:
        raise argparse.ArgumentTypeError(f"Expected format TIMEFRAME=VALUE, got '{window_str}'")
    tf, val = window_str.split("=", 1)
    try:
        window = int(val)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Window value must be an integer, got '{val}'")
    if window <= 0:
        raise argparse.ArgumentTypeError(f"Window must be positive, got {window}")
    try:
        timeframe_to_minutes(tf)  # validate timeframe
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e))
    return canonicalize_timeframe(tf), window


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generic multi-symbol, multi-timeframe event-driven preprocessing."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Root directory with per-symbol raw CSV subdirectories.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
        help="Root directory where processed outputs will be written.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument(
        "--default-window",
        type=int,
        default=32,
        help="Default window size for timeframes not explicitly configured.",
    )
    parser.add_argument(
        "--window",
        action="append",
        type=str,
        default=[],
        metavar="TF=WINDOW",
        help="Set window size for a specific timeframe (repeatable). "
        "Example: --window M15=32 --window H1=16.",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--symbol", type=str, help="Process a single symbol.")
    group.add_argument("--all", action="store_true", help="Process all discovered symbols.")

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)

    # Build windows dict from --window arguments
    windows: dict[str, int] = {}
    for w_str in args.window:
        tf, win = _parse_window_arg(w_str)
        if tf in windows:
            raise argparse.ArgumentTypeError(f"Duplicate window setting for timeframe '{tf}'")
        windows[tf] = win

    config = PreprocessingConfig(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        default_window=args.default_window,
        windows=windows,
    )

    if args.symbol:
        try:
            preprocess_symbol(args.symbol, config)
        except Exception as e:
            print(f"Error processing {args.symbol}: {e}", file=sys.stderr)
            sys.exit(1)
    else:  # --all
        preprocess_all_symbols(config)


if __name__ == "__main__":
    main()