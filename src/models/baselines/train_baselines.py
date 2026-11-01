"""Train PatchTST or TimeXer on the anchor timeframe of HARN data."""

from __future__ import annotations

import argparse
import pickle
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

try:
	from .PatchTST import PatchTST
	from .TimeXer import TimeXerForecaster
except ImportError:  # Support running this file directly as well as with -m.
	from src.models.baselines.PatchTST import PatchTST
	from src.models.baselines.TimeXer import TimeXerForecaster


MODEL_NAMES = ("PatchTST", "TimeXer")
INPUT_SIZE = 5
TARGET_INDEX = 3


def set_seed(seed: int) -> None:
	"""Seed Python, NumPy, and PyTorch random number generators."""
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)
	if hasattr(torch.backends, "cudnn"):
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False


def resolve_device(requested: str) -> torch.device:
	"""Resolve and validate a requested device name."""
	name = requested.lower()
	if name == "auto":
		if torch.cuda.is_available():
			name = "cuda"
		elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
			name = "mps"
		else:
			name = "cpu"
	if name.startswith("cuda") and not torch.cuda.is_available():
		raise ValueError("CUDA was requested but is not available.")
	if name == "mps" and (
		getattr(torch.backends, "mps", None) is None
		or not torch.backends.mps.is_available()
	):
		raise ValueError("MPS was requested but is not available.")
	if name not in {"cpu", "mps"} and not name.startswith("cuda"):
		raise ValueError(f"Unsupported device '{requested}'. Use auto, cpu, cuda, or mps.")
	return torch.device(name)


def _load_payload(path: Path) -> dict[str, Any]:
	if not path.exists():
		raise FileNotFoundError(f"Missing processed split: {path}")
	with path.open("rb") as handle:
		payload = pickle.load(handle)
	if not isinstance(payload, dict):
		raise ValueError(f"Expected a dictionary payload in {path}")
	return payload


def load_anchor_splits(data_dir: Path, asset: str) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], str, int]:
	"""Load and validate anchor features and targets for all chronological splits."""
	splits: dict[str, tuple[np.ndarray, np.ndarray]] = {}
	anchor_tf: str | None = None
	window_size: int | None = None

	for split in ("train", "val", "test"):
		payload = _load_payload(data_dir / asset / f"{split}.pkl")
		current_tf = payload.get("anchor_timeframe")
		if not isinstance(current_tf, str):
			raise ValueError(f"{split}.pkl has no valid anchor_timeframe.")
		if anchor_tf is None:
			anchor_tf = current_tf
		elif current_tf != anchor_tf:
			raise ValueError(f"Anchor timeframe differs across splits: {anchor_tf} vs {current_tf}.")

		try:
			features = np.asarray(payload["features"][current_tf], dtype=np.float32)
			targets = np.asarray(payload["targets"][current_tf], dtype=np.float32)
		except KeyError as exc:
			raise KeyError(f"{split}.pkl is missing anchor data for '{current_tf}'.") from exc

		if features.ndim != 3 or features.shape[2] != INPUT_SIZE:
			raise ValueError(f"{split} features must have shape (N, T, 5), got {features.shape}.")
		if targets.ndim == 1:
			targets = targets[:, None]
		if targets.ndim != 2 or targets.shape[1] != 1:
			raise ValueError(f"{split} targets must have shape (N, 1), got {targets.shape}.")
		if features.shape[0] != targets.shape[0] or features.shape[0] == 0:
			raise ValueError(f"{split} has incompatible or empty feature/target arrays.")
		if not np.isfinite(features).all() or not np.isfinite(targets).all():
			raise ValueError(f"{split} contains NaN or infinite values.")
		if window_size is None:
			window_size = int(features.shape[1])
		elif features.shape[1] != window_size:
			raise ValueError("Anchor window size differs across splits.")
		splits[split] = (features, targets)

	assert anchor_tf is not None and window_size is not None
	return splits, anchor_tf, window_size


def make_loader(
	arrays: tuple[np.ndarray, np.ndarray],
	batch_size: int,
	shuffle: bool,
	num_workers: int,
	device: torch.device,
	seed: int,
) -> DataLoader:
	features, targets = arrays
	generator = torch.Generator()
	generator.manual_seed(seed)
	dataset = TensorDataset(torch.from_numpy(features), torch.from_numpy(targets))
	return DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=shuffle,
		num_workers=num_workers,
		pin_memory=device.type == "cuda",
		generator=generator,
		persistent_workers=num_workers > 0,
	)


def build_model(args: argparse.Namespace, window_size: int) -> nn.Module:
	"""Construct exactly one of the two supported baseline models."""
	if args.window_size is not None and args.window_size != window_size:
		raise ValueError(
			f"--window-size={args.window_size} does not match processed data ({window_size})."
		)
	if args.patch_len > window_size:
		raise ValueError("--patch-len cannot exceed the anchor window size.")
	if args.d_model % args.n_heads != 0:
		raise ValueError("--d-model must be divisible by --n-heads.")

	common = {
		"input_size": INPUT_SIZE,
		"window_size": window_size,
		"horizon": 1,
		"patch_len": args.patch_len,
		"d_model": args.d_model,
		"dropout": args.dropout,
		"e_layers": args.e_layers,
		"n_heads": args.n_heads,
		"d_ff": args.d_ff,
		"target_idx": args.target_idx,
	}
	if args.model == "PatchTST":
		return PatchTST(
			**common,
			stride=args.stride,
			revin=False,
			decomposition=False,
		)
	if args.model == "TimeXer":
		if window_size % args.patch_len != 0:
			raise ValueError("TimeXer requires window_size divisible by --patch-len.")
		return TimeXerForecaster(**common, use_norm=False)
	raise ValueError(f"Unsupported model '{args.model}'. Choose PatchTST or TimeXer.")


def run_epoch(
	model: nn.Module,
	loader: DataLoader,
	device: torch.device,
	optimizer: torch.optim.Optimizer | None = None,
	grad_clip: float | None = None,
) -> float:
	training = optimizer is not None
	model.train(training)
	total_loss = 0.0
	total_items = 0
	criterion = nn.MSELoss()

	context = torch.enable_grad() if training else torch.no_grad()
	with context:
		for features, targets in loader:
			features = features.to(device, non_blocking=device.type == "cuda")
			targets = targets.to(device, non_blocking=device.type == "cuda")
			if training:
				optimizer.zero_grad(set_to_none=True)
			predictions = model(features)
			if predictions.shape != targets.shape:
				raise RuntimeError(
					f"Model returned {tuple(predictions.shape)}, expected {tuple(targets.shape)}."
				)
			loss = criterion(predictions, targets)
			if training:
				loss.backward()
				if grad_clip is not None and grad_clip > 0:
					torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
				optimizer.step()
			count = features.shape[0]
			total_loss += loss.item() * count
			total_items += count
	return total_loss / total_items


def save_checkpoint(
	path: Path,
	model: nn.Module,
	optimizer: torch.optim.Optimizer,
	epoch: int,
	best_val_loss: float,
	args: argparse.Namespace,
	anchor_tf: str,
	window_size: int,
) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	torch.save(
		{
			"epoch": epoch,
			"model": args.model,
			"anchor_timeframe": anchor_tf,
			"window_size": window_size,
			"model_state_dict": model.state_dict(),
			"optimizer_state_dict": optimizer.state_dict(),
			"best_val_loss": best_val_loss,
			"args": vars(args),
		},
		path,
	)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train PatchTST or TimeXer baselines.")
	parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
	parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/Baselines"))
	parser.add_argument("--asset", required=True, choices=("AAPL", "EURUSD", "USDCHF", "XAUUSD"))
	parser.add_argument("--model", required=True, choices=MODEL_NAMES)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--epochs", type=int, default=50)
	parser.add_argument("--batch-size", type=int, default=256)
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument("--weight-decay", type=float, default=1e-5)
	parser.add_argument("--patience", type=int, default=5)
	parser.add_argument("--grad-clip", type=float, default=1.0)
	parser.add_argument("--num-workers", type=int, default=0)
	parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
	parser.add_argument("--window-size", type=int, default=None)
	parser.add_argument("--horizon", type=int, default=1)
	parser.add_argument("--patch-len", type=int, default=16)
	parser.add_argument("--stride", type=int, default=8)
	parser.add_argument("--d-model", type=int, default=128)
	parser.add_argument("--d-ff", type=int, default=256)
	parser.add_argument("--e-layers", type=int, default=2)
	parser.add_argument("--n-heads", type=int, default=8)
	parser.add_argument("--dropout", type=float, default=0.1)
	parser.add_argument("--target-idx", type=int, default=TARGET_INDEX)
	args = parser.parse_args()
	if args.horizon != 1:
		parser.error("The preprocessed target contract requires --horizon 1.")
	if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
		parser.error("epochs, batch-size, and patience must be positive.")
	if args.patch_len < 1 or args.stride < 1 or args.n_heads < 1:
		parser.error("patch-len, stride, and n-heads must be positive.")
	if not 0 <= args.target_idx < INPUT_SIZE:
		parser.error(f"target-idx must be in [0, {INPUT_SIZE - 1}].")
	return args


def main() -> None:
	args = parse_args()
	set_seed(args.seed)
	device = resolve_device(args.device)
	splits, anchor_tf, window_size = load_anchor_splits(args.data_dir, args.asset)

	loaders = {
		split: make_loader(
			arrays,
			args.batch_size,
			shuffle=split == "train",
			num_workers=args.num_workers,
			device=device,
			seed=args.seed,
		)
		for split, arrays in splits.items()
	}
	model = build_model(args, window_size).to(device)
	optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

	output_dir = args.checkpoint_dir / args.model / args.asset / str(args.seed)
	output_dir.mkdir(parents=True, exist_ok=True)
	best_path = output_dir / "best_model.pt"
	last_path = output_dir / "last_model.pt"
	history_path = output_dir / "training_history.csv"
	history: list[dict[str, float | int]] = []
	best_val_loss = float("inf")
	best_epoch = 0
	stale_epochs = 0

	print(f"Model: {args.model} | Asset: {args.asset} | Anchor: {anchor_tf}")
	print(f"Window: {window_size} | Device: {device} | Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

	# Initialize tqdm progress bar
	pbar = tqdm(
		total=args.epochs,
		desc="Epoch 000/000",
		unit="epoch",
		dynamic_ncols=True,
		bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
	)

	stopped_early = False
	for epoch in range(1, args.epochs + 1):
		train_loss = run_epoch(model, loaders["train"], device, optimizer, args.grad_clip)
		val_loss = run_epoch(model, loaders["val"], device)
		learning_rate = optimizer.param_groups[0]["lr"]
		history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": learning_rate})
		pd.DataFrame(history).to_csv(history_path, index=False)

		improved = val_loss < best_val_loss
		if improved:
			best_val_loss = val_loss
			best_epoch = epoch
			stale_epochs = 0
			save_checkpoint(best_path, model, optimizer, epoch, best_val_loss, args, anchor_tf, window_size)
		else:
			stale_epochs += 1
		save_checkpoint(last_path, model, optimizer, epoch, best_val_loss, args, anchor_tf, window_size)

		# Update progress bar
		pbar.set_description(f"Epoch {epoch:03d}/{args.epochs}")
		pbar.set_postfix_str(f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {learning_rate:.2e}")
		pbar.update(1)

		if stale_epochs >= args.patience:
			stopped_early = True
			break

	pbar.close()
	if stopped_early:
		print(f"Early stopping after {args.patience} epochs without validation improvement.")

	checkpoint = torch.load(best_path, map_location=device, weights_only=False)
	model.load_state_dict(checkpoint["model_state_dict"])
	test_loss = run_epoch(model, loaders["test"], device)
	print(f"Best epoch: {best_epoch} | test loss: {test_loss:.6f}")
	print(f"Artifacts: {output_dir.resolve()}")


if __name__ == "__main__":
	main()