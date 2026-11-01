"""
HARN — Hierarchical Associative Resonance Network.

HARN is a causal hierarchical forecasting architecture for streaming
financial time series. The model combines multi-scale temporal encoding,
associative memory, hierarchical message passing, and event-driven state
updates to model market dynamics across multiple temporal resolutions.

The architecture operates online: each forward step consumes the latest
window from every hierarchy level, updates only the levels whose bars have
closed, exchanges information across the hierarchy, and produces forecasts
for every level simultaneously.

Architecture
------------
The processing pipeline consists of five stages:

1. Causal Multi-Scale Encoding
   Each hierarchy level independently extracts local temporal patterns
   using gated dilated convolutions, derived candlestick features and
   exponential temporal pooling.

2. Surprise-Gated Associative Memory
   Encoded representations update a persistent matrix-valued memory using
   bounded surprise-driven writes and residual state updates.

3. Cross-Scale Resonance
   Hidden states communicate through a shared low-rank attention mechanism,
   allowing information to flow across all hierarchy levels.

4. Bottom-Up Evidence Aggregation
   Recent lower-level states are summarized into contextual evidence for
   higher-level memory updates.

5. Forecasting
   Each hierarchy level predicts stationary price changes relative to the
   latest observed close, from which absolute prices are reconstructed.

Key Characteristics
-------------------
- causal computation
- streaming inference
- event-driven updates
- hierarchical memory
- cross-scale information exchange
- stationary target prediction
- end-to-end differentiable training

Ablation Variants
------------------
This module additionally supports component-wise ablation experiments via
``ModelConfig.variant``. Every variant shares the exact same architecture,
hyperparameters, and forward-pass control flow — only a single component is
swapped out for a minimal drop-in replacement that removes its function:

    A0  Full HARN (unchanged).
    A1  Remove Surprise-Gated Associative Memory.
        ``SurpriseGatedMemoryCell`` -> ``NullAssociativeMemoryCell``
        (no key-value memory matrix, no surprise-driven writes; state is
        updated with a plain gated recurrent update instead).
    A2  Remove Cross-Scale Resonance.
        ``CrossScaleResonance`` -> ``IdentityResonance``
        (hierarchy levels no longer exchange information; states pass
        through unchanged).
    A3  Remove Bottom-Up Evidence Aggregation.
        ``HistoryReadout`` -> ``ZeroEvidenceReadout``
        (higher levels no longer receive a summary of lower-level history;
        the evidence slot is zero-filled).

Variant selection is centralized in three factory functions
(``build_memory_cell``, ``build_resonance``, ``build_history_readout``) so
that ``HARN.__init__`` and ``HARN.forward`` never need to branch on
``variant`` directly — every variant produces a model with identical tensor
shapes, parameter groups (module names), and I/O contract, which keeps
checkpoints, training code, and evaluation code fully variant-agnostic.
"""

from __future__ import annotations

import math
import warnings
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from ..datasets.timeframes import resolve_timeframes
except ImportError:
    from src.datasets.timeframes import resolve_timeframes


# ---------------------------------------------------------------------------
# Ablation variants
# ---------------------------------------------------------------------------
#: Valid ``ModelConfig.variant`` values. "A0" is the unablated, full model.
VALID_VARIANTS: Tuple[str, ...] = ("A0", "A1", "A2", "A3")

#: Human-readable description of what each variant removes, relative to A0.
ABLATION_DESCRIPTIONS: Dict[str, str] = {
    "A0": "Full HARN (no component removed)",
    "A1": "Surprise-Gated Associative Memory removed",
    "A2": "Cross-Scale Resonance removed",
    "A3": "Bottom-Up Evidence Aggregation removed",
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    """
    Configuration for the Hierarchical Associative Resonance Network.

    The hierarchy is defined by ``timeframes``, ordered from the finest
    resolution to the coarsest. Every hierarchy level shares the same
    architecture while maintaining independent parameters and persistent
    states.

    All model components are constructed directly from this hierarchy.

    ``variant`` selects one of the component-wise ablations described in
    ``ABLATION_DESCRIPTIONS`` (default ``"A0"``, the full model). Every
    variant keeps identical hyperparameters, dimensions, and pipeline
    structure -- only the ablated component's implementation changes.
    """

    timeframes: Tuple[str, ...] = ("M15", "H1", "H4")
    input_dim: int = 5
    d_model: int = 72
    state_dim: int = 72
    mem_key_dim: int = 18
    coupling_rank: int = 36
    dilations: Tuple[int, ...] = (1, 2, 4, 8)
    kernel_size: int = 3
    head_hidden: int = 80
    dropout: float = 0.1
    history_len: int = 4
    close_index: int = 3
    target_scale: float = 10000.0
    variant: str = "A0"

    def __post_init__(self) -> None:
        self.timeframes = resolve_timeframes(self.timeframes, context="model timeframes")
        if len(self.timeframes) < 1:
            raise ValueError("`timeframes` must contain at least one hierarchy level.")
        if len(set(self.timeframes)) != len(self.timeframes):
            raise ValueError(f"`timeframes` must be unique, got {self.timeframes}.")
        if self.variant not in VALID_VARIANTS:
            raise ValueError(
                f"`variant` must be one of {VALID_VARIANTS}, got {self.variant!r}."
            )
        # `CausalMultiScaleEncoder._derived_features` and `HARN._build_anchor`
        # both unbind windows as a *fixed* (open, high, low, close, volume)
        # column order rather than reading `close_index` dynamically. Rather
        # than silently mis-deriving candlestick features for a config that
        # implies otherwise, refuse configs that would violate that baked-in
        # assumption.
        if self.input_dim != 5:
            raise ValueError(
                "`input_dim` must be 5 (OHLCV) -- the encoder's derived "
                f"candlestick features and anchor construction assume a fixed "
                f"(open, high, low, close, volume) column layout, got input_dim={self.input_dim}."
            )
        if self.close_index != 3:
            raise ValueError(
                "`close_index` must be 3 -- derived features and price anchors "
                f"assume 'close' is column index 3 of (open, high, low, close, volume), "
                f"got close_index={self.close_index}."
            )
        if self.history_len < 1:
            raise ValueError(f"`history_len` must be >= 1, got {self.history_len}.")

    @property
    def num_levels(self) -> int:
        """Number of hierarchy levels. Derived from `timeframes`, never set directly."""
        return len(self.timeframes)

    @property
    def ablation_description(self) -> str:
        """Human-readable description of what this variant removes."""
        return ABLATION_DESCRIPTIONS[self.variant]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def _safe_denorm(values: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Element-wise denormalization: ``values * std + mean``."""
    return values * std + mean


def _safe_norm(values: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Element-wise normalization: ``(values - mean) / std``."""
    return (values - mean) / std


# ---------------------------------------------------------------------------
# Building block 1: causal multi-scale local encoder
# ---------------------------------------------------------------------------
class GatedDilatedBranch(nn.Module):
    """
    A single causal, dilated depthwise-separable convolution branch with
    gated linear units (GLU).

    Applies a depthwise convolution followed by a pointwise convolution
    that produces two channels: a value and a gate. The output is the
    value modulated by the sigmoid of the gate.
    """

    def __init__(self, channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            dilation=dilation,
            groups=channels,
            padding=0,
        )
        self.pointwise = nn.Conv1d(channels, 2 * channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        pad = (self.kernel_size - 1) * self.dilation
        x = F.pad(x, (pad, 0))
        h = self.depthwise(x)
        value, gate = self.pointwise(h).chunk(2, dim=1)
        return value * torch.sigmoid(gate)


class CausalDecayPool(nn.Module):
    """
    Exponential temporal pooling with learnable per-channel decay rates.

    Given a tensor of shape ``(B, C, T)``, computes a weighted sum over
    the time axis where weights decay exponentially toward the past. Each
    channel learns its own decay factor via a trainable parameter, allowing
    the network to adapt the effective temporal context per feature.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.raw_decay = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        T = x.size(-1)
        decay = torch.sigmoid(self.raw_decay).clamp(1e-4, 1 - 1e-4)  # (C,)
        steps = torch.arange(T - 1, -1, -1, device=x.device, dtype=x.dtype)  # (T,)
        log_decay = torch.log(decay).unsqueeze(-1)  # (C, 1)
        # Weight for time step t: (1-decay) * decay^(T-1-t)
        weight = (1.0 - decay).unsqueeze(-1) * torch.exp(steps.unsqueeze(0) * log_decay)  # (C, T)
        return torch.einsum("bct,ct->bc", x, weight)


class CausalMultiScaleEncoder(nn.Module):
    """
    Multi-scale causal encoder for a single hierarchy level.

    Combines:
    - linear projections of raw OHLCV features,
    - derived candlestick features (close change, body, range, wicks, volume),
    - multiple gated dilated convolution branches,
    - learnable branch mixing,
    - exponential temporal pooling,
    - a short-history residual path (tail projection),

    to produce a fixed-length representation for each observation window.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        dilations: Tuple[int, ...],
        kernel_size: int,
        dropout: float,
        close_index: int,
    ) -> None:
        super().__init__()
        self.close_index = close_index
        self.input_proj = nn.Linear(input_dim, d_model)
        self.derived_proj = nn.Linear(6, d_model)
        self.branches = nn.ModuleList(
            [GatedDilatedBranch(d_model, kernel_size, d) for d in dilations]
        )
        self.branch_mix = nn.Parameter(torch.zeros(len(dilations)))
        self.post_mix = nn.Conv1d(d_model, d_model, 1)
        self.tail_proj = nn.Linear(2 * d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.pool = CausalDecayPool(d_model)
        self.dropout = nn.Dropout(dropout)

    def _derived_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute derived candlestick features from raw OHLCV bars.

        Args:
            x: (B, T, 5) with columns (open, high, low, close, volume).

        Returns:
            (B, T, 6) with columns (close_change, body, candle_range,
            upper_wick, lower_wick, volume).
        """
        open_, high, low, close, volume = x.unbind(dim=-1)
        prev_close = torch.cat([close[:, :1], close[:, :-1]], dim=1)
        close_change = close - prev_close
        body = close - open_
        candle_range = high - low
        upper_wick = high - torch.maximum(open_, close)
        lower_wick = torch.minimum(open_, close) - low
        return torch.stack(
            [close_change, body, candle_range, upper_wick, lower_wick, volume],
            dim=-1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode a window of bars for a single hierarchy level.

        Args:
            x: (B, T, input_dim) raw OHLCV window.

        Returns:
            (B, d_model) fixed-length latent representation.
        """
        h = self.input_proj(x) + self.derived_proj(self._derived_features(x))
        h = self.dropout(h)
        h = h.transpose(1, 2)  # (B, d_model, T)

        # Multi-branch dilated convolutions with learnable softmax mixing.
        branch_outs = torch.stack([branch(h) for branch in self.branches], dim=0)
        weights = torch.softmax(self.branch_mix, dim=0).view(-1, 1, 1, 1)
        mixed = (branch_outs * weights).sum(dim=0)
        mixed = self.post_mix(mixed)
        mixed = mixed + h  # residual connection
        mixed = self.norm(mixed.transpose(1, 2)).transpose(1, 2)

        # Summarise with exponential pooling plus a short-history tail.
        pooled = self.pool(mixed)
        tail_len = min(4, mixed.size(-1))
        tail = mixed[:, :, -tail_len:].mean(dim=-1)
        summary = pooled + self.tail_proj(torch.cat([mixed[:, :, -1], tail], dim=-1))
        return self.norm(summary)


# ---------------------------------------------------------------------------
# Building block 2: Surprise-Gated Delta-Memory cell
# ---------------------------------------------------------------------------
class SurpriseGatedMemoryCell(nn.Module):
    """
    Persistent associative memory for one hierarchy level.

    The memory stores key-value associations accumulated over time. Incoming
    representations generate keys, values and queries used to retrieve prior
    knowledge, compute prediction surprise, and update memory through bounded
    writes.

    The cell simultaneously produces:
    - an updated associative memory (M_new),
    - an updated latent state (state),

    which become the persistent state for the next streaming step.
    """

    def __init__(
        self,
        input_dim_cell: int,
        state_dim: int,
        mem_key_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.mem_key_dim = mem_key_dim
        self.input_norm = nn.LayerNorm(input_dim_cell)
        self.state_norm = nn.LayerNorm(state_dim)
        self.to_key = nn.Linear(input_dim_cell, mem_key_dim)
        self.to_value = nn.Linear(input_dim_cell, state_dim)
        self.to_query = nn.Linear(input_dim_cell, mem_key_dim)
        self.to_gate = nn.Linear(input_dim_cell + 2 * state_dim, 2 * state_dim)
        self.read_proj = nn.Linear(state_dim, state_dim)
        self.candidate_proj = nn.Linear(input_dim_cell + 2 * state_dim, state_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        M_prev: torch.Tensor,
        state_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update the memory and latent state.

        Args:
            x: (B, input_dim_cell) current input features (encoder output
                possibly concatenated with context/evidence).
            M_prev: (B, state_dim, mem_key_dim) previous memory matrix.
            state_prev: (B, state_dim) previous latent state.

        Returns:
            state: (B, state_dim) new latent state.
            M_new: (B, state_dim, mem_key_dim) updated memory matrix.
        """
        x = self.input_norm(x)
        k = F.normalize(self.to_key(x), dim=-1)          # (B, mem_key_dim)
        v = torch.tanh(self.to_value(x))                 # (B, state_dim)
        q = F.normalize(self.to_query(x), dim=-1)        # (B, mem_key_dim)

        # Expected value retrieved from memory using key k.
        expected = torch.bmm(M_prev, k.unsqueeze(-1)).squeeze(-1)  # (B, state_dim)
        surprise = v - expected

        # Gating signals: write strength eta and read gate.
        gate_in = torch.cat([x, state_prev, expected], dim=-1)
        eta_raw, read_raw = self.to_gate(gate_in).chunk(2, dim=-1)
        eta = torch.sigmoid(eta_raw)                     # (B, state_dim)
        read_gate = torch.sigmoid(read_raw)               # (B, state_dim)

        # Bounded write using the surprise signal.
        write = torch.bmm(surprise.unsqueeze(-1), k.unsqueeze(1))  # (B, state_dim, mem_key_dim)
        write = torch.tanh(write / math.sqrt(self.mem_key_dim))
        M_new = (1.0 - eta).unsqueeze(-1) * M_prev + eta.unsqueeze(-1) * write

        # Retrieve information from the updated memory.
        read = torch.bmm(M_new, q.unsqueeze(-1)).squeeze(-1)  # (B, state_dim)
        candidate = torch.tanh(self.candidate_proj(gate_in))
        state = state_prev + read_gate * self.read_proj(read) + (1 - read_gate) * candidate
        state = self.state_norm(state)
        state = self.dropout(state)
        return state, M_new


class NullAssociativeMemoryCell(nn.Module):
    """
    Ablation A1 — drop-in replacement for ``SurpriseGatedMemoryCell`` with
    the Surprise-Gated Associative Memory component removed.

    There is no persistent key-value memory matrix, no key/query/value
    retrieval, and no surprise-driven bounded write. The latent state is
    instead updated with a plain gated-recurrent-style update computed
    directly from the current input and previous state (comparable in
    spirit to a GRU cell). The memory tensor ``M`` is threaded through
    unchanged so that state management (``reset_states``/``detach_states``)
    and the surrounding pipeline require no special-casing: it becomes an
    inert, unused buffer for this variant.
    """

    def __init__(
        self,
        input_dim_cell: int,
        state_dim: int,
        mem_key_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        # Kept only so the constructor signature matches the ablated
        # component and `M` buffers stay shape-compatible with A0/A2/A3.
        self.mem_key_dim = mem_key_dim
        self.input_norm = nn.LayerNorm(input_dim_cell)
        self.state_norm = nn.LayerNorm(state_dim)
        self.candidate_proj = nn.Linear(input_dim_cell + state_dim, state_dim)
        self.update_gate = nn.Linear(input_dim_cell + state_dim, state_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        M_prev: torch.Tensor,
        state_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update the latent state without any associative memory.

        Args:
            x: (B, input_dim_cell) current input features.
            M_prev: (B, state_dim, mem_key_dim) unused, passed through as-is.
            state_prev: (B, state_dim) previous latent state.

        Returns:
            state: (B, state_dim) new latent state.
            M_new: (B, state_dim, mem_key_dim) identical to ``M_prev``
                (the associative memory is ablated, so it is never written).
        """
        x = self.input_norm(x)
        gate_in = torch.cat([x, state_prev], dim=-1)
        candidate = torch.tanh(self.candidate_proj(gate_in))
        update = torch.sigmoid(self.update_gate(gate_in))
        state = (1.0 - update) * state_prev + update * candidate
        state = self.state_norm(state)
        state = self.dropout(state)
        return state, M_prev


def build_memory_cell(
    variant: str,
    input_dim_cell: int,
    state_dim: int,
    mem_key_dim: int,
    dropout: float,
) -> nn.Module:
    """
    Factory for the per-level memory cell, selected by ablation variant.

    All variants expose the identical interface
    ``forward(x, M_prev, state_prev) -> (state, M_new)``, so callers never
    need to know which concrete implementation is active.
    """
    if variant == "A1":
        return NullAssociativeMemoryCell(input_dim_cell, state_dim, mem_key_dim, dropout)
    return SurpriseGatedMemoryCell(input_dim_cell, state_dim, mem_key_dim, dropout)


# ---------------------------------------------------------------------------
# Building block 3: Cross-Scale Resonance
# ---------------------------------------------------------------------------
class CrossScaleResonance(nn.Module):
    """
    Performs hierarchical communication across all latent states.

    Each hierarchy level exchanges information with every other level through
    a shared low-rank attention mechanism. The resulting residual updates
    allow both fine- and coarse-scale representations to influence one
    another while preserving each level's individual dynamics.
    """

    def __init__(self, state_dim: int, rank: int, dropout: float, num_levels: int) -> None:
        super().__init__()
        self.rank = rank
        self.num_levels = num_levels
        self.q_proj = nn.Linear(state_dim, rank, bias=False)
        self.k_proj = nn.Linear(state_dim, rank, bias=False)
        self.v_proj = nn.Linear(state_dim, rank, bias=False)
        self.out_proj = nn.Linear(rank, state_dim, bias=False)
        # Learnable pairwise bias to encourage cross-talk between levels.
        self.strength = nn.Parameter(torch.eye(num_levels) * 1.2 + 0.1)
        self.norm = nn.LayerNorm(state_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Cross-scale attention residual update.

        Args:
            states: (B, num_levels, state_dim) latent states ordered
                    from finest to coarsest resolution.

        Returns:
            (B, num_levels, state_dim) updated states after cross-scale
            communication.
        """
        q = self.q_proj(states)  # (B, num_levels, rank)
        k = self.k_proj(states)
        v = self.v_proj(states)

        energy = torch.einsum("bir,bjr->bij", q, k) / math.sqrt(self.rank)
        energy = energy + self.strength.unsqueeze(0)
        weights = torch.softmax(energy, dim=-1)
        messages = torch.einsum("bij,bjr->bir", weights, v)
        update = self.out_proj(messages)
        return self.norm(states + self.dropout(update))


class IdentityResonance(nn.Module):
    """
    Ablation A2 — drop-in replacement for ``CrossScaleResonance`` with the
    Cross-Scale Resonance component removed.

    Hierarchy levels no longer exchange information through shared
    attention: the module is a pure passthrough with no learnable
    parameters, so each level's latent state evolves independently of every
    other level.
    """

    def __init__(self, state_dim: int, num_levels: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.num_levels = num_levels

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Passthrough: returns the input states unchanged.

        Args:
            states: (B, num_levels, state_dim) latent states.

        Returns:
            (B, num_levels, state_dim) identical to ``states``.
        """
        return states


def build_resonance(
    variant: str,
    state_dim: int,
    rank: int,
    dropout: float,
    num_levels: int,
) -> nn.Module:
    """
    Factory for the cross-scale communication module, selected by variant.

    All variants expose the identical interface
    ``forward(states) -> updated_states`` with matching tensor shapes, so
    callers never need to know which concrete implementation is active.
    """
    if variant == "A2":
        return IdentityResonance(state_dim, num_levels)
    return CrossScaleResonance(state_dim, rank, dropout, num_levels)


# ---------------------------------------------------------------------------
# Building block 4: bottom-up history readout
# ---------------------------------------------------------------------------
class HistoryReadout(nn.Module):
    """
    Summarizes recent lower-level hidden states into bottom-up evidence.

    The module applies attention over a fixed-length history buffer while
    biasing recent observations. The resulting representation provides
    bottom-up evidence for updating the next hierarchy level.
    """

    def __init__(self, state_dim: int, dropout: float) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, state_dim) * 0.02)
        self.key_proj = nn.Linear(state_dim, state_dim)
        self.value_proj = nn.Linear(state_dim, state_dim)
        self.scale = state_dim ** -0.5
        self.recency_bias = nn.Parameter(torch.tensor(0.2))
        self.norm = nn.LayerNorm(state_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """
        Attend over a history buffer.

        Args:
            history: (B, K, state_dim) buffer of past states from a
                     lower hierarchy level (K = history length).

        Returns:
            (B, state_dim) context vector summarising the lower-level
            history.
        """
        B, K, _ = history.shape
        q = self.query.expand(B, -1).unsqueeze(1)                # (B, 1, state_dim)
        k = self.key_proj(history)                               # (B, K, state_dim)
        v = self.value_proj(history)
        recency = torch.linspace(-1.0, 1.0, steps=K, device=history.device, dtype=history.dtype)
        scores = torch.bmm(q, k.transpose(1, 2)) * self.scale \
                 + self.recency_bias * recency.view(1, 1, K)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        pooled = torch.bmm(attn, v).squeeze(1)                  # (B, state_dim)
        ema = history.mean(dim=1)                               # (B, state_dim)
        latest = history[:, -1]                                 # (B, state_dim)
        return self.norm(pooled + 0.35 * latest + 0.15 * ema)


class ZeroEvidenceReadout(nn.Module):
    """
    Ablation A3 — drop-in replacement for ``HistoryReadout`` with the
    Bottom-Up Evidence Aggregation component removed.

    Ignores the lower-level history buffer entirely and returns a zero
    vector, so the higher hierarchy level's memory cell receives no
    summarized bottom-up evidence in its evidence slot (equivalent to that
    input being absent, while keeping tensor shapes -- and therefore the
    memory cell's input width -- identical across all variants).
    """

    def __init__(self, state_dim: int) -> None:
        super().__init__()
        self.state_dim = state_dim

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """
        Args:
            history: (B, K, state_dim) buffer of past states from a
                     lower hierarchy level (ignored).

        Returns:
            (B, state_dim) zero vector -- no bottom-up evidence.
        """
        return history.new_zeros(history.size(0), self.state_dim)


def build_history_readout(variant: str, state_dim: int, dropout: float) -> nn.Module:
    """
    Factory for the bottom-up evidence readout, selected by ablation variant.

    All variants expose the identical interface
    ``forward(history) -> evidence`` with matching tensor shapes, so callers
    never need to know which concrete implementation is active.
    """
    if variant == "A3":
        return ZeroEvidenceReadout(state_dim)
    return HistoryReadout(state_dim, dropout)


# ---------------------------------------------------------------------------
# Building block 5: forecasting head
# ---------------------------------------------------------------------------
class ForecastingHead(nn.Module):
    """
    Predicts stationary price changes for one hierarchy level.

    The forecasting head combines:
    - the current latent state,
    - cross-scale context,
    - recent price anchors,

    to estimate price movement in basis points relative to the latest close.
    Absolute prices are reconstructed from the predicted relative change.
    """

    def __init__(
        self,
        state_dim: int,
        context_dim: int,
        anchor_dim: int,
        hidden: int,
        dropout: float,
        target_scale: float,
    ) -> None:
        super().__init__()
        self.target_scale = float(target_scale)
        self.net = nn.Sequential(
            nn.Linear(state_dim + context_dim + anchor_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.residual = nn.Linear(state_dim + anchor_dim, 1)

    def forward(
        self,
        state: torch.Tensor,
        context: torch.Tensor,
        anchor: torch.Tensor,
        baseline_price: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute prediction.

        Args:
            state: (B, state_dim) latent state for this level.
            context: (B, context_dim) cross-scale context for this level.
            anchor: (B, anchor_dim) recent-price anchor features.
            baseline_price: (B,) absolute latest close price.

        Returns:
            delta_bps: (B, 1) predicted stationary change in basis points.
            abs_price: (B, 1) reconstructed absolute price:
                       baseline * (1 + delta_bps / target_scale).
        """
        features = torch.cat([state, context, anchor], dim=-1)
        delta_bps = self.net(features) + self.residual(torch.cat([state, anchor], dim=-1))
        baseline_price = baseline_price.view(-1, 1)
        abs_price = baseline_price * (1.0 + delta_bps / self.target_scale)
        return delta_bps, abs_price


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------
class HARN(nn.Module):
    """
    Hierarchical Associative Resonance Network.

    The model maintains one processing pipeline for every hierarchy level.
    Each pipeline consists of:

        Encoder
            ↓
        Associative Memory
            ↓
        Cross-Scale Resonance
            ↓
        Forecasting Head

    During streaming inference, hierarchy levels update asynchronously
    according to event flags while persistent memories remain active across
    time.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.timeframes: Tuple[str, ...] = config.timeframes
        self.variant: str = config.variant
        L = config.num_levels
        d, s = config.d_model, config.state_dim
        mk, rk = config.mem_key_dim, config.coupling_rank
        self.anchor_dim = 6

        # Per-level causal encoders. All levels share the same architecture
        # but maintain separate weights.
        self.encoders = nn.ModuleList(
            [
                CausalMultiScaleEncoder(
                    config.input_dim, d, config.dilations, config.kernel_size, config.dropout, config.close_index
                )
                for _ in range(L)
            ]
        )

        # Per-level memory cells (Surprise-Gated Associative Memory, or its
        # A1 ablation replacement). The finest level (index 0) receives only
        # its encoder features and cross-scale context; every higher level
        # additionally receives bottom-up evidence from the level directly
        # below, making its cell input wider by `state_dim`. Variant
        # selection is centralized in `build_memory_cell` -- this is the
        # only place that needs to know it exists.
        self.cells = nn.ModuleList(
            [
                build_memory_cell(
                    config.variant,
                    d + s if level == 0 else d + s + s,
                    s,
                    mk,
                    config.dropout,
                )
                for level in range(L)
            ]
        )

        # Shared cross-scale resonance coupling across all L levels (or its
        # A2 ablation replacement).
        self.resonance = build_resonance(config.variant, s, rk, config.dropout, num_levels=L)

        # Bottom-up history readouts: one per adjacent level transition (or
        # their A3 ablation replacement). `history_readouts[i]` summarizes
        # level `i`'s recent states into evidence for level `i + 1`. No
        # readout exists above the top level.
        self.history_readouts = nn.ModuleList(
            [build_history_readout(config.variant, s, config.dropout) for _ in range(max(L - 1, 0))]
        )

        # Forecasting heads: stationary delta + reconstructed absolute price,
        # one per level.
        self.heads = nn.ModuleList(
            [
                ForecastingHead(s, s, self.anchor_dim, config.head_hidden, config.dropout, config.target_scale)
                for _ in range(L)
            ]
        )

        # Persistent streaming state, indexed by hierarchy level.
        self.M: List[Optional[torch.Tensor]] = [None] * L
        self.reads: List[Optional[torch.Tensor]] = [None] * L
        # History buffers only exist for levels that feed a level above them
        # (i.e. every level except the top one).
        self.history: List[deque] = [deque(maxlen=config.history_len) for _ in range(max(L - 1, 0))]

        # Dataset calibration buffers for reconstruction, one row per level.
        # (`target_scale` is not stored as a buffer here: it is a plain
        # config-derived float already threaded into every `ForecastingHead`
        # at construction time, so a second copy on `HARN` itself would be
        # dead weight in the state dict.)
        self.register_buffer("feature_mean", torch.zeros(L, config.input_dim))
        self.register_buffer("feature_std", torch.ones(L, config.input_dim))

        # Tracks whether `set_feature_statistics` has ever been called. Until
        # it is, `feature_mean`/`feature_std` are identity (0/1), so anchors
        # and denormalized windows are computed in *normalized* units rather
        # than real price units -- silently wrong for anything but a smoke
        # test. `forward` warns (once) if this is still False.
        self._stats_calibrated: bool = False

        self.apply(self._init_weights)

    @property
    def device(self) -> torch.device:
        """Convenience accessor: the device holding this model's parameters."""
        return self.feature_mean.device

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Initialize weights using Xavier/Kaiming schemes appropriate for the layer type."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_uniform_(module.weight, nonlinearity="linear")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    # Calibration and state management
    # ------------------------------------------------------------------
    def set_feature_statistics(self, stats: Dict[str, Dict[str, list]]) -> None:
        """
        Load per-level feature mean/std statistics from a processed payload.

        `stats` must contain one entry per name in `self.timeframes`
        (e.g. ``stats["m15"]``, ``stats["h1"]``, ...), each a dict with
        ``"mean"`` and ``"std"`` lists of length ``config.input_dim``.
        """
        with torch.no_grad():
            for idx, key in enumerate(self.timeframes):
                if key not in stats:
                    raise KeyError(f"Missing feature statistics for '{key}'.")
                block = stats[key]
                mean = torch.as_tensor(block["mean"], dtype=self.feature_mean.dtype, device=self.feature_mean.device)
                std = torch.as_tensor(block["std"], dtype=self.feature_std.dtype, device=self.feature_std.device)
                std = torch.where(std == 0.0, torch.ones_like(std), std)
                self.feature_mean[idx].copy_(mean)
                self.feature_std[idx].copy_(std)
        self._stats_calibrated = True

    def _denormalize_window(self, window: torch.Tensor, level: int) -> torch.Tensor:
        """Convert a normalized window back to absolute scale using stored statistics."""
        mean = self.feature_mean[level].view(1, 1, -1)
        std = self.feature_std[level].view(1, 1, -1)
        return _safe_denorm(window, mean, std)

    def _build_anchor(self, window: torch.Tensor, level: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build price anchors for one hierarchy level.

        Args:
            window: (B, T, input_dim) normalized OHLCV window.
            level: integer index of the hierarchy level.

        Returns:
            baseline_close: (B,) absolute latest close price.
            anchor_features: (B, 6) stationary basis-point features for
                the forecasting head.
        """
        abs_window = self._denormalize_window(window, level)
        open_, high, low, close, _ = abs_window.unbind(dim=-1)

        last_close = close[:, -1]
        prev_close = close[:, -2] if close.size(1) > 1 else last_close
        prev2_close = close[:, -3] if close.size(1) > 2 else prev_close

        tail_len = min(4, close.size(1))
        close_tail = close[:, -tail_len:]
        high_tail = high[:, -tail_len:]
        low_tail = low[:, -tail_len:]
        open_tail = open_[:, -tail_len:]

        eps = torch.finfo(last_close.dtype).eps
        safe_close = torch.clamp(last_close, min=eps)
        scale = self.config.target_scale

        # All anchor features are expressed in the same stationary bps scale
        # as the prediction target.  Baseline close is kept in absolute price
        # space because it is used multiplicatively to reconstruct absolute
        # price from the predicted relative change, a scale-invariant
        # operation.
        delta_1_bps = (last_close - prev_close) / safe_close * scale
        delta_2_bps = (prev_close - prev2_close) / safe_close * scale
        mean_dev_bps = (close_tail.mean(dim=1) - last_close) / safe_close * scale
        range_bps = (high_tail.max(dim=1).values - low_tail.min(dim=1).values) / safe_close * scale
        body_bps = (close_tail - open_tail).mean(dim=1) / safe_close * scale
        volume_z = window[:, -tail_len:, 4].mean(dim=1)   # already normalized

        anchor = torch.stack(
            [delta_1_bps, delta_2_bps, mean_dev_bps, range_bps, body_bps, volume_z],
            dim=-1,
        )
        return last_close, anchor

    def reset_states(self, batch_size: int, device: torch.device) -> None:
        """Reset persistent streaming state for all hierarchy levels."""
        s, mk = self.config.state_dim, self.config.mem_key_dim
        L = self.config.num_levels
        self.M = [torch.zeros(batch_size, s, mk, device=device) for _ in range(L)]
        self.reads = [torch.zeros(batch_size, s, device=device) for _ in range(L)]

        for buf in self.history:
            buf.clear()
        for buf in self.history:
            for _ in range(self.config.history_len):
                buf.append(torch.zeros(batch_size, s, device=device))

    def detach_states(self) -> None:
        """
        Detach all persistent state tensors from the autograd graph.

        Used for truncated BPTT: call after every ``backward()`` so the next
        window of steps does not try to backprop through already-freed graphs.
        """
        if self.M[0] is None:
            return
        self.M = [m.detach() if m is not None else None for m in self.M]
        self.reads = [r.detach() if r is not None else None for r in self.reads]
        for buf in self.history:
            for i in range(len(buf)):
                buf[i] = buf[i].detach()

    # ------------------------------------------------------------------
    def forward(
        self,
        windows: Dict[str, torch.Tensor],
        update_flags: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Execute one streaming update.

        Processing sequence:
        1. Encode every hierarchy window.
        2. Exchange contextual information across previous hidden states.
        3. Update associative memories.
        4. Aggregate bottom-up evidence.
        5. Exchange information across updated states.
        6. Produce forecasts for every hierarchy level.

        Only hierarchy levels whose update flag is active perform memory
        updates; inactive levels preserve their previous persistent state.

        Args:
            windows: dict mapping timeframe name to tensor of shape
                     (B, T_level, input_dim).
            update_flags: optional dict mapping timeframe name to boolean
                         tensor of shape (B,) indicating whether the bar
                         for that level has closed.

        Returns:
            dict with keys:
                pred_delta_{name}: (B, 1) stationary bps prediction.
                pred_{name}: (B, 1) absolute price forecast.
                baseline_{name}: (B, 1) latest close used for reconstruction.
        """
        names = self.timeframes
        L = len(names)
        for name in names:
            if name not in windows:
                raise KeyError(f"`windows` is missing required level '{name}'.")

        if not self._stats_calibrated:
            warnings.warn(
                "HARN.forward() called before set_feature_statistics(): price "
                "anchors and denormalized windows will be computed as if "
                "mean=0/std=1, which is only correct for already-unnormalized "
                "input or synthetic smoke tests. Call model.set_feature_statistics(...) "
                "with the dataset's real statistics before training/evaluating.",
                stacklevel=2,
            )

        B = windows[names[0]].size(0)
        if self.M[0] is None or self.M[0].size(0) != B:
            self.reset_states(B, windows[names[0]].device)

        # 1) Encode every level's raw window independently.
        encoded = [self.encoders[i](windows[names[i]]) for i in range(L)]

        # 2) Top-down prior: resonate the *previous* step's states before
        #    any memory updates happen this step.
        prior_states = torch.stack(self.reads, dim=1)  # (B, L, state_dim)
        prior_ctx = self.resonance(prior_states)

        # 3) Build every level's price anchor up front (independent of
        #    memory state, so order doesn't matter).
        baselines: List[torch.Tensor] = []
        anchors: List[torch.Tensor] = []
        for i in range(L):
            baseline, anchor = self._build_anchor(windows[names[i]], i)
            baselines.append(baseline)
            anchors.append(anchor)

        new_reads: List[torch.Tensor] = [None] * L  # type: ignore[list-item]
        new_M: List[torch.Tensor] = [None] * L  # type: ignore[list-item]

        # 4) Base level (index 0) updates unconditionally on every step.
        x0 = torch.cat([encoded[0], prior_ctx[:, 0]], dim=-1)
        new_reads[0], new_M[0] = self.cells[0](x0, self.M[0], self.reads[0])
        if self.history:  # only if there is a level above to feed
            self.history[0].append(new_reads[0])

        # 5) Every level above updates conditionally on its update flag,
        #    consuming bottom-up evidence summarized from the level below.
        for i in range(1, L):
            name = names[i]
            flag = None if update_flags is None else update_flags.get(name)
            if flag is None:
                # No flag supplied for this level: default to updating on
                # every step (lock-step hierarchy).
                should_consider_update = True
                mask = None
            else:
                should_consider_update = bool(flag.any())
                mask = flag.float().view(-1, 1)

            if should_consider_update:
                lower_hist = torch.stack(list(self.history[i - 1]), dim=1)
                evidence = self.history_readouts[i - 1](lower_hist)
                x_i = torch.cat([encoded[i], evidence, prior_ctx[:, i]], dim=-1)
                cand_read, cand_M = self.cells[i](x_i, self.M[i], self.reads[i])
                if mask is None:
                    new_read_i, new_M_i = cand_read, cand_M
                else:
                    # Masked update: only overwrite state for bars that closed.
                    new_read_i = (1 - mask) * self.reads[i] + mask * cand_read
                    new_M_i = (1 - mask.unsqueeze(-1)) * self.M[i] + mask.unsqueeze(-1) * cand_M
            else:
                new_read_i, new_M_i = self.reads[i], self.M[i]

            new_reads[i] = new_read_i
            new_M[i] = new_M_i
            if i < L - 1:  # only if there is a level above to feed
                self.history[i].append(new_read_i)

        self.reads = new_reads
        self.M = new_M

        # 6) Bottom-up fusion: resonate the just-updated states for prediction.
        posterior_states = torch.stack(new_reads, dim=1)
        posterior_ctx = self.resonance(posterior_states)

        # 7) Forecasting heads, one per level.
        outputs: Dict[str, torch.Tensor] = {}
        for i, name in enumerate(names):
            pred_delta, pred_abs = self.heads[i](
                new_reads[i], posterior_ctx[:, i], anchors[i], baselines[i]
            )
            outputs[f"pred_delta_{name}"] = pred_delta
            outputs[f"pred_{name}"] = pred_abs
            outputs[f"baseline_{name}"] = baselines[i].view(-1, 1)

        return outputs

    def forward_positional(
        self,
        *windows: torch.Tensor,
        update_flags: Optional[Sequence[torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Backward-compatible convenience wrapper.

        Accepts windows positionally in hierarchy order (finest to
        coarsest) and update flags positionally for every level above the
        base.  Internally builds the generic dict-based call — it does
        not hardcode any level names or level count.

        Example for a 3-level config:
            ``model.forward_positional(m15, h1, h4, update_flags=(is_h1_update, is_h4_update))``
        """
        names = self.timeframes
        if len(windows) != len(names):
            raise ValueError(f"Expected {len(names)} windows for levels {names}, got {len(windows)}.")
        windows_dict = dict(zip(names, windows))
        flags_dict = None
        if update_flags is not None:
            if len(update_flags) != len(names) - 1:
                raise ValueError(
                    f"Expected {len(names) - 1} update flags for levels {names[1:]}, got {len(update_flags)}."
                )
            flags_dict = dict(zip(names[1:], update_flags))
        return self.forward(windows_dict, flags_dict)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def count_parameters(model: nn.Module) -> int:
    """Total number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def parameter_breakdown(model: "HARN") -> Dict[str, int]:
    """Breakdown of trainable parameters by component group.

    Group labels are annotated with ``[ablated]`` when the corresponding
    component has been replaced under the model's active variant, so the
    printed breakdown makes the active ablation immediately visible.
    """
    names = model.timeframes
    v = model.variant
    tag = lambda label, ablated_in: f"{label} [ablated]" if v in ablated_in else label
    groups = {
        f"encoders ({'+'.join(names)})": list(model.encoders),
        tag(f"memory cells ({'+'.join(names)})", {"A1"}): list(model.cells),
        tag("cross-scale resonance", {"A2"}): [model.resonance],
        tag("history readouts", {"A3"}): list(model.history_readouts),
        "forecasting heads": list(model.heads),
    }
    return {name: sum(count_parameters(m) for m in mods) for name, mods in groups.items()}


# ---------------------------------------------------------------------------
# Self-contained demo / smoke test
# ---------------------------------------------------------------------------
def _synthetic_ohlcv(batch_size: int, length: int, input_dim: int) -> torch.Tensor:
    """
    Generate a plausible positive-price OHLCV batch for smoke-testing.

    Prices start around 1.1 and drift with small steps, guaranteeing
    positive closes so that bps-relative anchor features remain meaningful.
    """
    steps = torch.randn(batch_size, length) * 0.0005
    close = 1.1 + torch.cumsum(steps, dim=1)
    open_ = close + torch.randn(batch_size, length) * 0.0002
    high = torch.maximum(open_, close) + torch.rand(batch_size, length) * 0.0003
    low = torch.minimum(open_, close) - torch.rand(batch_size, length) * 0.0003
    volume = torch.rand(batch_size, length) * 100 + 10
    bars = torch.stack([open_, high, low, close, volume], dim=-1)
    if input_dim != 5:
        raise ValueError("Synthetic generator assumes OHLCV (input_dim=5).")
    return bars


if __name__ == "__main__":
    import argparse

    demo_parser = argparse.ArgumentParser(description="HARN ablation smoke test.")
    demo_parser.add_argument("--variant", type=str, default="A0", choices=VALID_VARIANTS,
                              help="Ablation variant to instantiate (default: A0, full model).")
    demo_args = demo_parser.parse_args()

    torch.manual_seed(0)

    # The hierarchy is defined entirely by `timeframes` here.  Changing this
    # to e.g. ("m5", "m15", "h1", "h4", "d1") creates a 5-level hierarchy
    # without any other code changes.
    config = ModelConfig(timeframes=("m15", "h1", "h4"), variant=demo_args.variant)
    model = HARN(config)
    level_names = config.timeframes
    num_levels = config.num_levels

    total_params = count_parameters(model)
    print(f"Variant: {config.variant} — {config.ablation_description}")
    print(f"Hierarchy levels: {level_names}")
    print(f"Total trainable parameters: {total_params:,}")
    for name, n in parameter_breakdown(model).items():
        print(f"  {name}: {n:,}")

    batch_size = 4
    window_lengths = [16] + [8] * (num_levels - 1)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.reset_states(batch_size, torch.device("cpu"))

    num_steps = 20
    detach_every = 5

    print("\nSimulated online-streaming training loop:")
    # Truncated BPTT: accumulate loss over `detach_every` steps and call
    # backward() only at the boundary, then detach states so the next window
    # of steps does not try to backprop through already-freed graphs.
    optimizer.zero_grad()
    window_loss = 0.0
    for step in range(1, num_steps + 1):
        windows = {
            name: _synthetic_ohlcv(batch_size, window_lengths[i], config.input_dim)
            for i, name in enumerate(level_names)
        }
        # Higher levels update less frequently, mimicking real bar closures.
        update_flags = {
            name: torch.rand(batch_size) < (0.5 ** i)
            for i, name in enumerate(level_names[1:], start=1)
        }

        out = model(windows, update_flags)

        targets = {name: torch.randn(batch_size, 1) * 10.0 for name in level_names}
        loss = sum(
            F.mse_loss(out[f"pred_delta_{name}"], targets[name]) for name in level_names
        )
        window_loss = window_loss + loss

        update_counts = " | ".join(
            f"{name} updates {int(update_flags[name].sum())}/{batch_size}"
            for name in level_names[1:]
        )
        print(f"  step {step:02d} | loss {loss.item():.4f}" + (f" | {update_counts}" if update_counts else ""))

        if step % detach_every == 0:
            window_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            model.detach_states()
            window_loss = 0.0