"""Canonical timeframe names and ordered hierarchy resolution."""

from __future__ import annotations

from typing import Iterable, Sequence

# Canonical spelling used by preprocessing payloads.
_TIMEFRAME_TO_MINUTES: dict[str, int] = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "M30": 30,
    "H1": 60,
    "H2": 120,
    "H4": 240,
    "H6": 360,
    "H8": 480,
    "H12": 720,
    "D1": 1440,
    "W1": 10080,
}


def canonicalize_timeframe(timeframe: str) -> str:
    """Return the canonical, case-insensitive spelling of *timeframe*."""
    if not isinstance(timeframe, str):
        raise ValueError(f"Timeframe names must be strings, got {type(timeframe).__name__}.")
    name = timeframe.strip().upper()
    if not name:
        raise ValueError("Timeframe names must not be empty.")
    if name not in _TIMEFRAME_TO_MINUTES:
        raise ValueError(f"Unsupported timeframe '{timeframe}'.")
    return name


def timeframe_to_minutes(timeframe: str) -> int:
    """Return the duration in minutes for a timeframe alias."""
    return _TIMEFRAME_TO_MINUTES[canonicalize_timeframe(timeframe)]


def resolve_timeframes(
    requested: Sequence[str] | Iterable[str],
    declared: Sequence[str] | Iterable[str] | None = None,
    *,
    context: str = "timeframes",
) -> tuple[str, ...]:
    """Resolve aliases while preserving the declared hierarchy order.

    If *declared* is supplied, it is the authoritative ordered hierarchy. The
    requested sequence must match it position-for-position after case folding;
    no levels are sorted, duplicated, reordered, or silently omitted.
    """
    requested_raw = tuple(requested)
    requested_canonical = tuple(canonicalize_timeframe(tf) for tf in requested_raw)
    if len(set(requested_canonical)) != len(requested_canonical):
        raise ValueError(f"{context} contains duplicate levels after alias resolution: {requested_raw}.")
    if not requested_canonical:
        raise ValueError(f"{context} must contain at least one hierarchy level.")

    if declared is None:
        return requested_canonical

    declared_raw = tuple(declared)
    declared_canonical = tuple(canonicalize_timeframe(tf) for tf in declared_raw)
    if len(set(declared_canonical)) != len(declared_canonical):
        raise ValueError(f"Declared {context} contains duplicate levels: {declared_raw}.")
    if requested_canonical != declared_canonical:
        raise ValueError(
            f"Requested {context} {requested_raw} resolve to {requested_canonical}, "
            f"but the payload declares ordered levels {declared_raw} "
            f"({declared_canonical}). Refusing to reorder or subset hierarchy levels."
        )
    return declared_canonical


def canonical_key_map(container: object, *, label: str) -> dict[str, object]:
    """Map canonical timeframe keys to existing values without copying them."""
    if not isinstance(container, dict):
        raise TypeError(f"{label} must be a dictionary.")
    result: dict[str, object] = {}
    for raw_key, value in container.items():
        canonical = canonicalize_timeframe(raw_key)
        if canonical in result:
            raise ValueError(f"{label} contains duplicate aliases for '{canonical}'.")
        result[canonical] = value
    return result
