"""
Dataset Version Management for MCP Server.

This module provides an LRU-evicting version store that saves DataFrame snapshots
at every mutation point. It enables listing versions, rolling back to any snapshot,
and computing structured diffs between arbitrary version pairs.

Classes:
    VersionEntry: Immutable record of a single version snapshot.
    VersionManager: Ordered store with LRU eviction and pinning support.
"""

import pandas as pd
import numpy as np
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set


@dataclass(frozen=True)
class VersionEntry:
    """Immutable record of a dataset snapshot at a point in the pipeline."""

    version: int
    df: pd.DataFrame
    timestamp: str
    tool: str
    params: Dict[str, Any]
    rows: int
    columns: int
    column_names: tuple  # tuple for immutability
    memory_mb: float

    class Config:
        arbitrary_types_allowed = True


class VersionManager:
    """
    LRU-evicting version store for DataFrame snapshots.

    Responsibilities:
        - Auto-increment version numbers on each snapshot.
        - Evict the oldest *non-pinned* version when capacity is reached.
        - Pin/unpin individual versions to protect them from eviction.
        - Compute rich structural diffs between any two stored versions.

    Args:
        max_versions: Maximum number of snapshots to keep (default 10).
    """

    def __init__(self, max_versions: int = 10):
        self._versions: OrderedDict[int, VersionEntry] = OrderedDict()
        self._next_version: int = 0
        self._max_versions: int = max_versions
        self._pinned: Set[int] = set()

    # =========================================================================
    # Snapshot Management
    # =========================================================================

    def snapshot(
        self,
        df: pd.DataFrame,
        tool: str = "unknown",
        params: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Save a deep-copy snapshot of the DataFrame.

        Args:
            df: The DataFrame to snapshot.
            tool: Name of the tool that triggered this snapshot.
            params: Parameters passed to the tool (for audit trail).

        Returns:
            The version number assigned to this snapshot.
        """
        version = self._next_version
        self._next_version += 1

        memory_bytes = df.memory_usage(deep=True).sum()
        memory_mb = round(memory_bytes / (1024 * 1024), 4)

        entry = VersionEntry(
            version=version,
            df=df.copy(),
            timestamp=datetime.now(timezone.utc).isoformat(),
            tool=tool,
            params=params or {},
            rows=len(df),
            columns=len(df.columns),
            column_names=tuple(df.columns.tolist()),
            memory_mb=memory_mb,
        )

        self._versions[version] = entry

        # Auto-pin version 0 (the original dataset)
        if version == 0:
            self._pinned.add(0)

        # Evict oldest non-pinned if over capacity
        # Never evict the version we just added
        self._evict_if_needed(exclude=version)

        return version

    def _evict_if_needed(self, exclude: Optional[int] = None) -> None:
        """Evict the oldest non-pinned version if we exceed max capacity."""
        while len(self._versions) > self._max_versions:
            evicted = False
            for v in list(self._versions.keys()):
                if v not in self._pinned and v != exclude:
                    del self._versions[v]
                    evicted = True
                    break
            if not evicted:
                # All versions are pinned or protected — allow over-capacity
                break

    # =========================================================================
    # Retrieval
    # =========================================================================

    def get(self, version: int) -> VersionEntry:
        """
        Retrieve a version entry by number.

        Args:
            version: The version number to retrieve.

        Returns:
            The VersionEntry for that version.

        Raises:
            KeyError: If the version doesn't exist (never created or evicted).
        """
        if version not in self._versions:
            available = sorted(self._versions.keys())
            raise KeyError(
                f"Version {version} not found. "
                f"Available versions: {available}. "
                f"It may have been evicted by LRU policy."
            )
        return self._versions[version]

    def get_latest_version(self) -> Optional[int]:
        """Return the latest version number, or None if no versions exist."""
        if not self._versions:
            return None
        return max(self._versions.keys())

    def list_all(self) -> List[Dict[str, Any]]:
        """
        Return metadata for all stored versions (DataFrames excluded).

        Returns:
            List of dicts with version info, ordered by version number.
        """
        result = []
        for v, entry in self._versions.items():
            result.append(
                {
                    "version": entry.version,
                    "tool": entry.tool,
                    "params": entry.params,
                    "timestamp": entry.timestamp,
                    "rows": entry.rows,
                    "columns": entry.columns,
                    "column_names": list(entry.column_names),
                    "memory_mb": entry.memory_mb,
                    "pinned": v in self._pinned,
                }
            )
        return result

    @property
    def total_memory_mb(self) -> float:
        """Total memory consumed by all stored snapshots."""
        return round(sum(e.memory_mb for e in self._versions.values()), 4)

    @property
    def current_version(self) -> int:
        """The most recently created version number (even if evicted)."""
        return self._next_version - 1 if self._next_version > 0 else -1

    @property
    def stored_count(self) -> int:
        """Number of versions currently stored (not evicted)."""
        return len(self._versions)

    # =========================================================================
    # Pinning
    # =========================================================================

    def pin(self, version: int) -> None:
        """
        Protect a version from LRU eviction.

        Raises:
            KeyError: If the version doesn't exist.
        """
        if version not in self._versions:
            raise KeyError(f"Cannot pin version {version}: not found.")
        self._pinned.add(version)

    def unpin(self, version: int) -> None:
        """
        Remove eviction protection from a version.

        Raises:
            KeyError: If the version doesn't exist.
        """
        if version not in self._versions:
            raise KeyError(f"Cannot unpin version {version}: not found.")
        self._pinned.discard(version)

    def is_pinned(self, version: int) -> bool:
        """Check if a version is pinned."""
        return version in self._pinned

    # =========================================================================
    # Diff Engine
    # =========================================================================

    def diff(self, version_a: int, version_b: int) -> Dict[str, Any]:
        """
        Compute a structured diff between two stored versions.

        Produces:
            - Shape delta (rows, columns)
            - Columns added / removed
            - Data type changes for shared columns
            - Missing value changes for shared columns
            - Basic statistical changes (mean, std, min, max) for shared numeric columns
            - Pipeline steps that occurred between the two versions

        Args:
            version_a: The "before" version.
            version_b: The "after" version.

        Returns:
            A rich diff dictionary suitable for display or visualization.

        Raises:
            KeyError: If either version doesn't exist.
        """
        entry_a = self.get(version_a)
        entry_b = self.get(version_b)

        df_a = entry_a.df
        df_b = entry_b.df

        cols_a = set(entry_a.column_names)
        cols_b = set(entry_b.column_names)
        shared_cols = sorted(cols_a & cols_b)

        result: Dict[str, Any] = {
            "version_a": version_a,
            "version_b": version_b,
        }

        # --- Shape diff ---
        result["shape_diff"] = {
            "rows": {
                "before": entry_a.rows,
                "after": entry_b.rows,
                "delta": entry_b.rows - entry_a.rows,
            },
            "columns": {
                "before": entry_a.columns,
                "after": entry_b.columns,
                "delta": entry_b.columns - entry_a.columns,
            },
        }

        # --- Column changes ---
        result["columns_added"] = sorted(cols_b - cols_a)
        result["columns_removed"] = sorted(cols_a - cols_b)

        # --- Dtype changes for shared columns ---
        dtype_changes = {}
        for col in shared_cols:
            dtype_before = str(df_a[col].dtype)
            dtype_after = str(df_b[col].dtype)
            if dtype_before != dtype_after:
                dtype_changes[col] = {"before": dtype_before, "after": dtype_after}
        result["dtype_changes"] = dtype_changes

        # --- Missing value changes for shared columns ---
        missing_diff = {}
        for col in shared_cols:
            missing_before = int(df_a[col].isna().sum())
            missing_after = int(df_b[col].isna().sum())
            if missing_before != missing_after:
                missing_diff[col] = {
                    "before": missing_before,
                    "after": missing_after,
                    "delta": missing_after - missing_before,
                }
        result["missing_values_diff"] = missing_diff

        # --- Stats diff for shared numeric columns ---
        stats_diff = {}
        numeric_shared = [
            col
            for col in shared_cols
            if pd.api.types.is_numeric_dtype(df_a[col])
            and pd.api.types.is_numeric_dtype(df_b[col])
        ]
        for col in numeric_shared:
            stats_before = {
                "mean": _safe_float(df_a[col].mean()),
                "std": _safe_float(df_a[col].std()),
                "min": _safe_float(df_a[col].min()),
                "max": _safe_float(df_a[col].max()),
            }
            stats_after = {
                "mean": _safe_float(df_b[col].mean()),
                "std": _safe_float(df_b[col].std()),
                "min": _safe_float(df_b[col].min()),
                "max": _safe_float(df_b[col].max()),
            }
            # Only include if stats actually changed
            if stats_before != stats_after:
                stats_diff[col] = {"before": stats_before, "after": stats_after}
        result["stats_diff"] = stats_diff

        # --- Pipeline steps between versions ---
        steps_between = []
        # Collect all stored versions between a and b (inclusive of b, exclusive of a)
        lo, hi = min(version_a, version_b), max(version_a, version_b)
        for ver in range(lo + 1, hi + 1):
            if ver in self._versions:
                e = self._versions[ver]
                param_str = ", ".join(
                    f"{pk}={pv!r}"
                    for pk, pv in e.params.items()
                    if pk not in ("dataset_name",)
                )
                step_desc = f"v{e.version}: {e.tool}"
                if param_str:
                    step_desc += f"({param_str})"
                steps_between.append(step_desc)
        result["steps_between"] = steps_between

        # --- Memory diff ---
        result["memory_diff_mb"] = {
            "before": entry_a.memory_mb,
            "after": entry_b.memory_mb,
            "delta": round(entry_b.memory_mb - entry_a.memory_mb, 2),
        }

        return result

    # =========================================================================
    # Reset
    # =========================================================================

    def clear(self) -> None:
        """Wipe all versions and reset counters."""
        self._versions.clear()
        self._pinned.clear()
        self._next_version = 0


# =============================================================================
# Helpers
# =============================================================================


def _safe_float(value: Any) -> Optional[float]:
    """Convert a numpy/pandas scalar to a plain float, handling NaN."""
    if value is None:
        return None
    try:
        f = float(value)
        if np.isnan(f) or np.isinf(f):
            return None
        return round(f, 4)
    except (TypeError, ValueError):
        return None
