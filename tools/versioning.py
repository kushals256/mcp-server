"""
Dataset Versioning Tools for MCP Server.

This module provides MCP tools for inspecting, comparing, and rolling back
dataset versions. Versions are created automatically whenever the dataset
is mutated by any cleaning or transformation tool.

Tools:
    list_versions: Show all available version snapshots with metadata.
    rollback_version: Restore the dataset to a previous version.
    diff_versions: Compare two versions and return a structured audit diff.
"""

from typing import Dict, Any
from utils.state_manager import GlobalStateManager


def list_versions() -> Dict[str, Any]:
    """
    List all available dataset version snapshots.

    Shows version number, which tool created it, timestamp, row/column counts,
    memory usage, and whether the version is pinned (protected from eviction).

    Returns:
        Dictionary containing current version, total memory, and version list.
    """
    manager = GlobalStateManager()
    vm = manager.versions

    if vm.stored_count == 0:
        return {
            "message": "No versions available. Load a dataset first.",
            "versions": [],
        }

    versions = vm.list_all()
    dataset_name = manager.get_dataset_name() or "unknown"

    return {
        "dataset_name": dataset_name,
        "current_version": vm.current_version,
        "total_versions_stored": vm.stored_count,
        "total_memory_mb": vm.total_memory_mb,
        "max_versions": vm._max_versions,
        "versions": versions,
    }


def rollback_version(version: int) -> Dict[str, Any]:
    """
    Restore the dataset to a previous version snapshot.

    This creates a NEW version entry so the rollback itself appears in the audit
    trail. The target version is auto-pinned to protect it from eviction.

    Args:
        version: The version number to roll back to. Use list_versions() to see
                 available versions.

    Returns:
        Dictionary with rollback details including the new version number.
    """
    manager = GlobalStateManager()
    vm = manager.versions

    if vm.stored_count == 0:
        return {"error": "No versions available. Load a dataset first."}

    # Validate version exists
    try:
        target = vm.get(version)
    except KeyError as e:
        return {"error": str(e)}

    # Restore the DataFrame into the state manager
    # Use the internal _current_df to avoid triggering another snapshot via load_data
    restored_df = target.df.copy()
    manager._current_df = restored_df

    # Pin the target so it survives future evictions
    vm.pin(version)

    # Create a new snapshot recording the rollback action
    new_version = vm.snapshot(
        df=restored_df,
        tool="rollback",
        params={"rolled_back_to": version},
    )

    # Log the rollback in pipeline history
    manager.log_action(
        "rollback",
        {
            "rolled_back_to": version,
            "new_version": new_version,
            "rows": len(restored_df),
            "columns": len(restored_df.columns),
        },
    )

    return {
        "rolled_back_to": version,
        "new_version": new_version,
        "rows": len(restored_df),
        "columns": len(restored_df.columns),
        "column_names": list(restored_df.columns),
        "message": (
            f"Rolled back to version {version}. "
            f"Current state is now version {new_version}. "
            f"Version {version} has been pinned to prevent eviction."
        ),
    }


def diff_versions(version_a: int, version_b: int) -> Dict[str, Any]:
    """
    Compare two dataset versions and return a structured diff (audit trail).

    Computes shape changes, columns added/removed, dtype changes, missing value
    changes, statistical shifts for numeric columns, and lists the pipeline steps
    that occurred between the two versions.

    Args:
        version_a: The 'before' version number to compare.
        version_b: The 'after' version number to compare.

    Returns:
        Rich diff dictionary with shape_diff, columns_added, columns_removed,
        dtype_changes, missing_values_diff, stats_diff, steps_between, and
        memory_diff_mb.
    """
    manager = GlobalStateManager()
    vm = manager.versions

    if vm.stored_count == 0:
        return {"error": "No versions available. Load a dataset first."}

    try:
        result = vm.diff(version_a, version_b)
    except KeyError as e:
        return {"error": str(e)}

    # Add dataset context
    result["dataset_name"] = manager.get_dataset_name() or "unknown"

    return result
