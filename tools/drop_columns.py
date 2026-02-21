"""
Column Dropping Tool for MCP Server.

This module provides functionality to drop one or more columns from the dataset.
It implements Phase 4 (Transformation) of the dataset analysis workflow.

Functions:
    drop_columns: Drop specified columns from the current dataset
"""

import pandas as pd
from typing import Dict, Any, List, Optional

from utils.state_manager import GlobalStateManager
from tools.discovery import load_dataset_metadata


def drop_columns(
    dataset_name: str,
    columns: List[str],
    errors: str = "raise",
) -> Dict[str, Any]:
    """
    Drop one or more columns from the current dataset.

    Removes the specified columns from the in-memory DataFrame and updates
    global state.  Supports both strict mode (raise on missing columns)
    and lenient mode (silently ignore missing columns).

    Args:
        dataset_name: Name of the dataset file (e.g., 'data.csv').
        columns: List of column names to drop.
            Example: ["Cabin", "Ticket"]
        errors: How to handle missing column names.
            - 'raise' (default): Raise an error if any column is not found.
            - 'ignore': Silently skip columns that don't exist.

    Returns:
        Dictionary with structure:
        {
            "columns_dropped": List[str],     # Successfully dropped columns
            "columns_not_found": List[str],   # Columns that were not in the dataset
            "remaining_columns": List[str],   # Columns still in the dataset
            "remaining_column_count": int,
            "remaining_rows": int
        }
    """
    # Validation: columns parameter
    if not isinstance(columns, list) or len(columns) == 0:
        return {"error": "Parameter 'columns' must be a non-empty list of column names."}

    if errors not in ("raise", "ignore"):
        return {"error": "Parameter 'errors' must be 'raise' or 'ignore'."}

    manager = GlobalStateManager()

    # Ensure dataset is loaded
    if manager.get_dataset_name() != dataset_name:
        try:
            load_dataset_metadata(dataset_name)
        except Exception as e:
            return {"error": f"Failed to load dataset: {str(e)}"}

    df = manager.get_data()
    if df is None:
        return {"error": "Dataset loaded but DataFrame is None."}

    # Classify columns
    existing_columns = [c for c in columns if c in df.columns]
    missing_columns = [c for c in columns if c not in df.columns]

    # Strict mode: fail if any column is missing
    if errors == "raise" and missing_columns:
        return {
            "error": (
                f"Columns not found in dataset: {missing_columns}. "
                "Use errors='ignore' to skip missing columns."
            )
        }

    # Guard: don't drop ALL columns
    if len(existing_columns) >= len(df.columns):
        return {"error": "Cannot drop all columns from the dataset."}

    # Nothing to drop
    if not existing_columns:
        return {
            "columns_dropped": [],
            "columns_not_found": missing_columns,
            "remaining_columns": list(df.columns),
            "remaining_column_count": len(df.columns),
            "remaining_rows": len(df),
        }

    # Drop columns
    df_out = df.drop(columns=existing_columns)

    # Update global state (preserve split if active)
    manager.load_data(df_out, dataset_name, reset_split=False)
    manager.log_action("drop_columns", {
        "columns_dropped": existing_columns,
        "columns_not_found": missing_columns if missing_columns else None,
        "errors_mode": errors,
    })

    return {
        "columns_dropped": existing_columns,
        "columns_not_found": missing_columns if missing_columns else None,
        "remaining_columns": list(df_out.columns),
        "remaining_column_count": len(df_out.columns),
        "remaining_rows": len(df_out),
    }
