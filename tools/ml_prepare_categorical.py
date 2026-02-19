"""
ML-Aware Categorical Preparation — Layer 4 (Optional / Experimental).

Uses skrub for advanced ML-driven normalization: auto-deduplication via
embeddings and GapEncoder for dense numerical representations.  This layer
is **optional** and only relevant for high-cardinality or very messy data.

Functions:
    ml_prepare_categorical: ML-driven deduplication or gap encoding

Internal Libraries:
    skrub — Lazy-imported to avoid heavy startup cost
"""

from typing import Dict, Any, List, Optional, Literal

import pandas as pd

from utils.state_manager import GlobalStateManager
from tools.discovery import load_dataset_metadata


# ============================================================================
# MCP Tool
# ============================================================================


def ml_prepare_categorical(
    dataset_name: str,
    column: str,
    method: Literal["deduplicate", "gap_encoder"] = "deduplicate",
    n_components: int = 10,
) -> Dict[str, Any]:
    """
    Apply ML-aware normalization to a categorical column (experimental).

    Supports two modes:
    - **deduplicate**: Uses skrub.deduplicate() to auto-merge similar strings
      using ML embeddings.  Reduces cardinality without manual synonym maps.
    - **gap_encoder**: Uses skrub.GapEncoder to produce dense numerical
      representations of the categorical column, replacing it with
      n_components float features.

    This is **Layer 4** of the normalization pipeline (optional).

    Args:
        dataset_name: Name of the dataset file (e.g., 'data.csv').
        column: The categorical column to process.
        method: 'deduplicate' or 'gap_encoder' (default 'deduplicate').
        n_components: Number of output features for gap_encoder (default 10).

    Returns:
        Dictionary with processing results and metrics.
    """
    # ---- Load & validate ----
    manager = GlobalStateManager()

    if manager.get_dataset_name() != dataset_name:
        return {
            "error": f"Dataset '{dataset_name}' is not currently loaded. "
                     "Call load_dataset_metadata() explicitly to load it first."
        }

    df = manager.get_data()
    if df is None:
        return {"error": "Dataset loaded but DataFrame is None."}

    if column not in df.columns:
        return {"error": f"Column '{column}' not found in dataset."}

    if pd.api.types.is_numeric_dtype(df[column]) and not isinstance(
        df[column].dtype, pd.CategoricalDtype
    ):
        return {
            "error": (
                f"Column '{column}' is numeric (dtype={df[column].dtype}). "
                "ML preparation is intended for categorical/string columns."
            )
        }

    if method not in ("deduplicate", "gap_encoder"):
        return {"error": f"Unknown method '{method}'. Supported: ['deduplicate', 'gap_encoder']"}

    # ---- Lazy-import skrub ----
    try:
        import skrub
    except ImportError:
        return {
            "error": (
                "The 'skrub' package is required for ml_prepare_categorical "
                "but is not installed. Install it with: pip install skrub>=0.3.0"
            )
        }

    # ---- Prepare ----
    df_out = df.copy()
    nan_mask = df_out[column].isna()
    non_null_mask = ~nan_mask
    unique_before = int(df_out[column].nunique(dropna=True))

    if method == "deduplicate":
        return _deduplicate(
            df_out, column, nan_mask, non_null_mask, unique_before,
            dataset_name, manager, skrub,
        )
    else:
        return _gap_encode(
            df_out, column, nan_mask, non_null_mask, unique_before,
            n_components, dataset_name, manager, skrub,
        )


# ============================================================================
# Private helpers
# ============================================================================


def _deduplicate(
    df: pd.DataFrame,
    column: str,
    nan_mask: pd.Series,
    non_null_mask: pd.Series,
    unique_before: int,
    dataset_name: str,
    manager: GlobalStateManager,
    skrub_module: Any,
) -> Dict[str, Any]:
    """Apply skrub.deduplicate to merge similar string values."""
    original_values = df[column].copy()
    series = df.loc[non_null_mask, column].astype(str)

    deduplicated = skrub_module.deduplicate(series)
    df.loc[non_null_mask, column] = deduplicated.values

    unique_after = int(df[column].nunique(dropna=True))
    changed_mask = non_null_mask & (original_values != df[column])
    values_changed = int(changed_mask.sum())

    sample_changes: List[Dict[str, str]] = []
    changed_indices = changed_mask[changed_mask].index[:5]
    for idx in changed_indices:
        sample_changes.append({
            "before": str(original_values[idx]),
            "after": str(df[column][idx]),
        })

    manager.load_data(df, dataset_name)
    manager.log_action("ml_prepare_categorical", {
        "column": column,
        "method": "deduplicate",
    })

    return {
        "column": column,
        "method": "deduplicate",
        "rows_affected": int(non_null_mask.sum()),
        "unique_before": unique_before,
        "unique_after": unique_after,
        "values_changed": values_changed,
        "sample_changes": sample_changes,
        "new_columns_created": [],
    }


def _gap_encode(
    df: pd.DataFrame,
    column: str,
    nan_mask: pd.Series,
    non_null_mask: pd.Series,
    unique_before: int,
    n_components: int,
    dataset_name: str,
    manager: GlobalStateManager,
    skrub_module: Any,
) -> Dict[str, Any]:
    """Apply skrub.GapEncoder for dense vector representations."""
    import numpy as np

    encoder = skrub_module.GapEncoder(n_components=n_components)

    # Fit-transform on non-null values — skrub expects a pandas Series
    series = df.loc[non_null_mask, column].astype(str).reset_index(drop=True)
    encoded_result = encoder.fit_transform(series)

    # skrub 0.7+ returns a DataFrame; older versions may return ndarray
    if isinstance(encoded_result, pd.DataFrame):
        encoded_df = encoded_result.reset_index(drop=True)
    else:
        # Fallback for ndarray
        encoded_df = pd.DataFrame(
            encoded_result,
            columns=[f"{column}_gap_{i}" for i in range(encoded_result.shape[1])],
        )

    # Create new columns
    new_columns: List[str] = []
    non_null_indices = df.index[non_null_mask]

    for i, src_col in enumerate(encoded_df.columns):
        col_name = f"{column}_gap_{i}"
        new_columns.append(col_name)
        df[col_name] = np.nan
        df.loc[non_null_indices, col_name] = encoded_df[src_col].values

    # Drop original column
    df = df.drop(columns=[column])

    unique_after = unique_before  # structural transform, cardinality preserved semantically

    manager.load_data(df, dataset_name)
    manager.log_action("ml_prepare_categorical", {
        "column": column,
        "method": "gap_encoder",
        "n_components": n_components,
    })

    return {
        "column": column,
        "method": "gap_encoder",
        "rows_affected": int(non_null_mask.sum()),
        "unique_before": unique_before,
        "unique_after": unique_after,
        "new_columns_created": new_columns,
        "sample_changes": [],
    }
