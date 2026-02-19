"""
Categorical Feature Encoding Tools for MCP Server.

This module provides comprehensive categorical variable encoding supporting
8 methods from simple (one-hot) to advanced (target, leave-one-out).

Functions:
    encode_categorical_feature: Encode a categorical column using the specified method

Internal Helpers:
    _encode_one_hot: One-hot (dummy) encoding
    _encode_label: Label encoding (arbitrary integer assignment)
    _encode_ordinal: Ordinal encoding (user-defined order)
    _encode_frequency: Frequency-based encoding
    _encode_target: Target (mean) encoding
    _encode_binary: Binary encoding
    _encode_hashing: Feature hashing
    _encode_leave_one_out: Leave-one-out target encoding
    _compute_risk_indicators: Compute memory/leakage/overfit risk signals
"""

import pandas as pd
import numpy as np
import math
from typing import Dict, Any, List, Optional, Literal, Tuple

from sklearn.preprocessing import LabelEncoder
import category_encoders as ce

from config import (
    ONE_HOT_MAX_CARDINALITY,
    ONE_HOT_BLOCK_CARDINALITY,
    DEFAULT_HASH_N_COMPONENTS,
    DEFAULT_TARGET_SMOOTHING,
    MEMORY_RISK_THRESHOLDS,
)
from utils.state_manager import GlobalStateManager


# ============================================================================
# Supported encoding methods
# ============================================================================

SUPPORTED_METHODS = [
    "one_hot", "label", "ordinal", "frequency",
    "target", "binary", "hashing", "leave_one_out",
]


def encode_categorical_feature(
    dataset_name: str,
    column: str,
    method: Literal[
        "one_hot", "label", "ordinal", "frequency",
        "target", "binary", "hashing", "leave_one_out"
    ],
    handle_unknown: Literal["ignore", "error", "new_category"] = "ignore",
    drop_original: bool = True,
    prefix: Optional[str] = None,
    ordinal_mapping: Optional[Dict[str, int]] = None,
    n_components: Optional[int] = None,
    target_column: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Encode a categorical column into ML-compatible numerical representations.

    Supports 8 encoding methods from simple (one-hot) to advanced (target,
    leave-one-out). Preserves NaN values throughout and returns rich metadata
    for LLM reasoning including risk indicators and dimensionality impact.

    Args:
        dataset_name: Name of the dataset file (e.g., 'data.csv').
        column: The categorical column to encode.
        method: Encoding method to use. One of:
            'one_hot', 'label', 'ordinal', 'frequency',
            'target', 'binary', 'hashing', 'leave_one_out'.
        handle_unknown: How to handle unseen categories.
            'ignore' maps to 0/NaN, 'error' raises, 'new_category' creates sentinel.
        drop_original: Whether to drop the original column after encoding (default True).
        prefix: Custom prefix for new column names (defaults to column name).
        ordinal_mapping: Required for 'ordinal' method. Maps categories to integers.
            e.g. {"low": 0, "medium": 1, "high": 2}
        n_components: Number of components for 'hashing' method (default 8).
        target_column: Required for 'target' and 'leave_one_out' methods.

    Returns:
        Dictionary containing encoding results, metadata, and risk indicators.
    """
    # ---- Input validation ----
    if method not in SUPPORTED_METHODS:
        return {"error": f"Unknown method '{method}'. Supported: {SUPPORTED_METHODS}"}

    manager = GlobalStateManager()

    # Ensure dataset is loaded
    if manager.get_dataset_name() != dataset_name:
        return {
            "error": f"Dataset '{dataset_name}' is not currently loaded. "
                     "Call load_dataset_metadata() explicitly to load it first."
        }

    df = manager.get_data()
    if df is None:
        return {"error": "Dataset loaded but DataFrame is None."}

    # Column validation
    if column not in df.columns:
        return {"error": f"Column '{column}' not found in dataset."}

    # Check column is categorical-like (non-numeric or object/category)
    if pd.api.types.is_numeric_dtype(df[column]) and not isinstance(
        df[column].dtype, pd.CategoricalDtype
    ):
        return {
            "error": (
                f"Column '{column}' is numeric (dtype={df[column].dtype}). "
                "Encoding is intended for categorical/string columns."
            )
        }

    # Method-specific parameter validation
    if method == "ordinal" and ordinal_mapping is None:
        return {"error": "ordinal_mapping is required for method='ordinal'."}

    if method in ("target", "leave_one_out"):
        if target_column is None:
            return {"error": f"target_column is required for method='{method}'."}
        if target_column not in df.columns:
            return {"error": f"Target column '{target_column}' not found in dataset."}

    # ---- Prepare ----
    df_encoded = df.copy()
    cardinality_before = df[column].nunique(dropna=True)
    nan_mask = df[column].isna()
    total_rows = len(df)
    col_prefix = prefix if prefix else column

    # Handle unknown categories (add sentinel if requested)
    if handle_unknown == "new_category":
        df_encoded[column] = df_encoded[column].fillna("__unknown__")

    # ---- Dispatch to encoder ----
    dispatch = {
        "one_hot": _encode_one_hot,
        "label": _encode_label,
        "ordinal": _encode_ordinal,
        "frequency": _encode_frequency,
        "target": _encode_target,
        "binary": _encode_binary,
        "hashing": _encode_hashing,
        "leave_one_out": _encode_leave_one_out,
    }

    result = dispatch[method](
        df=df_encoded,
        column=column,
        prefix=col_prefix,
        nan_mask=nan_mask,
        cardinality=cardinality_before,
        ordinal_mapping=ordinal_mapping,
        n_components=n_components,
        target_column=target_column,
        handle_unknown=handle_unknown,
    )

    # Check for error from encoder
    if "error" in result:
        return result

    df_encoded = result["df"]
    new_columns = result["new_columns"]
    encoding_metadata = result["encoding_metadata"]

    # ---- Drop original column ----
    original_dropped = False
    if drop_original and column in df_encoded.columns:
        df_encoded = df_encoded.drop(columns=[column])
        original_dropped = True

    # ---- Restore NaN rows ----
    if handle_unknown != "new_category":
        for nc in new_columns:
            if nc in df_encoded.columns:
                df_encoded.loc[nan_mask, nc] = np.nan

    # ---- Compute risk indicators ----
    n_new = len(new_columns)
    risks = _compute_risk_indicators(
        method=method,
        n_new_columns=n_new,
        cardinality=cardinality_before,
        total_rows=total_rows,
    )

    # Cardinality after
    if n_new == 1 and new_columns[0] in df_encoded.columns:
        cardinality_after = int(df_encoded[new_columns[0]].nunique(dropna=True))
    else:
        cardinality_after = cardinality_before

    # ---- Dimensionality change ----
    cols_removed = 1 if original_dropped else 0
    feature_space_delta = n_new - cols_removed
    dimensionality_change = feature_space_delta

    # ---- Update global state ----
    manager.load_data(df_encoded, dataset_name)

    return {
        "column": column,
        "method": method,
        "new_columns_created": new_columns,
        "original_column_dropped": original_dropped,
        "cardinality_before": cardinality_before,
        "cardinality_after": cardinality_after,
        "rows_affected": total_rows,
        "encoding_metadata": encoding_metadata,
        "memory_risk": risks["memory_risk"],
        "leakage_risk": risks["leakage_risk"],
        "overfit_risk": risks["overfit_risk"],
        "feature_space_delta": feature_space_delta,
        "dimensionality_change": dimensionality_change,
    }


# ============================================================================
# Private encoder helpers
# ============================================================================


def _encode_one_hot(
    df: pd.DataFrame,
    column: str,
    prefix: str,
    nan_mask: pd.Series,
    cardinality: int,
    **kwargs,
) -> Dict[str, Any]:
    """One-hot (dummy) encoding using pd.get_dummies."""

    if cardinality > ONE_HOT_BLOCK_CARDINALITY:
        return {
            "error": (
                f"Column '{column}' has cardinality {cardinality} which exceeds "
                f"the maximum allowed for one-hot encoding ({ONE_HOT_BLOCK_CARDINALITY}). "
                "Consider using 'binary', 'hashing', or 'frequency' instead."
            )
        }

    sparse_recommended = cardinality > ONE_HOT_MAX_CARDINALITY

    dummies = pd.get_dummies(df[column], prefix=prefix, dtype=float)
    new_columns = list(dummies.columns)

    df_out = pd.concat([df, dummies], axis=1)

    return {
        "df": df_out,
        "new_columns": new_columns,
        "encoding_metadata": {
            "categories": list(df[column].dropna().unique()),
            "columns_created": len(new_columns),
            "sparse_recommended": sparse_recommended,
        },
    }


def _encode_label(
    df: pd.DataFrame,
    column: str,
    prefix: str,
    nan_mask: pd.Series,
    **kwargs,
) -> Dict[str, Any]:
    """Label encoding using sklearn LabelEncoder."""

    le = LabelEncoder()
    new_col = f"{prefix}_encoded"

    non_null = df[column].dropna()
    le.fit(non_null)

    encoded = pd.Series(index=df.index, dtype="Int64")
    encoded.loc[~nan_mask] = le.transform(df.loc[~nan_mask, column])
    encoded.loc[nan_mask] = pd.NA

    df[new_col] = encoded
    mapping = {str(cls): int(i) for i, cls in enumerate(le.classes_)}

    return {
        "df": df,
        "new_columns": [new_col],
        "encoding_metadata": {
            "mapping": mapping,
        },
    }


def _encode_ordinal(
    df: pd.DataFrame,
    column: str,
    prefix: str,
    nan_mask: pd.Series,
    ordinal_mapping: Optional[Dict[str, int]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Ordinal encoding using user-provided mapping."""

    new_col = f"{prefix}_ordinal"
    df[new_col] = df[column].map(ordinal_mapping)

    unmapped = df.loc[~nan_mask & df[new_col].isna(), column].unique()

    metadata: Dict[str, Any] = {"mapping": ordinal_mapping}
    if len(unmapped) > 0:
        metadata["unmapped_categories"] = list(unmapped)

    return {
        "df": df,
        "new_columns": [new_col],
        "encoding_metadata": metadata,
    }


def _encode_frequency(
    df: pd.DataFrame,
    column: str,
    prefix: str,
    nan_mask: pd.Series,
    **kwargs,
) -> Dict[str, Any]:
    """Frequency (count-based) encoding."""

    new_col = f"{prefix}_freq"
    freq_map = df[column].value_counts(normalize=True, dropna=True)
    df[new_col] = df[column].map(freq_map)

    return {
        "df": df,
        "new_columns": [new_col],
        "encoding_metadata": {
            "value_counts": {str(k): round(float(v), 6) for k, v in freq_map.items()},
        },
    }


def _encode_target(
    df: pd.DataFrame,
    column: str,
    prefix: str,
    nan_mask: pd.Series,
    target_column: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Target (mean) encoding using category_encoders."""

    new_col = f"{prefix}_target"
    smoothing = DEFAULT_TARGET_SMOOTHING

    encoder = ce.TargetEncoder(
        cols=[column],
        smoothing=smoothing,
        return_df=True,
    )

    y = df[target_column]
    encoded_df = encoder.fit_transform(df[[column]], y)
    df[new_col] = encoded_df[column].values

    return {
        "df": df,
        "new_columns": [new_col],
        "encoding_metadata": {
            "target_column": target_column,
            "smoothing": smoothing,
            "leakage_safe": False,
        },
    }


def _encode_binary(
    df: pd.DataFrame,
    column: str,
    prefix: str,
    nan_mask: pd.Series,
    cardinality: int,
    **kwargs,
) -> Dict[str, Any]:
    """Binary encoding using category_encoders."""

    encoder = ce.BinaryEncoder(cols=[column], return_df=True)

    encoded_df = encoder.fit_transform(df[[column]])
    new_columns = [c for c in encoded_df.columns if c != column]

    rename_map = {c: f"{prefix}_bin_{i}" for i, c in enumerate(new_columns)}
    encoded_df = encoded_df.rename(columns=rename_map)
    new_columns = list(rename_map.values())

    for nc in new_columns:
        df[nc] = encoded_df[nc].values

    expected_cols = max(1, math.ceil(math.log2(cardinality + 1)))

    return {
        "df": df,
        "new_columns": new_columns,
        "encoding_metadata": {
            "n_binary_columns": len(new_columns),
            "expected_columns": expected_cols,
        },
    }


def _encode_hashing(
    df: pd.DataFrame,
    column: str,
    prefix: str,
    nan_mask: pd.Series,
    cardinality: int,
    n_components: Optional[int] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Feature hashing using category_encoders."""

    n_comp = n_components if n_components else DEFAULT_HASH_N_COMPONENTS

    encoder = ce.HashingEncoder(
        cols=[column],
        n_components=n_comp,
        return_df=True,
    )

    encoded_df = encoder.fit_transform(df[[column]])
    raw_cols = [c for c in encoded_df.columns if c != column]

    rename_map = {c: f"{prefix}_hash_{i}" for i, c in enumerate(raw_cols)}
    encoded_df = encoded_df.rename(columns=rename_map)
    new_columns = list(rename_map.values())

    for nc in new_columns:
        df[nc] = encoded_df[nc].values

    collision_risk = round(1.0 - (n_comp / max(cardinality, 1)), 4)
    collision_risk = max(0.0, min(1.0, collision_risk))

    return {
        "df": df,
        "new_columns": new_columns,
        "encoding_metadata": {
            "n_buckets": n_comp,
            "collision_risk": collision_risk,
        },
    }


def _encode_leave_one_out(
    df: pd.DataFrame,
    column: str,
    prefix: str,
    nan_mask: pd.Series,
    target_column: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Leave-one-out target encoding using category_encoders."""

    new_col = f"{prefix}_loo"
    smoothing = DEFAULT_TARGET_SMOOTHING

    encoder = ce.LeaveOneOutEncoder(
        cols=[column],
        sigma=smoothing,
        return_df=True,
    )

    y = df[target_column]
    encoded_df = encoder.fit_transform(df[[column]], y)
    df[new_col] = encoded_df[column].values

    return {
        "df": df,
        "new_columns": [new_col],
        "encoding_metadata": {
            "target_column": target_column,
            "smoothing": smoothing,
            "leakage_safe": True,
        },
    }


# ============================================================================
# Risk computation
# ============================================================================


def _compute_risk_indicators(
    method: str,
    n_new_columns: int,
    cardinality: int,
    total_rows: int,
) -> Dict[str, Any]:
    """Compute memory, leakage, and overfit risk indicators."""

    if n_new_columns <= MEMORY_RISK_THRESHOLDS["low"]:
        memory_risk = "low"
    elif n_new_columns <= MEMORY_RISK_THRESHOLDS["medium"]:
        memory_risk = "medium"
    else:
        memory_risk = "high"

    leakage_risk = method in ("target",)

    ratio = cardinality / max(total_rows, 1)
    if ratio < 0.01:
        overfit_risk = "low"
    elif ratio < 0.1:
        overfit_risk = "medium"
    else:
        overfit_risk = "high"

    return {
        "memory_risk": memory_risk,
        "leakage_risk": leakage_risk,
        "overfit_risk": overfit_risk,
    }