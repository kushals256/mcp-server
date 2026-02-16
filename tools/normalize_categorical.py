"""
Surface Normalization for Categorical Text — Layer 1.

Deterministic text cleanup: Unicode normalization, accent stripping,
whitespace collapsing, and case folding.  Designed to run *before*
harmonize_categorical_values → cluster_similar_categories → encode.

Functions:
    normalize_categorical_text: Clean a categorical column's string values

Internal Libraries:
    unicodedata  — Unicode normalization (NFKC)
    text_unidecode — Accent / diacritic stripping
    cleantext — Fix mojibake, strip whitespace, remove control chars
"""

import unicodedata
from typing import Dict, Any, List

import pandas as pd
from text_unidecode import unidecode as strip_accents
from cleantext import clean as clean_text

from utils.state_manager import GlobalStateManager
from tools.discovery import load_dataset_metadata


# ============================================================================
# MCP Tool
# ============================================================================


def normalize_categorical_text(
    dataset_name: str,
    column: str,
    lowercase: bool = True,
    strip_accents_flag: bool = True,
    unicode_normalize: bool = True,
    fix_unicode: bool = True,
    strip_whitespace: bool = True,
) -> Dict[str, Any]:
    """
    Apply deterministic surface-level text normalization to a categorical column.

    Cleans string values through a composable pipeline of operations:
    Unicode normalization → accent stripping → mojibake/control-char fixing →
    whitespace collapsing → case folding.

    This is designed as **Layer 1** of the normalization pipeline, feeding clean
    tokens into harmonize_categorical_values and cluster_similar_categories.

    Args:
        dataset_name: Name of the dataset file (e.g., 'data.csv').
        column: The categorical column to normalize.
        lowercase: Fold all text to lowercase (default True).
        strip_accents_flag: Remove diacritics / accents (café → cafe).
        unicode_normalize: Apply NFKC Unicode normalization.
        fix_unicode: Use clean-text to fix mojibake and strip control chars.
        strip_whitespace: Collapse and strip extra whitespace.

    Returns:
        Dictionary containing normalization results and before/after samples.
    """
    # ---- Load & validate ----
    manager = GlobalStateManager()

    if manager.get_dataset_name() != dataset_name:
        try:
            load_dataset_metadata(dataset_name)
        except Exception as e:
            return {"error": f"Failed to load dataset: {str(e)}"}

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
                "Normalization is intended for categorical/string columns."
            )
        }

    # ---- Prepare ----
    df_out = df.copy()
    original_values = df_out[column].copy()
    nan_mask = df_out[column].isna()
    unique_before = int(df_out[column].nunique(dropna=True))

    # Build the list of operations that will be applied
    operations_applied: List[str] = []

    # ---- Apply normalization pipeline to non-null values ----
    non_null_mask = ~nan_mask
    values = df_out.loc[non_null_mask, column].astype(str)

    if unicode_normalize:
        values = values.apply(lambda v: unicodedata.normalize("NFKC", v))
        operations_applied.append("unicode_normalize")

    if strip_accents_flag:
        values = values.apply(strip_accents)
        operations_applied.append("strip_accents")

    if fix_unicode:
        values = values.apply(
            lambda v: clean_text(
                v,
                fix_unicode=True,
                to_ascii=False,
                lower=False,
                no_line_breaks=True,
                no_urls=False,
                no_emails=False,
                no_phone_numbers=False,
                no_numbers=False,
                no_digits=False,
                no_currency_symbols=False,
                no_punct=False,
            )
        )
        operations_applied.append("fix_unicode")

    if strip_whitespace:
        values = values.str.strip().str.replace(r"\s+", " ", regex=True)
        operations_applied.append("strip_whitespace")

    if lowercase:
        values = values.str.lower()
        operations_applied.append("lowercase")

    df_out.loc[non_null_mask, column] = values

    # ---- Compute metrics ----
    unique_after = int(df_out[column].nunique(dropna=True))
    changed_mask = non_null_mask & (original_values != df_out[column])
    values_changed = int(changed_mask.sum())

    # Collect up to 5 sample changes
    sample_changes: List[Dict[str, str]] = []
    changed_indices = changed_mask[changed_mask].index[:5]
    for idx in changed_indices:
        sample_changes.append({
            "before": str(original_values[idx]),
            "after": str(df_out[column][idx]),
        })

    # ---- Update state ----
    manager.load_data(df_out, dataset_name)
    manager.log_action("normalize_categorical_text", {
        "column": column,
        "operations": operations_applied,
    })

    return {
        "column": column,
        "rows_affected": int(non_null_mask.sum()),
        "values_changed": values_changed,
        "unique_before": unique_before,
        "unique_after": unique_after,
        "sample_changes": sample_changes,
        "operations_applied": operations_applied,
    }
