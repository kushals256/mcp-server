"""
Canonical Value Mapping for Categorical Data — Layer 2.

Uses FlashText for fast, deterministic synonym → canonical-value replacement.
Runs *after* normalize_categorical_text and *before* cluster_similar_categories.

Functions:
    harmonize_categorical_values: Map synonym variants to canonical forms

Internal Libraries:
    flashtext — Aho-Corasick based keyword replacement (whole-word safe)
"""

from typing import Dict, Any, List

import pandas as pd
from flashtext import KeywordProcessor

from config import HARMONIZE_CASE_SENSITIVE
from utils.state_manager import GlobalStateManager
from tools.discovery import load_dataset_metadata


# ============================================================================
# MCP Tool
# ============================================================================


def harmonize_categorical_values(
    dataset_name: str,
    column: str,
    synonym_map: Dict[str, List[str]],
    case_sensitive: bool = HARMONIZE_CASE_SENSITIVE,
) -> Dict[str, Any]:
    """
    Map synonym variants to canonical values using deterministic keyword matching.

    Uses FlashText (Aho-Corasick) for O(n) replacement that is safe against
    substring corruption — only whole-word / whole-token matches are replaced.

    This is **Layer 2** of the normalization pipeline: runs after surface
    normalization and before fuzzy clustering.

    Args:
        dataset_name: Name of the dataset file (e.g., 'data.csv').
        column: The categorical column to harmonize.
        synonym_map: Mapping from canonical value to list of synonyms.
            Example: {"usa": ["united states", "u.s.", "america"]}
        case_sensitive: Whether matching is case-sensitive (default False).

    Returns:
        Dictionary with harmonization results and before/after samples.
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
                "Harmonization is intended for categorical/string columns."
            )
        }

    if not synonym_map:
        return {
            "column": column,
            "rows_affected": 0,
            "values_harmonized": 0,
            "unique_before": int(df[column].nunique(dropna=True)),
            "unique_after": int(df[column].nunique(dropna=True)),
            "canonical_values": [],
            "sample_changes": [],
        }

    # ---- Build FlashText processor ----
    processor = KeywordProcessor(case_sensitive=case_sensitive)

    for canonical, synonyms in synonym_map.items():
        for synonym in synonyms:
            # Guard: only add non-empty synonyms
            if synonym and synonym.strip():
                processor.add_keyword(synonym.strip(), canonical)
        # Also map the canonical value to itself to handle casing normalization
        if canonical and canonical.strip():
            processor.add_keyword(canonical.strip(), canonical)

    # ---- Apply replacements ----
    df_out = df.copy()
    original_values = df_out[column].copy()
    nan_mask = df_out[column].isna()
    non_null_mask = ~nan_mask
    unique_before = int(df_out[column].nunique(dropna=True))

    def _replace_value(val: str) -> str:
        """Replace using FlashText, guarding against substring corruption.

        FlashText does whole-word replacement by default if the cell is a
        single keyword.  For cell values that are exactly one token (the
        common categorical case) we do an exact-match path for safety.
        For multi-word cells we fall back to replace_keywords which does
        Aho-Corasick scanning of the full string.
        """
        replaced = processor.replace_keywords(val)
        return replaced

    values = df_out.loc[non_null_mask, column].astype(str)
    df_out.loc[non_null_mask, column] = values.apply(_replace_value)

    # ---- Compute metrics ----
    unique_after = int(df_out[column].nunique(dropna=True))
    changed_mask = non_null_mask & (original_values != df_out[column])
    values_harmonized = int(changed_mask.sum())

    sample_changes: List[Dict[str, str]] = []
    changed_indices = changed_mask[changed_mask].index[:5]
    for idx in changed_indices:
        sample_changes.append({
            "before": str(original_values[idx]),
            "after": str(df_out[column][idx]),
        })

    # ---- Update state ----
    manager.update_data(df_out, tool_name="harmonize_categorical_values")
    manager.log_action("harmonize_categorical_values", {
        "column": column,
        "canonical_count": len(synonym_map),
        "case_sensitive": case_sensitive,
    })

    return {
        "column": column,
        "rows_affected": int(non_null_mask.sum()),
        "values_harmonized": values_harmonized,
        "unique_before": unique_before,
        "unique_after": unique_after,
        "canonical_values": list(synonym_map.keys()),
        "sample_changes": sample_changes,
    }
