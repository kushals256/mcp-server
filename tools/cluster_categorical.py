"""
Fuzzy Grouping for Categorical Data — Layer 3.

Clusters similar string values (typos, abbreviation variants) using RapidFuzz
and replaces them with the most-frequent canonical representative.  Runs
*after* harmonize_categorical_values and *before* encode_categorical_feature.

Functions:
    cluster_similar_categories: Group and merge similar category strings

Internal Libraries:
    rapidfuzz — Fast Levenshtein / token-sort ratio computation
"""

from typing import Dict, Any, List, Optional, Literal
from pydantic import BaseModel, Field

import pandas as pd
from rapidfuzz import fuzz, process

from config import (
    FUZZY_SCORE_THRESHOLD,
    FUZZY_MIN_GROUP_SIZE,
    FUZZY_MAX_COMPARISONS,
)
from utils.state_manager import GlobalStateManager
from tools.discovery import load_dataset_metadata


# ============================================================================
# Similarity function dispatch
# ============================================================================

_SCORERS = {
    "ratio": fuzz.ratio,
    "partial_ratio": fuzz.partial_ratio,
    "token_sort_ratio": fuzz.token_sort_ratio,
}


# ============================================================================
# MCP Tool
# ============================================================================


class ClusterCategoricalRequest(BaseModel):
    dataset_name: str = Field(..., description="Name of the dataset file (e.g., 'data.csv').")
    column: str = Field(..., description="The categorical column to cluster.")
    threshold: int = Field(FUZZY_SCORE_THRESHOLD, description="Minimum similarity score (0-100) to merge values.")
    method: Literal["ratio", "partial_ratio", "token_sort_ratio"] = Field("ratio", description="Similarity metric.")
    canonical_override: Optional[Dict[str, str]] = Field(None, description="Optional dict mapping values to canonical form.")


def cluster_similar_categories(request: ClusterCategoricalRequest) -> Dict[str, Any]:
    """
    Cluster similar category strings and replace them with a canonical form.

    Uses RapidFuzz for fast similarity scoring and a greedy frequency-first
    clustering strategy with deterministic tie-breaking (alphabetical).

    This is **Layer 3** of the normalization pipeline: runs after
    harmonize_categorical_values and before encode_categorical_feature.

    Args:
        request: ClusterCategoricalRequest containing parameters.

    Returns:
        Dictionary with clustering results, cluster details, and metrics.
    """
    dataset_name = request.dataset_name
    column = request.column
    threshold = request.threshold
    method = request.method
    canonical_override = request.canonical_override

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
                "Fuzzy clustering is intended for categorical/string columns."
            )
        }

    if method not in _SCORERS:
        return {
            "error": f"Unknown method '{method}'. Supported: {list(_SCORERS.keys())}"
        }

    # ---- Prepare ----
    df_out = df.copy()
    nan_mask = df_out[column].isna()
    non_null_mask = ~nan_mask
    unique_before = int(df_out[column].nunique(dropna=True))

    # Get unique values with their frequencies
    value_counts = df_out.loc[non_null_mask, column].astype(str).value_counts()
    unique_values = list(value_counts.index)

    # Guard: refuse if too many unique values (O(n²) comparisons)
    if len(unique_values) > FUZZY_MAX_COMPARISONS:
        return {
            "error": (
                f"Column '{column}' has {len(unique_values)} unique values, "
                f"exceeding FUZZY_MAX_COMPARISONS ({FUZZY_MAX_COMPARISONS}). "
                "Apply normalize_categorical_text and harmonize_categorical_values "
                "first to reduce cardinality, or increase the limit in config."
            )
        }

    # ---- Greedy frequency-first clustering with deterministic tie-break ----
    scorer = _SCORERS[method]
    assigned: Dict[str, str] = {}  # value → canonical representative
    clusters: List[Dict[str, Any]] = []

    # Sort: primary by frequency (descending), secondary by value (ascending)
    # This ensures deterministic tie-breaking for equal-frequency values.
    sorted_values = sorted(unique_values, key=lambda v: (-value_counts[v], v))

    for value in sorted_values:
        if value in assigned:
            continue

        # This value becomes the canonical representative of its cluster
        cluster_members = [value]
        assigned[value] = value

        # Find all unassigned values similar to this one
        remaining = [v for v in sorted_values if v not in assigned]
        if remaining:
            matches = process.extract(
                value,
                remaining,
                scorer=scorer,
                score_cutoff=threshold,
                limit=None,
            )
            for match_val, score, _ in matches:
                if match_val not in assigned:
                    assigned[match_val] = value
                    cluster_members.append(match_val)

        # Only record multi-member clusters
        if len(cluster_members) >= max(2, FUZZY_MIN_GROUP_SIZE):
            clusters.append({
                "canonical": value,
                "members": cluster_members,
                "frequency": int(sum(value_counts.get(m, 0) for m in cluster_members)),
            })

    # ---- Apply canonical overrides ----
    if canonical_override:
        for source, target in canonical_override.items():
            assigned[source] = target

    # ---- Replace values ----
    original_values = df_out[column].copy()
    values = df_out.loc[non_null_mask, column].astype(str)
    df_out.loc[non_null_mask, column] = values.map(
        lambda v: assigned.get(v, v)
    )

    # ---- Compute metrics ----
    unique_after = int(df_out[column].nunique(dropna=True))

    # ---- Update state ----
    manager.load_data(df_out, dataset_name)
    manager.log_action("cluster_similar_categories", {
        "column": column,
        "threshold": threshold,
        "method": method,
        "clusters_found": len(clusters),
    })

    return {
        "column": column,
        "rows_affected": int(non_null_mask.sum()),
        "clusters_found": len(clusters),
        "unique_before": unique_before,
        "unique_after": unique_after,
        "cluster_details": clusters,
        "threshold_used": threshold,
        "method_used": method,
    }
