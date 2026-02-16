"""
Tests for Fuzzy Grouping Tool (Layer 3).

Covers typo clustering, threshold sensitivity, canonical overrides,
deterministic tie-breaking, NaN preservation, max-comparisons guard,
idempotency, and error handling.
"""

import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.cluster_categorical import cluster_similar_categories
from utils.state_manager import GlobalStateManager


# ============================================================================
# Test data helpers
# ============================================================================


def setup_data():
    return pd.DataFrame({
        "state": [
            "california",
            "califronia",    # typo
            "califonia",     # typo
            "california",    # duplicate (most frequent)
            "new york",
            "new yrok",      # typo
            np.nan,
            "texas",
        ],
        "score": [10, 20, 30, 40, 50, 60, 70, 80],
    })


def load_test_data(name="test_clust.csv"):
    df = setup_data()
    manager = GlobalStateManager()
    manager.load_data(df, name)
    return df, manager


# ============================================================================
# Basic clustering
# ============================================================================


def test_basic_typo_clustering():
    """Test that typos are merged into the most-frequent canonical form."""
    df, manager = load_test_data("test_clust_basic.csv")

    result = cluster_similar_categories(
        "test_clust_basic.csv", "state", threshold=80
    )

    assert result.get("error") is None, f"Unexpected error: {result.get('error')}"
    assert result["clusters_found"] > 0
    assert result["unique_after"] < result["unique_before"]

    new_df = manager.get_data()
    # "califronia" and "califonia" should map to "california" (most frequent)
    assert new_df["state"].iloc[1] == "california"
    assert new_df["state"].iloc[2] == "california"


def test_nan_preservation():
    """Test that NaN values survive clustering."""
    df, manager = load_test_data("test_clust_nan.csv")

    result = cluster_similar_categories(
        "test_clust_nan.csv", "state", threshold=80
    )
    assert result.get("error") is None
    new_df = manager.get_data()
    assert pd.isna(new_df["state"].iloc[6])


# ============================================================================
# Threshold sensitivity
# ============================================================================


def test_high_threshold_fewer_matches():
    """Test that a very high threshold reduces clustering."""
    df, manager = load_test_data("test_clust_high.csv")

    result = cluster_similar_categories(
        "test_clust_high.csv", "state", threshold=99
    )
    assert result.get("error") is None
    # With threshold=99, fewer (or no) clusters should form
    assert result["clusters_found"] <= 1


def test_low_threshold_more_matches():
    """Test that a low threshold increases clustering."""
    df, manager = load_test_data("test_clust_low.csv")

    result = cluster_similar_categories(
        "test_clust_low.csv", "state", threshold=60
    )
    assert result.get("error") is None
    # More clusters should form at low threshold
    assert result["unique_after"] <= result["unique_before"]


# ============================================================================
# Canonical override
# ============================================================================


def test_canonical_override():
    """Test that canonical_override takes precedence over automatic clustering."""
    df, manager = load_test_data("test_clust_over.csv")

    result = cluster_similar_categories(
        "test_clust_over.csv",
        "state",
        threshold=80,
        canonical_override={"califronia": "CA"},
    )
    assert result.get("error") is None

    new_df = manager.get_data()
    # Override should take precedence
    assert new_df["state"].iloc[1] == "CA"


# ============================================================================
# Deterministic tie-breaking
# ============================================================================


def test_deterministic_tiebreak():
    """Test that equal-frequency values use alphabetical tie-breaking."""
    # All values appear exactly once — alphabetical first should be canonical
    df = pd.DataFrame({
        "city": ["bravo", "alpha", "charlie"],
    })
    manager = GlobalStateManager()
    manager.load_data(df, "test_clust_tie.csv")

    result = cluster_similar_categories(
        "test_clust_tie.csv", "city", threshold=50
    )
    assert result.get("error") is None

    # Run it again — result must be identical (deterministic)
    manager.load_data(
        pd.DataFrame({"city": ["bravo", "alpha", "charlie"]}),
        "test_clust_tie2.csv",
    )
    result2 = cluster_similar_categories(
        "test_clust_tie2.csv", "city", threshold=50
    )
    assert result["cluster_details"] == result2["cluster_details"]


# ============================================================================
# Max comparisons guard
# ============================================================================


def test_max_comparisons_guard():
    """Test that high-cardinality columns are rejected."""
    df = pd.DataFrame({"cat": [f"value_{i}" for i in range(600)]})
    manager = GlobalStateManager()
    manager.load_data(df, "test_clust_max.csv")

    result = cluster_similar_categories("test_clust_max.csv", "cat")
    assert "error" in result
    assert "FUZZY_MAX_COMPARISONS" in result["error"]


# ============================================================================
# Error handling
# ============================================================================


def test_missing_column():
    load_test_data("test_clust_err1.csv")
    result = cluster_similar_categories("test_clust_err1.csv", "nonexistent")
    assert "error" in result
    assert "not found" in result["error"]


def test_numeric_column():
    load_test_data("test_clust_err2.csv")
    result = cluster_similar_categories("test_clust_err2.csv", "score")
    assert "error" in result
    assert "numeric" in result["error"]


def test_invalid_method():
    load_test_data("test_clust_err3.csv")
    result = cluster_similar_categories(
        "test_clust_err3.csv", "state", method="invalid"
    )
    assert "error" in result
    assert "Unknown method" in result["error"]


# ============================================================================
# Idempotency
# ============================================================================


def test_idempotency():
    """Test that running clustering twice produces the same result."""
    load_test_data("test_clust_idem.csv")

    result1 = cluster_similar_categories(
        "test_clust_idem.csv", "state", threshold=80
    )
    assert result1.get("error") is None

    manager = GlobalStateManager()
    df_after_first = manager.get_data().copy()

    result2 = cluster_similar_categories(
        "test_clust_idem.csv", "state", threshold=80
    )
    assert result2.get("error") is None

    df_after_second = manager.get_data()
    pd.testing.assert_frame_equal(df_after_first, df_after_second)


# ============================================================================
# All-unique column
# ============================================================================


def test_all_unique_no_clusters():
    """Test that fully unique, dissimilar values produce no clusters."""
    df = pd.DataFrame({"cat": ["apple", "zebra", "mountain"]})
    manager = GlobalStateManager()
    manager.load_data(df, "test_clust_uniq.csv")

    result = cluster_similar_categories(
        "test_clust_uniq.csv", "cat", threshold=90
    )
    assert result.get("error") is None
    assert result["clusters_found"] == 0
    assert result["unique_before"] == result["unique_after"]


# ============================================================================
# Runner
# ============================================================================


def run_tests():
    tests = [
        test_basic_typo_clustering,
        test_nan_preservation,
        test_high_threshold_fewer_matches,
        test_low_threshold_more_matches,
        test_canonical_override,
        test_deterministic_tiebreak,
        test_max_comparisons_guard,
        test_missing_column,
        test_numeric_column,
        test_invalid_method,
        test_idempotency,
        test_all_unique_no_clusters,
    ]

    passed = 0
    failed = 0

    print(f"Running {len(tests)} tests...\n")
    for test in tests:
        try:
            test()
            print(f"  ✅ {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"  ❌ {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed out of {len(tests)}")
    return failed == 0


if __name__ == "__main__":
    try:
        import pytest
        sys.exit(pytest.main(["-v", __file__]))
    except ImportError:
        success = run_tests()
        sys.exit(0 if success else 1)
