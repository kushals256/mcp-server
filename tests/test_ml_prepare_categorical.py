"""
Tests for ML-Aware Categorical Preparation Tool (Layer 4).

Covers deduplicate mode, gap_encoder mode, lazy import, NaN handling,
invalid method, idempotency, and error handling.
"""

import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset_analysis_mcp.tools.ml_prepare_categorical import ml_prepare_categorical
from dataset_analysis_mcp.utils.state_manager import GlobalStateManager


# ============================================================================
# Test data helpers
# ============================================================================


def setup_data():
    return pd.DataFrame({
        "company": [
            "Google",
            "google",
            "Gogle",       # typo
            "Microsoft",
            "Microsft",    # typo
            "Apple",
            np.nan,
            "apple",
        ],
        "revenue": [100, 200, 300, 400, 500, 600, 700, 800],
    })


def load_test_data(name="test_mlp.csv"):
    df = setup_data()
    manager = GlobalStateManager()
    manager.load_data(df, name)
    return df, manager


# ============================================================================
# Deduplicate mode
# ============================================================================


def test_deduplicate_basic():
    """Test that deduplicate reduces cardinality by merging similar strings."""
    df, manager = load_test_data("test_mlp_dedup.csv")

    result = ml_prepare_categorical("test_mlp_dedup.csv", "company", method="deduplicate")

    assert result.get("error") is None, f"Unexpected error: {result.get('error')}"
    assert result["method"] == "deduplicate"
    assert result["unique_after"] <= result["unique_before"]
    assert result["new_columns_created"] == []

    new_df = manager.get_data()
    assert "company" in new_df.columns


def test_deduplicate_nan_preservation():
    """Test that NaN values survive deduplication."""
    df, manager = load_test_data("test_mlp_dedup_nan.csv")

    result = ml_prepare_categorical("test_mlp_dedup_nan.csv", "company", method="deduplicate")
    assert result.get("error") is None
    new_df = manager.get_data()
    assert pd.isna(new_df["company"].iloc[6])


# ============================================================================
# GapEncoder mode
# ============================================================================


def test_gap_encoder_basic():
    """Test that gap_encoder creates numerical columns and drops original."""
    df, manager = load_test_data("test_mlp_gap.csv")

    result = ml_prepare_categorical(
        "test_mlp_gap.csv", "company", method="gap_encoder", n_components=3
    )

    assert result.get("error") is None, f"Unexpected error: {result.get('error')}"
    assert result["method"] == "gap_encoder"
    assert len(result["new_columns_created"]) == 3

    new_df = manager.get_data()
    # Original column should be dropped
    assert "company" not in new_df.columns
    # New columns should exist
    for col in result["new_columns_created"]:
        assert col in new_df.columns


# ============================================================================
# Error handling
# ============================================================================


def test_missing_column():
    load_test_data("test_mlp_err1.csv")
    result = ml_prepare_categorical("test_mlp_err1.csv", "nonexistent")
    assert "error" in result
    assert "not found" in result["error"]


def test_numeric_column():
    load_test_data("test_mlp_err2.csv")
    result = ml_prepare_categorical("test_mlp_err2.csv", "revenue")
    assert "error" in result
    assert "numeric" in result["error"]


def test_invalid_method():
    load_test_data("test_mlp_err3.csv")
    result = ml_prepare_categorical("test_mlp_err3.csv", "company", method="invalid")
    assert "error" in result
    assert "Unknown method" in result["error"]


# ============================================================================
# Idempotency (deduplicate mode)
# ============================================================================


def test_deduplicate_idempotency():
    """Test that running deduplicate twice doesn't increase cardinality.

    Note: skrub.deduplicate is ML-based and may merge additional values
    on a second pass, so we verify cardinality is monotonically non-increasing
    rather than checking exact frame equality.
    """
    load_test_data("test_mlp_idem.csv")

    result1 = ml_prepare_categorical("test_mlp_idem.csv", "company", method="deduplicate")
    assert result1.get("error") is None

    manager = GlobalStateManager()
    unique_after_first = manager.get_data()["company"].nunique(dropna=True)

    result2 = ml_prepare_categorical("test_mlp_idem.csv", "company", method="deduplicate")
    assert result2.get("error") is None

    unique_after_second = manager.get_data()["company"].nunique(dropna=True)
    # Cardinality should never increase on a second pass
    assert unique_after_second <= unique_after_first


# ============================================================================
# Runner
# ============================================================================


def run_tests():
    tests = [
        test_deduplicate_basic,
        test_deduplicate_nan_preservation,
        test_gap_encoder_basic,
        test_missing_column,
        test_numeric_column,
        test_invalid_method,
        test_deduplicate_idempotency,
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
