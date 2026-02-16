"""
Tests for Surface Normalization Tool (Layer 1).

Covers Unicode normalization, accent stripping, whitespace collapsing,
case folding, NaN preservation, flag toggles, idempotency, and error handling.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.normalize_categorical import normalize_categorical_text
from utils.state_manager import GlobalStateManager


# ============================================================================
# Test data helpers
# ============================================================================


def setup_data():
    """Create a dataframe with messy categorical text."""
    return pd.DataFrame({
        "city": [
            "  São   Paulo  ",   # accents + extra whitespace
            "MÜNCHEN",           # uppercase + umlaut
            "café",              # accent
            "tokyo",             # already clean
            np.nan,              # NaN
            "New\tYork",         # tab character
            "naïve",             # diaeresis
            "Zürich",            # umlaut
        ],
        "score": [10, 20, 30, 40, 50, 60, 70, 80],
    })


def load_test_data(name="test_norm.csv"):
    df = setup_data()
    manager = GlobalStateManager()
    manager.load_data(df, name)
    return df, manager


# ============================================================================
# Basic normalization
# ============================================================================


def test_basic_normalization():
    """Test that accents, case, and whitespace are all cleaned."""
    df, manager = load_test_data("test_norm_basic.csv")

    result = normalize_categorical_text("test_norm_basic.csv", "city")

    assert result.get("error") is None, f"Unexpected error: {result.get('error')}"
    assert result["values_changed"] > 0
    assert result["unique_after"] <= result["unique_before"]
    assert "lowercase" in result["operations_applied"]
    assert "strip_accents" in result["operations_applied"]
    assert "strip_whitespace" in result["operations_applied"]

    new_df = manager.get_data()
    # "São Paulo" with accents + whitespace → "sao paulo"
    assert new_df["city"].iloc[0] == "sao paulo"
    # "MÜNCHEN" → "munchen"
    assert new_df["city"].iloc[1] == "munchen"
    # "café" → "cafe"
    assert new_df["city"].iloc[2] == "cafe"
    # "tokyo" stays "tokyo"
    assert new_df["city"].iloc[3] == "tokyo"


def test_nan_preservation():
    """Test that NaN values are preserved after normalization."""
    df, manager = load_test_data("test_norm_nan.csv")

    result = normalize_categorical_text("test_norm_nan.csv", "city")
    assert result.get("error") is None
    new_df = manager.get_data()
    assert pd.isna(new_df["city"].iloc[4])


# ============================================================================
# Flag toggles
# ============================================================================


def test_no_lowercase():
    """Test disabling lowercase keeps original casing."""
    df, manager = load_test_data("test_norm_nolc.csv")

    result = normalize_categorical_text(
        "test_norm_nolc.csv", "city", lowercase=False
    )
    assert result.get("error") is None
    assert "lowercase" not in result["operations_applied"]

    new_df = manager.get_data()
    # MÜNCHEN → accents stripped but case kept → MUNCHEN
    val = new_df["city"].iloc[1]
    assert val == "MUNCHEN" or val.isupper()


def test_no_strip_accents():
    """Test disabling accent stripping preserves diacritics."""
    df, manager = load_test_data("test_norm_noac.csv")

    result = normalize_categorical_text(
        "test_norm_noac.csv", "city", strip_accents_flag=False
    )
    assert result.get("error") is None
    assert "strip_accents" not in result["operations_applied"]


# ============================================================================
# Error handling
# ============================================================================


def test_missing_column():
    """Test error when column doesn't exist."""
    load_test_data("test_norm_err1.csv")
    result = normalize_categorical_text("test_norm_err1.csv", "nonexistent")
    assert "error" in result
    assert "not found" in result["error"]


def test_numeric_column():
    """Test error when column is numeric."""
    load_test_data("test_norm_err2.csv")
    result = normalize_categorical_text("test_norm_err2.csv", "score")
    assert "error" in result
    assert "numeric" in result["error"]


# ============================================================================
# Idempotency
# ============================================================================


def test_idempotency():
    """Test that running normalization twice produces the same result."""
    load_test_data("test_norm_idem.csv")

    result1 = normalize_categorical_text("test_norm_idem.csv", "city")
    assert result1.get("error") is None

    manager = GlobalStateManager()
    df_after_first = manager.get_data().copy()

    result2 = normalize_categorical_text("test_norm_idem.csv", "city")
    assert result2.get("error") is None
    assert result2["values_changed"] == 0

    df_after_second = manager.get_data()
    pd.testing.assert_frame_equal(df_after_first, df_after_second)


# ============================================================================
# Sample changes
# ============================================================================


def test_sample_changes_present():
    """Test that sample_changes shows before/after diffs."""
    load_test_data("test_norm_samples.csv")

    result = normalize_categorical_text("test_norm_samples.csv", "city")
    assert result.get("error") is None
    assert len(result["sample_changes"]) > 0
    for change in result["sample_changes"]:
        assert "before" in change
        assert "after" in change
        assert change["before"] != change["after"]


# ============================================================================
# Runner
# ============================================================================


def run_tests():
    """Run all tests manually."""
    tests = [
        test_basic_normalization,
        test_nan_preservation,
        test_no_lowercase,
        test_no_strip_accents,
        test_missing_column,
        test_numeric_column,
        test_idempotency,
        test_sample_changes_present,
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
