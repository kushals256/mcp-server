"""
Tests for Canonical Value Mapping Tool (Layer 2).

Covers synonym replacement, case sensitivity, empty maps, NaN preservation,
substring safety, idempotency, and error handling.
"""

import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.harmonize_categorical import harmonize_categorical_values
from utils.state_manager import GlobalStateManager


# ============================================================================
# Test data helpers
# ============================================================================


def setup_data():
    return pd.DataFrame({
        "country": [
            "united states",
            "u.s.",
            "usa",
            "united kingdom",
            "uk",
            "france",
            np.nan,
            "america",
        ],
        "score": [10, 20, 30, 40, 50, 60, 70, 80],
    })


def load_test_data(name="test_harm.csv"):
    df = setup_data()
    manager = GlobalStateManager()
    manager.load_data(df, name)
    return df, manager


SYNONYM_MAP = {
    "usa": ["united states", "u.s.", "america"],
    "uk": ["united kingdom", "great britain"],
}


# ============================================================================
# Basic harmonization
# ============================================================================


def test_basic_synonym_replacement():
    """Test that synonyms are replaced with canonical values."""
    df, manager = load_test_data("test_harm_basic.csv")

    result = harmonize_categorical_values(
        "test_harm_basic.csv", "country", SYNONYM_MAP
    )

    assert result.get("error") is None, f"Unexpected error: {result.get('error')}"
    assert result["values_harmonized"] > 0
    assert result["unique_after"] < result["unique_before"]

    new_df = manager.get_data()
    # "united states" → "usa"
    assert new_df["country"].iloc[0] == "usa"
    # "u.s." → "usa"
    assert new_df["country"].iloc[1] == "usa"
    # "usa" stays "usa"
    assert new_df["country"].iloc[2] == "usa"
    # "united kingdom" → "uk"
    assert new_df["country"].iloc[3] == "uk"
    # "france" unchanged
    assert new_df["country"].iloc[5] == "france"
    # "america" → "usa"
    assert new_df["country"].iloc[7] == "usa"


def test_nan_preservation():
    """Test that NaN values survive harmonization."""
    df, manager = load_test_data("test_harm_nan.csv")

    result = harmonize_categorical_values(
        "test_harm_nan.csv", "country", SYNONYM_MAP
    )
    assert result.get("error") is None
    new_df = manager.get_data()
    assert pd.isna(new_df["country"].iloc[6])


# ============================================================================
# Edge cases
# ============================================================================


def test_empty_synonym_map():
    """Test that an empty synonym map is a no-op."""
    df, manager = load_test_data("test_harm_empty.csv")

    result = harmonize_categorical_values(
        "test_harm_empty.csv", "country", {}
    )
    assert result.get("error") is None
    assert result["values_harmonized"] == 0
    assert result["unique_before"] == result["unique_after"]


def test_case_insensitive_matching():
    """Test case-insensitive synonym matching (default)."""
    df = pd.DataFrame({"country": ["United States", "USA", "AMERICA"]})
    manager = GlobalStateManager()
    manager.load_data(df, "test_harm_ci.csv")

    result = harmonize_categorical_values(
        "test_harm_ci.csv", "country", SYNONYM_MAP, case_sensitive=False
    )
    assert result.get("error") is None

    new_df = manager.get_data()
    # All should map to "usa"
    for i in range(3):
        assert new_df["country"].iloc[i] == "usa"


def test_substring_safety():
    """Test that FlashText doesn't corrupt substrings.

    'usa' should NOT match inside 'unusual' or 'causality'.
    """
    df = pd.DataFrame({
        "word": ["usa", "unusual", "causality", "united states"],
    })
    manager = GlobalStateManager()
    manager.load_data(df, "test_harm_substr.csv")

    synonym_map = {"usa": ["united states"]}
    result = harmonize_categorical_values(
        "test_harm_substr.csv", "word", synonym_map
    )
    assert result.get("error") is None

    new_df = manager.get_data()
    # "usa" → "usa" (self-mapped)
    assert new_df["word"].iloc[0] == "usa"
    # "unusual" should remain unchanged
    assert new_df["word"].iloc[1] == "unusual"
    # "causality" should remain unchanged
    assert new_df["word"].iloc[2] == "causality"
    # "united states" → "usa"
    assert new_df["word"].iloc[3] == "usa"


# ============================================================================
# Error handling
# ============================================================================


def test_missing_column():
    """Test error when column doesn't exist."""
    load_test_data("test_harm_err1.csv")
    result = harmonize_categorical_values(
        "test_harm_err1.csv", "nonexistent", SYNONYM_MAP
    )
    assert "error" in result
    assert "not found" in result["error"]


def test_numeric_column():
    """Test error when column is numeric."""
    load_test_data("test_harm_err2.csv")
    result = harmonize_categorical_values(
        "test_harm_err2.csv", "score", SYNONYM_MAP
    )
    assert "error" in result
    assert "numeric" in result["error"]


# ============================================================================
# Idempotency
# ============================================================================


def test_idempotency():
    """Test that running harmonization twice produces the same result."""
    load_test_data("test_harm_idem.csv")

    result1 = harmonize_categorical_values(
        "test_harm_idem.csv", "country", SYNONYM_MAP
    )
    assert result1.get("error") is None

    manager = GlobalStateManager()
    df_after_first = manager.get_data().copy()

    result2 = harmonize_categorical_values(
        "test_harm_idem.csv", "country", SYNONYM_MAP
    )
    assert result2.get("error") is None
    assert result2["values_harmonized"] == 0

    df_after_second = manager.get_data()
    pd.testing.assert_frame_equal(df_after_first, df_after_second)


# ============================================================================
# Runner
# ============================================================================


def run_tests():
    tests = [
        test_basic_synonym_replacement,
        test_nan_preservation,
        test_empty_synonym_map,
        test_case_insensitive_matching,
        test_substring_safety,
        test_missing_column,
        test_numeric_column,
        test_idempotency,
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
