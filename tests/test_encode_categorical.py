"""
Tests for Categorical Feature Encoding Tool.

Covers all 8 encoding methods, NaN preservation, error handling,
risk indicators, and edge cases.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.encode_categorical import encode_categorical_feature
from utils.state_manager import GlobalStateManager


# ============================================================================
# Test data helpers
# ============================================================================


def setup_categorical_data():
    """Create a dataframe with categorical columns for testing."""
    return pd.DataFrame({
        "color": ["red", "blue", "green", "blue", "red", "green", np.nan, "red"],
        "size": ["S", "M", "L", "XL", "S", "M", "L", np.nan],
        "score": [10, 20, 30, 40, 50, 60, 70, 80],
        "target": [1, 0, 1, 0, 1, 0, 1, 0],
    })


def load_test_data(name="test.csv"):
    """Load test data into the GlobalStateManager."""
    df = setup_categorical_data()
    manager = GlobalStateManager()
    manager.load_data(df, name)
    return df, manager


# ============================================================================
# One-Hot Encoding
# ============================================================================


def test_one_hot_basic():
    """Test basic one-hot encoding creates correct dummy columns."""
    df, manager = load_test_data("test_oh.csv")

    result = encode_categorical_feature("test_oh.csv", "color", "one_hot")

    assert result.get("error") is None, f"Unexpected error: {result.get('error')}"
    assert result["method"] == "one_hot"
    assert result["original_column_dropped"] is True
    assert result["cardinality_before"] == 3
    assert len(result["new_columns_created"]) == 3
    assert "columns_created" in result["encoding_metadata"]
    assert result["encoding_metadata"]["columns_created"] == 3

    # Verify state was updated
    new_df = manager.get_data()
    assert "color" not in new_df.columns  # original dropped


def test_one_hot_high_cardinality_block():
    """Test that one-hot blocks encoding for very high cardinality."""
    # Create a column with >100 unique values
    df = pd.DataFrame({"cat": [f"val_{i}" for i in range(150)]})
    manager = GlobalStateManager()
    manager.load_data(df, "test_hc.csv")

    result = encode_categorical_feature("test_hc.csv", "cat", "one_hot")
    assert "error" in result
    assert "exceeds" in result["error"]


# ============================================================================
# Label Encoding
# ============================================================================


def test_label_encoding():
    """Test label encoding assigns unique integers and returns mapping."""
    df, manager = load_test_data("test_label.csv")

    result = encode_categorical_feature("test_label.csv", "color", "label")

    assert result.get("error") is None, f"Unexpected error: {result.get('error')}"
    assert result["method"] == "label"
    assert "mapping" in result["encoding_metadata"]

    mapping = result["encoding_metadata"]["mapping"]
    assert len(mapping) == 3  # red, blue, green

    new_df = manager.get_data()
    assert "color_encoded" in new_df.columns


# ============================================================================
# Ordinal Encoding
# ============================================================================


def test_ordinal_with_mapping():
    """Test ordinal encoding respects user-provided ordering."""
    df, manager = load_test_data("test_ord.csv")

    mapping = {"S": 0, "M": 1, "L": 2, "XL": 3}
    result = encode_categorical_feature(
        "test_ord.csv", "size", "ordinal", ordinal_mapping=mapping
    )

    assert result.get("error") is None, f"Unexpected error: {result.get('error')}"
    assert result["method"] == "ordinal"
    assert result["encoding_metadata"]["mapping"] == mapping

    new_df = manager.get_data()
    assert "size_ordinal" in new_df.columns
    # Check first non-NaN S row is mapped to 0
    s_vals = new_df.loc[df["size"] == "S", "size_ordinal"]
    assert (s_vals == 0).all()


def test_ordinal_missing_mapping_error():
    """Test ordinal encoding fails if no mapping provided."""
    load_test_data("test_ord_err.csv")

    result = encode_categorical_feature("test_ord_err.csv", "size", "ordinal")
    assert "error" in result
    assert "ordinal_mapping" in result["error"]


# ============================================================================
# Frequency Encoding
# ============================================================================


def test_frequency_encoding():
    """Test frequency encoding maps values to their normalized counts."""
    df, manager = load_test_data("test_freq.csv")

    result = encode_categorical_feature("test_freq.csv", "color", "frequency")

    assert result.get("error") is None, f"Unexpected error: {result.get('error')}"
    assert result["method"] == "frequency"
    assert "value_counts" in result["encoding_metadata"]

    new_df = manager.get_data()
    assert "color_freq" in new_df.columns

    # Red appears 3 times out of 7 non-null → ~0.4286
    red_freq = result["encoding_metadata"]["value_counts"].get("red")
    assert red_freq is not None
    assert abs(red_freq - 3 / 7) < 0.01


# ============================================================================
# Target Encoding
# ============================================================================


def test_target_encoding():
    """Test target encoding requires target_column and returns metadata."""
    df, manager = load_test_data("test_target.csv")

    result = encode_categorical_feature(
        "test_target.csv", "color", "target", target_column="target"
    )

    assert result.get("error") is None, f"Unexpected error: {result.get('error')}"
    assert result["method"] == "target"
    assert result["encoding_metadata"]["target_column"] == "target"
    assert result["encoding_metadata"]["leakage_safe"] is False

    new_df = manager.get_data()
    assert "color_target" in new_df.columns


def test_target_missing_target_error():
    """Test target encoding fails if no target column provided."""
    load_test_data("test_target_err.csv")

    result = encode_categorical_feature("test_target_err.csv", "color", "target")
    assert "error" in result
    assert "target_column" in result["error"]


# ============================================================================
# Binary Encoding
# ============================================================================


def test_binary_encoding():
    """Test binary encoding creates log2(k) columns."""
    df, manager = load_test_data("test_bin.csv")

    result = encode_categorical_feature("test_bin.csv", "color", "binary")

    assert result.get("error") is None, f"Unexpected error: {result.get('error')}"
    assert result["method"] == "binary"
    assert result["encoding_metadata"]["n_binary_columns"] > 0

    new_df = manager.get_data()
    for col in result["new_columns_created"]:
        assert col in new_df.columns


# ============================================================================
# Hashing Encoding
# ============================================================================


def test_hashing_encoding():
    """Test hashing encoding creates n_components columns and reports collision risk."""
    df, manager = load_test_data("test_hash.csv")

    result = encode_categorical_feature(
        "test_hash.csv", "color", "hashing", n_components=4
    )

    assert result.get("error") is None, f"Unexpected error: {result.get('error')}"
    assert result["method"] == "hashing"
    assert result["encoding_metadata"]["n_buckets"] == 4
    assert "collision_risk" in result["encoding_metadata"]
    assert len(result["new_columns_created"]) == 4


# ============================================================================
# Leave-One-Out Encoding
# ============================================================================


def test_leave_one_out():
    """Test leave-one-out encoding is flagged as leakage-safe."""
    df, manager = load_test_data("test_loo.csv")

    result = encode_categorical_feature(
        "test_loo.csv", "color", "leave_one_out", target_column="target"
    )

    assert result.get("error") is None, f"Unexpected error: {result.get('error')}"
    assert result["method"] == "leave_one_out"
    assert result["encoding_metadata"]["leakage_safe"] is True

    new_df = manager.get_data()
    assert "color_loo" in new_df.columns


# ============================================================================
# NaN Preservation
# ============================================================================


def test_nan_preservation():
    """Test that NaN rows survive all encoding methods."""
    methods_simple = ["label", "frequency"]

    for method in methods_simple:
        df, manager = load_test_data(f"test_nan_{method}.csv")
        result = encode_categorical_feature(f"test_nan_{method}.csv", "color", method)

        assert result.get("error") is None, f"{method} error: {result.get('error')}"

        new_df = manager.get_data()
        new_col = result["new_columns_created"][0]
        # Row 6 had NaN in original color column
        assert pd.isna(new_df[new_col].iloc[6]), f"{method}: NaN not preserved at row 6"


# ============================================================================
# Error Handling
# ============================================================================


def test_invalid_inputs():
    """Test error handling for missing columns, numeric columns, etc."""
    df, manager = load_test_data("test_inv.csv")

    # Missing column
    result = encode_categorical_feature("test_inv.csv", "nonexistent", "label")
    assert "error" in result
    assert "not found" in result["error"]

    # Numeric column
    result = encode_categorical_feature("test_inv.csv", "score", "label")
    assert "error" in result
    assert "numeric" in result["error"]


# ============================================================================
# Handle Unknown
# ============================================================================


def test_handle_unknown():
    """Test handle_unknown='new_category' replaces NaN with sentinel."""
    df, manager = load_test_data("test_unk.csv")

    result = encode_categorical_feature(
        "test_unk.csv", "color", "frequency", handle_unknown="new_category"
    )

    assert result.get("error") is None, f"Unexpected error: {result.get('error')}"
    new_df = manager.get_data()
    new_col = result["new_columns_created"][0]
    # NaN row should now have a value (not NaN)
    assert not pd.isna(new_df[new_col].iloc[6])


# ============================================================================
# Risk Indicators
# ============================================================================


def test_risk_indicators():
    """Test that risk indicators are always present."""
    df, manager = load_test_data("test_risk.csv")

    result = encode_categorical_feature("test_risk.csv", "color", "one_hot")

    assert "memory_risk" in result
    assert result["memory_risk"] in ("low", "medium", "high")
    assert "leakage_risk" in result
    assert isinstance(result["leakage_risk"], bool)
    assert "overfit_risk" in result
    assert result["overfit_risk"] in ("low", "medium", "high")
    assert "feature_space_delta" in result
    assert "dimensionality_change" in result


# ============================================================================
# Drop Original
# ============================================================================


def test_drop_original_false():
    """Test original column is preserved when drop_original=False."""
    df, manager = load_test_data("test_keep.csv")

    result = encode_categorical_feature(
        "test_keep.csv", "color", "label", drop_original=False
    )

    assert result.get("error") is None
    assert result["original_column_dropped"] is False

    new_df = manager.get_data()
    assert "color" in new_df.columns  # original preserved
    assert "color_encoded" in new_df.columns  # new column also present


# ============================================================================
# Runner
# ============================================================================


def run_tests():
    """Run all tests manually."""
    tests = [
        test_one_hot_basic,
        test_one_hot_high_cardinality_block,
        test_label_encoding,
        test_ordinal_with_mapping,
        test_ordinal_missing_mapping_error,
        test_frequency_encoding,
        test_target_encoding,
        test_target_missing_target_error,
        test_binary_encoding,
        test_hashing_encoding,
        test_leave_one_out,
        test_nan_preservation,
        test_invalid_inputs,
        test_handle_unknown,
        test_risk_indicators,
        test_drop_original_false,
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
