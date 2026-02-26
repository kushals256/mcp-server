"""
Tests for Handle Missing Values Tool.

Covers basic strategies (mean, median, mode, constant, drop_rows),
fill strategies (ffill, bfill), interpolation, and error handling.
"""

import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.handle_missing_values import handle_missing_values, HandleMissingValuesRequest
from utils.state_manager import GlobalStateManager


def load_numeric_data(name="test_missing.csv"):
    """Load test data with missing values in a numeric column."""
    df = pd.DataFrame({
        "val": [1.0, np.nan, 3.0, np.nan, 5.0],
        "cat": ["a", "b", np.nan, "d", "e"],
        "all_present": [10, 20, 30, 40, 50],
    })
    manager = GlobalStateManager()
    manager.load_data(df, name)
    return df, manager


def test_mean_imputation():
    """Test mean strategy fills with column mean."""
    df, manager = load_numeric_data("test_mean.csv")

    result = handle_missing_values(HandleMissingValuesRequest(
        dataset_name="test_mean.csv",
        column="val",
        strategy="mean"
    ))

    assert result.get("error") is None
    assert result["strategy"] == "mean"
    assert result["rows_affected"] == 2

    new_df = manager.get_data()
    assert new_df["val"].isna().sum() == 0
    # Mean of [1, 3, 5] = 3.0
    assert new_df["val"].iloc[1] == 3.0


def test_median_imputation():
    """Test median strategy fills with column median."""
    df, manager = load_numeric_data("test_median.csv")

    result = handle_missing_values(HandleMissingValuesRequest(
        dataset_name="test_median.csv",
        column="val",
        strategy="median"
    ))

    assert result.get("error") is None
    assert result["rows_affected"] == 2

    new_df = manager.get_data()
    assert new_df["val"].isna().sum() == 0


def test_mode_imputation():
    """Test mode strategy fills with most frequent value."""
    df = pd.DataFrame({
        "cat": ["a", "a", "b", np.nan, "a"],
    })
    manager = GlobalStateManager()
    manager.load_data(df, "test_mode.csv")

    result = handle_missing_values(HandleMissingValuesRequest(
        dataset_name="test_mode.csv",
        column="cat",
        strategy="mode"
    ))

    assert result.get("error") is None
    assert result["rows_affected"] == 1

    new_df = manager.get_data()
    assert new_df["cat"].isna().sum() == 0
    assert new_df["cat"].iloc[3] == "a"  # Mode is "a"


def test_constant_imputation():
    """Test constant strategy fills with provided value."""
    df, manager = load_numeric_data("test_const.csv")

    result = handle_missing_values(HandleMissingValuesRequest(
        dataset_name="test_const.csv",
        column="val",
        strategy="constant",
        constant_value=0.0
    ))

    assert result.get("error") is None
    assert result["rows_affected"] == 2

    new_df = manager.get_data()
    assert new_df["val"].isna().sum() == 0
    assert new_df["val"].iloc[1] == 0.0


def test_drop_rows_strategy():
    """Test drop_rows removes rows with missing values."""
    df, manager = load_numeric_data("test_drop.csv")

    result = handle_missing_values(HandleMissingValuesRequest(
        dataset_name="test_drop.csv",
        column="val",
        strategy="drop_rows"
    ))

    assert result.get("error") is None
    assert result["rows_affected"] == 2

    new_df = manager.get_data()
    assert len(new_df) == 3  # 5 - 2 dropped
    assert new_df["val"].isna().sum() == 0


def test_forward_fill():
    """Test forward fill propagates last valid value."""
    df = pd.DataFrame({
        "val": [1.0, np.nan, np.nan, 4.0, np.nan],
    })
    manager = GlobalStateManager()
    manager.load_data(df, "test_ffill.csv")

    result = handle_missing_values(HandleMissingValuesRequest(
        dataset_name="test_ffill.csv",
        column="val",
        strategy="forward_fill"
    ))

    assert result.get("error") is None

    new_df = manager.get_data()
    # ffill: [1, 1, 1, 4, 4]
    assert new_df["val"].iloc[1] == 1.0
    assert new_df["val"].iloc[2] == 1.0
    assert new_df["val"].iloc[4] == 4.0


def test_backward_fill():
    """Test backward fill propagates next valid value."""
    df = pd.DataFrame({
        "val": [np.nan, np.nan, 3.0, np.nan, 5.0],
    })
    manager = GlobalStateManager()
    manager.load_data(df, "test_bfill.csv")

    result = handle_missing_values(HandleMissingValuesRequest(
        dataset_name="test_bfill.csv",
        column="val",
        strategy="backward_fill"
    ))

    assert result.get("error") is None

    new_df = manager.get_data()
    # bfill: [3, 3, 3, 5, 5]
    assert new_df["val"].iloc[0] == 3.0
    assert new_df["val"].iloc[1] == 3.0
    assert new_df["val"].iloc[3] == 5.0


def test_interpolate():
    """Test interpolation fills with interpolated values."""
    df = pd.DataFrame({
        "val": [0.0, np.nan, 2.0, np.nan, 4.0],
    })
    manager = GlobalStateManager()
    manager.load_data(df, "test_interp.csv")

    result = handle_missing_values(HandleMissingValuesRequest(
        dataset_name="test_interp.csv",
        column="val",
        strategy="interpolate"
    ))

    assert result.get("error") is None

    new_df = manager.get_data()
    assert new_df["val"].isna().sum() == 0
    # Linear interpolation: [0, 1, 2, 3, 4]
    assert abs(new_df["val"].iloc[1] - 1.0) < 0.01
    assert abs(new_df["val"].iloc[3] - 3.0) < 0.01


def test_no_missing_values():
    """Test handling a column with no missing values."""
    df, manager = load_numeric_data("test_no_miss.csv")

    result = handle_missing_values(HandleMissingValuesRequest(
        dataset_name="test_no_miss.csv",
        column="all_present",
        strategy="mean"
    ))

    # Should succeed but affect 0 rows
    assert result.get("error") is None
    assert result["rows_affected"] == 0


def test_missing_column_error():
    """Test error when column doesn't exist."""
    load_numeric_data("test_miss_col.csv")

    result = handle_missing_values(HandleMissingValuesRequest(
        dataset_name="test_miss_col.csv",
        column="nonexistent",
        strategy="mean"
    ))

    assert "error" in result


def test_incompatible_strategy_error():
    """Test error when strategy is incompatible with column type."""
    df, manager = load_numeric_data("test_incompat.csv")

    # Mean on a categorical column should fail
    result = handle_missing_values(HandleMissingValuesRequest(
        dataset_name="test_incompat.csv",
        column="cat",
        strategy="mean"
    ))

    assert "error" in result


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main(["-v", __file__]))
