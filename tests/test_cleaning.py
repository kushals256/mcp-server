"""
Tests for Duplicate Row Removal Tool.

Covers basic deduplication, subset columns, keep strategies,
empty columns list, and error handling.
"""

import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.cleaning import drop_duplicate_rows, DropDuplicateRowsRequest
from utils.state_manager import GlobalStateManager


def load_test_data(name="test_cleaning.csv"):
    """Load test data with duplicates into GlobalStateManager."""
    df = pd.DataFrame({
        "A": [1, 1, 2, 3, 2],
        "B": ["x", "x", "y", "z", "y"],
        "C": [10, 10, 20, 30, 20],
    })
    manager = GlobalStateManager()
    manager.load_data(df, name)
    return df, manager


def test_drop_duplicate_rows_basic():
    """Test basic deduplication with keep='first'."""
    df, manager = load_test_data("test_dup_basic.csv")

    result = drop_duplicate_rows(DropDuplicateRowsRequest(
        dataset_name="test_dup_basic.csv",
        subset_columns=None,
        keep="first"
    ))

    assert result.get("error") is None, f"Unexpected error: {result.get('error')}"
    assert result["rows_removed"] == 2  # rows 1 and 4 are dupes of 0 and 2
    assert result["remaining_rows"] == 3

    new_df = manager.get_data()
    assert len(new_df) == 3
    assert new_df.duplicated().sum() == 0


def test_drop_duplicate_rows_keep_last():
    """Test deduplication with keep='last'."""
    df, manager = load_test_data("test_dup_last.csv")

    result = drop_duplicate_rows(DropDuplicateRowsRequest(
        dataset_name="test_dup_last.csv",
        subset_columns=None,
        keep="last"
    ))

    assert result.get("error") is None
    assert result["rows_removed"] == 2
    assert result["remaining_rows"] == 3


def test_drop_duplicate_rows_keep_none():
    """Test deduplication with keep='none' (remove all duplicates)."""
    df, manager = load_test_data("test_dup_none.csv")

    result = drop_duplicate_rows(DropDuplicateRowsRequest(
        dataset_name="test_dup_none.csv",
        subset_columns=None,
        keep="none"
    ))

    assert result.get("error") is None
    # Rows [1,1,x,10] and [2,y,20] each have 2 copies → all 4 removed, 1 remains
    assert result["rows_removed"] == 4
    assert result["remaining_rows"] == 1


def test_drop_duplicate_rows_subset():
    """Test deduplication on a subset of columns."""
    df = pd.DataFrame({
        "A": [1, 1, 1],
        "B": ["x", "x", "y"],
    })
    manager = GlobalStateManager()
    manager.load_data(df, "test_dup_subset.csv")

    result = drop_duplicate_rows(DropDuplicateRowsRequest(
        dataset_name="test_dup_subset.csv",
        subset_columns=["A"],
        keep="first"
    ))

    assert result.get("error") is None
    assert result["rows_removed"] == 2  # A has 3 copies of 1
    assert result["remaining_rows"] == 1


def test_drop_duplicate_rows_no_duplicates():
    """Test when no duplicates exist."""
    df = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]})
    manager = GlobalStateManager()
    manager.load_data(df, "test_dup_nodup.csv")

    result = drop_duplicate_rows(DropDuplicateRowsRequest(
        dataset_name="test_dup_nodup.csv",
        subset_columns=None,
        keep="first"
    ))

    assert result.get("error") is None
    assert result["rows_removed"] == 0
    assert result["remaining_rows"] == 3


def test_drop_duplicate_rows_wrong_dataset():
    """Test error when dataset name doesn't match loaded dataset."""
    load_test_data("test_correct.csv")

    result = drop_duplicate_rows(DropDuplicateRowsRequest(
        dataset_name="wrong_name.csv",
        subset_columns=None,
        keep="first"
    ))

    assert "error" in result
    assert "not currently loaded" in result["error"]


if __name__ == "__main__":
    try:
        import pytest
        sys.exit(pytest.main(["-v", __file__]))
    except ImportError:
        tests = [
            test_drop_duplicate_rows_basic,
            test_drop_duplicate_rows_keep_last,
            test_drop_duplicate_rows_keep_none,
            test_drop_duplicate_rows_subset,
            test_drop_duplicate_rows_no_duplicates,
            test_drop_duplicate_rows_wrong_dataset,
        ]
        for t in tests:
            try:
                t()
                print(f"  ✅ {t.__name__}")
            except Exception as e:
                print(f"  ❌ {t.__name__}: {e}")
