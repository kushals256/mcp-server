"""
Tests for Feature Extraction Tool.

Covers datetime, text, and math extraction methods,
plus error handling for missing columns and no dataset.
"""

import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset_analysis_mcp.tools.extract_features import extract_features
from dataset_analysis_mcp.utils.state_manager import GlobalStateManager


def test_datetime_year_extraction():
    """Test extracting year from a datetime column."""
    df = pd.DataFrame({
        "date": pd.to_datetime(["2023-01-15", "2024-06-30", "2025-12-01"]),
    })
    manager = GlobalStateManager()
    manager.load_data(df, "test_dt.csv")

    result = extract_features(
        method="datetime",
        columns=["date"],
        operation="year"
    )

    assert result.get("error") is None
    new_df = manager.get_data()
    assert any("year" in c.lower() for c in new_df.columns)


def test_datetime_month_extraction():
    """Test extracting month from a datetime column."""
    df = pd.DataFrame({
        "date": pd.to_datetime(["2023-01-15", "2023-06-30", "2023-12-01"]),
    })
    manager = GlobalStateManager()
    manager.load_data(df, "test_dt_month.csv")

    result = extract_features(
        method="datetime",
        columns=["date"],
        operation="month"
    )

    assert result.get("error") is None
    new_df = manager.get_data()
    assert any("month" in c.lower() for c in new_df.columns)


def test_text_length_extraction():
    """Test extracting text length features."""
    df = pd.DataFrame({
        "text": ["hello world", "hi", "this is a longer sentence"],
    })
    manager = GlobalStateManager()
    manager.load_data(df, "test_text.csv")

    result = extract_features(
        method="text",
        columns=["text"],
        operation="length"
    )

    assert result.get("error") is None
    new_df = manager.get_data()
    assert len(new_df.columns) > 1  # At least one new feature added


def test_math_log_extraction():
    """Test log transformation."""
    df = pd.DataFrame({
        "val": [1.0, 10.0, 100.0, 1000.0],
    })
    manager = GlobalStateManager()
    manager.load_data(df, "test_log.csv")

    result = extract_features(
        method="math",
        columns=["val"],
        operation="log"
    )

    assert result.get("error") is None
    new_df = manager.get_data()
    assert len(new_df.columns) > 1


def test_missing_column_error():
    """Test error when column doesn't exist."""
    df = pd.DataFrame({"A": [1, 2, 3]})
    manager = GlobalStateManager()
    manager.load_data(df, "test_err.csv")

    result = extract_features(
        method="math",
        columns=["nonexistent"],
        operation="log"
    )

    assert "error" in result
    assert "not found" in result["error"]


def test_no_dataset_error():
    """Test error when no dataset is loaded."""
    result = extract_features(
        method="math",
        columns=["A"],
        operation="log"
    )

    assert "error" in result


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main(["-v", __file__]))
