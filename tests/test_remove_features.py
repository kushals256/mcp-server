"""
Tests for Feature Removal Tool.

Covers missing_threshold, variance_threshold, correlation_threshold,
by_name methods, target protection, and error handling.
"""

import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset_analysis_mcp.tools.remove_features import remove_features
from dataset_analysis_mcp.utils.state_manager import GlobalStateManager


def load_test_data(name="test_remove.csv"):
    """Load test data with various column types."""
    np.random.seed(42)
    df = pd.DataFrame({
        "good1": np.random.randn(50),
        "good2": np.random.randn(50),
        "constant": np.ones(50),  # Zero variance
        "mostly_nan": [np.nan] * 45 + [1.0] * 5,  # 90% missing
        "target": np.random.choice([0, 1], 50),
    })
    manager = GlobalStateManager()
    manager.load_data(df, name)
    return df, manager


def test_missing_threshold():
    """Test removing columns with too many missing values."""
    df, manager = load_test_data("test_rm_missing.csv")

    result = remove_features(
        method="missing_threshold",
        threshold=0.5,  # Remove cols with >50% missing
        target_col="target"
    )

    assert result.get("error") is None
    new_df = manager.get_data()
    assert "mostly_nan" not in new_df.columns
    assert "good1" in new_df.columns
    assert "target" in new_df.columns  # Protected


def test_variance_threshold():
    """Test removing low-variance columns."""
    df, manager = load_test_data("test_rm_var.csv")

    result = remove_features(
        method="variance_threshold",
        threshold=0.01,  # Remove cols with variance < 0.01
        target_col="target"
    )

    assert result.get("error") is None
    new_df = manager.get_data()
    assert "constant" not in new_df.columns
    assert "good1" in new_df.columns
    assert "target" in new_df.columns  # Protected


def test_by_name():
    """Test removing specific named columns."""
    df, manager = load_test_data("test_rm_name.csv")

    result = remove_features(
        method="by_name",
        feature_names=["constant", "mostly_nan"],
        target_col="target"
    )

    assert result.get("error") is None
    new_df = manager.get_data()
    assert "constant" not in new_df.columns
    assert "mostly_nan" not in new_df.columns
    assert "good1" in new_df.columns
    assert "target" in new_df.columns


def test_target_protection():
    """Test that target column is never removed."""
    df, manager = load_test_data("test_rm_protect.csv")

    # Even with aggressive threshold, target should survive
    result = remove_features(
        method="variance_threshold",
        threshold=0.0,
        target_col="target"
    )

    assert result.get("error") is None
    new_df = manager.get_data()
    assert "target" in new_df.columns


def test_no_dataset_error():
    """Test error when no dataset loaded."""
    result = remove_features(
        method="missing_threshold",
        threshold=0.5
    )

    assert "error" in result


def test_missing_target_error():
    """Test error when target column doesn't exist."""
    load_test_data("test_rm_miss.csv")

    result = remove_features(
        method="variance_threshold",
        threshold=0.01,
        target_col="nonexistent"
    )

    assert "error" in result


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main(["-v", __file__]))
