"""
Tests for Feature Reduction Tool (PCA, LDA, TruncatedSVD).

Covers basic dimensionality reduction, validation, and error handling.
"""

import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset_analysis_mcp.tools.reduce_features import reduce_features
from dataset_analysis_mcp.utils.state_manager import GlobalStateManager


def load_numeric_data(name="test_reduce.csv", n_features=5, n_rows=50):
    """Load a dataset with multiple numeric features for reduction."""
    np.random.seed(42)
    data = {f"feat_{i}": np.random.randn(n_rows) for i in range(n_features)}
    data["target"] = np.random.choice([0, 1], n_rows)
    df = pd.DataFrame(data)
    manager = GlobalStateManager()
    manager.load_data(df, name)
    return df, manager


def test_pca_basic():
    """Test basic PCA reduction."""
    df, manager = load_numeric_data("test_pca.csv")

    result = reduce_features(
        method="pca",
        n_components=2,
        target_col="target"
    )

    assert result.get("error") is None
    new_df = manager.get_data()
    # Should have 2 PCA components + target
    assert "target" in new_df.columns
    assert len(new_df.columns) == 3  # PC1, PC2, target


def test_pca_explained_variance():
    """Test that PCA returns explained variance info."""
    df, manager = load_numeric_data("test_pca_ev.csv")

    result = reduce_features(
        method="pca",
        n_components=3,
        target_col="target"
    )

    assert result.get("error") is None
    assert "explained_variance" in result or "explained_variance_ratio" in result


def test_truncated_svd():
    """Test TruncatedSVD reduction."""
    np.random.seed(42)
    # Use a larger, well-conditioned dataset to avoid numerical issues
    data = {f"feat_{i}": np.random.randn(100) * (i + 1) for i in range(5)}
    data["target"] = np.random.choice([0, 1], 100)
    df = pd.DataFrame(data)
    manager = GlobalStateManager()
    manager.load_data(df, "test_svd.csv")

    result = reduce_features(
        method="truncated_svd",
        n_components=2,
        target_col="target"
    )

    assert result.get("error") is None
    new_df = manager.get_data()
    assert "target" in new_df.columns


def test_invalid_components():
    """Test error when n_components <= 0."""
    load_numeric_data("test_inv.csv")

    result = reduce_features(
        method="pca",
        n_components=0
    )

    assert "error" in result


def test_no_dataset_error():
    """Test error when no dataset is loaded."""
    result = reduce_features(
        method="pca",
        n_components=2
    )

    assert "error" in result


def test_missing_target_column():
    """Test error when target column doesn't exist."""
    load_numeric_data("test_miss_target.csv")

    result = reduce_features(
        method="pca",
        n_components=2,
        target_col="nonexistent"
    )

    assert "error" in result


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main(["-v", __file__]))
