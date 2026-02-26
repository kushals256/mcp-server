"""
Tests for Persistence / Preprocessing Report Tool.

Covers report generation with various action history entries,
empty history, and handler coverage.
"""

import sys
import os
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.persistence import generate_preprocessing_report
from utils.state_manager import GlobalStateManager


def test_empty_history():
    """Test report with no actions recorded."""
    result = generate_preprocessing_report()
    assert len(result.steps) == 1
    assert "No actions recorded" in result.steps[0]


def test_load_data_step():
    """Test report includes load_data step."""
    manager = GlobalStateManager()
    df = pd.DataFrame({"A": [1, 2]})
    manager.load_data(df, "test.csv")

    result = generate_preprocessing_report()
    assert any("Loaded dataset" in s for s in result.steps)


def test_drop_duplicate_step():
    """Test report includes drop_duplicate_rows step."""
    manager = GlobalStateManager()
    df = pd.DataFrame({"A": [1, 2]})
    manager.load_data(df, "test.csv")
    manager.log_action("drop_duplicate_rows", {
        "rows_removed": 5,
        "subset_columns": ["A"],
    })

    result = generate_preprocessing_report()
    assert any("duplicate" in s.lower() for s in result.steps)
    assert any("5" in s for s in result.steps)


def test_drop_columns_step():
    """Test report includes drop_columns step."""
    manager = GlobalStateManager()
    df = pd.DataFrame({"A": [1], "B": [2]})
    manager.load_data(df, "test.csv")
    manager.log_action("drop_columns", {
        "columns_dropped": ["B"],
    })

    result = generate_preprocessing_report()
    assert any("Dropped" in s and "B" in s for s in result.steps)


def test_remove_outliers_step():
    """Test report includes remove_outliers step."""
    manager = GlobalStateManager()
    df = pd.DataFrame({"A": [1]})
    manager.load_data(df, "test.csv")
    manager.log_action("remove_outliers", {
        "column": "price",
        "method": "zscore",
        "rows_removed": 3,
    })

    result = generate_preprocessing_report()
    assert any("outlier" in s.lower() for s in result.steps)
    assert any("zscore" in s for s in result.steps)


def test_create_feature_step():
    """Test report includes create_feature step."""
    manager = GlobalStateManager()
    df = pd.DataFrame({"A": [1]})
    manager.load_data(df, "test.csv")
    manager.log_action("create_feature", {
        "name": "total",
        "expression": "A * 2",
    })

    result = generate_preprocessing_report()
    assert any("total" in s and "A * 2" in s for s in result.steps)


def test_handle_missing_values_step():
    """Test report includes handle_missing_values step."""
    manager = GlobalStateManager()
    df = pd.DataFrame({"A": [1]})
    manager.load_data(df, "test.csv")
    manager.log_action("handle_missing_values", {
        "column": "age",
        "strategy": "mean",
        "rows_affected": 10,
    })

    result = generate_preprocessing_report()
    assert any("missing" in s.lower() and "age" in s for s in result.steps)


def test_multiple_steps():
    """Test report with multiple actions in sequence."""
    manager = GlobalStateManager()
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    manager.load_data(df, "multi.csv")
    manager.log_action("drop_columns", {"columns_dropped": ["B"]})
    manager.log_action("remove_outliers", {
        "column": "A", "method": "iqr", "rows_removed": 1
    })

    result = generate_preprocessing_report()
    # Should have: load_data + drop_columns + remove_outliers = 3 steps
    assert len(result.steps) == 3


def test_rollback_step():
    """Test report includes rollback step."""
    manager = GlobalStateManager()
    df = pd.DataFrame({"A": [1]})
    manager.load_data(df, "test.csv")
    manager.log_action("rollback", {
        "rolled_back_to": 0,
        "new_version": 2,
    })

    result = generate_preprocessing_report()
    assert any("Rolled back" in s for s in result.steps)


def test_validate_action_step():
    """Test report includes validate_action step."""
    manager = GlobalStateManager()
    df = pd.DataFrame({"A": [1]})
    manager.load_data(df, "test.csv")
    manager.log_action("validate_action", {
        "target_tool": "drop_columns",
        "allowed": True,
    })

    result = generate_preprocessing_report()
    assert any("Validated" in s for s in result.steps)


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main(["-v", __file__]))
