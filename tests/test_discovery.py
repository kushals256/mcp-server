"""
Tests for Discovery Tools (list_datasets, load_dataset_metadata, peek_dataset_metadata).

Covers dataset listing, loading into state, read-only peek, and error handling.
"""

import sys
import os
import tempfile
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.discovery import list_datasets, load_dataset_metadata, peek_dataset_metadata
from utils.state_manager import GlobalStateManager


def _create_temp_csv(tmp_dir, name="test.csv"):
    """Create a temporary CSV file for testing."""
    df = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]})
    path = os.path.join(tmp_dir, name)
    df.to_csv(path, index=False)
    return path


def test_list_datasets_empty(monkeypatch, tmp_path):
    """Test listing datasets in an empty directory."""
    monkeypatch.setattr("tools.discovery.DATA_DIR", str(tmp_path))
    result = list_datasets()
    assert len(result) == 1
    assert "No files found" in result[0].filename


def test_list_datasets_with_files(monkeypatch, tmp_path):
    """Test listing datasets finds CSV files."""
    monkeypatch.setattr("tools.discovery.DATA_DIR", str(tmp_path))
    _create_temp_csv(str(tmp_path), "data1.csv")
    _create_temp_csv(str(tmp_path), "data2.csv")
    # Non-CSV file should be ignored
    (tmp_path / "readme.txt").write_text("hello")

    result = list_datasets()
    filenames = [r.filename for r in result]
    assert "data1.csv" in filenames
    assert "data2.csv" in filenames
    assert "readme.txt" not in filenames


def test_load_dataset_metadata_success(monkeypatch, tmp_path):
    """Test loading a dataset into GlobalStateManager."""
    monkeypatch.setattr("tools.discovery.DATA_DIR", str(tmp_path))
    _create_temp_csv(str(tmp_path), "test.csv")

    result = load_dataset_metadata("test.csv")

    assert result.error is None
    assert result.filename == "test.csv"
    assert "A" in result.columns
    assert "B" in result.columns
    assert result.estimated_row_count == 3
    assert len(result.preview) > 0

    # Verify state was updated
    manager = GlobalStateManager()
    assert manager.get_dataset_name() == "test.csv"
    assert manager.get_data() is not None
    assert len(manager.get_data()) == 3


def test_load_dataset_metadata_not_found(monkeypatch, tmp_path):
    """Test loading a non-existent dataset returns error."""
    monkeypatch.setattr("tools.discovery.DATA_DIR", str(tmp_path))

    result = load_dataset_metadata("nonexistent.csv")
    assert result.error is not None
    assert "not found" in result.error


def test_peek_dataset_metadata_no_state_change(monkeypatch, tmp_path):
    """Test that peek does NOT modify GlobalStateManager state."""
    monkeypatch.setattr("tools.discovery.DATA_DIR", str(tmp_path))
    _create_temp_csv(str(tmp_path), "peek_test.csv")

    manager = GlobalStateManager()
    assert manager.get_data() is None  # No data loaded

    result = peek_dataset_metadata("peek_test.csv")

    assert result.error is None
    assert result.filename == "peek_test.csv"
    assert result.estimated_row_count == 3

    # State should NOT have changed
    assert manager.get_data() is None
    assert manager.get_dataset_name() is None


def test_peek_dataset_metadata_not_found(monkeypatch, tmp_path):
    """Test peeking a non-existent file returns error."""
    monkeypatch.setattr("tools.discovery.DATA_DIR", str(tmp_path))

    result = peek_dataset_metadata("nonexistent.csv")
    assert result.error is not None
    assert "not found" in result.error


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main(["-v", __file__]))
