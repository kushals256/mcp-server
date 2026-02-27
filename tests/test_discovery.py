"""
Tests for Discovery Tools (list_datasets, load_dataset_metadata,
peek_dataset_metadata, load_dataset).

Covers dataset listing, loading into state, read-only peek, global path
loading, and error handling.
"""

import sys
import os
import stat
import tempfile
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.discovery import list_datasets, load_dataset_metadata, peek_dataset_metadata, load_dataset
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


# =========================================================================
# load_dataset tests — global path support
# =========================================================================


def test_load_dataset_absolute_path(tmp_path):
    """Test loading a CSV from an absolute path."""
    csv_path = _create_temp_csv(str(tmp_path), "abs_test.csv")

    result = load_dataset(csv_path)

    assert result.error is None
    assert result.filename == "abs_test.csv"
    assert "A" in result.columns
    assert "B" in result.columns
    assert result.estimated_row_count == 3
    assert len(result.preview) > 0

    # Verify state was updated
    manager = GlobalStateManager()
    assert manager.get_dataset_name() == "abs_test.csv"
    assert manager.get_data() is not None


def test_load_dataset_tilde_expansion(tmp_path, monkeypatch):
    """Test that ~ is expanded to the home directory."""
    csv_path = _create_temp_csv(str(tmp_path), "tilde_test.csv")
    # Simulate ~ by monkeypatching HOME to tmp_path
    monkeypatch.setenv("HOME", str(tmp_path))

    result = load_dataset("~/tilde_test.csv")

    assert result.error is None
    assert result.filename == "tilde_test.csv"
    assert result.estimated_row_count == 3


def test_load_dataset_relative_path(tmp_path, monkeypatch):
    """Test that relative paths are resolved from CWD."""
    csv_path = _create_temp_csv(str(tmp_path), "rel_test.csv")
    monkeypatch.chdir(tmp_path)

    result = load_dataset("rel_test.csv")

    assert result.error is None
    assert result.filename == "rel_test.csv"
    assert result.estimated_row_count == 3


def test_load_dataset_json_format(tmp_path):
    """Test loading a JSON file from a global path."""
    df = pd.DataFrame({"X": [10, 20], "Y": ["a", "b"]})
    json_path = os.path.join(str(tmp_path), "data.json")
    df.to_json(json_path, orient="records")

    result = load_dataset(json_path)

    assert result.error is None
    assert result.filename == "data.json"
    assert "X" in result.columns
    assert result.estimated_row_count == 2


def test_load_dataset_parquet_format(tmp_path):
    """Test loading a Parquet file from a global path."""
    pytest = __import__("pytest")
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        pytest.skip("pyarrow not installed")

    df = pd.DataFrame({"P": [1, 2, 3], "Q": [4.0, 5.0, 6.0]})
    parquet_path = os.path.join(str(tmp_path), "data.parquet")
    df.to_parquet(parquet_path, index=False)

    result = load_dataset(parquet_path)

    assert result.error is None
    assert result.filename == "data.parquet"
    assert "P" in result.columns
    assert result.estimated_row_count == 3


def test_load_dataset_not_found():
    """Test loading a non-existent file returns error."""
    result = load_dataset("/nonexistent/path/to/file.csv")

    assert result.error is not None
    assert "File not found" in result.error


def test_load_dataset_unsupported_format(tmp_path):
    """Test loading an unsupported file format returns error."""
    txt_path = os.path.join(str(tmp_path), "notes.txt")
    with open(txt_path, "w") as f:
        f.write("hello world")

    result = load_dataset(txt_path)

    assert result.error is not None
    assert "Unsupported file format" in result.error


def test_load_dataset_permission_error(tmp_path):
    """Test loading a file without read permission returns error."""
    csv_path = _create_temp_csv(str(tmp_path), "locked.csv")
    os.chmod(csv_path, 0o000)

    try:
        result = load_dataset(csv_path)
        assert result.error is not None
        assert "Permission denied" in result.error or "Permission" in result.error
    finally:
        # Restore permissions so tmp_path cleanup works
        os.chmod(csv_path, stat.S_IRUSR | stat.S_IWUSR)


def test_load_dataset_stores_in_state(tmp_path):
    """Test that load_dataset correctly stores data in GlobalStateManager."""
    manager = GlobalStateManager()
    assert manager.get_data() is None

    csv_path = _create_temp_csv(str(tmp_path), "state_test.csv")
    result = load_dataset(csv_path)

    assert result.error is None
    assert manager.get_data() is not None
    assert len(manager.get_data()) == 3
    assert manager.get_dataset_name() == "state_test.csv"
    assert list(manager.get_data().columns) == ["A", "B"]


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main(["-v", __file__]))
