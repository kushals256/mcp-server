import sys
import os
import pandas as pd
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.state_manager import GlobalStateManager
from tools.save_dataset import save_processed_dataset, SaveDatasetRequest


@pytest.fixture
def manager():
    mgr = GlobalStateManager()
    mgr.clear_state()
    return mgr


# ============================================================================
# load_data() defensive copy
# ============================================================================


def test_load_data_defensive_copy(manager):
    """Mutating the original DataFrame after load_data() must NOT affect stored data."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    manager.load_data(df, "test.csv")

    # Mutate the original DataFrame
    df["a"] = [999, 999, 999]

    # Stored data should be untouched
    stored = manager.get_data()
    assert list(stored["a"]) == [1, 2, 3]


# ============================================================================
# get_data() returns a copy
# ============================================================================


def test_get_data_returns_copy(manager):
    """Mutating the DataFrame returned by get_data() must NOT affect stored data."""
    df = pd.DataFrame({"a": [10, 20, 30]})
    manager.load_data(df, "test.csv")

    # Get data and mutate it
    retrieved = manager.get_data()
    retrieved.loc[0, "a"] = 999

    # Stored data should be untouched
    stored = manager.get_data()
    assert stored.loc[0, "a"] == 10


def test_get_data_returns_none_when_empty(manager):
    """get_data() should return None when no dataset is loaded."""
    assert manager.get_data() is None


# ============================================================================
# save_processed_dataset overwrite guard
# ============================================================================


def test_save_blocks_source_overwrite(manager):
    """Saving to the same filename as the loaded source should be blocked."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    manager.load_data(df, "data.csv")

    req = SaveDatasetRequest(format="csv", path="data.csv")
    result = save_processed_dataset(req)

    assert not result.success
    assert "matches the currently loaded source dataset" in result.message


def test_save_allows_different_filename(manager, tmp_path):
    """Saving to a different filename should work normally."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    manager.load_data(df, "original.csv")

    target = tmp_path / "processed.csv"
    req = SaveDatasetRequest(format="csv", path=str(target))
    result = save_processed_dataset(req)

    assert result.success
    assert os.path.exists(target)

    saved = pd.read_csv(target)
    assert len(saved) == 3
    assert list(saved["a"]) == [1, 2, 3]


if __name__ == "__main__":
    try:
        import pytest
        sys.exit(pytest.main(["-v", __file__]))
    except ImportError:
        print("Pytest not found.")
