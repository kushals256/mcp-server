"""
Tests for the Dataset Versioning System.

Covers:
    - VersionManager: snapshot, LRU eviction, pinning, diff, clear
    - MCP tools: list_versions, rollback_version, diff_versions
    - Integration: auto-snapshot on GlobalStateManager mutations
"""

import pytest
import pandas as pd
import numpy as np
from utils.version_manager import VersionManager, VersionEntry
from utils.state_manager import GlobalStateManager


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def reset_state():
    """Reset the GlobalStateManager singleton before each test."""
    manager = GlobalStateManager()
    manager.clear_state()
    yield
    manager.clear_state()


@pytest.fixture
def sample_df():
    """A simple DataFrame for testing."""
    return pd.DataFrame(
        {
            "age": [25, 30, 35, 40, 45],
            "income": [50000, 60000, 70000, 80000, 90000],
            "city": ["NYC", "LA", "NYC", "SF", "LA"],
        }
    )


@pytest.fixture
def sample_df_cleaned():
    """A 'cleaned' version with fewer rows and a type change."""
    return pd.DataFrame(
        {
            "age": [25, 30, 35, 40],
            "income": [50000, 60000, 70000, 80000],
            "city": ["NYC", "LA", "NYC", "SF"],
        }
    )


@pytest.fixture
def sample_df_with_new_col():
    """A version with an extra column added."""
    return pd.DataFrame(
        {
            "age": [25, 30, 35],
            "income": [50000, 60000, 70000],
            "city": ["NYC", "LA", "NYC"],
            "age_squared": [625, 900, 1225],
        }
    )


@pytest.fixture
def vm():
    """A fresh VersionManager with default max_versions."""
    return VersionManager(max_versions=10)


@pytest.fixture
def small_vm():
    """A VersionManager with a small capacity for eviction tests."""
    return VersionManager(max_versions=3)


# =============================================================================
# VersionManager Unit Tests
# =============================================================================


class TestVersionManagerSnapshot:
    """Tests for the snapshot method."""

    def test_first_snapshot_is_v0(self, vm, sample_df):
        v = vm.snapshot(sample_df, tool="load_data", params={"name": "test.csv"})
        assert v == 0

    def test_increments_versions(self, vm, sample_df, sample_df_cleaned):
        v0 = vm.snapshot(sample_df, tool="load_data")
        v1 = vm.snapshot(sample_df_cleaned, tool="remove_outliers")
        assert v0 == 0
        assert v1 == 1

    def test_snapshot_stores_copy(self, vm, sample_df):
        vm.snapshot(sample_df, tool="load_data")
        # Mutate original — snapshot should be unaffected
        sample_df["age"] = [0, 0, 0, 0, 0]
        entry = vm.get(0)
        assert list(entry.df["age"]) == [25, 30, 35, 40, 45]

    def test_snapshot_records_metadata(self, vm, sample_df):
        vm.snapshot(sample_df, tool="load_data", params={"dataset_name": "test.csv"})
        entry = vm.get(0)
        assert entry.tool == "load_data"
        assert entry.params == {"dataset_name": "test.csv"}
        assert entry.rows == 5
        assert entry.columns == 3
        assert entry.memory_mb > 0
        assert entry.timestamp is not None

    def test_snapshot_column_names(self, vm, sample_df):
        vm.snapshot(sample_df, tool="load_data")
        entry = vm.get(0)
        assert entry.column_names == ("age", "income", "city")


class TestVersionManagerLRU:
    """Tests for LRU eviction."""

    def test_evicts_oldest_non_pinned(self, small_vm, sample_df):
        # v0 is auto-pinned
        small_vm.snapshot(sample_df, tool="load_data")  # v0 (pinned)
        small_vm.snapshot(sample_df, tool="step1")  # v1
        small_vm.snapshot(sample_df, tool="step2")  # v2
        # Now at capacity (3). Adding v3 should evict v1 (oldest non-pinned).
        small_vm.snapshot(sample_df, tool="step3")  # v3 → evicts v1
        assert small_vm.stored_count == 3
        assert 0 in small_vm._versions  # pinned, not evicted
        assert 1 not in small_vm._versions  # evicted
        assert 2 in small_vm._versions
        assert 3 in small_vm._versions

    def test_v0_auto_pinned(self, vm, sample_df):
        vm.snapshot(sample_df, tool="load_data")
        assert vm.is_pinned(0)

    def test_pinned_versions_survive_eviction(self, small_vm, sample_df):
        small_vm.snapshot(sample_df, tool="load_data")  # v0 (auto-pinned)
        small_vm.snapshot(sample_df, tool="step1")  # v1
        small_vm.pin(1)  # pin v1 too
        small_vm.snapshot(sample_df, tool="step2")  # v2
        # At capacity. v3 should evict v2 (only non-pinned).
        small_vm.snapshot(sample_df, tool="step3")  # v3 → evicts v2
        assert 0 in small_vm._versions
        assert 1 in small_vm._versions
        assert 2 not in small_vm._versions
        assert 3 in small_vm._versions

    def test_all_pinned_allows_over_capacity(self, small_vm, sample_df):
        small_vm.snapshot(sample_df, tool="load_data")  # v0 (auto-pinned)
        small_vm.snapshot(sample_df, tool="step1")  # v1
        small_vm.pin(1)
        small_vm.snapshot(sample_df, tool="step2")  # v2
        small_vm.pin(2)
        # All pinned. Adding v3 cannot evict anyone.
        small_vm.snapshot(sample_df, tool="step3")  # v3
        assert small_vm.stored_count == 4  # over max, but all pinned


class TestVersionManagerRetrieval:
    """Tests for get and list methods."""

    def test_get_existing(self, vm, sample_df):
        vm.snapshot(sample_df, tool="load_data")
        entry = vm.get(0)
        assert isinstance(entry, VersionEntry)
        assert entry.version == 0

    def test_get_missing_raises(self, vm):
        with pytest.raises(KeyError, match="Version 99 not found"):
            vm.get(99)

    def test_get_evicted_raises(self, small_vm, sample_df):
        small_vm.snapshot(sample_df, tool="load_data")  # v0
        small_vm.snapshot(sample_df, tool="step1")  # v1
        small_vm.snapshot(sample_df, tool="step2")  # v2
        small_vm.snapshot(sample_df, tool="step3")  # v3 → evicts v1
        with pytest.raises(KeyError, match="Version 1 not found"):
            small_vm.get(1)

    def test_list_all_metadata(self, vm, sample_df, sample_df_cleaned):
        vm.snapshot(sample_df, tool="load_data", params={"name": "test.csv"})
        vm.snapshot(sample_df_cleaned, tool="remove_outliers", params={"column": "age"})
        versions = vm.list_all()
        assert len(versions) == 2
        assert versions[0]["version"] == 0
        assert versions[0]["rows"] == 5
        assert versions[0]["pinned"] is True
        assert versions[1]["version"] == 1
        assert versions[1]["rows"] == 4
        assert versions[1]["pinned"] is False
        # Check no DataFrame in metadata
        for v in versions:
            assert "df" not in v

    def test_get_latest_version(self, vm, sample_df):
        assert vm.get_latest_version() is None
        vm.snapshot(sample_df, tool="load_data")
        assert vm.get_latest_version() == 0
        vm.snapshot(sample_df, tool="step1")
        assert vm.get_latest_version() == 1

    def test_current_version_property(self, vm, sample_df):
        assert vm.current_version == -1
        vm.snapshot(sample_df, tool="load_data")
        assert vm.current_version == 0
        vm.snapshot(sample_df, tool="step1")
        assert vm.current_version == 1

    def test_total_memory(self, vm, sample_df):
        vm.snapshot(sample_df, tool="load_data")
        assert vm.total_memory_mb > 0


class TestVersionManagerPinning:
    """Tests for pin/unpin."""

    def test_pin_and_unpin(self, vm, sample_df):
        vm.snapshot(sample_df, tool="load_data")
        vm.snapshot(sample_df, tool="step1")
        assert not vm.is_pinned(1)
        vm.pin(1)
        assert vm.is_pinned(1)
        vm.unpin(1)
        assert not vm.is_pinned(1)

    def test_pin_missing_raises(self, vm):
        with pytest.raises(KeyError):
            vm.pin(99)

    def test_unpin_missing_raises(self, vm):
        with pytest.raises(KeyError):
            vm.unpin(99)


class TestVersionManagerDiff:
    """Tests for the diff engine."""

    def test_diff_shape(self, vm, sample_df, sample_df_cleaned):
        vm.snapshot(sample_df, tool="load_data")
        vm.snapshot(sample_df_cleaned, tool="remove_outliers")
        d = vm.diff(0, 1)
        assert d["shape_diff"]["rows"]["before"] == 5
        assert d["shape_diff"]["rows"]["after"] == 4
        assert d["shape_diff"]["rows"]["delta"] == -1

    def test_diff_columns_added(self, vm, sample_df, sample_df_with_new_col):
        vm.snapshot(sample_df, tool="load_data")
        vm.snapshot(sample_df_with_new_col, tool="create_feature")
        d = vm.diff(0, 1)
        assert "age_squared" in d["columns_added"]
        assert d["columns_removed"] == []

    def test_diff_columns_removed(self, vm, sample_df):
        df_dropped = sample_df.drop(columns=["city"])
        vm.snapshot(sample_df, tool="load_data")
        vm.snapshot(df_dropped, tool="drop_columns")
        d = vm.diff(0, 1)
        assert "city" in d["columns_removed"]
        assert d["columns_added"] == []

    def test_diff_dtype_changes(self, vm, sample_df):
        df_cast = sample_df.copy()
        df_cast["age"] = df_cast["age"].astype(float)
        vm.snapshot(sample_df, tool="load_data")
        vm.snapshot(df_cast, tool="cast_column_type")
        d = vm.diff(0, 1)
        assert "age" in d["dtype_changes"]
        assert d["dtype_changes"]["age"]["before"] == "int64"
        assert d["dtype_changes"]["age"]["after"] == "float64"

    def test_diff_missing_values(self, vm, sample_df):
        df_with_missing = sample_df.copy()
        df_with_missing.loc[0, "income"] = np.nan
        df_with_missing.loc[2, "income"] = np.nan
        vm.snapshot(df_with_missing, tool="load_data")
        # "Fill" the missing values
        vm.snapshot(sample_df, tool="handle_missing_values")
        d = vm.diff(0, 1)
        assert "income" in d["missing_values_diff"]
        assert d["missing_values_diff"]["income"]["before"] == 2
        assert d["missing_values_diff"]["income"]["after"] == 0

    def test_diff_stats(self, vm, sample_df, sample_df_cleaned):
        vm.snapshot(sample_df, tool="load_data")
        vm.snapshot(sample_df_cleaned, tool="remove_outliers")
        d = vm.diff(0, 1)
        # age stats should differ because we removed one row
        assert "age" in d["stats_diff"]
        assert "before" in d["stats_diff"]["age"]
        assert "after" in d["stats_diff"]["age"]

    def test_diff_steps_between(self, vm, sample_df, sample_df_cleaned):
        vm.snapshot(sample_df, tool="load_data", params={"dataset_name": "test.csv"})
        vm.snapshot(sample_df_cleaned, tool="remove_outliers", params={"column": "age"})
        vm.snapshot(sample_df_cleaned, tool="drop_columns", params={"columns": ["id"]})
        d = vm.diff(0, 2)
        assert len(d["steps_between"]) == 2
        assert "remove_outliers" in d["steps_between"][0]
        assert "drop_columns" in d["steps_between"][1]

    def test_diff_memory(self, vm, sample_df, sample_df_cleaned):
        vm.snapshot(sample_df, tool="load_data")
        vm.snapshot(sample_df_cleaned, tool="remove_outliers")
        d = vm.diff(0, 1)
        assert "memory_diff_mb" in d
        assert "before" in d["memory_diff_mb"]
        assert "after" in d["memory_diff_mb"]
        assert "delta" in d["memory_diff_mb"]

    def test_diff_missing_version_raises(self, vm, sample_df):
        vm.snapshot(sample_df, tool="load_data")
        with pytest.raises(KeyError):
            vm.diff(0, 99)


class TestVersionManagerClear:
    """Tests for clear."""

    def test_clear_wipes_all(self, vm, sample_df):
        vm.snapshot(sample_df, tool="load_data")
        vm.snapshot(sample_df, tool="step1")
        vm.clear()
        assert vm.stored_count == 0
        assert vm.current_version == -1
        assert len(vm._pinned) == 0


# =============================================================================
# GlobalStateManager Integration Tests
# =============================================================================


class TestStateManagerAutoSnapshot:
    """Test that GlobalStateManager auto-creates versions."""

    def test_load_data_creates_v0(self, sample_df):
        manager = GlobalStateManager()
        manager.load_data(sample_df, "test.csv")
        assert manager.versions.stored_count == 1
        assert manager.versions.current_version == 0
        entry = manager.versions.get(0)
        assert entry.tool == "load_data"

    def test_update_data_creates_version(self, sample_df, sample_df_cleaned):
        manager = GlobalStateManager()
        manager.load_data(sample_df, "test.csv")  # v0
        manager.update_data(sample_df_cleaned, tool_name="drop_duplicates")  # v1
        assert manager.versions.stored_count == 2
        assert manager.versions.current_version == 1
        entry = manager.versions.get(1)
        assert entry.tool == "drop_duplicates"

    def test_set_split_data_creates_version(self, sample_df):
        manager = GlobalStateManager()
        manager.load_data(sample_df, "test.csv")  # v0
        train = sample_df.iloc[:3]
        test = sample_df.iloc[3:]
        manager.set_split_data(train, test, {"test_size": 0.4})  # v1
        assert manager.versions.stored_count == 2
        entry = manager.versions.get(1)
        assert entry.tool == "train_test_split"

    def test_clear_state_clears_versions(self, sample_df):
        manager = GlobalStateManager()
        manager.load_data(sample_df, "test.csv")
        manager.clear_state()
        assert manager.versions.stored_count == 0

    def test_versions_property_returns_manager(self, sample_df):
        manager = GlobalStateManager()
        assert isinstance(manager.versions, VersionManager)


# =============================================================================
# MCP Tool Tests
# =============================================================================


class TestListVersionsTool:
    """Tests for the list_versions MCP tool."""

    def test_no_versions(self):
        from tools.versioning import list_versions

        result = list_versions()
        assert "No versions available" in result["message"]

    def test_with_versions(self, sample_df, sample_df_cleaned):
        from tools.versioning import list_versions

        manager = GlobalStateManager()
        manager.load_data(sample_df, "test.csv")
        manager.update_data(sample_df_cleaned, tool_name="clean")

        result = list_versions()
        assert result["dataset_name"] == "test.csv"
        assert result["current_version"] == 1
        assert result["total_versions_stored"] == 2
        assert len(result["versions"]) == 2
        assert result["versions"][0]["pinned"] is True  # v0 auto-pinned


class TestRollbackVersionTool:
    """Tests for the rollback_version MCP tool."""

    def test_rollback_to_v0(self, sample_df, sample_df_cleaned):
        from tools.versioning import rollback_version

        manager = GlobalStateManager()
        manager.load_data(sample_df, "test.csv")  # v0: 5 rows
        manager.update_data(sample_df_cleaned, tool_name="clean")  # v1: 4 rows

        result = rollback_version(0)
        assert result["rolled_back_to"] == 0
        assert result["new_version"] == 2  # rollback creates v2
        assert result["rows"] == 5  # restored to original

        # Verify the DataFrame in state manager is actually restored
        current = manager.get_data()
        assert len(current) == 5

    def test_rollback_pins_target(self, sample_df, sample_df_cleaned):
        from tools.versioning import rollback_version

        manager = GlobalStateManager()
        manager.load_data(sample_df, "test.csv")
        manager.update_data(sample_df_cleaned, tool_name="clean")

        rollback_version(0)
        assert manager.versions.is_pinned(0)

    def test_rollback_invalid_version(self, sample_df):
        from tools.versioning import rollback_version

        manager = GlobalStateManager()
        manager.load_data(sample_df, "test.csv")
        result = rollback_version(99)
        assert "error" in result

    def test_rollback_no_data(self):
        from tools.versioning import rollback_version

        result = rollback_version(0)
        assert "error" in result

    def test_rollback_appears_in_history(self, sample_df, sample_df_cleaned):
        from tools.versioning import rollback_version

        manager = GlobalStateManager()
        manager.load_data(sample_df, "test.csv")
        manager.update_data(sample_df_cleaned, tool_name="clean")
        rollback_version(0)
        history = manager.get_history()
        rollback_entries = [h for h in history if h["tool"] == "rollback"]
        assert len(rollback_entries) == 1
        assert rollback_entries[0]["params"]["rolled_back_to"] == 0


class TestDiffVersionsTool:
    """Tests for the diff_versions MCP tool."""

    def test_basic_diff(self, sample_df, sample_df_cleaned):
        from tools.versioning import diff_versions

        manager = GlobalStateManager()
        manager.load_data(sample_df, "test.csv")
        manager.update_data(sample_df_cleaned, tool_name="remove_outliers")

        result = diff_versions(0, 1)
        assert result["version_a"] == 0
        assert result["version_b"] == 1
        assert result["shape_diff"]["rows"]["delta"] == -1
        assert result["dataset_name"] == "test.csv"

    def test_diff_invalid_version(self, sample_df):
        from tools.versioning import diff_versions

        manager = GlobalStateManager()
        manager.load_data(sample_df, "test.csv")
        result = diff_versions(0, 99)
        assert "error" in result

    def test_diff_no_data(self):
        from tools.versioning import diff_versions

        result = diff_versions(0, 1)
        assert "error" in result
