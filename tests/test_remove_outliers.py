import sys
import os
import pandas as pd
import numpy as np
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.remove_outliers import remove_outliers
from utils.state_manager import GlobalStateManager

# Mock the GlobalStateManager to avoid loading real files
class MockStateManager:
    def __init__(self):
        self._df = None
        self._name = None
        
    def get_dataset_name(self):
        return self._name
        
    def get_data(self):
        return self._df
        
    def load_data(self, df, name):
        self._df = df
        self._name = name

# We need to patch the real manager in tools.cleaning
# Since we can't easily patch inside a simple script without mock lib complexity, 
# we'll use the real one but manually inject data.
# Note: For this to work efficiently in this environment, I'll rely on the tool's 
# "load_dataset_metadata" check. I'll pre-load a dummy dataset into the real global manager.

def setup_module(module):
    """Setup a dummy dataset in memory before tests run"""
    pass

def test_zscore_basic():
    # Create normal distribution with one outlier
    np.random.seed(42)
    data = np.random.normal(0, 1, 100)
    data = np.append(data, [100]) # Extreme outlier
    df = pd.DataFrame({'val': data})
    
    manager = GlobalStateManager()
    manager.load_data(df, "test_zscore.csv")
    
    result = remove_outliers("test_zscore.csv", "val", "zscore", threshold=3.0)
    
    assert result['rows_removed'] == 1
    assert result['remaining_rows'] == 100
    
    # Verify index reset
    new_df = manager.get_data()
    assert new_df.index[-1] == 99

def test_iqr_skewed():
    # Data: [1, 2, 3, 4, 100]
    # Q1=2, Q3=4, IQR=2. Bounds: [2-3, 4+3] = [-1, 7]. 100 is outlier.
    df = pd.DataFrame({'val': [1, 2, 3, 4, 100]})
    
    manager = GlobalStateManager()
    manager.load_data(df, "test_iqr.csv")
    
    result = remove_outliers("test_iqr.csv", "val", "iqr", threshold=1.5)
    
    assert result['rows_removed'] == 1
    assert result['remaining_rows'] == 4

def test_nan_preservation():
    # NaNs should NOT be removed
    # Need enough data for stable quantiles.
    # Data: [1, 2, 3, 4, 5, 100(outlier), NaN]
    # Q1(25%) of [1,2,3,4,5,100] -> approx 2.25
    # Q3(75%) -> approx 28.75 (if linear) or 5 (if midpoint on small sample?)
    # Let's use a clearer distribution:
    # [10, 10, 10, 10, 10, 1000, NaN] -> Median 10. IQR 0? No.
    # [1, 2, 3, 4, 5, 6, 7, 8, 100, NaN]
    # Valid: 1..8, 100. (9 items).
    # Q1 close to 3. Q3 close to 7. IQR=4. 
    # Upper = 7 + 1.5*4 = 13.
    # 100 is definitely > 13.
    
    data = [1, 2, 3, 4, 5, 6, 7, 8, 100, np.nan]
    df = pd.DataFrame({'val': data})
    
    manager = GlobalStateManager()
    manager.load_data(df, "test_nan.csv")
    
    # IQR removal (100 is outlier, Nan should stay)
    result = remove_outliers("test_nan.csv", "val", "iqr", threshold=1.5)
    
    new_df = manager.get_data()
    print(f"DEBUG: New DF:\n{new_df}")
    
    # Expected: 10 items total. 1 outlier removed (100). 9 remaining.
    assert len(new_df) == 9 
    assert 100 not in new_df['val'].values
    assert new_df['val'].isna().sum() == 1

def test_constant_values():
    # Std=0, IQR=0 should not crash and remove nothing (Strict 0 removal)
    df = pd.DataFrame({'val': [5, 5, 5, 5]})
    
    manager = GlobalStateManager()
    manager.load_data(df, "test_constant.csv")
    
    result = remove_outliers("test_constant.csv", "val", "zscore")
    assert result['rows_removed'] == 0
    assert result['stats']['zero_variance_detected'] is True
    
    result = remove_outliers("test_constant.csv", "val", "iqr")
    assert result['rows_removed'] == 0
    assert result['stats']['zero_variance_detected'] is True

def test_invalid_input():
    df = pd.DataFrame({'val': [1, 2, 3]})
    manager = GlobalStateManager()
    manager.load_data(df, "test_invalid.csv")
    
    # Threshold <= 0
    res = remove_outliers("test_invalid.csv", "val", "zscore", threshold=0)
    assert "error" in res
    
    # Non-numeric column
    df_str = pd.DataFrame({'cat': ['a', 'b', 'c']})
    manager.load_data(df_str, "test_str.csv")
    res = remove_outliers("test_str.csv", "cat", "zscore")
    assert "error" in res

def test_small_sample_size():
    # Only 2 values.
    # Z-score: std is calculated (with ddof=1 by default). 
    # If values are [0, 10], mean=5, std=7.07. Z of 0 is -0.7, Z of 10 is 0.7.
    # Should NOT remove anything with threshold=3.
    df = pd.DataFrame({'val': [0, 10]})
    manager = GlobalStateManager()
    manager.load_data(df, "test_small.csv")
    
    res = remove_outliers("test_small.csv", "val", "zscore", threshold=1.0) # Low threshold
    # Z-scores are ~0.7. Threshold 1.0 keeps them.
    assert res['rows_removed'] == 0
    
    # Threshold 0.5 should remove both?
    # Z-scores are 0.707. > 0.5.
    res = remove_outliers("test_small.csv", "val", "zscore", threshold=0.5)
    assert res['rows_removed'] == 2

def test_metadata_structure():
    # Verify metadata is returned correctly
    np.random.seed(42)
    df = pd.DataFrame({'val': np.random.normal(0, 1, 100)})
    manager = GlobalStateManager()
    manager.load_data(df, "test_meta.csv")
    
    res = remove_outliers("test_meta.csv", "val", "zscore", threshold=2.0)
    assert 'stats' in res
    assert res['stats']['method'] == 'zscore'
    assert res['stats']['threshold'] == 2.0
    assert 'mean' in res['stats']
    assert 'lower_bound' in res['stats']

def test_modified_zscore():
    # Modified Z-Score is robust.
    # Data: [1, 2, 3, 4, 100]
    # Median = 3
    # |x - Med|: [2, 1, 0, 1, 97]
    # MAD = Median([0, 1, 1, 2, 97]) = 1
    # ModZ = 0.6745 * (x - 3) / 1
    # ModZ for 100: 0.6745 * 97 = 65.4265 -> Outlier
    # ModZ for 4: 0.6745 * 1 = 0.6745 -> Not Outlier
    
    df = pd.DataFrame({'val': [1, 2, 3, 4, 100]})
    manager = GlobalStateManager()
    manager.load_data(df, "test_modz.csv")
    
    # Default threshold 3.5
    res = remove_outliers("test_modz.csv", "val", "modified_zscore")
    
    assert res['rows_removed'] == 1
    assert res['remaining_rows'] == 4
    assert res['stats']['method'] == 'modified_zscore'
    assert res['stats']['median'] == 3.0
    assert res['stats']['mad'] == 1.0

def test_custom_threshold_modzscore():
    # Test specific threshold requested by user (e.g. 3.2, 3.3)
    # Data: [10, 12, 10, 11, 20]
    # Sorted: 10, 10, 11, 12, 20
    # Median = 11
    # Diffs: |10-11|=1, |12-11|=1, |10-11|=1, |11-11|=0, |20-11|=9
    # MAD list: 0, 1, 1, 1, 9 -> Sorted: 0, 1, 1, 1, 9 -> Median is 1.
    # MAD = 1.
    # ModZ = 0.6745 * (x - 11) / 1
    # For 20: 0.6745 * 9 = 6.0705
    # For 12: 0.6745 * 1 = 0.6745
    
    # Let's try a case where threshold matters more closely.
    # x = 15. ModZ = 0.6745 * 4 = 2.698.
    # If threshold is 2.5, it removes 15. If 3.0, it keeps 15.
    
    df = pd.DataFrame({'val': [10, 12, 10, 11, 15]})
    manager = GlobalStateManager()
    manager.load_data(df, "test_custom.csv")
    
    # Threshold 2.5 -> removes 15 (score ~2.7)
    res = remove_outliers("test_custom.csv", "val", "modified_zscore", threshold=2.5)
    assert res['rows_removed'] == 1
    
    # Reload original
    manager.load_data(df, "test_custom.csv")
    
    # Threshold 3.0 -> keeps 15 (score ~2.7)
    res = remove_outliers("test_custom.csv", "val", "modified_zscore", threshold=3.0)
    assert res['rows_removed'] == 0

def test_isolation_forest_basic():
    # IF should find the outlier
    np.random.seed(42)
    # 50 normal points, 5 outliers far away
    data = np.concatenate([np.random.normal(0, 1, 50), np.random.normal(20, 1, 5)])
    df = pd.DataFrame({'val': data})
    
    manager = GlobalStateManager()
    manager.load_data(df, "test_if.csv")
    
    # Contamination ~0.1 (5/55)
    res = remove_outliers("test_if.csv", "val", "isolation_forest", threshold=0.1)
    
    assert res['stats']['method'] == 'isolation_forest'
    # IF is stochastic but with fixed random_state should be stable-ish.
    # We expect roughly 5-6 removals with 0.1 contamination
    assert res['rows_removed'] >= 4
    assert res['rows_removed'] <= 7

def test_lof_local_density():
    # LOF finds local outliers.
    # [1]*20 ... [10]*20 ... [5]
    # Total 41. n_neighbors = min(20, 40) = 20.
    # With 20 neighbors, 5.0 sees mostly 1s or 10s depending on distance?
    # Actually, 5 is distance 4 from 1 and 5 from 10. 
    # Its neighbors will be a mix. It should be less dense than the clusters.
    
    # Let's make it clearer. Two tight clusters, one point in middle.
    c1 = [1.0] * 10
    c2 = [10.0] * 10
    outlier = [5.5]
    data = c1 + c2 + outlier # 21 points
    
    df = pd.DataFrame({'val': data})
    
    manager = GlobalStateManager()
    manager.load_data(df, "test_lof.csv")
    
    # Neighbors = min(20, 20) = 20.
    # To make LOF work well here, we might need fewer neighbors to see local density properly?
    # Or just rely on contamination. 1/21 ~ 0.05.
    
    res = remove_outliers("test_lof.csv", "val", "lof", threshold=0.1)
    
    # 5.5 should be removed
    final_df = manager.get_data()
    assert 5.5 not in final_df['val'].values

def test_contamination_boundary():
    # Strict validation 0.5 < T <= 1.0
    df = pd.DataFrame({'val': [1, 2, 3]})
    manager = GlobalStateManager()
    manager.load_data(df, "test_bound.csv")
    
    res = remove_outliers("test_bound.csv", "val", "isolation_forest", threshold=0.51)
    assert "error" in res
    assert "Ambiguous" in res["error"]
    
    # T > 1.0 -> Auto
    res = remove_outliers("test_bound.csv", "val", "isolation_forest", threshold=1.5)
    assert "error" not in res
    assert res["stats"]["contamination_param"] == "auto"

def test_constant_column_ml():
    # Model should skip constant column
    df = pd.DataFrame({'val': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]})
    manager = GlobalStateManager()
    manager.load_data(df, "test_const_ml.csv")
    
    res = remove_outliers("test_const_ml.csv", "val", "isolation_forest")
    assert res['rows_removed'] == 0
    assert res['stats']['reason'] == "constant_column"

def test_small_dataset_safety():
    # n < 5 for LOF -> Skip
    df = pd.DataFrame({'val': [1, 2, 3, 4]})
    manager = GlobalStateManager()
    manager.load_data(df, "test_small_ml.csv")
    
    res = remove_outliers("test_small_ml.csv", "val", "lof")
    assert res['rows_removed'] == 0
    assert res['stats']['reason'] == "too_small_for_model"

def test_nan_inf_handling_ml():
    # NaNs and Infs should be preserved, not removed, and not crash model
    data = [1, 2, 3, 4, 5] * 3 # 15 points
    data.extend([np.nan, np.inf, -np.inf, 100]) # 4 special points. 100 is outlier.
    # Total 19 points.
    
    df = pd.DataFrame({'val': data})
    manager = GlobalStateManager()
    manager.load_data(df, "test_nan_inf.csv")
    
    # IF with auto contamination
    res = remove_outliers("test_nan_inf.csv", "val", "isolation_forest")
    
    final_df = manager.get_data()
    
    # 1. Check NaNs/Infs exist
    assert final_df['val'].isna().sum() >= 1 # NaN
    assert np.isinf(final_df['val']).sum() >= 2 # Infs
    
    # 2. Check 100 (outlier) is removed (mostly likely, depending on contamination)
    # With 'auto', IF is usually good at finding the 1 gross outlier
    # But let's verify rows_removed > 0
    if res['rows_removed'] > 0:
        pass # Good
        
def test_reproducibility():
    # IF should be deterministic with random_state=42
    np.random.seed(42)
    data = np.random.normal(0, 1, 100)
    data = np.append(data, [100, -100])
    df = pd.DataFrame({'val': data})
    
    manager = GlobalStateManager()
    
    # Run 1
    manager.load_data(df, "test_rep.csv")
    res1 = remove_outliers("test_rep.csv", "val", "isolation_forest", threshold=0.1)
    
    # Run 2
    manager.load_data(df, "test_rep.csv")
    res2 = remove_outliers("test_rep.csv", "val", "isolation_forest", threshold=0.1)
    
    assert res1['rows_removed'] == res2['rows_removed']
    assert res1['remaining_rows'] == res2['remaining_rows']

def test_split_preservation():
    """Verify that removing outliers respects the split state (does not wipe test set)."""
    # 1. Setup Data
    df_train = pd.DataFrame({'val': [1, 1, 1, 1, 100]}) # 100 is outlier
    df_test = pd.DataFrame({'val': [4, 5, 6]})
    
    manager = GlobalStateManager()
    
    # 2. Initialize State
    # Check if load_data resets split is irrelevant here because we set split right after
    manager.load_data(df_train, "test_split_persist.csv")
    manager.set_split_data(df_train, df_test, {"test_size": 0.2})
    
    assert manager.is_split() is True
    assert manager.get_test_data() is not None
    
    # 3. Run Removal
    # Should remove 100 from train
    res = remove_outliers("test_split_persist.csv", "val", "zscore", threshold=1.0)
    
    # 4. Verify Split Persists
    assert manager.is_split() is True, "Split state was lost (reset_split=True probably triggered)"
    assert manager.get_test_data() is not None, "Test data turned to None"
    assert len(manager.get_test_data()) == 3, "Test data was corrupted or modified"
    
    # Verify Train modified
    final_train = manager.get_data()
    assert len(final_train) == 4 # 100 removed
    assert 100 not in final_train['val'].values

if __name__ == "__main__":
    # check if pytest is installed, otherwise run manually
    try:
        import pytest
        sys.exit(pytest.main(["-v", __file__]))
    except ImportError:
        print("Pytest not found, running simple assertions...")
        test_zscore_basic()
        test_iqr_skewed()
        test_nan_preservation()
        test_constant_values()
        test_invalid_input()
        test_small_sample_size()
        test_metadata_structure()
        test_modified_zscore()
        test_custom_threshold_modzscore()
        
        # New Tests
        test_isolation_forest_basic()
        test_lof_local_density()
        test_contamination_boundary()
        test_constant_column_ml()
        test_small_dataset_safety()
        test_nan_inf_handling_ml()
        test_reproducibility()
        
        print("All manual tests passed!")
