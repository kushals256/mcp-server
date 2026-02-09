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
        print("All manual tests passed!")
