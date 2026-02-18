import pytest
import pandas as pd
import numpy as np
import os
from utils.state_manager import GlobalStateManager
from tools.train_test_split import train_test_split
from tools.remove_features import remove_features
from tools.extract_features import extract_features
from tools.reduce_features import reduce_features

@pytest.fixture
def manager():
    mgr = GlobalStateManager()
    mgr.initialize()
    return mgr

@pytest.fixture
def sample_data(manager):
    # Create dataset with specific properties for testing
    np.random.seed(42)
    rows = 100
    df = pd.DataFrame({
        'A': np.random.normal(0, 1, rows), # Normal
        'B': np.random.normal(0, 1, rows), # Normal
        'C': np.random.choice(['a', 'b', 'c'], rows), # Cat
        'Constant': np.zeros(rows), # Low variance
        'Correlated': np.zeros(rows), # Will overwrite
        'Target': np.random.choice([0, 1], rows) # Binary target
    })
    # Make 'Correlated' highly correlated with 'A'
    df['Correlated'] = df['A'] * 0.99 + np.random.normal(0, 0.01, rows)
    
    manager.load_data(df, "test_fe.csv")
    return manager

# =========================================================================
# 1. LEAKAGE & STATE CONSISTENCY TESTS
# =========================================================================

def test_split_consistency_remove(manager, sample_data):
    """Verify changes apply to BOTH train and test after split."""
    # Split first
    train_test_split(test_size=0.2, random_state=42)
    
    # Remove 'Constant' (stateless/stateful check)
    res = remove_features(method="variance_threshold", threshold=0.0)
    assert res['success'] is True
    
    train_df = manager.get_data()
    test_df = manager.get_test_data()
    
    # Check consistency
    assert 'Constant' not in train_df.columns
    assert 'Constant' not in test_df.columns
    assert list(train_df.columns) == list(test_df.columns)
    
    # Check persistence
    transformers = manager.list_transformers()
    assert any("variance_threshold" in name for name in transformers)

def test_target_protection_reduction(manager, sample_data):
    """Verify target is preservered even after PCA."""
    train_test_split(test_size=0.2)
    
    res = reduce_features(method="pca", n_components=2, target_col="Target")
    assert res['success'] is True
    
    train_df = manager.get_data()
    test_df = manager.get_test_data()
    
    # Target should exist
    assert "Target" in train_df.columns
    assert "Target" in test_df.columns
    
    # Feature columns should be replaced by pca_1, pca_2
    assert "pca_1" in train_df.columns
    assert "A" not in train_df.columns

# =========================================================================
# 2. STRICT VALIDATION TESTS
# =========================================================================

def test_nan_policy_pca(manager):
    """PCA should CRASH if NaNs present."""
    df = pd.DataFrame({
        'A': [1.0, 2.0, np.nan, 4.0],
        'B': [1.0, 2.0, 3.0, 4.0]
    })
    manager.load_data(df, "nan_test.csv")
    
    res = reduce_features(method="pca", n_components=1)
    assert "error" in res
    assert "Dataset contains NaNs" in res["error"] # Strict error check

def test_zero_feature_guard(manager):
    """Should error if all features removed."""
    df = pd.DataFrame({'A': [1,1,1], 'B': [1,1,1]}) # All constant
    manager.load_data(df, "const_test.csv")
    
    res = remove_features(method="variance_threshold")
    assert "error" in res
    # Sklearn raises ValueError if no features meet threshold
    assert "ZERO features" in res["error"] or "No feature in X meets" in res["error"]

# =========================================================================
# 3. DETERMINISM & TIE-BREAKING
# =========================================================================

def test_correlation_tie_breaker(manager):
    """Ensure alphabetical tie breaking works."""
    df = pd.DataFrame({
        'Z_Feat': [1, 2, 3, 4, 5],
        'A_Feat': [1, 2, 3, 4, 5] # Perfectly correlated
    })
    manager.load_data(df, "corr.csv")
    
    # A_Feat and Z_Feat are 1.0 correlated.
    # Sorted pair: [A_Feat, Z_Feat]. Drop index 1 -> Z_Feat. Keeping A_Feat.
    # Wait, my logic was: pair = sorted([col, row]); to_drop.add(pair[1])
    # So it drops the alphabetically LATER one.
    
    res = remove_features(method="correlation_threshold", threshold=0.99)
    assert res['success'] is True
    
    df_res = manager.get_data()
    assert 'A_Feat' in df_res.columns
    assert 'Z_Feat' not in df_res.columns # Should have dropped Z

# =========================================================================
# 4. EXTRACTION SAFETY
# =========================================================================

def test_math_guards(manager):
    """Log of negative should handle gracefully (NaN) or error? 
    Implementation implements NaN masking."""
    df = pd.DataFrame({'A': [10, 0, -5]})
    manager.load_data(df, "math.csv")
    
    res = extract_features(method="math", columns=['A'], operation="log")
    df_res = manager.get_data()
    
    # 0 -> -inf in numpy default, but we guarded?
    # extract_features logic: np.log(series.replace(0, np.nan).where(series > 0))
    # so 0 -> NaN, -5 -> NaN.
    
    assert np.isnan(df_res['A_log'].iloc[1]) # 0
    assert np.isnan(df_res['A_log'].iloc[2]) # -5
    assert not np.isnan(df_res['A_log'].iloc[0]) # 10

def test_naming_collision(manager):
    """Should error if new name exists."""
    df = pd.DataFrame({'A': [1], 'A_log': [1]})
    manager.load_data(df, "coll.csv")
    
    res = extract_features(method="math", columns=['A'], operation="log")
    assert "error" in res
    assert "already exists" in res["error"]
