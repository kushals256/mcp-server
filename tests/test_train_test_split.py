import sys
import os
import pandas as pd
import numpy as np
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.train_test_split import train_test_split
from utils.state_manager import GlobalStateManager

@pytest.fixture
def manager():
    mgr = GlobalStateManager()
    mgr.clear_state()
    return mgr

def test_basic_split(manager):
    # Setup 100-row dataset
    df = pd.DataFrame({'a': range(100), 'b': range(100)})
    manager.load_data(df, "test.csv")
    
    # Split
    result = train_test_split(test_size=0.2, random_state=42)
    
    # Check result metadata
    assert result['train_rows'] == 80
    assert result['test_rows'] == 20
    assert result['test_size'] == 0.2
    
    # Check Manager State
    train_df = manager.get_data()
    test_df = manager.get_test_data()
    
    # Verify rows
    assert len(train_df) == 80
    assert len(test_df) == 20
    assert len(train_df) + len(test_df) == 100
    
    # Verify index reset (should be 0..79 and 0..19)
    assert train_df.index[0] == 0
    assert train_df.index[-1] == 79
    assert test_df.index[0] == 0
    assert test_df.index[-1] == 19

def test_double_split_prevention(manager):
    df = pd.DataFrame({'a': range(50)})
    manager.load_data(df, "test.csv")
    
    # First split - OK
    train_test_split(test_size=0.2)
    
    # Second split - Should Fail
    result = train_test_split(test_size=0.2)
    assert "error" in result
    assert "already split" in result['error']
    
    # Reload -> Split again -> OK
    manager.load_data(df, "test.csv")
    result = train_test_split(test_size=0.2)
    assert "train_rows" in result

def test_stratified_split(manager):
    # Create unbalanced dataset: 90 'A', 10 'B'
    data = {'target': ['A']*90 + ['B']*10, 'val': range(100)}
    df = pd.DataFrame(data)
    manager.load_data(df, "strat.csv")
    
    # Split with stratification
    result = train_test_split(test_size=0.2, stratify='target', random_state=42)
    
    train_df = manager.get_data()
    test_df = manager.get_test_data()
    
    # Verify proportions in Test Set (should be approx 2 'B's out of 20)
    # 10 'B's total. 20% of 10 is 2.
    b_counts_test = test_df['target'].value_counts().get('B', 0)
    assert b_counts_test == 2

def test_immutability(manager):
    df = pd.DataFrame({'a': range(10)})
    manager.load_data(df, "test.csv")
    
    train_test_split(test_size=0.5)
    
    # Get test data and modify it
    test_df_copy = manager.get_test_data()
    test_df_copy.loc[0, 'a'] = 999
    
    # Verify stored test data is detecting NO change
    test_df_stored = manager.get_test_data()
    assert test_df_stored.loc[0, 'a'] != 999

def test_small_dataset(manager):
    # 5 rows. 20% test -> 1 row.
    df = pd.DataFrame({'a': range(5)})
    manager.load_data(df, "small.csv")
    
    result = train_test_split(test_size=0.2)
    assert result['train_rows'] == 4
    assert result['test_rows'] == 1

def test_invalid_inputs(manager):
    df = pd.DataFrame({'a': range(10)})
    manager.load_data(df, "test.csv")
    
    # Invalid Test Size
    res1 = train_test_split(test_size=1.5)
    assert "error" in res1
    
    # Invalid Stratify Column
    res2 = train_test_split(test_size=0.2, stratify='non_existent')
    assert "error" in res2

def test_stratification_failure(manager):
    # Class with too few samples (1 sample)
    data = {'target': ['A'] + ['B']*10, 'val': range(11)}
    df = pd.DataFrame(data)
    manager.load_data(df, "strat_fail.csv")
    
    result = train_test_split(test_size=0.2, stratify='target')
    assert "error" in result
    assert "Stratification failed" in result['error']

def test_deep_immutability(manager):
    df = pd.DataFrame({'a': range(10)})
    manager.load_data(df, "test.csv")
    
    train_test_split(test_size=0.5)
    
    # Get test data and modify it deeply
    test_df_copy = manager.get_test_data()
    test_df_copy.iloc[0, 0] = 999
    
    # Verify stored test data is detecting NO change
    test_df_stored = manager.get_test_data()
    assert test_df_stored.iloc[0, 0] != 999

def test_shuffle_false(manager):
    # Create ordered data: 0..9
    df = pd.DataFrame({'val': range(10)})
    manager.load_data(df, "ordered.csv")
    
    # Split without shuffle (first 80% train, last 20% test)
    result = train_test_split(test_size=0.2, shuffle=False)
    
    train_df = manager.get_data()
    test_df = manager.get_test_data()
    
    # Train should be 0..7, Test should be 8..9
    assert list(train_df['val']) == list(range(8))
    assert list(test_df['val']) == list(range(8, 10))

def test_stratify_nan(manager):
    # create data with NaN in stratify col
    data = {'target': ['A']*5 + ['B']*5 + [np.nan], 'val': range(11)}
    df = pd.DataFrame(data)
    manager.load_data(df, "strat_nan.csv")
    
    result = train_test_split(test_size=0.2, stratify='target')
    assert "error" in result
    assert "contains NaN values" in result['error']

if __name__ == "__main__":
    try:
        import pytest
        sys.exit(pytest.main(["-v", __file__]))
    except ImportError:
        print("Pytest not found.")
