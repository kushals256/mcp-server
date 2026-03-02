import sys
import os
import pandas as pd
import pytest
from unittest.mock import MagicMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_analysis_mcp.tools.save_dataset import save_processed_dataset, SaveDatasetRequest
from dataset_analysis_mcp.utils.state_manager import GlobalStateManager
from dataset_analysis_mcp.tools.train_test_split import train_test_split

@pytest.fixture
def manager():
    mgr = GlobalStateManager()
    mgr.clear_state()
    return mgr

def test_save_both_splits(manager, tmp_path):
    # Setup split state
    df = pd.DataFrame({'val': range(10)})
    manager.load_data(df, "data.csv")
    train_test_split(test_size=0.2)
    
    # Define paths
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    
    # 1. Save Training Set
    req_train = SaveDatasetRequest(format="csv", path=str(train_path), split_type="train")
    res_train = save_processed_dataset(req_train)
    assert res_train.success
    assert os.path.exists(train_path)
    
    # Verify content (should be 8 rows)
    saved_train = pd.read_csv(train_path)
    assert len(saved_train) == 8
    
    # 2. Save Test Set
    req_test = SaveDatasetRequest(format="csv", path=str(test_path), split_type="test")
    res_test = save_processed_dataset(req_test)
    assert res_test.success
    assert os.path.exists(test_path)
    
    # Verify content (should be 2 rows)
    saved_test = pd.read_csv(test_path)
    assert len(saved_test) == 2

def test_save_test_no_split(manager):
    # Load data but DON'T split
    df = pd.DataFrame({'val': range(10)})
    manager.load_data(df, "data.csv")
    
    # Try to save test set -> Should fail gracefully
    req = SaveDatasetRequest(format="csv", path="dummy.csv", split_type="test")
    res = save_processed_dataset(req)
    
    assert not res.success
    assert "No test dataset loaded" in res.message or "No None dataset" in res.message

if __name__ == "__main__":
    try:
        import pytest
        sys.exit(pytest.main(["-v", __file__]))
    except ImportError:
        print("Pytest not found.")
