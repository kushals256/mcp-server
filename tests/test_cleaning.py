import sys
import os
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.cleaning import drop_duplicate_rows, DropDuplicateRowsRequest
from utils.state_manager import GlobalStateManager

# Mock dependencies
sys.modules["utils.state_manager"] = MagicMock()
sys.modules["tools.discovery"] = MagicMock()

@patch("tools.cleaning.GlobalStateManager")
@patch("tools.cleaning.load_dataset_metadata")
@patch("tools.cleaning.DATA_DIR", "/tmp") # Mock DATA_DIR
def test_drop_duplicate_rows(mock_load_metadata, mock_state_manager_cls):
    # Setup mock manager
    mock_manager = MagicMock()
    mock_state_manager_cls.return_value = mock_manager
    mock_manager.get_dataset_name.return_value = "test.csv"
    
    # Create test data with duplicates
    df = pd.DataFrame({
        'A': [1, 1, 2, 3],
        'B': ['x', 'x', 'y', 'z']
    })
    mock_manager.get_data.return_value = df
    
    # Execute tool
    request = DropDuplicateRowsRequest(
        dataset_name="test.csv",
        subset_columns=None,
        keep="first"
    )
    
    # Mock to_csv so we don't actually write to /tmp in unit test (or we can let it write to a temp dir)
    # But for safety, let's just inspect the dataframe passed to load_data
    
    with patch("pandas.DataFrame.to_csv") as mock_to_csv:
        result = drop_duplicate_rows(request)
        
        # Verify result
        assert result["rows_removed"] == 1
        assert result["remaining_rows"] == 3
        
        # Verify manager called with cleaned df
        args, _ = mock_manager.load_data.call_args
        cleaned_df = args[0]
        assert len(cleaned_df) == 3
        assert cleaned_df.duplicated().sum() == 0

if __name__ == "__main__":
    try:
        import pytest
        sys.exit(pytest.main(["-v", __file__]))
    except ImportError:
        print("pytest not found")
