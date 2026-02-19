import pandas as pd
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field
from utils.state_manager import GlobalStateManager
from tools.discovery import load_dataset_metadata

class DropDuplicateRowsRequest(BaseModel):
    dataset_name: str = Field(..., description="Name of the dataset file (e.g. 'test.csv').")
    subset_columns: Optional[List[str]] = Field(None, description="List of columns to check for duplicates. If None, checks all columns.")
    keep: str = Field("first", description="'first', 'last', or 'none'.")

def drop_duplicate_rows(request: DropDuplicateRowsRequest) -> Dict[str, int]:
    """
    Remove duplicate rows from the dataset.
    
    This operation only updates the in-memory state. To persist changes to disk,
    use the 'save_dataset' tool explicitly.
    
    Args:
        request: DropDuplicateRowsRequest containing dataset_name, subset_columns, and keep strategy.
    """
    dataset_name = request.dataset_name
    subset_columns = request.subset_columns
    keep = request.keep
    manager = GlobalStateManager()
    
    # 1. Ensure Data is Loaded
    if manager.get_dataset_name() != dataset_name:
        try:
            load_dataset_metadata(dataset_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset: {str(e)}")
            
    df = manager.get_data()
    if df is None:
        raise RuntimeError("Dataset loaded but DataFrame is None.")

    initial_rows = len(df)
    
    # 2. Handle Logic
    if subset_columns is not None and len(subset_columns) == 0:
        subset_columns = None
        
    keep_arg: Union[str, bool] = False if keep == 'none' else keep
    
    try:
        # Create cleaned dataframe
        df_cleaned = df.drop_duplicates(subset=subset_columns, keep=keep_arg)
        
        # 3. Update State (Memory ONLY)
        # Use update_data to avoid duplicate 'load_data' logs
        if hasattr(manager, 'update_data'):
            manager.update_data(df_cleaned)
        else:
            manager.load_data(df_cleaned, dataset_name)
        
        rows_dropped = initial_rows - len(df_cleaned)
        remaining_rows = len(df_cleaned)
        
        manager.log_action("drop_duplicate_rows", {
            "subset_columns": subset_columns, 
            "keep": keep,
            "rows_removed": rows_dropped,
            "save_to_disk": False  # Explicitly logging that we didn't save
        })
        
        return {
            "rows_removed": rows_dropped,
            "remaining_rows": remaining_rows
        }

    except Exception as e:
        raise RuntimeError(f"Error dropping duplicates: {str(e)}")