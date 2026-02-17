import pandas as pd
import os
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field
from utils.state_manager import GlobalStateManager
from tools.discovery import load_dataset_metadata, DATA_DIR

class DropDuplicateRowsRequest(BaseModel):
    dataset_name: str = Field(..., description="Name of the dataset file (e.g. 'test.csv').")
    subset_columns: Optional[List[str]] = Field(None, description="List of columns to check for duplicates. If None, checks all columns.")
    keep: str = Field("first", description="'first', 'last', or 'none'.")

def drop_duplicate_rows(request: DropDuplicateRowsRequest) -> Dict[str, int]:
    """
    Remove duplicate rows from the dataset and SAVE the changes to the file.
    
    Args:
        request: DropDuplicateRowsRequest containing dataset_name, subset_columns, and keep strategy.
    """
    dataset_name = request.dataset_name
    subset_columns = request.subset_columns
    keep = request.keep
    manager = GlobalStateManager()
    
    # 1. Ensure Data is Loaded
    # Even if loaded, we want to ensure we are operating on the latest version if multiple tools are used.
    # However, for consistency with the state manager pattern, we check the name.
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
        
        # 3. SAVE to Disk (Overwrite original file)
        # Construct the full path based on DATA_DIR imported from discovery
        file_path = os.path.join(DATA_DIR, dataset_name)
        
        if dataset_name.lower().endswith(".csv"):
            df_cleaned.to_csv(file_path, index=False)
        elif dataset_name.lower().endswith(".json"):
            df_cleaned.to_json(file_path, orient="records", indent=4)
        else:
            raise ValueError("Unsupported file format for saving. Only CSV and JSON supported.")

        # 4. Update State (Memory)
        manager.load_data(df_cleaned, dataset_name)
        
        rows_dropped = initial_rows - len(df_cleaned)
        remaining_rows = len(df_cleaned)
        
        manager.log_action("drop_duplicate_rows", {
            "subset_columns": subset_columns, 
            "keep": keep,
            "rows_removed": rows_dropped,
            "save_to_disk": True
        })
        
        return {
            "rows_removed": rows_dropped,
            "remaining_rows": remaining_rows
        }

    except Exception as e:
        raise RuntimeError(f"Error dropping duplicates or saving file: {str(e)}")