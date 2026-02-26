import pandas as pd
import os
from typing import Dict, List, Any
from utils.state_manager import GlobalStateManager
from tools.discovery import load_dataset_metadata, DATA_DIR

def drop_columns(dataset_name: str, columns: List[str]) -> Dict[str, Any]:
    """
    Drop specific columns from the dataset and SAVE the changes to the file.
    
    Args:
        dataset_name: Name of the dataset file.
        columns: List of column names to remove.
        
    Returns:
        Dictionary containing the list of successfully dropped columns.
    """
    manager = GlobalStateManager()
    
    # 1. Ensure Data is Loaded
    if manager.get_dataset_name() != dataset_name:
        return {
            "dropped_columns": [],
            "error": f"Dataset '{dataset_name}' is not currently loaded. "
                     "Call load_dataset_metadata() explicitly to load it first."
        }
            
    df = manager.get_data()
    if df is None:
        return {"dropped_columns": [], "error": "Dataset loaded but DataFrame is None."}
    
    # 2. Validate Columns
    existing_cols_to_drop = [col for col in columns if col in df.columns]
    
    if not existing_cols_to_drop:
        return {"dropped_columns": [], "message": "No matching columns found to drop."}
    
    try:
        # 3. Drop Columns
        df_new = df.drop(columns=existing_cols_to_drop)
        
        # 4. SAVE to Disk (Overwrite original file)
        file_path = os.path.join(DATA_DIR, dataset_name)
        
        if dataset_name.lower().endswith(".csv"):
            df_new.to_csv(file_path, index=False)
        elif dataset_name.lower().endswith(".json"):
            df_new.to_json(file_path, orient="records", indent=4)
        else:
            # If format is weird, don't save but update memory (safer)
            pass

        # 5. Update Global State
        manager.update_data(df_new, tool_name="select_features")
        manager.log_action("drop_columns", {
            "columns": existing_cols_to_drop,
            "save_to_disk": True
        })
        
        return {
            "dropped_columns": existing_cols_to_drop
        }

    except Exception as e:
        return {"dropped_columns": [], "error": str(e)}