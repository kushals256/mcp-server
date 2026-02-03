import os
import pandas as pd
import json
import yaml
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from utils.state_manager import GlobalStateManager

# Resolve DATA_DIR relative to the project root (assuming tools/ is one level deep)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

class SaveDatasetRequest(BaseModel):
    format: str = Field(..., description="Format to save: 'csv', 'json', or 'parquet'")
    path: str = Field(..., description="Target filename or path")

class SavePipelineRequest(BaseModel):
    pipeline_name: str = Field(..., description="Name of the pipeline (will be saved as .json or .yaml)")
    format: str = Field("json", description="Format to save: 'json' or 'yaml'")

class OperationResult(BaseModel):
    success: bool
    message: str
    path: str

def save_processed_dataset(request: SaveDatasetRequest) -> OperationResult:
    """
    Save the current in-memory processed dataset to disk.
    Args:
        request: SaveDatasetRequest containing format and path.
    """
    manager = GlobalStateManager()
    df = manager.get_data()
    
    if df is None:
        return OperationResult(success=False, message="No dataset loaded in memory. Please load a dataset first.", path="")

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    path = os.path.join(DATA_DIR, request.path)
    
    try:
        if request.format == "csv":
            df.to_csv(path, index=False)
        elif request.format == "json":
            df.to_json(path, orient="records", indent=2) 
        elif request.format == "parquet":
             # Optional: might need pyarrow or fastparquet dependency if not present
             # We assume pandas logic handles it or raises helpful error
             df.to_parquet(path)
        else:
            return OperationResult(success=False, message="Unsupported format. Use 'csv', 'json', or 'parquet'", path="")
            
        return OperationResult(success=True, message="Dataset saved successfully", path=path)
    except Exception as e:
        return OperationResult(success=False, message=str(e), path="")

def export_pipeline_config(request: SavePipelineRequest) -> OperationResult:
    """
    Export the current transformation pipeline configuration from history.
    Args:
        request: SavePipelineRequest containing name and format.
    """
    manager = GlobalStateManager()
    steps = manager.get_history()
    
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    filename = f"{request.pipeline_name}.{request.format}"
    path = os.path.join(DATA_DIR, filename)
    
    try:
        if request.format == "json":
            with open(path, 'w') as f:
                json.dump(steps, f, indent=2)
        elif request.format == "yaml":
            with open(path, 'w') as f:
                yaml.dump(steps, f)
        else:
             return OperationResult(success=False, message="Unsupported format. Use 'json' or 'yaml'", path="")
             
        return OperationResult(success=True, message="Pipeline config exported successfully", path=path)
    except Exception as e:
         return OperationResult(success=False, message=str(e), path="")
