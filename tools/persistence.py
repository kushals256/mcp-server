"""
Dataset Persistence Tools for MCP Server.

This module provides tools for saving processed datasets and exporting pipeline
configurations. It implements Phase 2 of the dataset analysis workflow.

Functions:
    save_processed_dataset: Save the current in-memory dataset to disk
    export_pipeline_config: Export the transformation pipeline history
"""

import os
import pandas as pd
import json
import yaml
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from config import DATA_DIR, SUPPORTED_DATASET_FORMATS, SUPPORTED_PIPELINE_FORMATS
from utils.state_manager import GlobalStateManager


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
    
    This function saves the dataset currently stored in GlobalStateManager to the
    specified file format in the DATA_DIR directory.
    
    Args:
        request: SaveDatasetRequest containing:
            - format: Output format ('csv', 'json', or 'parquet')
            - path: Target filename (will be saved in DATA_DIR)
    
    Returns:
        OperationResult: Object containing:
            - success: Whether the save operation succeeded
            - message: Status message or error description
            - path: Full path where file was saved (empty on failure)
    
    Note:
        The parquet format requires pyarrow or fastparquet to be installed.
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
    
    This function saves the sequence of operations (pipeline steps) that have been
    performed on the dataset, allowing for reproducibility and documentation.
    
    Args:
        request: SavePipelineRequest containing:
            - pipeline_name: Base name for the config file (without extension)
            - format: Output format ('json' or 'yaml', default: 'json')
    
    Returns:
        OperationResult: Object containing:
            - success: Whether the export succeeded
            - message: Status message or error description
            - path: Full path where config was saved (empty on failure)
    
    Note:
        The pipeline history is managed by GlobalStateManager and includes
        all tool calls with their parameters in chronological order.
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
