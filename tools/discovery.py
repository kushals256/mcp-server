"""
Dataset Discovery Tools for MCP Server.

This module provides tools for listing available datasets and loading dataset metadata
into the server's global state. It implements Phase 1 of the dataset analysis workflow.

Functions:
    list_datasets: List all CSV/JSON files in the data directory
    load_dataset_metadata: Load a dataset and return its metadata
"""

import os
import pandas as pd
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from config import DATA_DIR
from utils.state_manager import GlobalStateManager


class DatasetInfo(BaseModel):
    filename: str = Field(..., description="Name of the file including extension")
    size_bytes: int = Field(..., description="Size of the file in bytes")

class DatasetMetadata(BaseModel):
    filename: str
    columns: List[str]
    dtypes: Dict[str, str]
    missing_percentages_sample: Dict[str, float]
    estimated_row_count: int
    preview: List[Dict[str, Any]]
    error: Optional[str] = None

def list_datasets() -> List[DatasetInfo]:
    """
    List all available CSV/JSON files in the data directory.
    
    This is typically the first tool called in the dataset analysis workflow.
    It scans the configured DATA_DIR for supported file formats.
    
    Returns:
        List[DatasetInfo]: List of DatasetInfo objects containing filename and size.
                          Returns a debug message if no files are found.
    
    Note:
        Supported formats: .csv, .json
    """
    # OPTIONAL: Log this action if you want "Listed Datasets" to appear in your report
    # manager = GlobalStateManager()
    # manager.log_action("list_datasets", {})

    if not os.path.exists(DATA_DIR):
        return []
    
    files = []
    for f in os.listdir(DATA_DIR):
        if f.endswith((".csv", ".json")):
            path = os.path.join(DATA_DIR, f)
            files.append(DatasetInfo(
                filename=f,
                size_bytes=os.path.getsize(path)
            ))
            
    if not files:
        # Debug info for the user/LLM to see where we looked
        return [DatasetInfo(filename=f"[DEBUG] No files found in: {DATA_DIR}", size_bytes=0)]
        
    return files

def load_dataset_metadata(filename: str) -> DatasetMetadata:
    """
    Load a dataset into the server's memory and return its metadata.
    
    This function loads the entire dataset into memory and stores it in the
    GlobalStateManager singleton for subsequent operations. It also calculates
    and returns comprehensive metadata about the dataset.
    
    Args:
        filename: Name of the file in the data directory (e.g., 'data.csv' or 'data.json').
    
    Returns:
        DatasetMetadata: Object containing:
            - filename: Name of the loaded file
            - columns: List of column names
            - dtypes: Dictionary mapping column names to data types
            - missing_percentages_sample: Dictionary of missing value percentages per column
            - estimated_row_count: Total number of rows
            - preview: First 5 rows as list of dictionaries
    
    Raises:
        FileNotFoundError: If the specified file doesn't exist in DATA_DIR.
        ValueError: If the file format is not supported.
        Exception: For other errors during file reading or processing.
    """
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset {filename} not found.")
    
    try:
        # Load the FULL dataset into memory as requested
        if filename.endswith(".csv"):
            df = pd.read_csv(path)
        elif filename.endswith(".json"):
            df = pd.read_json(path)
        else:
             raise ValueError("Unsupported file format")
            
        # Store in GlobalStateManager
        # Note: We use load_data() here because this is a FRESH load from disk.
        # This correctly logs "load_data" to the history.
        manager = GlobalStateManager()
        manager.load_data(df, filename)

        # Calculate stats from the loaded dataframe
        missing_stats = df.isnull().mean().to_dict()
        
        return DatasetMetadata(
            filename=filename,
            columns=list(df.columns),
            dtypes={k: str(v) for k, v in df.dtypes.items()},
            missing_percentages_sample=missing_stats,
            estimated_row_count=len(df),
            preview=df.head(5).to_dict(orient="records")
        )

    except Exception as e:
        raise e