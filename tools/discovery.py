import os
import pandas as pd
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from utils.state_manager import GlobalStateManager

# Resolve DATA_DIR relative to the project root (assuming tools/ is one level deep)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

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
    Returns a list of DatasetInfo models.
    """
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
    Args:
        filename: Name of the file in the data directory.
    Returns:
        DatasetMetadata model including columns, types, missing values, and row count.
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
