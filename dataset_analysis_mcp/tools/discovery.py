"""
Dataset Discovery Tools for MCP Server.

This module provides tools for listing available datasets and loading or
inspecting dataset metadata. It implements Phase 1 of the dataset analysis
workflow.

Functions:
    list_datasets: List all CSV/JSON files in the data directory.
    load_dataset_metadata: Load a dataset into global state and return metadata.
    peek_dataset_metadata: Inspect a dataset on disk without modifying state.
    load_dataset: Load a dataset from any path on the user's machine.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
from pydantic import BaseModel, Field

from dataset_analysis_mcp.config import DATA_DIR, SUPPORTED_LOAD_EXTENSIONS
from dataset_analysis_mcp.utils.state_manager import GlobalStateManager

logger = logging.getLogger(__name__)


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

    files: List[DatasetInfo] = []
    for f in os.listdir(DATA_DIR):
        if f.endswith((".csv", ".json")):
            path = os.path.join(DATA_DIR, f)
            files.append(
                DatasetInfo(
                    filename=f,
                    size_bytes=os.path.getsize(path),
                )
            )

    if not files:
        # Debug info for the user/LLM to see where we looked
        return [
            DatasetInfo(
                filename=f"[DEBUG] No files found in: {DATA_DIR}",
                size_bytes=0,
            )
        ]

    return files


def _read_dataset(path: str, filename: str) -> pd.DataFrame:
    """
    Internal helper to read a dataset from disk based on its extension.

    Supports: .csv, .json, .parquet, .xlsx
    """
    if filename.endswith(".csv"):
        return pd.read_csv(path)
    if filename.endswith(".json"):
        return pd.read_json(path)
    if filename.endswith(".parquet"):
        return pd.read_parquet(path)
    if filename.endswith(".xlsx"):
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file format: {Path(filename).suffix}")


def load_dataset_metadata(filename: str) -> DatasetMetadata:
    """
    Load a dataset into the server's memory and return its metadata.

    This function loads the entire dataset into memory and stores it in the
    GlobalStateManager singleton for subsequent operations. It also calculates
    and returns comprehensive metadata about the dataset.

    WARNING:
        This OVERWRITES the current in-memory dataset in GlobalStateManager.
        Any unsaved transformations will be lost. To inspect a file without
        modifying state, use peek_dataset_metadata() instead.

    Args:
        filename:
            Name of the file in the data directory (e.g., 'data.csv' or 'data.json').

    Returns:
        DatasetMetadata: Object containing:
            - filename: Name of the loaded file
            - columns: List of column names
            - dtypes: Dictionary mapping column names to data types
            - missing_percentages_sample: Dict of missing value percentages per column
            - estimated_row_count: Total number of rows
            - preview: First 5 rows as list of dictionaries
            - error: Error message if operation failed

    Note:
        Returns DatasetMetadata with error field populated if operation fails.
    """
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return DatasetMetadata(
            filename=filename,
            columns=[],
            dtypes={},
            missing_percentages_sample={},
            estimated_row_count=0,
            preview=[],
            error=f"Dataset {filename} not found."
        )

    try:
        # Load the FULL dataset into memory (destructive by design)
        # df = _read_dataset(path, filename) # code written by jaideep

        # Store in GlobalStateManager (this overwrites the current in-memory state)
        # Load the FULL dataset into memory as requested
        if filename.endswith(".csv"):
            df = pd.read_csv(path)
        elif filename.endswith(".json"):
            df = pd.read_json(path)
        else:
            return DatasetMetadata(
                filename=filename,
                columns=[],
                dtypes={},
                missing_percentages_sample={},
                estimated_row_count=0,
                preview=[],
                error="Unsupported file format"
            )
            
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
            preview=df.head(5).to_dict(orient="records"),
        )

    except Exception as e:
        return DatasetMetadata(
            filename=filename,
            columns=[],
            dtypes={},
            missing_percentages_sample={},
            estimated_row_count=0,
            preview=[],
            error=str(e)
        )


def peek_dataset_metadata(filename: str) -> DatasetMetadata:
    """
    Inspect a dataset on disk and return its metadata WITHOUT modifying
    the in-memory state managed by GlobalStateManager.

    Use this when you only want to check a file's schema or basic stats
    mid-pipeline without risking data loss.

    Args:
        filename:
            Name of the file in the data directory (e.g., 'data.csv' or 'data.json').

    Returns:
        DatasetMetadata: Same structure as load_dataset_metadata, but computed
        purely from the file on disk and without touching GlobalStateManager.
        Returns DatasetMetadata with error field populated if operation fails.

    Note:
        Returns DatasetMetadata with error field populated if operation fails.
    """
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return DatasetMetadata(
            filename=filename,
            columns=[],
            dtypes={},
            missing_percentages_sample={},
            estimated_row_count=0,
            preview=[],
            error=f"Dataset {filename} not found."
        )

    try:
        # Read from disk but DO NOT call GlobalStateManager
        df = _read_dataset(path, filename)

        missing_stats = df.isnull().mean().to_dict()

        return DatasetMetadata(
            filename=filename,
            columns=list(df.columns),
            dtypes={k: str(v) for k, v in df.dtypes.items()},
            missing_percentages_sample=missing_stats,
            estimated_row_count=len(df),
            preview=df.head(5).to_dict(orient="records"),
        )

    except Exception as e:
        return DatasetMetadata(
            filename=filename,
            columns=[],
            dtypes={},
            missing_percentages_sample={},
            estimated_row_count=0,
            preview=[],
            error=str(e)
        )


def load_dataset(path: str) -> DatasetMetadata:
    """
    Load a dataset from any absolute or relative path on the user's machine.

    Unlike load_dataset_metadata (which requires files in the data/ directory),
    this tool accepts any filesystem path. It resolves ~ home-directory
    shorthand, environment variables, and relative paths automatically.

    WARNING:
        This OVERWRITES the current in-memory dataset in GlobalStateManager.
        Any unsaved transformations will be lost.

    Args:
        path:
            File path to load. Accepts:
            - Absolute paths: /Users/me/data/sales.csv
            - Home shorthand: ~/Downloads/sales.csv
            - Relative paths: ../data/sales.csv (resolved from CWD)

    Returns:
        DatasetMetadata: Object containing filename, columns, dtypes,
        missing percentages, row count, and a 5-row preview.
        Returns DatasetMetadata with error field populated if operation fails.

    Note:
        Supported formats: .csv, .json, .parquet, .xlsx
    """
    # Resolve the path: expand ~ and make absolute
    resolved = Path(path).expanduser().resolve()
    filename = resolved.name

    logger.info("load_dataset: resolved '%s' -> '%s'", path, resolved)

    # Validate file existence
    if not resolved.is_file():
        return DatasetMetadata(
            filename=filename,
            columns=[],
            dtypes={},
            missing_percentages_sample={},
            estimated_row_count=0,
            preview=[],
            error=f"File not found: {resolved}"
        )

    # Validate supported format
    ext = resolved.suffix.lower()
    if ext not in SUPPORTED_LOAD_EXTENSIONS:
        return DatasetMetadata(
            filename=filename,
            columns=[],
            dtypes={},
            missing_percentages_sample={},
            estimated_row_count=0,
            preview=[],
            error=(
                f"Unsupported file format '{ext}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_LOAD_EXTENSIONS))}"
            )
        )

    try:
        df = _read_dataset(str(resolved), filename)

        # Store in GlobalStateManager
        manager = GlobalStateManager()
        manager.load_data(df, filename)

        missing_stats = df.isnull().mean().to_dict()

        return DatasetMetadata(
            filename=filename,
            columns=list(df.columns),
            dtypes={k: str(v) for k, v in df.dtypes.items()},
            missing_percentages_sample=missing_stats,
            estimated_row_count=len(df),
            preview=df.head(5).to_dict(orient="records"),
        )

    except PermissionError:
        return DatasetMetadata(
            filename=filename,
            columns=[],
            dtypes={},
            missing_percentages_sample={},
            estimated_row_count=0,
            preview=[],
            error=f"Permission denied: {resolved}"
        )

    except Exception as e:
        return DatasetMetadata(
            filename=filename,
            columns=[],
            dtypes={},
            missing_percentages_sample={},
            estimated_row_count=0,
            preview=[],
            error=str(e)
        )