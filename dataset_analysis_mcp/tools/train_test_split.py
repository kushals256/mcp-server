"""
Train/Test Split Tool for MCP Server.

This module provides the train_test_split tool for dataset partitioning.
It implements Phase 4 (Transformation) of the dataset analysis workflow.

Functions:
    train_test_split: Split the dataset into training and testing sets
"""

import pandas as pd
from typing import Dict, Any, Optional
from sklearn.model_selection import train_test_split as sklearn_split

from dataset_analysis_mcp.utils.state_manager import GlobalStateManager

def train_test_split(
    test_size: float,
    random_state: Optional[int] = None,
    stratify: Optional[str] = None,
    shuffle: bool = True
) -> Dict[str, Any]:
    """
    Split the current dataset into training and testing sets.
    
    This function:
    1. Splits the data using sklearn.model_selection.train_test_split.
    2. Updates GlobalStateManager to set the TRAINING set as the active dataset.
    3. Stores the TEST set securely in the manager for future evaluation.
    4. Resets indices for both sets to ensure clean subsequent operations.

    Args:
        test_size: Proportion of the dataset to include in the test split (0.0 to 1.0).
        random_state: Controls the shuffling applied to the data before applying the split.
                      Pass an int for reproducible output across multiple function calls.
        stratify: Column name to use for stratified sampling (e.g., target variable).
                  Ensure the column exists and has sufficient class counts.
        shuffle: Whether to shuffle the data before splitting. Default is True.
                 Set to False for time-series or ordered data.

    Returns:
        Dictionary containing split statistics:
        {
            "original_rows": int,
            "train_rows": int,
            "test_rows": int,
            "test_size": float,
            "random_state": int or None,
            "stratified_by": str or None
        }

    Raises:
        ValueError: If input parameters are invalid or dataset is too small.
    """
    # 1. Validation: Test Size
    if not (0.0 < test_size < 1.0):
        return {"error": "test_size must be strictly between 0.0 and 1.0"}

    manager = GlobalStateManager()
    df = manager.get_data()
    
    # 2. Validation: Dataset Loaded
    if df is None:
        return {"error": "No dataset loaded. Please load a dataset first."}
        
    original_rows = len(df)
    
    # 3. Validation: Dataset Size
    if original_rows < 2:
        return {"error": "Dataset has too few rows to split (minimum 2)."}
        
    if original_rows < 10:
        # Warning via return message (since we can't easily warn otherwise in this pattern)
        # We'll proceed but it's risky.
        pass

    # 4. Validation: Stratify Column
    stratify_col = None
    if stratify:
        if stratify not in df.columns:
            return {"error": f"Stratification column '{stratify}' not found in dataset."}
        
        # Check for NaNs in stratification column (sklearn error source)
        if df[stratify].isna().any():
             return {"error": f"Stratification column '{stratify}' contains NaN values. Please handle missing values before splitting."}
             
        stratify_col = df[stratify]

    try:
        # 5. Perform Split
        train_df, test_df = sklearn_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_col,
            shuffle=shuffle
        )
        
        # 6. Reset Indices (Important for ML pipelines to avoid index confusion)
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        
        # 7. Internal Consistency Check
        if len(train_df) + len(test_df) != original_rows:
            return {"error": "Critical Error: Split resulted in data loss. Operation aborted."}
        
        # 8. Update State
        # This will raise ValueError if already split, which we catch
        metadata = {
            "test_size": test_size,
            "random_state": random_state,
            "stratified_by": stratify,
            "shuffle": shuffle,
            "original_rows": original_rows,
            "train_rows": len(train_df),
            "test_rows": len(test_df)
        }
        
        if original_rows < 10:
            metadata["warning"] = "Dataset has fewer than 10 rows. Split may be unstable."
        
        manager.set_split_data(train_df, test_df, metadata)

        manager.log_action("train_test_split", metadata)
        
        return metadata

    except ValueError as e:
        # Catch sklearn stratification errors specifically
        msg = str(e)
        if "The least populated class" in msg or "train_test_split" in msg:
             return {"error": f"Stratification failed: {msg}. Each class must have enough samples."}
        return {"error": msg}
    except Exception as e:
        return {"error": f"Split failed: {str(e)}"}
