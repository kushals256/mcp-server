"""
Outlier Removal Tools for MCP Server.

This module provides statistical methods for removing outliers from datasets.
It implements Phase 4 (Transformation) of the dataset analysis workflow.

Supported methods:
    - Z-score: Removes data points with |Z| > threshold
    - IQR: Removes data outside [Q1 - threshold*IQR, Q3 + threshold*IQR]

Functions:
    remove_outliers: Remove outliers from a specified column using statistical methods
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Literal

from config import DEFAULT_ZSCORE_THRESHOLD, DEFAULT_IQR_MULTIPLIER
from utils.state_manager import GlobalStateManager


def remove_outliers(
    dataset_name: str,
    column: str,
    method: Literal["zscore", "iqr"],
    threshold: float = DEFAULT_ZSCORE_THRESHOLD
) -> Dict[str, Any]:
    """
    Remove outliers from a dataset using Statistical methods (Z-Score or IQR).
    
    Robustness Features:
    - Explicitly ignores NaNs (Does NOT remove them).
    - Resets index after removal.
    - Zero Variance Safety: If std=0 or IQR=0, removes 0 rows.
    - Validates threshold > 0.
    - Returns detailed metadata about the operation.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'data.csv').
        column: The numeric column to check for outliers.
        method: 'zscore' or 'iqr'.
            - zscore: Removes data where |Z| > threshold.
            - iqr: Removes data outside [Q1 - threshold*IQR, Q3 + threshold*IQR].
        threshold: The threshold value (must be > 0).
        
    Returns:
        Dictionary with structure:
        {
            "column": str,
            "rows_removed": int,
            "remaining_rows": int,
            "stats": { ... }
        }
    """
    # 1. Validation: Threshold
    if threshold <= 0:
        return {"error": "Threshold must be greater than 0."}

    manager = GlobalStateManager()
    
    # 2. Ensure dataset is loaded
    if manager.get_dataset_name() != dataset_name:
        return {
            "error": f"Dataset '{dataset_name}' is not currently loaded. "
                     "Call load_dataset_metadata() explicitly to load it first."
        }
            
    df = manager.get_data()
    if df is None:
        return {"error": "Dataset loaded but DataFrame is None."}
        
    # 3. Validation: Column
    if column not in df.columns:
        return {"error": f"Column '{column}' not found in dataset."}
        
    if not pd.api.types.is_numeric_dtype(df[column]):
        return {"error": f"Column '{column}' is not numeric. Outlier removal only works on numeric data."}
        
    initial_rows = len(df)
    series = df[column]
    
    # Metadata container
    stats_meta = {
        "method": method,
        "threshold": threshold,
        "zero_variance_detected": False
    }

    # 4. Outlier Logic (Pure Numpy/Pandas)
    
    if method == "zscore":
        std_dev = series.std(skipna=True)
        mean_val = series.mean(skipna=True)
        
        stats_meta["mean"] = round(mean_val, 4) if not np.isnan(mean_val) else None
        stats_meta["std"] = round(std_dev, 4) if not np.isnan(std_dev) else None
        
        if std_dev == 0 or np.isnan(std_dev):
            mask = pd.Series(True, index=df.index)
            stats_meta["zero_variance_detected"] = True
        else:
            z_scores = (series - mean_val) / std_dev
            mask = (z_scores.isna()) | (abs(z_scores) <= threshold)
            stats_meta["lower_bound"] = round(mean_val - (threshold * std_dev), 4)
            stats_meta["upper_bound"] = round(mean_val + (threshold * std_dev), 4)
        
    elif method == "iqr":
        Q1 = series.quantile(0.25, interpolation='linear')
        Q3 = series.quantile(0.75, interpolation='linear')
        IQR = Q3 - Q1
        
        stats_meta["q1"] = round(Q1, 4)
        stats_meta["q3"] = round(Q3, 4)
        stats_meta["iqr"] = round(IQR, 4)
        
        if IQR == 0:
             mask = pd.Series(True, index=df.index)
             stats_meta["zero_variance_detected"] = True
        else:
            lower_bound = Q1 - (threshold * IQR)
            upper_bound = Q3 + (threshold * IQR)
            
            stats_meta["lower_bound"] = round(lower_bound, 4)
            stats_meta["upper_bound"] = round(upper_bound, 4)
            
            mask = (series.isna()) | ((series >= lower_bound) & (series <= upper_bound))
        
    else:
        return {"error": f"Unknown method '{method}'. Use 'zscore' or 'iqr'."}
        
    # 5. Apply Filter & Update State
    dataset_cleaned = df[mask].copy()
    dataset_cleaned = dataset_cleaned.reset_index(drop=True)
    rows_removed = initial_rows - len(dataset_cleaned)
    
    # Update global state
    manager.load_data(dataset_cleaned, dataset_name)
    
    return {
        "column": column,
        "rows_removed": rows_removed,
        "remaining_rows": len(dataset_cleaned),
        "stats": stats_meta
    }