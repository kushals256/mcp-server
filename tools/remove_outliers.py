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
from typing import Dict, Any, Literal, Optional
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from config import (
    DEFAULT_ZSCORE_THRESHOLD, 
    DEFAULT_IQR_MULTIPLIER, 
    DEFAULT_MODIFIED_ZSCORE_THRESHOLD,
    DEFAULT_RANDOM_STATE
)
from utils.state_manager import GlobalStateManager
from tools.discovery import load_dataset_metadata


def remove_outliers(
    dataset_name: str,
    column: str,
    method: Literal["zscore", "iqr", "modified_zscore", "isolation_forest", "lof"],
    threshold: Optional[float] = None
) -> Dict[str, Any]:
    """
    Remove outliers from a dataset using Statistical or Model-Based methods.
    
    Robustness Features:
    - Explicitly ignores NaNs and Infs (Does NOT remove them).
    - Resets index after removal to guarantee 1:1 mapping.
    - Zero Variance Safety: If std=0 or IQR=0 or MAD=0, removes 0 rows.
    - Strict Validation:
      - Threshold 0 < T <= 0.5: Treated as contamination percentage.
      - Threshold > 1.0 (or None): Auto/Default.
      - Threshold in (0.5, 1.0]: Rises ValueError (ambiguous).
    - Model Safety:
      - Small Datasets: Skips model fitting if n < 5 (LOF) or n < 10 (IF).
      - Constant Columns: Skips model fitting.
      - Determinism: Uses random_state=42 and n_jobs=1.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'data.csv').
        column: The numeric column to check for outliers.
        method: 'zscore', 'iqr', 'modified_zscore', 'isolation_forest', 'lof'.
            - zscore: Removes data where |Z| > threshold.
            - iqr: Removes data outside [Q1 - threshold*IQR, Q3 + threshold*IQR].
            - modified_zscore: Removes data where |Modified Z| > threshold (robust to outliers).
            - isolation_forest: Model-based (univariate).
            - lof: Density-based (univariate).
        threshold: The threshold value. 
                   For Stats: Z-score value or IQR multiplier.
                   For Models: Contamination fraction (<= 0.5) or Auto (> 1.0/None).
        
    Returns:
        Dictionary with structure:
        {
            "column": str,
            "rows_removed": int,
            "remaining_rows": int,
            "stats": { ... }
        }
    """
    manager = GlobalStateManager()
    
    # 1. Ensure dataset is loaded
    if manager.get_dataset_name() != dataset_name:
        try:
            load_dataset_metadata(dataset_name)
        except Exception as e:
            return {"error": f"Failed to load dataset: {str(e)}"}
            
    df = manager.get_data()
    if df is None:
        return {"error": "Dataset loaded but DataFrame is None."}
        
    # Safety: Reset index immediately to ensure row alignment
    # We do NOT propagate this reset to the saved file unless we actually remove rows.
    # But for calculation, we need a clean index.
    df = df.reset_index(drop=True)
        
    # 2. Validation: Column
    if column not in df.columns:
        return {"error": f"Column '{column}' not found in dataset."}
        
    if not pd.api.types.is_numeric_dtype(df[column]):
        return {"error": f"Column '{column}' is not numeric. Outlier removal only works on numeric data."}
        
    initial_rows = len(df)
    series = df[column]
    
    # Metadata container
    stats_meta = {
        "method": method,
        "input_threshold": threshold,
        "zero_variance_detected": False
    }

    # 3. Default Threshold Handling per Method
    if threshold is None:
        if method == "zscore":
            threshold = DEFAULT_ZSCORE_THRESHOLD
        elif method == "iqr":
            threshold = DEFAULT_IQR_MULTIPLIER
        elif method == "modified_zscore":
            threshold = DEFAULT_MODIFIED_ZSCORE_THRESHOLD
        # For ML methods, threshold acts as contamination. None -> Auto.
        # We handle this below in the specific logic block.

    # 4. Outlier Logic
    
    if method in ["zscore", "iqr", "modified_zscore"]:
        # Statistical Methods (unchanged logic mostly)
        if threshold is None or threshold <= 0:
             return {"error": "Threshold must be greater than 0 for statistical methods."}
        
        stats_meta["threshold"] = threshold

        if method == "zscore":
            std_dev = series.std(skipna=True)
            mean_val = series.mean(skipna=True)
            stats_meta["mean"] = round(mean_val, 4) if not np.isnan(mean_val) else None
            stats_meta["std"] = round(std_dev, 4) if not np.isnan(std_dev) else None
            
            if std_dev == 0 or np.isnan(std_dev):
                # Constant value -> No outliers (Conservative safety)
                mask = pd.Series(True, index=df.index)
                stats_meta["zero_variance_detected"] = True
            else:
                z_scores = (series - mean_val) / std_dev
                # Keep if Nan OR |Z| <= threshold
                mask = (z_scores.isna()) | (abs(z_scores) <= threshold)
                
                # Record bounds for reference (implied)
                stats_meta["lower_bound"] = round(mean_val - (threshold * std_dev), 4)
                stats_meta["upper_bound"] = round(mean_val + (threshold * std_dev), 4)
            
        elif method == "iqr":
            # Calculate IQR bounds using LINEAR interpolation regarding requested refactor
            Q1 = series.quantile(0.25, interpolation='linear')
            Q3 = series.quantile(0.75, interpolation='linear')
            IQR = Q3 - Q1
            
            stats_meta["q1"] = round(Q1, 4)
            stats_meta["q3"] = round(Q3, 4)
            stats_meta["iqr"] = round(IQR, 4)
            
            if IQR == 0:
                 # Zero IQR -> No outliers (Conservative safety)
                 mask = pd.Series(True, index=df.index)
                 stats_meta["zero_variance_detected"] = True
            else:
                lower_bound = Q1 - (threshold * IQR)
                upper_bound = Q3 + (threshold * IQR)
                
                stats_meta["lower_bound"] = round(lower_bound, 4)
                stats_meta["upper_bound"] = round(upper_bound, 4)
                
                # Keep if Nan OR within bounds
                mask = (series.isna()) | ((series >= lower_bound) & (series <= upper_bound))

        elif method == "modified_zscore":
            # Modified Z-Score = 0.6745 * (x - median) / MAD
            # MAD = median(|x - median|)
            median_val = series.median(skipna=True)
            diff = abs(series - median_val)
            mad_val = diff.median(skipna=True)
            
            stats_meta["median"] = round(median_val, 4) if not np.isnan(median_val) else None
            stats_meta["mad"] = round(mad_val, 4) if not np.isnan(mad_val) else None
            
            if mad_val == 0 or np.isnan(mad_val):
                # Constant value (or >50% values are same) -> MAD is 0.
                # Conservative safety: Do NOT remove anything if MAD is 0.
                mask = pd.Series(True, index=df.index)
                stats_meta["zero_variance_detected"] = True
            else:
                consistency_correction = 0.6745
                mod_z_scores = (consistency_correction * (series - median_val)) / mad_val
                
                # Keep if Nan OR |ModZ| <= threshold
                mask = (mod_z_scores.isna()) | (abs(mod_z_scores) <= threshold)
                
                stats_meta["min_mod_z"] = round(mod_z_scores.min(), 4)
                stats_meta["max_mod_z"] = round(mod_z_scores.max(), 4)

    elif method in ["isolation_forest", "lof"]:
        # Model-Based Methods (Univariate)
        
        # A. Strict Contamination/Threshold Validation
        contamination = "auto" # default for IF
        
        if threshold is not None:
            if threshold <= 0:
                return {"error": "Threshold (contamination) must be > 0."}
            elif 0 < threshold <= 0.5:
                # Interpret as contamination percentage
                contamination = threshold
            elif threshold > 1.0:
                 # Interpret as Auto/Default
                 contamination = "auto"
            else:
                # 0.5 < threshold <= 1.0 is ambiguous. Reject strict.
                return {"error": f"Ambiguous threshold {threshold}. Use <= 0.5 for contamination or > 1.0 for auto."}
        
        # For LOF, 'auto' is different. Default is often 0.1 logic or 'auto' in newer sklearn.
        # But LOF doesn't support 'auto' string in all versions.
        if method == "lof" and contamination == "auto":
            contamination = 0.1 # LOF default equivalent if user passed > 1.0 or None

        stats_meta["contamination_param"] = contamination
        stats_meta["random_state"] = DEFAULT_RANDOM_STATE

        # B. Data Prep & Valid Mask
        # Drop NaNs and Infs explicitly
        # We need a clean numeric array for sklearn
        valid_mask = series.notna() & ~series.isin([np.inf, -np.inf])
        valid_data = series[valid_mask]
        
        # C. Small Dataset & Constant Column Guards
        n_samples = len(valid_data)
        stats_meta["n_valid_samples"] = n_samples
        
        if n_samples == 0:
             # Nothing to detect
             mask = pd.Series(True, index=df.index)
             stats_meta["reason"] = "empty_valid_data"
        
        elif valid_data.nunique() <= 1:
             # Constant column -> No outliers model-wise
             mask = pd.Series(True, index=df.index)
             stats_meta["zero_variance_detected"] = True
             stats_meta["reason"] = "constant_column"
             
        elif (method == "lof" and n_samples < 5) or (method == "isolation_forest" and n_samples < 10):
             # Too small for reliable detection
             mask = pd.Series(True, index=df.index)
             stats_meta["reason"] = "too_small_for_model"
             
        else:
            # Reshape for sklearn
            X = valid_data.values.reshape(-1, 1)
            
            # Prediction map: -1 (outlier), 1 (inlier)
            if method == "isolation_forest":
                model = IsolationForest(
                    contamination=contamination, 
                    random_state=DEFAULT_RANDOM_STATE,
                    n_jobs=1 # Deterministic
                )
                preds = model.fit_predict(X)
            else: # lof
                # Adjust neighbors if dataset is small but > 5
                n_neighbors = min(20, n_samples - 1)
                stats_meta["n_neighbors"] = n_neighbors
                
                model = LocalOutlierFactor(
                    n_neighbors=n_neighbors,
                    contamination=contamination,
                    n_jobs=1 # Deterministic
                )
                preds = model.fit_predict(X)
                
            # D. Map Predictions back to DataFrame
            # preds is aligned with valid_data.index
            # We want False (remove) where preds == -1
            
            # Start strict: All True (keep everything)
            mask = pd.Series(True, index=df.index)
            
            # Identify outlier INDICES in original DF
            outlier_indices = valid_data.index[preds == -1]
            
            # Mark them False
            mask.loc[outlier_indices] = False
            
            stats_meta["outliers_detected"] = len(outlier_indices)

    else:
        return {"error": f"Unknown method '{method}'."}
        
    # 5. Apply Filter & Update State (Immutability pattern)
    # create new dataframe only after mask is fully computed
    dataset_cleaned = df[mask].copy()
    
    # Important: Reset Index
    dataset_cleaned = dataset_cleaned.reset_index(drop=True)
    
    rows_removed = initial_rows - len(dataset_cleaned)
    
    # Update global state
    if hasattr(manager, 'load_data'):
        manager.load_data(dataset_cleaned, dataset_name, preserve_split=True)
    
    # LOG THE ACTION FOR REPORT GENERATION
    manager.log_action("remove_data_outliers", {
        "dataset_name": dataset_name,
        "column": column,
        "method": method,
        "threshold": threshold,
        "rows_removed": rows_removed
    })
    
    return {
        "column": column,
        "rows_removed": rows_removed,
        "remaining_rows": len(dataset_cleaned),
        "stats": stats_meta
    }