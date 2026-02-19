"""
Feature Removal Tool for MCP Server.

Implements leakage-proof feature selection/removal:
1. Stateless methods (missing_threshold) apply to both Train/Test.
2. Stateful methods (variance, correlation, RFE) FIT on Train, TRANSFORM both.
3. Target column is ALWAYS protected (removed before, re-attached after).
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Literal
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, f_regression, RFE
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.base import clone

from utils.state_manager import GlobalStateManager

def remove_features(
    method: Literal["missing_threshold", "variance_threshold", "correlation_threshold", "select_k_best", "rfe", "by_name"],
    confirm_action: bool = True,  # Ignored, but kept for compatibility patterns if needed
    threshold: Optional[float] = None,
    k: Optional[int] = None,
    target_col: Optional[str] = None,
    feature_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Remove features from the dataset using various statistical or model-based methods.
    
    CRITICAL:
    - If dataset is SPLIT: Fits on Train, Transforms Train & Test.
    - Target column is PROTECTED (never removed/used in unsupervised steps).
    - Transformers are SAVED in StateManager.
    
    Args:
        method: Selection method.
        threshold: 
            - missing_threshold: Max % of NaNs allowed (0.0-1.0). e.g., 0.2 removes cols with >20% missing.
            - variance_threshold: Min variance required.
            - correlation_threshold: Max correlation allowed.
        k: Number of features to keep (for SelectKBest, RFE).
        target_col: Target variable (Required for SelectKBest, RFE). Protected from removal.
        feature_names: List of columns to drop (for 'by_name' method).
    """
    manager = GlobalStateManager()
    train_df = manager.get_data()
    
    if train_df is None:
        return {"error": "No dataset loaded."}
    
    is_split = manager.is_split()
    test_df = manager.get_test_data() if is_split else None
    
    # 0. Validate Target
    if target_col and target_col not in train_df.columns:
        return {"error": f"Target column '{target_col}' not found in dataset."}
        
    # Separate Target (PROTECTION)
    train_target = None
    test_target = None
    
    if target_col:
        train_target = train_df[target_col].copy()
        train_df = train_df.drop(columns=[target_col])
        
        if is_split and test_df is not None:
            if target_col in test_df.columns:
                test_target = test_df[target_col].copy()
                test_df = test_df.drop(columns=[target_col])
            else:
                 # It's possible test set doesn't have target (e.g. Kaggle test set)
                 pass

    # Helper to re-attach target
    def reattach_target(df_tr, df_te):
        if train_target is not None:
            # Re-attach to same position? Standard append is safer for now.
            # To preserve order perfectly we'd need index tracking, but append is standard for "X, y" separation tools.
            df_tr[target_col] = train_target
        if test_target is not None and df_te is not None:
            df_te[target_col] = test_target
        return df_tr, df_te

    # 1. Method Implementation
    
    removed_features = []
    
    try:
        # ---------------------------------------------------------------------
        # A. Missing Threshold (Stateless)
        # ---------------------------------------------------------------------
        if method == "missing_threshold":
            if threshold is None or not (0 <= threshold <= 1.0):
                return {"error": "missing_threshold requires 'threshold' between 0.0 and 1.0"}
                
            # Calculate on TRAIN
            missing_series = train_df.isnull().mean()
            cols_to_drop = missing_series[missing_series > threshold].index.tolist()
            
            if not cols_to_drop:
                 train_df, test_df = reattach_target(train_df, test_df)
                 return {"message": "No columns exceeded missing threshold.", "features_removed": 0}
            
            train_df = train_df.drop(columns=cols_to_drop)
            if is_split:
                test_df = test_df.drop(columns=cols_to_drop)
                
            removed_features = cols_to_drop

        # ---------------------------------------------------------------------
        # B. Variance Threshold (Stateful)
        # ---------------------------------------------------------------------
        elif method == "variance_threshold":
            th = threshold if threshold is not None else 0.0
            
            # Numeric only strictness
            numeric_cols = train_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                 return {"error": "No numeric columns found for VarianceThreshold."}
            
            selector = VarianceThreshold(threshold=th)
            
            # Fit on Train (Numeric Only)
            selector.fit(train_df[numeric_cols])
            
            # Get support
            keep_mask = selector.get_support()
            keep_cols = numeric_cols[keep_mask]
            drop_cols = numeric_cols[~keep_mask]
            
            # Drop from full df
            train_df = train_df.drop(columns=drop_cols)
            if is_split:
                test_df = test_df.drop(columns=drop_cols)
                
            removed_features = drop_cols.tolist()
            
            # Persist
            manager.save_transformer("variance_threshold", selector)

        # ---------------------------------------------------------------------
        # C. Correlation Threshold (Stateful + Deterministic Tie-Break)
        # ---------------------------------------------------------------------
        elif method == "correlation_threshold":
            if threshold is None or not (0 < threshold <= 1.0):
                 return {"error": "correlation_threshold requires 'threshold' between 0.0 and 1.0"}
            
            numeric_df = train_df.select_dtypes(include=[np.number])
            if numeric_df.empty:
                return {"error": "No numeric columns for correlation analysis."}
                
            corr_matrix = numeric_df.corr().abs()
            
            # Select upper triangle
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            
            # Find columns to drop
            to_drop = set()
            for col in upper.columns:
                # If col is correlated with any row feature
                high_corr_rows = upper.index[upper[col] > threshold]
                
                for row in high_corr_rows:
                    # Tie-Breaker: Drop the one that comes LATER in alphabetical order
                    # (To be strictly deterministic regardless of existing column order)
                    pair = sorted([col, row]) # [A, B]
                    to_drop.add(pair[1]) # Drop B
            
            drop_list = list(to_drop)
            
            train_df = train_df.drop(columns=drop_list)
            if is_split:
                test_df = test_df.drop(columns=drop_list)
                
            removed_features = drop_list

        # ---------------------------------------------------------------------
        # D. Select K Best (Stateful + Supervised)
        # ---------------------------------------------------------------------
        elif method == "select_k_best":
            if train_target is None:
                return {"error": "select_k_best requires 'target_col'."}
            if k is None or k <= 0:
                return {"error": "k must be > 0"}
            if k >= len(train_df.columns):
                return {"error": f"k ({k}) must be less than feature count ({len(train_df.columns)})."}
            
            # Numeric enforce
            numeric_cols = train_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                return {"error": "No numeric columns for SelectKBest."}
            
            # Check NaNs
            if train_df[numeric_cols].isnull().any().any():
                return {"error": "SelectKBest cannot handle NaNs. Impute missing values first."}
            
            # Determine score function (Regression or Classif)
            # Simple heuristic: if target numeric and unique > 20 -> Regression
            is_regression = pd.api.types.is_numeric_dtype(train_target) and train_target.nunique() > 20
            score_func = f_regression if is_regression else f_classif
            
            selector = SelectKBest(score_func=score_func, k=k)
            selector.fit(train_df[numeric_cols], train_target)
            
            cols_idxs = selector.get_support(indices=True)
            keep_cols = numeric_cols[cols_idxs]
            
            # Construct new df (keep ONLY selected numeric + original non-numeric?)
            # Usually KBest implies we only care about these. Let's strictly keep ONLY these.
            # BUT user might have categorical features they want to keep. 
            # Safest MVP: Apply ONLY to numeric, drop rejected numeric, keep original other types.
            
            rejected_numeric = set(numeric_cols) - set(keep_cols)
            
            train_df = train_df.drop(columns=rejected_numeric)
            if is_split:
                test_df = test_df.drop(columns=rejected_numeric)
                
            removed_features = list(rejected_numeric)
            manager.save_transformer(f"kbest_{k}", selector)

        # ---------------------------------------------------------------------
        # E. RFE (Stateful + Supervised)
        # ---------------------------------------------------------------------
        elif method == "rfe":
            if train_target is None:
                return {"error": "rfe requires 'target_col'."}
            if k is None or k <= 0:
                 return {"error": "k must be > 0"}
            
            # Numeric enforce
            numeric_cols = train_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                return {"error": "No numeric columns for RFE."}
             # Check NaNs
            if train_df[numeric_cols].isnull().any().any():
                 return {"error": "RFE cannot handle NaNs. Impute first."}

            is_regression = pd.api.types.is_numeric_dtype(train_target) and train_target.nunique() > 20
            estimator = RandomForestRegressor(n_estimators=50, random_state=42) if is_regression else RandomForestClassifier(n_estimators=50, random_state=42)
            
            selector = RFE(estimator=estimator, n_features_to_select=k)
            selector.fit(train_df[numeric_cols], train_target)
            
            keep_mask = selector.support_
            keep_cols = numeric_cols[keep_mask]
            rejected_numeric = set(numeric_cols) - set(keep_cols)
            
            train_df = train_df.drop(columns=rejected_numeric)
            if is_split:
                test_df = test_df.drop(columns=rejected_numeric)
                
            removed_features = list(rejected_numeric)
            manager.save_transformer("rfe", selector)

        # ---------------------------------------------------------------------
        # F. By Name (Stateless)
        # ---------------------------------------------------------------------
        elif method == "by_name":
            if not feature_names:
                return {"error": "by_name requires 'feature_names' list."}
            
            # Validate existence
            missing_cols = [c for c in feature_names if c not in train_df.columns]
            if missing_cols:
                return {"error": f"Columns not found: {missing_cols}"}
            
            train_df = train_df.drop(columns=feature_names)
            if is_split:
                 test_df = test_df.drop(columns=feature_names)
            
            removed_features = feature_names

        else:
            return {"error": f"Unknown method: {method}"}
            
    except Exception as e:
        return {"error": f"Feature removal failed: {str(e)}"}

    # 4. Zero-Feature Guard
    if len(train_df.columns) == 0:
        return {"error": "Operation result in ZERO features. Operation aborted to protect dataset."}

    # 5. Re-attach Target
    train_df, test_df = reattach_target(train_df, test_df)
    
    # 6. Final State Update
    # 6. Final State Update
    manager.load_data(train_df, manager.get_dataset_name(), reset_split=False)
    if is_split:
        manager.update_test_data(test_df)
        
    return {
        "success": True,
        "method": method,
        "features_removed_count": len(removed_features),
        "features_removed": removed_features,
        "remaining_features": len(train_df.columns)
    }
