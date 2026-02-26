"""
Feature Reduction Tool for MCP Server.

Implements Dimensionality Reduction (PCA, LDA, SVD).

Strict Rules:
1. Stateful: Fit on Train, Transform Both.
2. Auto-Scaling: Always scales inputs (StandardScaler) inside pipeline.
3. NaN Policy: ERROR if NaNs present (User must run imputation first).
4. Target Protection: Target removed before reduction.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Literal
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from utils.state_manager import GlobalStateManager

def reduce_features(
    method: Literal["pca", "lda", "truncated_svd"],
    n_components: int,
    target_col: Optional[str] = None
) -> Dict[str, Any]:
    """
    Reduce dimensionality of the dataset.
    
    Args:
        method: Reduction algorithm.
        n_components: Number of components to keep.
        target_col: Target variable (Required for LDA). Protected from reduction.
    """
    manager = GlobalStateManager()
    train_df = manager.get_data()
    
    if train_df is None:
        return {"error": "No dataset loaded."}
    
    is_split = manager.is_split()
    test_df = manager.get_test_data() if is_split else None
    
    # 0. Validation: Components
    if n_components <= 0:
        return {"error": "n_components must be > 0"}
    if n_components >= len(train_df.columns): # Rough check, excludes target later
        pass # We let sklearn raise if it's strictly invalid, or check after target drop

    # 1. Target Protection
    train_target = None
    test_target = None
    
    if target_col:
        if target_col not in train_df.columns:
            return {"error": f"Target '{target_col}' not found."}
            
        train_target = train_df[target_col].copy()
        train_df = train_df.drop(columns=[target_col])
        
        if is_split and test_df is not None and target_col in test_df.columns:
            test_target = test_df[target_col].copy()
            test_df = test_df.drop(columns=[target_col])

    elif method == "lda":
        return {"error": "LDA requires 'target_col'."}

    # 2. Numeric Only & NaN Check (Strict)
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) != len(train_df.columns):
        # We could drop non-numeric, but for reduction it's usually better to Error so user knows they are losing data?
        # Or just operate on numeric?
        # "Strict" implies we should probably warn or error if user tries to PCA text columns.
        # Let's filter to numeric but ERROR if resulting set is empty.
        pass # Proceed with numeric subset
        
    X_train = train_df[numeric_cols]
    if is_split:
        X_test = test_df[numeric_cols]
        
    if X_train.shape[1] == 0:
        return {"error": "No numeric columns available for reduction."}
        
    # Check NaNs - STRICT ERROR
    if X_train.isnull().any().any():
        return {"error": "Dataset contains NaNs. Feature reduction requires complete data. Please run imputation first."}

    # Check n_components validity against feature count
    if n_components > X_train.shape[1]:
        return {"error": f"n_components ({n_components}) cannot be greater than feature count ({X_train.shape[1]})."}

    try:
        # 3. Pipeline Construction (Scale + Reduce)
        steps = [('scaler', StandardScaler())]
        
        if method == "pca":
            steps.append(('pca', PCA(n_components=n_components, random_state=42)))
        elif method == "lda":
            # LDA constraint check
            n_classes = train_target.nunique()
            max_components = n_classes - 1
            if n_components > max_components:
                return {"error": f"LDA n_components ({n_components}) must be < number of classes ({n_classes}). Max allowed: {max_components}"}
            
            steps.append(('lda', LinearDiscriminantAnalysis(n_components=n_components)))
        elif method == "truncated_svd":
            # SVD doesn't require centering, but scaler helps.
            # However SVD often used for Sparse where scaler breaks sparsity.
            # Assuming dense for now (CSV). Standard scaler is fine.
            steps.append(('truncated_svd', TruncatedSVD(n_components=n_components, random_state=42)))
            
        pipeline = Pipeline(steps)
        
        # 4. Fit (Train Only)
        if method == "lda":
            # Check target NaNs strictly
            if train_target.isnull().any():
                return {"error": "Target contains NaNs. LDA cannot proceed."}
            pipeline.fit(X_train, train_target)
        else:
            pipeline.fit(X_train)
            
        # 5. Transform (Both)
        X_train_reduced = pipeline.transform(X_train)
        
        # Create DataFrames
        new_cols = [f"{method}_{i+1}" for i in range(n_components)]
        train_reduced_df = pd.DataFrame(X_train_reduced, columns=new_cols, index=train_df.index)
        
        if is_split:
            X_test_reduced = pipeline.transform(X_test)
            test_reduced_df = pd.DataFrame(X_test_reduced, columns=new_cols, index=test_df.index)

        # 6. Re-attach Target (to the REDUCED dataset? Usually yes, we replace features with components)
        if train_target is not None:
             train_reduced_df[target_col] = train_target
             if is_split and test_target is not None:
                 test_reduced_df[target_col] = test_target
        
        # 7. Save Transformer
        manager.save_transformer(method, pipeline, list(numeric_cols))
        
        # 8. Update State
        # 8. Update State
        manager.update_data(train_reduced_df, tool_name="reduce_features")
        if is_split:
            manager.update_test_data(test_reduced_df)
            
        return {
            "success": True,
            "method": method,
            "original_features": X_train.shape[1],
            "reduced_features": n_components,
            "explained_variance": float(np.sum(pipeline.named_steps[method].explained_variance_ratio_)) if method in ['pca', 'truncated_svd'] else "N/A"
        }

    except Exception as e:
        return {"error": f"Reduction failed: {str(e)}"}
