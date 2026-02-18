"""
Feature Extraction Tool for MCP Server.

Implements safe feature generation:
1. Datetime extraction (Year, Month, etc.)
2. Text statistics (Length, Word Count)
3. Mathematical transformations (Log, Sqrt, Poly, Interaction)

Safety Guards:
- Invalid math (log<=0) -> NaN (no crash)
- Naming collisions -> Error
- NaNs -> Handled gracefully
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Literal
from utils.state_manager import GlobalStateManager

def extract_features(
    method: Literal["datetime", "text", "math"],
    columns: List[str],
    operation: Optional[str] = None, # e.g. "year", "log", "interaction"
    new_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Generate new features from existing ones.
    
    Args:
        method: extraction type.
        columns: input columns.
        operation: specific operation (e.g. 'year' for datetime, 'log' for math).
        new_names: Optional manual names. If None, auto-generated deterministic names.
    """
    manager = GlobalStateManager()
    train_df = manager.get_data()
    
    if train_df is None:
        return {"error": "No dataset loaded."}
        
    is_split = manager.is_split()
    test_df = manager.get_test_data() if is_split else None
    
    # Validation: Cols exist
    missing = [c for c in columns if c not in train_df.columns]
    if missing:
        return {"error": f"Columns not found: {missing}"}
        
    created_features = []
    
    try:
        # Helper to apply to both safely
        def apply_transform(func, col_name, suffix):
            # Deterministic Naming
            new_col = f"{col_name}_{suffix}"
            if new_names and len(new_names) == len(columns):
                 new_col = new_names[columns.index(col_name)]
            
            # Collision Check
            if new_col in train_df.columns:
                raise ValueError(f"Feature '{new_col}' already exists. Rename or drop first.")
            
            # Apply
            train_df[new_col] = func(train_df[col_name])
            if is_split:
                test_df[new_col] = func(test_df[col_name])
            
            created_features.append(new_col)

        # ---------------------------------------------------------------------
        # A. Datetime
        # ---------------------------------------------------------------------
        if method == "datetime":
            valid_ops = ["year", "month", "day", "dayofweek", "is_weekend", "hour"]
            if operation not in valid_ops:
                return {"error": f"Invalid datetime operation. Choose: {valid_ops}"}
            
            for col in columns:
                # Safe Coercion
                if not pd.api.types.is_datetime64_any_dtype(train_df[col]):
                    # We only coerce temporarily for extraction to avoid modifying original type permanently if unwanted?
                    # Actually, usually better to coerce valid date strings.
                    # We'll use a local coerced series.
                    pass
                
                def extract_dt(series):
                    # Coerce errors='coerce' turns invalid/out-of-bound to NaT
                    s_dt = pd.to_datetime(series, errors='coerce')
                    if operation == "year": return s_dt.dt.year
                    if operation == "month": return s_dt.dt.month
                    if operation == "day": return s_dt.dt.day
                    if operation == "dayofweek": return s_dt.dt.dayofweek
                    if operation == "hour": return s_dt.dt.hour
                    if operation == "is_weekend": return (s_dt.dt.dayofweek >= 5).astype(int) # 0 or 1
                    
                apply_transform(extract_dt, col, operation)

        # ---------------------------------------------------------------------
        # B. Text
        # ---------------------------------------------------------------------
        elif method == "text":
            valid_ops = ["length", "word_count"]
            if operation not in valid_ops:
                return {"error": f"Invalid text operation. Choose: {valid_ops}"}
            
            for col in columns:
                def extract_text(series):
                    s_str = series.astype(str) # Force string (handles numbers)
                    # Handle "nan" string if it was NaN
                    mask = series.isna()
                    
                    if operation == "length":
                        res = s_str.str.len()
                    elif operation == "word_count":
                        res = s_str.str.count(' ') + 1
                    
                    # Restore NaNs (don't give length to NaN)
                    res[mask] = np.nan
                    return res
                    
                apply_transform(extract_text, col, operation)

        # ---------------------------------------------------------------------
        # C. Math
        # ---------------------------------------------------------------------
        elif method == "math":
            valid_ops = ["log", "sqrt", "square", "interaction"]
            if operation not in valid_ops:
                return {"error": f"Invalid math operation. Choose: {valid_ops}"}
            
            # Interaction is special (combines cols)
            if operation == "interaction":
                if len(columns) < 2:
                    return {"error": "Interaction requires 2+ columns."}
                
                # Check numeric
                if not all(pd.api.types.is_numeric_dtype(train_df[c]) for c in columns):
                    return {"error": "Interaction columns must be numeric."}
                    
                new_col = "_x_".join(columns)
                if new_names: new_col = new_names[0]
                
                if new_col in train_df.columns:
                     return {"error": f"Feature '{new_col}' exists."}
                
                # Multiply all
                def interact(df_subset):
                     res = df_subset[columns[0]]
                     for c in columns[1:]:
                         res = res * df_subset[c]
                     return res
                
                train_df[new_col] = interact(train_df)
                if is_split:
                    test_df[new_col] = interact(test_df)
                created_features.append(new_col)
                
            else:
                # Unary ops
                for col in columns:
                    if not pd.api.types.is_numeric_dtype(train_df[col]):
                        return {"error": f"Column '{col}' is not numeric."}
                    
                    def extract_math(series):
                        if operation == "log":
                            # Guard: log(<=0) -> NaN
                            return np.log(series.replace(0, np.nan).where(series > 0))
                        if operation == "sqrt":
                             # Guard: sqrt(<0) -> NaN
                             return np.sqrt(series.where(series >= 0))
                        if operation == "square":
                            return np.square(series)
                            
                    apply_transform(extract_math, col, operation)

    except Exception as e:
        return {"error": f"Extraction failed: {str(e)}"}
        
    # Update State
    # Update State
    manager.load_data(train_df, manager.get_dataset_name(), preserve_split=True)
    if is_split:
        manager.update_test_data(test_df)
        
    return {
        "success": True,
        "features_created": created_features
    }
