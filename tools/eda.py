import pandas as pd
import numpy as np
from typing import Dict, Any, List, Union
from utils.state_manager import GlobalStateManager
from tools.discovery import load_dataset_metadata

def describe_dataset(dataset_name: str) -> Dict[str, Any]:
    """
    Generate a comprehensive statistical summary of the dataset.
    
    Args:
        dataset_name: Name of the dataset file (e.g., 'data.csv').
        
    Returns:
        Dictionary containing numerical and categorical summaries.
    """
    manager = GlobalStateManager()
    
    # 1. State Management: Load if needed
    if manager.get_dataset_name() != dataset_name:
        try:
            # Re-use discovery logic to load
            load_dataset_metadata(dataset_name)
        except Exception as e:
            return {"error": f"Failed to load dataset: {str(e)}"}
            
    df = manager.get_data()
    if df is None:
        return {"error": "Dataset loaded but DataFrame is None."}

    summary = {
        "dataset_name": dataset_name,
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "numerical_summary": {},
        "categorical_summary": {}
    }

    # 2. Numerical Analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        series = df[col]
        # Basic Stats
        desc = series.describe()
        
        # Advanced Stats
        skew = series.skew()
        kurt = series.kurt()
        
        # Data Quality
        n_zeros = (series == 0).sum()
        n_negatives = (series < 0).sum()
        
        summary["numerical_summary"][col] = {
            "mean": round(desc['mean'], 4),
            "std": round(desc['std'], 4),
            "min": desc['min'],
            "max": desc['max'],
            "quantiles": {
                "25%": desc['25%'],
                "50%": desc['50%'],
                "75%": desc['75%']
            },
            "distribution": {
                "skewness": round(skew, 4),
                "kurtosis": round(kurt, 4)
            },
            "quality_checks": {
                "num_zeros": int(n_zeros),
                "num_negatives": int(n_negatives)
            }
        }

    # 3. Categorical Analysis
    cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns
    for col in cat_cols:
        series = df[col].astype(str) # Ensure string for uniformity
        
        # Basic Stats
        n_unique = series.nunique()
        missing_count = df[col].isnull().sum()
        
        # Frequencies
        value_counts = series.value_counts()
        top_5 = value_counts.head(5).to_dict()
        top_5_with_pct = {k: {"count": v, "percentage": round((v/len(df))*100, 2)} for k, v in top_5.items()}
        
        # Rare Categories (< 5%)
        # Note: value_counts(normalize=True) gives proportions
        rare_mask = series.value_counts(normalize=True) < 0.05
        n_rare = rare_mask.sum()
        
        # String Metadata
        # Calculate avg length safely (skip NaNs for length calc if we didn't fill them)
        # But we cast to string above, so NaNs became 'nan'. 
        # Better: use original series for length calc, ignoring nulls
        original_series = df[col].dropna().astype(str)
        avg_len = original_series.str.len().mean() if not original_series.empty else 0
        
        summary["categorical_summary"][col] = {
            "unique_count": n_unique,
            "missing_count": int(missing_count),
            "top_frequencies": top_5_with_pct,
            "rare_categories_count": int(n_rare),
            "avg_string_length": round(avg_len, 2)
        }
        
    return summary
