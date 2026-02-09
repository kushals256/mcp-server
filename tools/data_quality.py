import pandas as pd
import numpy as np
from typing import Dict, Any, List
from utils.state_manager import GlobalStateManager
from tools.discovery import load_dataset_metadata


def detect_data_quality_issues(dataset_name: str) -> Dict[str, Any]:
    """
    Automatically detect data quality problems in a dataset.
    
    Detects:
    - Missing values
    - Outliers (using adaptive method selection based on distribution)
    - High cardinality columns
    - Duplicate rows
    
    Args:
        dataset_name: Name of the dataset file (e.g., 'data.csv').
        
    Returns:
        Dictionary containing array of detected issues with type, column, severity, method, and parameters.
    """
    manager = GlobalStateManager()
    
    # Load dataset if needed
    if manager.get_dataset_name() != dataset_name:
        try:
            load_dataset_metadata(dataset_name)
        except Exception as e:
            return {"error": f"Failed to load dataset: {str(e)}"}
            
    df = manager.get_data()
    if df is None:
        return {"error": "Dataset loaded but DataFrame is None."}
    
    issues = []
    total_rows = len(df)
    
    # 1. DETECT MISSING VALUES (all columns)
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            missing_pct = (missing_count / total_rows) * 100
            
            # Severity: <5% = low, 5-20% = medium, >20% = high
            if missing_pct < 5:
                severity = "low"
            elif missing_pct < 20:
                severity = "medium"
            else:
                severity = "high"
            
            issues.append({
                "type": "missing_values",
                "column": col,
                "severity": severity,
                "method": "Percentage",
                "parameters": {
                    "count": int(missing_count),
                    "percentage": round(missing_pct, 2),
                    "total_rows": total_rows
                }
            })
    
    # 2. DETECT OUTLIERS (numerical columns only)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        series = df[col].dropna()  # Remove NaN for outlier detection
        
        if len(series) < 3:  # Need at least 3 values for meaningful outlier detection
            continue
            
        # Calculate distribution metrics
        skewness = series.skew()
        kurtosis = series.kurtosis()
        n_samples = len(series)
        
        # Adaptive method selection based on distribution
        method, outlier_mask, parameters = _detect_outliers_adaptive(
            series, skewness, kurtosis, n_samples
        )
        
        outlier_count = outlier_mask.sum()
        
        if outlier_count > 0:
            outlier_pct = (outlier_count / len(series)) * 100
            
            # Severity: <1% = low, 1-5% = medium, >5% = high
            if outlier_pct < 1:
                severity = "low"
            elif outlier_pct < 5:
                severity = "medium"
            else:
                severity = "high"
            
            # Add distribution metrics to parameters for transparency
            parameters["outlier_count"] = int(outlier_count)
            parameters["outlier_percentage"] = round(outlier_pct, 2)
            parameters["distribution_metrics"] = {
                "skewness": round(skewness, 4),
                "kurtosis": round(kurtosis, 4),
                "sample_size": n_samples
            }
            
            issues.append({
                "type": "outliers",
                "column": col,
                "severity": severity,
                "method": method,
                "parameters": parameters
            })
    
    # 3. DETECT HIGH CARDINALITY (all columns)
    for col in df.columns:
        unique_count = df[col].nunique()
        unique_ratio = unique_count / total_rows
        
        # Determine if this is high cardinality
        # Ratio-based: >0.5 = low, >0.8 = medium, >0.95 = high
        # Absolute for categorical: >50 = low, >100 = medium, >200 = high
        is_numeric = col in numeric_cols
        
        if is_numeric:
            # For numeric columns, use ratio-based approach
            if unique_ratio > 0.95:
                severity = "high"
            elif unique_ratio > 0.8:
                severity = "medium"
            elif unique_ratio > 0.5:
                severity = "low"
            else:
                severity = None  # Not high cardinality
        else:
            # For categorical columns, use absolute count
            if unique_count > 200:
                severity = "high"
            elif unique_count > 100:
                severity = "medium"
            elif unique_count > 50:
                severity = "low"
            else:
                severity = None  # Not high cardinality
        
        if severity is not None:
            issues.append({
                "type": "high_cardinality",
                "column": col,
                "severity": severity,
                "method": "Unique_ratio",
                "parameters": {
                    "unique_count": int(unique_count),
                    "ratio": round(unique_ratio, 4),
                    "total_rows": total_rows
                }
            })
    
    # 4. DETECT DUPLICATE ROWS
    duplicate_mask = df.duplicated(keep='first')
    duplicate_count = duplicate_mask.sum()
    
    if duplicate_count > 0:
        # Count unique duplicated rows (rows that appear more than once)
        unique_duplicated_rows = df[df.duplicated(keep=False)].drop_duplicates().shape[0]
        
        # Severity: <10 = low, 10-100 = medium, >100 = high
        if duplicate_count < 10:
            severity = "low"
        elif duplicate_count < 100:
            severity = "medium"
        else:
            severity = "high"
        
        issues.append({
            "type": "duplicate_rows",
            "column": "ALL_COLUMNS",
            "severity": severity,
            "method": "Exact_match",
            "parameters": {
                "duplicate_count": int(duplicate_count),
                "unique_duplicated_rows": int(unique_duplicated_rows),
                "total_rows": total_rows,
                "duplicate_percentage": round((duplicate_count / total_rows) * 100, 2)
            }
        })
    
    return {"issues": issues}


def _detect_outliers_adaptive(series: pd.Series, skewness: float, kurtosis: float, n_samples: int):
    """
    Adaptively select and apply outlier detection method based on distribution characteristics.
    
    Uses only skewness and kurtosis (no statistical tests) for performance.
    
    Args:
        series: Pandas Series with numerical data (NaN already removed)
        skewness: Pre-calculated skewness
        kurtosis: Pre-calculated kurtosis (excess kurtosis)
        n_samples: Sample size
        
    Returns:
        Tuple of (method_name, outlier_mask, parameters_dict)
    """
    abs_skew = abs(skewness)
    abs_kurt = abs(kurtosis)
    
    # Decision tree for method selection
    if n_samples < 30:
        # Small sample: use robust IQR method
        method = "IQR"
        outlier_mask, parameters = _detect_outliers_iqr(series)
        parameters["method_reason"] = "small_sample_size"
        
    elif abs_skew >= 1.0 or abs_kurt >= 3.0:
        # Highly non-normal: use IQR
        method = "IQR"
        outlier_mask, parameters = _detect_outliers_iqr(series)
        if abs_skew >= 1.0:
            parameters["method_reason"] = "highly_skewed"
        else:
            parameters["method_reason"] = "heavy_tails"
            
    elif abs_skew < 0.5 and abs_kurt < 1.0:
        # Approximately normal: use Z-score
        method = "Z-score"
        outlier_mask, parameters = _detect_outliers_zscore(series)
        parameters["method_reason"] = "approximately_normal"
        
    else:
        # Borderline case: use both methods with intersection (conservative)
        method = "Both_intersection"
        iqr_mask, iqr_params = _detect_outliers_iqr(series)
        z_mask, z_params = _detect_outliers_zscore(series)
        
        # Only flag outliers that both methods agree on
        outlier_mask = iqr_mask & z_mask
        
        parameters = {
            "method_reason": "borderline_distribution",
            "iqr_method": iqr_params,
            "zscore_method": z_params,
            "agreement_count": int(outlier_mask.sum()),
            "iqr_only_count": int((iqr_mask & ~z_mask).sum()),
            "zscore_only_count": int((z_mask & ~iqr_mask).sum())
        }
    
    return method, outlier_mask, parameters


def _detect_outliers_iqr(series: pd.Series, multiplier: float = 1.5):
    """
    Detect outliers using Interquartile Range (IQR) method.
    
    Args:
        series: Pandas Series with numerical data
        multiplier: IQR multiplier for fence calculation (default 1.5)
        
    Returns:
        Tuple of (outlier_mask, parameters_dict)
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_fence = Q1 - multiplier * IQR
    upper_fence = Q3 + multiplier * IQR
    
    outlier_mask = (series < lower_fence) | (series > upper_fence)
    
    parameters = {
        "multiplier": multiplier,
        "Q1": round(Q1, 4),
        "Q3": round(Q3, 4),
        "IQR": round(IQR, 4),
        "lower_fence": round(lower_fence, 4),
        "upper_fence": round(upper_fence, 4)
    }
    
    return outlier_mask, parameters


def _detect_outliers_zscore(series: pd.Series, threshold: float = 3.0):
    """
    Detect outliers using Z-score method.
    
    Args:
        series: Pandas Series with numerical data
        threshold: Z-score threshold (default 3.0 for 99.7% coverage)
        
    Returns:
        Tuple of (outlier_mask, parameters_dict)
    """
    mean = series.mean()
    std = series.std()
    
    # Avoid division by zero
    if std == 0:
        return pd.Series([False] * len(series), index=series.index), {
            "threshold": threshold,
            "mean": round(mean, 4),
            "std": 0.0,
            "note": "zero_variance_no_outliers"
        }
    
    z_scores = np.abs((series - mean) / std)
    outlier_mask = z_scores > threshold
    
    parameters = {
        "threshold": threshold,
        "mean": round(mean, 4),
        "std": round(std, 4),
        "max_z_score": round(z_scores.max(), 4)
    }
    
    return outlier_mask, parameters
