"""
Data Quality Detection Tools for MCP Server.

This module provides automated detection of data quality issues including
missing values, outliers, high cardinality columns, duplicate rows,
and zero/near-zero variance columns.
It implements Phase 3 (Analysis) of the dataset analysis workflow.

Functions:
    detect_data_quality_issues: Main function to detect all quality issues
    
Helper Functions (Internal):
    _detect_outliers_adaptive: Adaptively select outlier detection method
    _detect_outliers_iqr: IQR-based outlier detection
    _detect_outliers_zscore: Z-score based outlier detection
"""

import pandas as pd
import numpy as np
from typing import Dict, Any

from config import (
    MISSING_VALUES_THRESHOLDS,
    OUTLIER_THRESHOLDS,
    HIGH_CARDINALITY_RATIO,
    HIGH_CARDINALITY_ABSOLUTE,
    DUPLICATE_ROWS_THRESHOLDS,
    MIN_OUTLIER_DETECTION_SAMPLES,
    SKEWNESS_THRESHOLD,
    KURTOSIS_THRESHOLD,
    NEAR_NORMAL_SKEWNESS,
    NEAR_NORMAL_KURTOSIS,
    DEFAULT_IQR_MULTIPLIER,
    DEFAULT_ZSCORE_THRESHOLD, MIN_SAMPLE_SIZE_STATS
)
from utils.state_manager import GlobalStateManager

def detect_data_quality_issues(dataset_name: str) -> Dict[str, Any]:
    """
    Automatically detect data quality problems in a dataset.
    
    Detects:
    - Missing values
    - Outliers (using adaptive method selection based on distribution)
    - High cardinality columns
    - Duplicate rows
    - Zero-variance columns (constant value, no predictive signal)
    - Near-zero-variance columns (>99% dominant value)
    
    Args:
        dataset_name: Name of the dataset file (e.g., 'data.csv').
        
    Returns:
        Dictionary containing array of detected issues with type, column, severity, method, and parameters.
    """
    manager = GlobalStateManager()
    
    if manager.get_dataset_name() != dataset_name:
        return {
            "error": f"Dataset '{dataset_name}' is not currently loaded. "
                     "Call load_dataset_metadata() explicitly to load it first."
        }
            
    df = manager.get_data()
    if df is None:
        return {"error": "Dataset loaded but DataFrame is None."}
    
    # Log the check action (Analysis step)
    manager.log_action("detect_data_quality_issues", {
        "dataset_name": dataset_name,
        "rows_checked": len(df),
        "columns_checked": len(df.columns)
    })

    issues = []
    total_rows = len(df)
    
    # 1. DETECT MISSING VALUES (all columns)
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            missing_pct = (missing_count / total_rows) * 100
            
            if missing_pct < MISSING_VALUES_THRESHOLDS["low"]:
                severity = "low"
            elif missing_pct < MISSING_VALUES_THRESHOLDS["medium"]:
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
        series = df[col].dropna()
        
        if len(series) < MIN_OUTLIER_DETECTION_SAMPLES:
            continue
            
        skewness = series.skew()
        kurtosis = series.kurtosis()
        n_samples = len(series)
        
        method, outlier_mask, parameters = _detect_outliers_adaptive(
            series, skewness, kurtosis, n_samples
        )
        
        outlier_count = outlier_mask.sum()
        
        if outlier_count > 0:
            outlier_pct = (outlier_count / len(series)) * 100
            
            if outlier_pct < OUTLIER_THRESHOLDS["low"]:
                severity = "low"
            elif outlier_pct < OUTLIER_THRESHOLDS["medium"]:
                severity = "medium"
            else:
                severity = "high"
            
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
        is_numeric = col in numeric_cols
        
        if is_numeric:
            if unique_ratio > HIGH_CARDINALITY_RATIO["high"]:
                severity = "high"
            elif unique_ratio > HIGH_CARDINALITY_RATIO["medium"]:
                severity = "medium"
            elif unique_ratio > HIGH_CARDINALITY_RATIO["low"]:
                severity = "low"
            else:
                severity = None
        else:
            if unique_count > HIGH_CARDINALITY_ABSOLUTE["high"]:
                severity = "high"
            elif unique_count > HIGH_CARDINALITY_ABSOLUTE["medium"]:
                severity = "medium"
            elif unique_count > HIGH_CARDINALITY_ABSOLUTE["low"]:
                severity = "low"
            else:
                severity = None
        
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
        unique_duplicated_rows = df[df.duplicated(keep=False)].drop_duplicates().shape[0]
        
        if duplicate_count < DUPLICATE_ROWS_THRESHOLDS["low"]:
            severity = "low"
        elif duplicate_count < DUPLICATE_ROWS_THRESHOLDS["medium"]:
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
    # 5. DETECT ZERO / NEAR-ZERO VARIANCE COLUMNS
    for col in df.columns:
        n_unique = df[col].nunique(dropna=True)
        non_null_count = df[col].notna().sum()

        if non_null_count == 0:
            # Entirely null column — already caught by missing values check
            continue

        if n_unique <= 1:
            # Zero variance: constant column (or all-null which is caught above)
            constant_value = df[col].dropna().iloc[0] if non_null_count > 0 else None
            issues.append({
                "type": "zero_variance",
                "column": col,
                "severity": "high",
                "method": "Unique_count",
                "parameters": {
                    "unique_values": n_unique,
                    "constant_value": str(constant_value),
                    "non_null_count": int(non_null_count),
                    "total_rows": total_rows,
                    "recommendation": (
                        "Column has zero variance (constant value). "
                        "Consider dropping it — it provides no predictive signal."
                    ),
                },
            })
        elif n_unique == 2 and non_null_count > 0:
            # Check if one value dominates >99% (near-constant with two values)
            value_counts = df[col].value_counts(dropna=True)
            dominant_ratio = value_counts.iloc[0] / non_null_count
            if dominant_ratio > 0.99:
                issues.append({
                    "type": "near_zero_variance",
                    "column": col,
                    "severity": "medium",
                    "method": "Dominant_value_ratio",
                    "parameters": {
                        "unique_values": n_unique,
                        "dominant_value": str(value_counts.index[0]),
                        "dominant_ratio": round(dominant_ratio, 4),
                        "minority_count": int(value_counts.iloc[1]) if len(value_counts) > 1 else 0,
                        "total_rows": total_rows,
                        "recommendation": (
                            f"Column is near-constant ({round(dominant_ratio * 100, 1)}% "
                            f"one value). Low predictive power — consider dropping."
                        ),
                    },
                })
        else:
            # General near-zero-variance: check if top value dominates >99%
            if non_null_count > 0:
                value_counts = df[col].value_counts(dropna=True)
                dominant_ratio = value_counts.iloc[0] / non_null_count
                if dominant_ratio > 0.99:
                    issues.append({
                        "type": "near_zero_variance",
                        "column": col,
                        "severity": "medium",
                        "method": "Dominant_value_ratio",
                        "parameters": {
                            "unique_values": n_unique,
                            "dominant_value": str(value_counts.index[0]),
                            "dominant_ratio": round(dominant_ratio, 4),
                            "total_rows": total_rows,
                            "recommendation": (
                                f"Column is near-constant ({round(dominant_ratio * 100, 1)}% "
                                f"one value). Low predictive power — consider dropping."
                            ),
                        },
                    })

    return {"issues": issues}


def _detect_outliers_adaptive(series: pd.Series, skewness: float, kurtosis: float, n_samples: int):
    """
    Adaptively select and apply outlier detection method based on distribution characteristics.
    """
    abs_skew = abs(skewness)
    abs_kurt = abs(kurtosis)
    
    if n_samples < MIN_SAMPLE_SIZE_STATS:
        method = "IQR"
        outlier_mask, parameters = _detect_outliers_iqr(series)
        parameters["method_reason"] = "small_sample_size"
        
    elif abs_skew >= SKEWNESS_THRESHOLD or abs_kurt >= KURTOSIS_THRESHOLD:
        method = "IQR"
        outlier_mask, parameters = _detect_outliers_iqr(series)
        if abs_skew >= SKEWNESS_THRESHOLD:
            parameters["method_reason"] = "highly_skewed"
        else:
            parameters["method_reason"] = "heavy_tails"
            
    elif abs_skew < NEAR_NORMAL_SKEWNESS and abs_kurt < NEAR_NORMAL_KURTOSIS:
        method = "Z-score"
        outlier_mask, parameters = _detect_outliers_zscore(series)
        parameters["method_reason"] = "approximately_normal"
        
    else:
        method = "Both_intersection"
        iqr_mask, iqr_params = _detect_outliers_iqr(series)
        z_mask, z_params = _detect_outliers_zscore(series)
        
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


def _detect_outliers_iqr(series: pd.Series, multiplier: float = DEFAULT_IQR_MULTIPLIER):
    """
    Detect outliers using Interquartile Range (IQR) method.
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


def _detect_outliers_zscore(series: pd.Series, threshold: float = DEFAULT_ZSCORE_THRESHOLD):
    """
    Detect outliers using Z-score method.
    """
    mean = series.mean()
    std = series.std()
    
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