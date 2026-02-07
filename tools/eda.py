import pandas as pd
import numpy as np
from typing import Dict, Any, List, Union, Optional
from scipy import stats
from sklearn.metrics import mutual_info_score
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

def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Calculate Cramér's V statistic for categorical-categorical association."""
    confusion_matrix = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    with np.errstate(divide='ignore', invalid='ignore'):
        phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
        rcorr = r - ((r-1)**2)/(n-1)
        kcorr = k - ((k-1)**2)/(n-1)
        return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

def correlation_ratio(categories: pd.Series, measurements: pd.Series) -> float:
    """Calculate Correlation Ratio (eta) for categorical-numerical association."""
    fcat, _ = pd.factorize(categories)
    cat_num = np.max(fcat) + 1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0, cat_num):
        cat_measures = measurements[fcat == i]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.mean(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array, n_array)) / np.sum(n_array)
    numerator = np.sum(np.multiply(n_array, np.power(np.subtract(y_avg_array, y_total_avg), 2)))
    denominator = np.sum(np.power(np.subtract(measurements, y_total_avg), 2))
    if numerator == 0:
        return 0.0
    return np.sqrt(numerator / denominator)

def correlation_analysis(dataset_name: str, method: str = 'pearson') -> Dict[str, Any]:
    """
    Analyze correlations between features in the dataset.
    Supports Num-Num (Pearson/Spearman/Kendall), Cat-Cat (Cramér's V/MI), and Num-Cat (Eta/ANOVA).
    """
    manager = GlobalStateManager()
    if manager.get_dataset_name() != dataset_name:
        try:
            load_dataset_metadata(dataset_name)
        except Exception as e:
            return {"error": f"Failed to load dataset: {str(e)}"}
            
    df = manager.get_data()
    if df is None:
        return {"error": "Dataset loaded but DataFrame is None."}

    # Prepare outputs
    results = {
        "dataset_name": dataset_name,
        "numerical_correlations": [],
        "categorical_correlations": [],
        "numerical_categorical_correlations": []
    }
    
    # 1. Numerical vs Numerical
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 1:
        corr_matrix = df[num_cols].corr(method=method)
        # Iterate over triangle to avoid duplicates
        for i in range(len(num_cols)):
            for j in range(i+1, len(num_cols)):
                col1, col2 = num_cols[i], num_cols[j]
                val = corr_matrix.loc[col1, col2]
                sample_n = df[[col1, col2]].dropna().shape[0]
                
                results["numerical_correlations"].append({
                    "feature_1": col1,
                    "feature_2": col2,
                    "value": round(val, 4),
                    "abs_value": round(abs(val), 4),
                    "sample_size": int(sample_n),
                    "method": method
                })

    # 2. Categorical vs Categorical (Cramér's V & MI)
    # Filter for low cardinality (< 20 unique) to avoid ID columns
    cat_cols = [c for c in df.select_dtypes(include=['object', 'category', 'bool']).columns 
                if df[c].nunique() < 20 and df[c].nunique() > 1]
    
    if len(cat_cols) > 1:
        for i in range(len(cat_cols)):
            for j in range(i+1, len(cat_cols)):
                col1, col2 = cat_cols[i], cat_cols[j]
                # Drop NAs for calculation
                clean_df = df[[col1, col2]].dropna()
                if clean_df.empty: continue
                
                cramers = cramers_v(clean_df[col1], clean_df[col2])
                mi = mutual_info_score(clean_df[col1], clean_df[col2])
                
                results["categorical_correlations"].append({
                    "feature_1": col1,
                    "feature_2": col2,
                    "value": round(cramers, 4),
                    "abs_value": round(abs(cramers), 4),
                    "sample_size": int(len(clean_df)),
                    "method": "cramers_v",
                    "mutual_info": round(mi, 4)
                })

    # 3. Numerical vs Categorical (Correlation Ratio & ANOVA)
    if len(num_cols) > 0 and len(cat_cols) > 0:
        for num_col in num_cols:
            for cat_col in cat_cols:
                clean_df = df[[num_col, cat_col]].dropna()
                if clean_df.empty: continue
                
                # Correlation Ratio (Eta)
                eta = correlation_ratio(clean_df[cat_col], clean_df[num_col])
                
                # ANOVA F-Test
                groups = [group[num_col].values for name, group in clean_df.groupby(cat_col)]
                if len(groups) > 1:
                    f_stat, p_val = stats.f_oneway(*groups)
                else:
                    f_stat, p_val = 0, 1.0 # Cannot compare 1 group
                
                results["numerical_categorical_correlations"].append({
                    "feature_1": num_col,
                    "feature_2": cat_col,
                    "value": round(eta, 4),
                    "abs_value": round(abs(eta), 4),
                    "sample_size": int(len(clean_df)),
                    "method": "correlation_ratio",
                    "anova_f_stat": round(f_stat, 4) if not np.isnan(f_stat) else 0,
                    "anova_p_val": round(p_val, 6) if not np.isnan(p_val) else 1.0
                })
                
    return results
