"""
Handle Missing Values Tool for Dataset Analysis MCP Server.

This module provides comprehensive missing data imputation capabilities with
11 distinct strategies: 5 basic (mean, median, mode, drop_rows, constant) and
6 advanced (knn, iterative, forward_fill, backward_fill, interpolate, random_sample).

The tool follows the established pattern:
    load → validate → transform → update state → log action

Key features:
- Train/test split consistency (prevents data leakage)
- Type preservation and validation
- Edge case handling (all-NaN, constant columns, etc.)
- Immutable DataFrame operations
- Comprehensive statistics reporting
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Literal
from pydantic import BaseModel, Field
from sklearn.impute import KNNImputer
# IterativeImputer is experimental and requires explicit import
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

from utils.state_manager import GlobalStateManager
from tools.discovery import load_dataset_metadata


def _convert_to_native_types(value):
    """
    Convert numpy types to Python native types for JSON serialization.
    
    Args:
        value: Value to convert (can be numpy type or Python type)
    
    Returns:
        Python native type (int, float, str, etc.)
    """
    if isinstance(value, (np.integer, np.int64, np.int32)):
        return int(value)
    elif isinstance(value, (np.floating, np.float64, np.float32)):
        return float(value)
    elif isinstance(value, np.bool_):
        return bool(value)
    elif isinstance(value, np.ndarray):
        return value.tolist()
    else:
        return value


class HandleMissingValuesRequest(BaseModel):
    """Request model for handle_missing_values tool."""
    
    dataset_name: str = Field(
        ..., 
        description="Name of the dataset file (e.g., 'data.csv')"
    )
    column: str = Field(
        ..., 
        description="Column name to process for missing values"
    )
    strategy: Literal[
        "mean", "median", "mode", "drop_rows", "constant",  # Basic strategies
        "knn", "iterative", "forward_fill", "backward_fill", "interpolate", "random_sample"  # Advanced strategies
    ] = Field(
        ..., 
        description="Imputation strategy to apply"
    )
    constant_value: Optional[Any] = Field(
        None, 
        description="Value to use when strategy is 'constant' (required for constant strategy)"
    )
    n_neighbors: Optional[int] = Field(
        5, 
        description="Number of neighbors for KNN imputation (default: 5)"
    )
    max_iter: Optional[int] = Field(
        10, 
        description="Maximum iterations for iterative imputation (default: 10)"
    )
    random_state: Optional[int] = Field(
        42, 
        description="Random seed for reproducible random_sample strategy (default: 42)"
    )


def handle_missing_values(request: HandleMissingValuesRequest) -> Dict[str, Any]:
    """
    Handle missing values in a dataset column using specified strategy.
    
    This function implements comprehensive missing value imputation with support for
    both basic statistical methods and advanced ML-based approaches. It maintains
    train/test split consistency to prevent data leakage.
    
    Args:
        request: HandleMissingValuesRequest containing:
            - dataset_name: Name of the dataset file
            - column: Column to process
            - strategy: Imputation strategy (mean, median, mode, drop_rows, constant,
                       knn, iterative, forward_fill, backward_fill, interpolate, random_sample)
            - constant_value: Value for 'constant' strategy (required if strategy='constant')
            - n_neighbors: Number of neighbors for KNN (default: 5)
            - max_iter: Max iterations for iterative imputation (default: 10)
            - random_state: Random seed for random_sample (default: 42)
    
    Returns:
        Dictionary containing:
            - column: Column that was processed
            - rows_affected: Number of rows imputed or removed
            - strategy: Strategy that was applied
            - imputation_value: Value used (for mean/median/mode/constant)
            - rows_removed: Number of rows removed (for drop_rows)
            - remaining_rows: Rows remaining after drop (for drop_rows)
            - stats: Detailed statistics about the operation
    
    Raises:
        ValueError: For invalid inputs, incompatible strategies, or edge cases
        FileNotFoundError: If dataset file doesn't exist
    
    Examples:
        >>> # Mean imputation
        >>> request = HandleMissingValuesRequest(
        ...     dataset_name="data.csv",
        ...     column="age",
        ...     strategy="mean"
        ... )
        >>> result = handle_missing_values(request)
        
        >>> # KNN imputation with custom neighbors
        >>> request = HandleMissingValuesRequest(
        ...     dataset_name="data.csv",
        ...     column="salary",
        ...     strategy="knn",
        ...     n_neighbors=10
        ... )
        >>> result = handle_missing_values(request)
    """
    manager = GlobalStateManager()
    
    # =========================================================================
    # 1. Dataset Loading and Validation (Requirements 5.2, 5.3, 5.5)
    # =========================================================================
    
    # Check if dataset is already loaded, load if not
    if manager.get_dataset_name() != request.dataset_name:
        try:
            load_dataset_metadata(request.dataset_name)
        except FileNotFoundError as e:
            return {
                "error": f"Dataset '{request.dataset_name}' not found: {str(e)}"
            }
        except Exception as e:
            return {
                "error": f"Failed to load dataset '{request.dataset_name}': {str(e)}"
            }
    
    # Get the DataFrame from state
    df = manager.get_data()
    
    # Validate DataFrame is not None (Requirement 5.5)
    if df is None:
        return {
            "error": f"Dataset '{request.dataset_name}' loaded but DataFrame is None"
        }
    
    # =========================================================================
    # 2. Column Existence Validation (Requirement 5.1)
    # =========================================================================
    
    if request.column not in df.columns:
        return {
            "error": f"Column '{request.column}' not found in dataset. "
                    f"Available columns: {list(df.columns)}"
        }
    
    # =========================================================================
    # 3. Strategy-Dtype Compatibility Validation (Requirements 2.1-2.11)
    # =========================================================================
    
    error_msg = _validate_strategy_compatibility(df[request.column], request.strategy)
    if error_msg:
        return {"error": error_msg}
    
    # =========================================================================
    # 4. Constant Value Validation (Requirement 3.5)
    # =========================================================================
    
    if request.strategy == "constant" and request.constant_value is None:
        return {
            "error": "constant_value parameter is required when strategy is 'constant'"
        }
    
    # =========================================================================
    # 5. Check for No Missing Values (Requirement 3.4)
    # =========================================================================
    
    initial_missing_count = _convert_to_native_types(df[request.column].isna().sum())
    if initial_missing_count == 0:
        return {
            "column": request.column,
            "rows_affected": 0,
            "strategy": request.strategy,
            "message": f"No missing values found in column '{request.column}'"
        }
    
    # =========================================================================
    # 6. Apply KNN Imputation Strategy (Requirement 1A.1)
    # =========================================================================
    
    if request.strategy == "knn":
        # Check if dataset has train/test split
        is_split = manager.is_split()
        
        try:
            if is_split:
                # For split datasets: fit on training data, transform both sets
                train_df = manager.get_data()
                test_df = manager.get_test_data()
                
                # Apply KNN imputation to training set (fit and transform)
                train_transformed, train_affected = _apply_knn_imputation(
                    train_df, 
                    request.column, 
                    request.n_neighbors
                )
                
                # For test set, we need to fit on training and transform test
                # Extract numeric columns
                numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
                
                if len(numeric_cols) < 2:
                    return {
                        "error": f"KNN imputation requires at least 2 numeric columns, "
                                f"but only found {len(numeric_cols)}"
                    }
                
                # Fit imputer on training data
                imputer = KNNImputer(n_neighbors=request.n_neighbors, weights='uniform')
                imputer.fit(train_df[numeric_cols])
                
                # Transform test data
                test_numeric = test_df[numeric_cols].copy()
                test_imputed_values = imputer.transform(test_numeric)
                test_imputed = pd.DataFrame(
                    test_imputed_values,
                    columns=numeric_cols,
                    index=test_df.index
                )
                
                # Update test DataFrame with imputed column
                test_transformed = test_df.copy()
                test_affected = _convert_to_native_types(test_transformed[request.column].isna().sum())
                test_transformed[request.column] = test_imputed[request.column]
                
                # Update state with both transformed datasets
                manager.load_data(train_transformed, request.dataset_name, reset_split=False)
                manager.update_test_data(test_transformed)
                
                total_affected = train_affected + test_affected
                
                # Return statistics
                return {
                    "column": request.column,
                    "rows_affected": _convert_to_native_types(total_affected),
                    "strategy": request.strategy,
                    "n_neighbors": request.n_neighbors,
                    "stats": {
                        "initial_missing_count": _convert_to_native_types(initial_missing_count),
                        "train_missing_count": _convert_to_native_types(train_affected),
                        "test_missing_count": _convert_to_native_types(test_affected),
                        "train_rows_affected": _convert_to_native_types(train_affected),
                        "test_rows_affected": _convert_to_native_types(test_affected),
                        "column_dtype": str(df[request.column].dtype),
                        "numeric_features_used": _convert_to_native_types(len(numeric_cols))
                    }
                }
            else:
                # For non-split datasets: simple fit and transform
                df_transformed, rows_affected = _apply_knn_imputation(
                    df, 
                    request.column, 
                    request.n_neighbors
                )
                
                # Update state
                manager.load_data(df_transformed, request.dataset_name)
                
                # Get numeric columns count for stats
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                
                # Return statistics
                return {
                    "column": request.column,
                    "rows_affected": _convert_to_native_types(rows_affected),
                    "strategy": request.strategy,
                    "n_neighbors": request.n_neighbors,
                    "stats": {
                        "initial_missing_count": _convert_to_native_types(initial_missing_count),
                        "column_dtype": str(df[request.column].dtype),
                        "numeric_features_used": _convert_to_native_types(len(numeric_cols))
                    }
                }
        
        except ValueError as e:
            return {"error": str(e)}
    
    # =========================================================================
    # 7. Apply Iterative (MICE) Imputation Strategy (Requirement 1A.2)
    # =========================================================================
    
    if request.strategy == "iterative":
        # Check if dataset has train/test split
        is_split = manager.is_split()
        
        try:
            if is_split:
                # For split datasets: fit on training data, transform both sets
                train_df = manager.get_data()
                test_df = manager.get_test_data()
                
                # Apply iterative imputation to training set (fit and transform)
                train_transformed, train_affected = _apply_iterative_imputation(
                    train_df, 
                    request.column, 
                    request.max_iter,
                    request.random_state
                )
                
                # For test set, we need to fit on training and transform test
                # Extract numeric columns
                numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
                
                if len(numeric_cols) < 2:
                    return {
                        "error": f"Iterative imputation requires at least 2 numeric columns, "
                                f"but only found {len(numeric_cols)}"
                    }
                
                # Fit imputer on training data
                imputer = IterativeImputer(
                    max_iter=request.max_iter,
                    random_state=request.random_state,
                    initial_strategy='mean'
                )
                imputer.fit(train_df[numeric_cols])
                
                # Transform test data
                test_numeric = test_df[numeric_cols].copy()
                test_imputed_values = imputer.transform(test_numeric)
                test_imputed = pd.DataFrame(
                    test_imputed_values,
                    columns=numeric_cols,
                    index=test_df.index
                )
                
                # Update test DataFrame with imputed column
                test_transformed = test_df.copy()
                test_affected = _convert_to_native_types(test_transformed[request.column].isna().sum())
                test_transformed[request.column] = test_imputed[request.column]
                
                # Update state with both transformed datasets
                manager.load_data(train_transformed, request.dataset_name, reset_split=False)
                manager.update_test_data(test_transformed)
                
                total_affected = train_affected + test_affected
                
                # Return statistics
                return {
                    "column": request.column,
                    "rows_affected": _convert_to_native_types(total_affected),
                    "strategy": request.strategy,
                    "max_iter": request.max_iter,
                    "random_state": request.random_state,
                    "stats": {
                        "initial_missing_count": _convert_to_native_types(initial_missing_count),
                        "train_missing_count": _convert_to_native_types(train_affected),
                        "test_missing_count": _convert_to_native_types(test_affected),
                        "train_rows_affected": _convert_to_native_types(train_affected),
                        "test_rows_affected": _convert_to_native_types(test_affected),
                        "column_dtype": str(df[request.column].dtype),
                        "numeric_features_used": _convert_to_native_types(len(numeric_cols))
                    }
                }
            else:
                # For non-split datasets: simple fit and transform
                df_transformed, rows_affected = _apply_iterative_imputation(
                    df, 
                    request.column, 
                    request.max_iter,
                    request.random_state
                )
                
                # Update state
                manager.load_data(df_transformed, request.dataset_name)
                
                # Get numeric columns count for stats
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                
                # Return statistics
                return {
                    "column": request.column,
                    "rows_affected": _convert_to_native_types(rows_affected),
                    "strategy": request.strategy,
                    "max_iter": request.max_iter,
                    "random_state": request.random_state,
                    "stats": {
                        "initial_missing_count": _convert_to_native_types(initial_missing_count),
                        "column_dtype": str(df[request.column].dtype),
                        "numeric_features_used": _convert_to_native_types(len(numeric_cols))
                    }
                }
        
        except ValueError as e:
            return {"error": str(e)}
    
    # =========================================================================
    # 8. Apply Forward Fill Strategy (Requirement 1A.3)
    # =========================================================================
    
    if request.strategy == "forward_fill":
        # Check if dataset has train/test split
        is_split = manager.is_split()
        
        if is_split:
            # For split datasets: apply forward fill independently to each set
            train_df = manager.get_data()
            test_df = manager.get_test_data()
            
            # Apply forward fill to training set
            train_transformed, train_affected = _apply_forward_fill(
                train_df, 
                request.column
            )
            
            # Apply forward fill to test set
            test_transformed, test_affected = _apply_forward_fill(
                test_df, 
                request.column
            )
            
            # Update state with both transformed datasets
            manager.load_data(train_transformed, request.dataset_name, reset_split=False)
            manager.update_test_data(test_transformed)
            
            total_affected = train_affected + test_affected
            
            # Return statistics
            return {
                "column": request.column,
                "rows_affected": _convert_to_native_types(total_affected),
                "strategy": request.strategy,
                "stats": {
                    "initial_missing_count": _convert_to_native_types(initial_missing_count),
                    "train_missing_count": _convert_to_native_types(train_df[request.column].isna().sum()),
                    "test_missing_count": _convert_to_native_types(test_df[request.column].isna().sum()),
                    "train_rows_affected": _convert_to_native_types(train_affected),
                    "test_rows_affected": _convert_to_native_types(test_affected),
                    "column_dtype": str(df[request.column].dtype)
                }
            }
        else:
            # For non-split datasets: simple forward fill
            df_transformed, rows_affected = _apply_forward_fill(
                df, 
                request.column
            )
            
            # Update state
            manager.load_data(df_transformed, request.dataset_name)
            
            # Return statistics
            return {
                "column": request.column,
                "rows_affected": _convert_to_native_types(rows_affected),
                "strategy": request.strategy,
                "stats": {
                    "initial_missing_count": _convert_to_native_types(initial_missing_count),
                    "column_dtype": str(df[request.column].dtype)
                }
            }
    
    # =========================================================================
    # 9. Apply Backward Fill Strategy (Requirement 1A.4)
    # =========================================================================
    
    if request.strategy == "backward_fill":
        # Check if dataset has train/test split
        is_split = manager.is_split()
        
        if is_split:
            # For split datasets: apply backward fill independently to each set
            train_df = manager.get_data()
            test_df = manager.get_test_data()
            
            # Apply backward fill to training set
            train_transformed, train_affected = _apply_backward_fill(
                train_df, 
                request.column
            )
            
            # Apply backward fill to test set
            test_transformed, test_affected = _apply_backward_fill(
                test_df, 
                request.column
            )
            
            # Update state with both transformed datasets
            manager.load_data(train_transformed, request.dataset_name, reset_split=False)
            manager.update_test_data(test_transformed)
            
            total_affected = train_affected + test_affected
            
            # Return statistics
            return {
                "column": request.column,
                "rows_affected": _convert_to_native_types(total_affected),
                "strategy": request.strategy,
                "stats": {
                    "initial_missing_count": _convert_to_native_types(initial_missing_count),
                    "train_missing_count": _convert_to_native_types(train_df[request.column].isna().sum()),
                    "test_missing_count": _convert_to_native_types(test_df[request.column].isna().sum()),
                    "train_rows_affected": _convert_to_native_types(train_affected),
                    "test_rows_affected": _convert_to_native_types(test_affected),
                    "column_dtype": str(df[request.column].dtype)
                }
            }
        else:
            # For non-split datasets: simple backward fill
            df_transformed, rows_affected = _apply_backward_fill(
                df, 
                request.column
            )
            
            # Update state
            manager.load_data(df_transformed, request.dataset_name)
            
            # Return statistics
            return {
                "column": request.column,
                "rows_affected": _convert_to_native_types(rows_affected),
                "strategy": request.strategy,
                "stats": {
                    "initial_missing_count": _convert_to_native_types(initial_missing_count),
                    "column_dtype": str(df[request.column].dtype)
                }
            }
    
    # =========================================================================
    # 10. Apply Interpolate Strategy (Requirement 1A.5)
    # =========================================================================
    
    if request.strategy == "interpolate":
        # Check if dataset has train/test split
        is_split = manager.is_split()
        
        try:
            if is_split:
                # For split datasets: apply interpolate independently to each set
                train_df = manager.get_data()
                test_df = manager.get_test_data()
                
                # Apply interpolate to training set
                train_transformed, train_affected = _apply_interpolate(
                    train_df, 
                    request.column
                )
                
                # Apply interpolate to test set
                test_transformed, test_affected = _apply_interpolate(
                    test_df, 
                    request.column
                )
                
                # Update state with both transformed datasets
                manager.load_data(train_transformed, request.dataset_name, reset_split=False)
                manager.update_test_data(test_transformed)
                
                total_affected = train_affected + test_affected
                
                # Return statistics
                return {
                    "column": request.column,
                    "rows_affected": _convert_to_native_types(total_affected),
                    "strategy": request.strategy,
                    "stats": {
                        "initial_missing_count": _convert_to_native_types(initial_missing_count),
                        "train_missing_count": _convert_to_native_types(train_df[request.column].isna().sum()),
                        "test_missing_count": _convert_to_native_types(test_df[request.column].isna().sum()),
                        "train_rows_affected": _convert_to_native_types(train_affected),
                        "test_rows_affected": _convert_to_native_types(test_affected),
                        "column_dtype": str(df[request.column].dtype)
                    }
                }
            else:
                # For non-split datasets: simple interpolate
                df_transformed, rows_affected = _apply_interpolate(
                    df, 
                    request.column
                )
                
                # Update state
                manager.load_data(df_transformed, request.dataset_name)
                
                # Return statistics
                return {
                    "column": request.column,
                    "rows_affected": _convert_to_native_types(rows_affected),
                    "strategy": request.strategy,
                    "stats": {
                        "initial_missing_count": _convert_to_native_types(initial_missing_count),
                        "column_dtype": str(df[request.column].dtype)
                    }
                }
        
        except ValueError as e:
            return {"error": str(e)}
    
    # =========================================================================
    # 11. Apply Random Sample Strategy (Requirement 1A.6)
    # =========================================================================
    
    if request.strategy == "random_sample":
        # Check if dataset has train/test split
        is_split = manager.is_split()
        
        try:
            if is_split:
                # For split datasets: apply random_sample independently to each set
                # Each set gets its own random samples from its own valid values
                train_df = manager.get_data()
                test_df = manager.get_test_data()
                
                # Apply random_sample to training set
                train_transformed, train_affected = _apply_random_sample(
                    train_df, 
                    request.column,
                    request.random_state
                )
                
                # Apply random_sample to test set
                # Use different seed for test set to ensure independence
                test_transformed, test_affected = _apply_random_sample(
                    test_df, 
                    request.column,
                    request.random_state + 1  # Different seed for test set
                )
                
                # Update state with both transformed datasets
                manager.load_data(train_transformed, request.dataset_name, reset_split=False)
                manager.update_test_data(test_transformed)
                
                total_affected = train_affected + test_affected
                
                # Return statistics
                return {
                    "column": request.column,
                    "rows_affected": _convert_to_native_types(total_affected),
                    "strategy": request.strategy,
                    "random_state": request.random_state,
                    "stats": {
                        "initial_missing_count": _convert_to_native_types(initial_missing_count),
                        "train_missing_count": _convert_to_native_types(train_df[request.column].isna().sum()),
                        "test_missing_count": _convert_to_native_types(test_df[request.column].isna().sum()),
                        "train_rows_affected": _convert_to_native_types(train_affected),
                        "test_rows_affected": _convert_to_native_types(test_affected),
                        "column_dtype": str(df[request.column].dtype)
                    }
                }
            else:
                # For non-split datasets: simple random sample
                df_transformed, rows_affected = _apply_random_sample(
                    df, 
                    request.column,
                    request.random_state
                )
                
                # Update state
                manager.load_data(df_transformed, request.dataset_name)
                
                # Return statistics
                return {
                    "column": request.column,
                    "rows_affected": _convert_to_native_types(rows_affected),
                    "strategy": request.strategy,
                    "random_state": request.random_state,
                    "stats": {
                        "initial_missing_count": _convert_to_native_types(initial_missing_count),
                        "column_dtype": str(df[request.column].dtype)
                    }
                }
        
        except ValueError as e:
            return {"error": str(e)}
    
    # =========================================================================
    # 12. Basic Strategies: mean, median, mode, drop_rows, constant
    # =========================================================================
    
    # For basic strategies, we need to:
    # 1. Calculate imputation value (for mean/median/mode/constant)
    # 2. Apply transformation using _apply_imputation
    # 3. Update GlobalStateManager with transformed DataFrame
    # 4. Reset index with drop=True before updating state
    # 5. Return statistics
    
    try:
        # Calculate imputation value for basic strategies
        # For drop_rows, imputation_value is not used
        if request.strategy in ["mean", "median", "mode", "constant"]:
            imputation_value = _calculate_imputation_value(
                df[request.column],
                request.strategy,
                request.constant_value
            )
        else:
            imputation_value = None
        
        # Apply transformation
        df_transformed, rows_affected = _apply_imputation(
            df,
            request.column,
            imputation_value,
            request.strategy
        )
        
        # Reset index with drop=True to ensure clean row numbering (Requirement 4.4)
        df_transformed = df_transformed.reset_index(drop=True)
        
        # Update GlobalStateManager with transformed DataFrame (Requirement 4.1)
        manager.load_data(df_transformed, request.dataset_name)
        
        # Build output statistics
        result = {
            "column": request.column,
            "rows_affected": _convert_to_native_types(rows_affected),
            "strategy": request.strategy,
            "stats": {
                "initial_missing_count": _convert_to_native_types(initial_missing_count),
                "column_dtype": str(df[request.column].dtype)
            }
        }
        
        # Add strategy-specific statistics
        if request.strategy in ["mean", "median", "mode", "constant"]:
            result["imputation_value"] = _convert_to_native_types(imputation_value)
        
        if request.strategy == "drop_rows":
            result["rows_removed"] = _convert_to_native_types(rows_affected)
            result["remaining_rows"] = _convert_to_native_types(len(df_transformed))
        
        return result
    
    except ValueError as e:
        return {"error": str(e)}
    except TypeError as e:
        return {"error": str(e)}


def _validate_strategy_compatibility(
    column: pd.Series, 
    strategy: str
) -> Optional[str]:
    """
    Validate that the imputation strategy is compatible with the column data type.
    
    Numeric-only strategies: mean, median, knn, iterative, interpolate
    Universal strategies: mode, constant, drop_rows, forward_fill, backward_fill, random_sample
    
    Args:
        column: The pandas Series to validate
        strategy: The imputation strategy to check
    
    Returns:
        None if valid, error message string if incompatible
    
    Validates:
        - Requirements 2.1, 2.2: mean/median require numeric columns
        - Requirements 2.3, 2.4, 2.5: knn/iterative/interpolate require numeric columns
        - Requirements 2.6-2.11: mode/constant/drop_rows/forward_fill/backward_fill/random_sample work on any type
    """
    # Define numeric-only strategies
    numeric_only_strategies = {"mean", "median", "knn", "iterative", "interpolate"}
    
    # Check if strategy requires numeric data
    if strategy in numeric_only_strategies:
        if not pd.api.types.is_numeric_dtype(column):
            return (
                f"Strategy '{strategy}' requires a numeric column, "
                f"but column has dtype '{column.dtype}'. "
                f"Use mode, constant, drop_rows, forward_fill, backward_fill, or random_sample for non-numeric columns."
            )
    
    # Universal strategies (mode, constant, drop_rows, forward_fill, backward_fill, random_sample) 
    # work on any column type - no validation needed
    
    return None


def _calculate_imputation_value(
    series: pd.Series,
    strategy: str,
    constant_value: Any = None
) -> Any:
    """
    Calculate the imputation value based on the specified strategy.
    
    This function computes the value to use for imputing missing data in a series
    based on basic statistical strategies (mean, median, mode, constant).
    
    Args:
        series: The pandas Series to calculate imputation value from
        strategy: The imputation strategy ('mean', 'median', 'mode', 'constant')
        constant_value: The constant value to use (required for 'constant' strategy)
    
    Returns:
        The calculated imputation value (numeric for mean/median, any type for mode/constant)
    
    Raises:
        ValueError: If all values are NaN (for mean/median/mode) or if constant_value
                   is missing for 'constant' strategy
    
    Validates:
        - Requirements 1.1: Mean calculation with skipna=True
        - Requirements 1.2: Median calculation with skipna=True
        - Requirements 1.3: Mode calculation, uses first mode if multiple
        - Requirements 1.5: Constant strategy validation
        - Requirements 3.1: All-NaN edge case handling
        - Requirements 3.5: Missing constant_value validation
    
    Examples:
        >>> series = pd.Series([1, 2, 3, np.nan, 5])
        >>> _calculate_imputation_value(series, 'mean')
        2.75
        
        >>> series = pd.Series(['a', 'b', 'a', np.nan])
        >>> _calculate_imputation_value(series, 'mode')
        'a'
        
        >>> _calculate_imputation_value(series, 'constant', constant_value=0)
        0
    """
    # =========================================================================
    # Mean Strategy (Requirement 1.1)
    # =========================================================================
    if strategy == "mean":
        # Calculate mean from valid (non-NaN) values using skipna=True
        mean_value = series.mean(skipna=True)
        
        # Check for all-NaN case (Requirement 3.1)
        if pd.isna(mean_value):
            raise ValueError(
                f"Cannot calculate mean: all values in the series are missing"
            )
        
        return mean_value
    
    # =========================================================================
    # Median Strategy (Requirement 1.2)
    # =========================================================================
    elif strategy == "median":
        # Calculate median from valid values using skipna=True
        median_value = series.median(skipna=True)
        
        # Check for all-NaN case (Requirement 3.1)
        if pd.isna(median_value):
            raise ValueError(
                f"Cannot calculate median: all values in the series are missing"
            )
        
        return median_value
    
    # =========================================================================
    # Mode Strategy (Requirement 1.3)
    # =========================================================================
    elif strategy == "mode":
        # Calculate mode (most frequent value)
        # pandas.Series.mode() returns a Series of mode values
        mode_result = series.mode()
        
        # Check for all-NaN case (Requirement 3.1)
        if len(mode_result) == 0:
            raise ValueError(
                f"Cannot calculate mode: all values in the series are missing"
            )
        
        # Use first mode if multiple exist (Requirement 1.3, pandas default behavior)
        mode_value = mode_result.iloc[0]
        
        return mode_value
    
    # =========================================================================
    # Constant Strategy (Requirement 1.5)
    # =========================================================================
    elif strategy == "constant":
        # Validate constant_value is provided (Requirement 3.5)
        if constant_value is None:
            raise ValueError(
                f"constant_value parameter is required when strategy is 'constant'"
            )
        
        return constant_value
    
    # =========================================================================
    # Invalid Strategy (should not reach here due to Pydantic validation)
    # =========================================================================
    else:
        raise ValueError(
            f"Strategy '{strategy}' is not supported by _calculate_imputation_value. "
            f"Use 'mean', 'median', 'mode', or 'constant'."
        )


def _apply_imputation(
    df: pd.DataFrame,
    column: str,
    imputation_value: Any,
    strategy: str
) -> tuple[pd.DataFrame, int]:
    """
    Apply imputation to a DataFrame column using the specified strategy.
    
    This function handles the actual transformation of the DataFrame by applying
    the imputation strategy. For basic strategies (mean, median, mode, constant),
    it uses fillna() with the provided imputation_value. For drop_rows, it filters
    out rows with missing values. For advanced strategies, it delegates to
    strategy-specific implementations.
    
    The function follows an immutable pattern by creating a copy of the DataFrame
    before modification, ensuring the original data is not altered.
    
    Args:
        df: The DataFrame to impute (will not be modified)
        column: The target column to impute missing values in
        imputation_value: The value to use for imputation (for basic strategies)
        strategy: The imputation strategy to apply
    
    Returns:
        Tuple of (transformed_df, rows_affected) where:
            - transformed_df: New DataFrame with imputed values or rows removed
            - rows_affected: Number of missing values filled or rows removed
    
    Validates:
        - Requirements 1.1, 1.2, 1.3, 1.5: Basic strategies use fillna()
        - Requirement 1.4: drop_rows filters rows and resets index
        - Requirements 1A.1-1A.6: Advanced strategies delegate to specific implementations
        - Requirement 9.2: Immutable pattern with copy-on-write
    
    Examples:
        >>> df = pd.DataFrame({'age': [25, np.nan, 35, np.nan]})
        >>> result_df, affected = _apply_imputation(df, 'age', 30.0, 'mean')
        >>> result_df['age'].tolist()
        [25.0, 30.0, 35.0, 30.0]
        >>> affected
        2
        
        >>> df = pd.DataFrame({'status': ['active', np.nan, 'inactive']})
        >>> result_df, affected = _apply_imputation(df, 'status', None, 'drop_rows')
        >>> len(result_df)
        2
        >>> affected
        1
    """
    # Create a copy to avoid modifying the original DataFrame (Requirement 9.2)
    df_copy = df.copy()
    
    # Count missing values before imputation
    missing_count = _convert_to_native_types(df_copy[column].isna().sum())
    
    # =========================================================================
    # Basic Strategies: mean, median, mode, constant (Requirements 1.1-1.3, 1.5)
    # =========================================================================
    if strategy in ["mean", "median", "mode", "constant"]:
        # Use fillna() to replace missing values with the imputation value
        df_copy[column] = df_copy[column].fillna(imputation_value)
        
        # Count how many values were filled
        rows_affected = missing_count
        
        return df_copy, rows_affected
    
    # =========================================================================
    # Drop Rows Strategy (Requirement 1.4)
    # =========================================================================
    elif strategy == "drop_rows":
        # Filter rows where column is not NaN
        # Use boolean indexing to keep only rows with non-missing values
        df_copy = df_copy[df_copy[column].notna()].copy()
        
        # Reset index with drop=True to avoid creating an extra index column
        df_copy = df_copy.reset_index(drop=True)
        
        # For drop_rows, rows_affected is the number of rows removed
        rows_affected = missing_count
        
        return df_copy, rows_affected
    
    # =========================================================================
    # Advanced Strategies: Delegate to strategy-specific implementations
    # =========================================================================
    # Note: These strategies don't use the imputation_value parameter
    # They calculate their own values or apply their own logic
    
    elif strategy == "knn":
        # Delegate to KNN imputation (Requirement 1A.1)
        # Note: This function expects n_neighbors parameter, but we don't have it here
        # This case should be handled in the main function, not here
        raise ValueError(
            f"Strategy '{strategy}' should be handled by _apply_knn_imputation, "
            f"not by _apply_imputation"
        )
    
    elif strategy == "iterative":
        # Delegate to iterative imputation (Requirement 1A.2)
        raise ValueError(
            f"Strategy '{strategy}' should be handled by _apply_iterative_imputation, "
            f"not by _apply_imputation"
        )
    
    elif strategy == "forward_fill":
        # Delegate to forward fill (Requirement 1A.3)
        raise ValueError(
            f"Strategy '{strategy}' should be handled by _apply_forward_fill, "
            f"not by _apply_imputation"
        )
    
    elif strategy == "backward_fill":
        # Delegate to backward fill (Requirement 1A.4)
        raise ValueError(
            f"Strategy '{strategy}' should be handled by _apply_backward_fill, "
            f"not by _apply_imputation"
        )
    
    elif strategy == "interpolate":
        # Delegate to interpolate (Requirement 1A.5)
        raise ValueError(
            f"Strategy '{strategy}' should be handled by _apply_interpolate, "
            f"not by _apply_imputation"
        )
    
    elif strategy == "random_sample":
        # Delegate to random sample (Requirement 1A.6)
        raise ValueError(
            f"Strategy '{strategy}' should be handled by _apply_random_sample, "
            f"not by _apply_imputation"
        )
    
    # =========================================================================
    # Invalid Strategy (should not reach here due to Pydantic validation)
    # =========================================================================
    else:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Supported strategies: "
            f"mean, median, mode, drop_rows, constant, knn, iterative, "
            f"forward_fill, backward_fill, interpolate, random_sample"
        )


def _coerce_constant_value(
    constant_value: Any,
    target_dtype: np.dtype
) -> Any:
    """
    Attempt to coerce a constant value to match the target column dtype.
    
    This function ensures type compatibility when using the 'constant' strategy
    for imputation. It attempts to convert the provided constant_value to match
    the dtype of the target column using pandas type coercion.
    
    Args:
        constant_value: The value to coerce (can be any type)
        target_dtype: The target numpy dtype to coerce to
    
    Returns:
        The coerced value matching the target dtype
    
    Raises:
        TypeError: If the value cannot be coerced to the target dtype
    
    Validates:
        - Requirement 8.5: Type coercion for constant values with error handling
    
    Examples:
        >>> _coerce_constant_value(0, np.dtype('int64'))
        0
        
        >>> _coerce_constant_value('5', np.dtype('int64'))
        5
        
        >>> _coerce_constant_value(3.14, np.dtype('float64'))
        3.14
        
        >>> _coerce_constant_value('hello', np.dtype('object'))
        'hello'
        
        >>> _coerce_constant_value('invalid', np.dtype('int64'))
        Traceback (most recent call last):
            ...
        TypeError: Cannot coerce constant_value 'invalid' to column dtype int64: ...
    """
    try:
        # Use pandas Series to perform type coercion
        # Create a single-element series with the constant value
        # Then attempt to convert it to the target dtype
        coerced_series = pd.Series([constant_value]).astype(target_dtype)
        
        # Extract the coerced value from the series
        coerced_value = coerced_series.iloc[0]
        
        return coerced_value
    
    except (ValueError, TypeError) as e:
        # Catch both ValueError and TypeError from pandas astype()
        # Raise a descriptive TypeError with context
        raise TypeError(
            f"Cannot coerce constant_value {repr(constant_value)} "
            f"to column dtype {target_dtype}: {str(e)}"
        )


def _apply_knn_imputation(
    df: pd.DataFrame,
    column: str,
    n_neighbors: int = 5
) -> tuple[pd.DataFrame, int]:
    """
    Apply KNN imputation to a numeric column using all numeric features as context.
    
    KNN imputation uses K-Nearest Neighbors to predict missing values based on
    similar rows. It requires multiple numeric columns to establish feature context.
    
    Args:
        df: The DataFrame to impute (will not be modified)
        column: The target column to impute missing values in
        n_neighbors: Number of neighbors to use for imputation (default: 5)
    
    Returns:
        Tuple of (transformed_df, rows_affected) where:
            - transformed_df: New DataFrame with imputed values
            - rows_affected: Number of missing values that were filled
    
    Raises:
        ValueError: If column is not numeric or if insufficient numeric columns exist
    
    Validates:
        - Requirement 1A.1: KNN imputation with n_neighbors parameter
        - Uses all numeric columns for feature context
        - Fits on provided data, transforms the same data
        - Handles edge case of insufficient numeric columns
    
    Examples:
        >>> df = pd.DataFrame({
        ...     'age': [25, 30, np.nan, 40],
        ...     'salary': [50000, 60000, 55000, np.nan],
        ...     'years_exp': [2, 5, 3, 8]
        ... })
        >>> result_df, affected = _apply_knn_imputation(df, 'age', n_neighbors=2)
        >>> result_df['age'].isna().sum()
        0
    """
    # Create a copy to avoid modifying the original DataFrame
    df_copy = df.copy()
    
    # Validate column is numeric
    if not pd.api.types.is_numeric_dtype(df_copy[column]):
        raise ValueError(
            f"KNN imputation requires a numeric column, but '{column}' has dtype '{df_copy[column].dtype}'"
        )
    
    # Extract all numeric columns for feature context
    numeric_cols = df_copy.select_dtypes(include=[np.number]).columns.tolist()
    
    # Check if we have enough numeric columns for KNN
    # Need at least 2 columns (target + at least one feature for context)
    if len(numeric_cols) < 2:
        raise ValueError(
            f"KNN imputation requires at least 2 numeric columns for feature context, "
            f"but only found {len(numeric_cols)} numeric column(s): {numeric_cols}. "
            f"Consider using mean, median, or mode strategies instead."
        )
    
    # Ensure target column is in numeric columns
    if column not in numeric_cols:
        raise ValueError(
            f"Target column '{column}' is not in the list of numeric columns"
        )
    
    # Count missing values before imputation
    missing_count = df_copy[column].isna().sum()
    
    # If no missing values, return early
    if missing_count == 0:
        return df_copy, 0
    
    # Create KNN imputer with specified number of neighbors
    # Use uniform weights (all neighbors contribute equally)
    imputer = KNNImputer(n_neighbors=n_neighbors, weights='uniform')
    
    # Extract numeric columns for imputation
    df_numeric = df_copy[numeric_cols].copy()
    
    # Fit and transform the numeric data
    # KNN will use all numeric columns as features to find nearest neighbors
    df_imputed_values = imputer.fit_transform(df_numeric)
    
    # Create DataFrame from imputed values
    df_imputed = pd.DataFrame(
        df_imputed_values,
        columns=numeric_cols,
        index=df_copy.index
    )
    
    # Update only the target column in the original DataFrame copy
    # This preserves non-numeric columns and only updates the imputed column
    df_copy[column] = df_imputed[column]
    
    # Return the transformed DataFrame and count of affected rows
    return df_copy, missing_count


def _apply_iterative_imputation(
    df: pd.DataFrame,
    column: str,
    max_iter: int = 10,
    random_state: int = 42
) -> tuple[pd.DataFrame, int]:
    """
    Apply iterative (MICE) imputation to a numeric column using multivariate approach.
    
    Iterative imputation models each feature with missing values as a function of other
    features in a round-robin fashion. This is also known as MICE (Multivariate Imputation
    by Chained Equations).
    
    Args:
        df: The DataFrame to impute (will not be modified)
        column: The target column to impute missing values in
        max_iter: Maximum number of imputation iterations (default: 10)
        random_state: Random seed for reproducibility (default: 42)
    
    Returns:
        Tuple of (transformed_df, rows_affected) where:
            - transformed_df: New DataFrame with imputed values
            - rows_affected: Number of missing values that were filled
    
    Raises:
        ValueError: If column is not numeric or if insufficient numeric columns exist
    
    Validates:
        - Requirement 1A.2: Iterative (MICE) imputation with max_iter and random_state
        - Uses all numeric columns for multivariate imputation
        - Uses 'mean' as initial_strategy
        - Fits on provided data, transforms the same data
        - Handles edge case of insufficient numeric columns
    
    Examples:
        >>> df = pd.DataFrame({
        ...     'age': [25, 30, np.nan, 40],
        ...     'salary': [50000, 60000, 55000, np.nan],
        ...     'years_exp': [2, 5, 3, 8]
        ... })
        >>> result_df, affected = _apply_iterative_imputation(df, 'age', max_iter=10)
        >>> result_df['age'].isna().sum()
        0
    """
    # Create a copy to avoid modifying the original DataFrame
    df_copy = df.copy()
    
    # Validate column is numeric
    if not pd.api.types.is_numeric_dtype(df_copy[column]):
        raise ValueError(
            f"Iterative imputation requires a numeric column, but '{column}' has dtype '{df_copy[column].dtype}'"
        )
    
    # Extract all numeric columns for multivariate imputation
    numeric_cols = df_copy.select_dtypes(include=[np.number]).columns.tolist()
    
    # Check if we have enough numeric columns for iterative imputation
    # Need at least 2 columns (target + at least one feature for modeling)
    if len(numeric_cols) < 2:
        raise ValueError(
            f"Iterative imputation requires at least 2 numeric columns for multivariate modeling, "
            f"but only found {len(numeric_cols)} numeric column(s): {numeric_cols}. "
            f"Consider using mean, median, or mode strategies instead."
        )
    
    # Ensure target column is in numeric columns
    if column not in numeric_cols:
        raise ValueError(
            f"Target column '{column}' is not in the list of numeric columns"
        )
    
    # Count missing values before imputation
    missing_count = df_copy[column].isna().sum()
    
    # If no missing values, return early
    if missing_count == 0:
        return df_copy, 0
    
    # Create iterative imputer with specified parameters
    # - max_iter: maximum number of imputation rounds
    # - random_state: for reproducibility
    # - initial_strategy: 'mean' to initialize missing values before iteration
    imputer = IterativeImputer(
        max_iter=max_iter,
        random_state=random_state,
        initial_strategy='mean'
    )
    
    # Extract numeric columns for imputation
    df_numeric = df_copy[numeric_cols].copy()
    
    # Fit and transform the numeric data
    # Iterative imputation will model each feature with missing values
    # as a function of other features in a round-robin fashion
    df_imputed_values = imputer.fit_transform(df_numeric)
    
    # Create DataFrame from imputed values
    df_imputed = pd.DataFrame(
        df_imputed_values,
        columns=numeric_cols,
        index=df_copy.index
    )
    
    # Update only the target column in the original DataFrame copy
    # This preserves non-numeric columns and only updates the imputed column
    df_copy[column] = df_imputed[column]
    
    # Return the transformed DataFrame and count of affected rows
    return df_copy, missing_count


def _apply_forward_fill(
    df: pd.DataFrame,
    column: str
) -> tuple[pd.DataFrame, int]:
    """
    Apply forward fill imputation to a column by propagating last valid observation forward.
    
    Forward fill (also known as "last observation carried forward" or LOCF) propagates
    the last valid (non-missing) value forward to fill subsequent missing values.
    If the first values are NaN (no previous value to fill from), they are optionally
    backfilled using the next valid observation.
    
    This strategy works for any column type (numeric, categorical, object) and is
    particularly useful for time-series or ordered data where values tend to persist.
    
    Args:
        df: The DataFrame to impute (will not be modified)
        column: The target column to impute missing values in
    
    Returns:
        Tuple of (transformed_df, rows_affected) where:
            - transformed_df: New DataFrame with imputed values
            - rows_affected: Number of missing values that were filled
    
    Validates:
        - Requirement 1A.3: Forward fill using pandas fillna(method='ffill')
        - Optionally backfills remaining NaNs at the start
        - Works for any column type
    
    Examples:
        >>> df = pd.DataFrame({
        ...     'status': [np.nan, 'active', 'active', np.nan, 'inactive', np.nan]
        ... })
        >>> result_df, affected = _apply_forward_fill(df, 'status')
        >>> result_df['status'].tolist()
        ['active', 'active', 'active', 'active', 'inactive', 'inactive']
        
        >>> df = pd.DataFrame({
        ...     'value': [np.nan, np.nan, 10, 20, np.nan, 30]
        ... })
        >>> result_df, affected = _apply_forward_fill(df, 'value')
        >>> result_df['value'].tolist()
        [10.0, 10.0, 10.0, 20.0, 20.0, 30.0]
    """
    # Create a copy to avoid modifying the original DataFrame
    df_copy = df.copy()
    
    # Count missing values before imputation
    missing_count = df_copy[column].isna().sum()
    
    # If no missing values, return early
    if missing_count == 0:
        return df_copy, 0
    
    # Apply forward fill: propagate last valid observation forward
    # This fills NaN values with the most recent non-NaN value
    # Note: pandas deprecated method='ffill', use ffill() directly
    df_copy[column] = df_copy[column].ffill()
    
    # Check if there are still NaN values remaining
    # This happens when the first values are NaN (no previous value to fill from)
    remaining_nans = _convert_to_native_types(df_copy[column].isna().sum())
    
    if remaining_nans > 0:
        # Backfill remaining NaNs at the start using the next valid observation
        # Note: pandas deprecated method='bfill', use bfill() directly
        df_copy[column] = df_copy[column].bfill()
    
    # Count how many values were actually filled
    # Some values might still be NaN if the entire column was NaN
    final_missing_count = _convert_to_native_types(df_copy[column].isna().sum())
    rows_affected = missing_count - final_missing_count
    
    # Return the transformed DataFrame and count of affected rows
    return df_copy, rows_affected


def _apply_backward_fill(
    df: pd.DataFrame,
    column: str
) -> tuple[pd.DataFrame, int]:
    """
    Apply backward fill imputation to a column by propagating next valid observation backward.
    
    Backward fill (also known as "next observation carried backward" or NOCB) propagates
    the next valid (non-missing) value backward to fill preceding missing values.
    If the last values are NaN (no next value to fill from), they are optionally
    forward filled using the previous valid observation.
    
    This strategy works for any column type (numeric, categorical, object) and is
    particularly useful for time-series or ordered data where future values can inform
    past missing values.
    
    Args:
        df: The DataFrame to impute (will not be modified)
        column: The target column to impute missing values in
    
    Returns:
        Tuple of (transformed_df, rows_affected) where:
            - transformed_df: New DataFrame with imputed values
            - rows_affected: Number of missing values that were filled
    
    Validates:
        - Requirement 1A.4: Backward fill using pandas fillna(method='bfill')
        - Optionally forward fills remaining NaNs at the end
        - Works for any column type
    
    Examples:
        >>> df = pd.DataFrame({
        ...     'status': [np.nan, 'active', 'active', np.nan, 'inactive', np.nan]
        ... })
        >>> result_df, affected = _apply_backward_fill(df, 'status')
        >>> result_df['status'].tolist()
        ['active', 'active', 'active', 'inactive', 'inactive', 'inactive']
        
        >>> df = pd.DataFrame({
        ...     'value': [np.nan, np.nan, 10, 20, np.nan, 30]
        ... })
        >>> result_df, affected = _apply_backward_fill(df, 'value')
        >>> result_df['value'].tolist()
        [10.0, 10.0, 10.0, 20.0, 30.0, 30.0]
    """
    # Create a copy to avoid modifying the original DataFrame
    df_copy = df.copy()
    
    # Count missing values before imputation
    missing_count = df_copy[column].isna().sum()
    
    # If no missing values, return early
    if missing_count == 0:
        return df_copy, 0
    
    # Apply backward fill: propagate next valid observation backward
    # This fills NaN values with the next non-NaN value
    # Note: pandas deprecated method='bfill', use bfill() directly
    df_copy[column] = df_copy[column].bfill()
    
    # Check if there are still NaN values remaining
    # This happens when the last values are NaN (no next value to fill from)
    remaining_nans = _convert_to_native_types(df_copy[column].isna().sum())
    
    if remaining_nans > 0:
        # Forward fill remaining NaNs at the end using the previous valid observation
        # Note: pandas deprecated method='ffill', use ffill() directly
        df_copy[column] = df_copy[column].ffill()
    
    # Count how many values were actually filled
    # Some values might still be NaN if the entire column was NaN
    final_missing_count = _convert_to_native_types(df_copy[column].isna().sum())
    rows_affected = missing_count - final_missing_count
    
    # Return the transformed DataFrame and count of affected rows
    return df_copy, rows_affected


def _apply_interpolate(
    df: pd.DataFrame,
    column: str
) -> tuple[pd.DataFrame, int]:
    """
    Apply linear interpolation to a numeric column to fill missing values.
    
    Interpolation estimates missing values by fitting a line between known values.
    This strategy requires a numeric column and is particularly useful for time-series
    or ordered data where values change gradually.
    
    The function uses pandas interpolate(method='linear', limit_direction='both')
    to fill gaps from both directions. If any NaN values remain after interpolation
    (e.g., at the edges or if all values are NaN), they are filled with the column mean
    as a fallback.
    
    Args:
        df: The DataFrame to impute (will not be modified)
        column: The target column to impute missing values in
    
    Returns:
        Tuple of (transformed_df, rows_affected) where:
            - transformed_df: New DataFrame with imputed values
            - rows_affected: Number of missing values that were filled
    
    Raises:
        ValueError: If column is not numeric
    
    Validates:
        - Requirement 1A.5: Interpolate using pandas interpolate(method='linear', limit_direction='both')
        - Requires numeric column
        - Fallback to mean for remaining NaNs
    
    Examples:
        >>> df = pd.DataFrame({
        ...     'temperature': [20.0, 22.0, np.nan, np.nan, 28.0, 30.0]
        ... })
        >>> result_df, affected = _apply_interpolate(df, 'temperature')
        >>> result_df['temperature'].tolist()
        [20.0, 22.0, 24.0, 26.0, 28.0, 30.0]
        
        >>> df = pd.DataFrame({
        ...     'value': [np.nan, 10.0, 20.0, np.nan, 40.0, np.nan]
        ... })
        >>> result_df, affected = _apply_interpolate(df, 'value')
        >>> # First NaN backfilled to 10.0, middle NaN interpolated to 30.0, last NaN forward filled to 40.0
    """
    # Create a copy to avoid modifying the original DataFrame
    df_copy = df.copy()
    
    # Validate column is numeric
    if not pd.api.types.is_numeric_dtype(df_copy[column]):
        raise ValueError(
            f"Interpolate strategy requires a numeric column, but '{column}' has dtype '{df_copy[column].dtype}'"
        )
    
    # Count missing values before imputation
    missing_count = df_copy[column].isna().sum()
    
    # If no missing values, return early
    if missing_count == 0:
        return df_copy, 0
    
    # Apply linear interpolation with limit_direction='both'
    # This fills gaps from both forward and backward directions
    df_copy[column] = df_copy[column].interpolate(
        method='linear', 
        limit_direction='both'
    )
    
    # Check if there are still NaN values remaining
    # This can happen at the edges (first/last values) or if all values were NaN
    remaining_nans = _convert_to_native_types(df_copy[column].isna().sum())
    
    if remaining_nans > 0:
        # Fallback to mean for remaining NaNs
        mean_value = df_copy[column].mean(skipna=True)
        
        # Only fill if mean is valid (not NaN)
        if not pd.isna(mean_value):
            df_copy[column] = df_copy[column].fillna(mean_value)
    
    # Count how many values were actually filled
    final_missing_count = _convert_to_native_types(df_copy[column].isna().sum())
    rows_affected = missing_count - final_missing_count
    
    # Return the transformed DataFrame and count of affected rows
    return df_copy, rows_affected


def _apply_random_sample(
    df: pd.DataFrame,
    column: str,
    random_state: int = 42
) -> tuple[pd.DataFrame, int]:
    """
    Apply random sampling imputation to a column by sampling from non-missing values.
    
    Random sampling replaces missing values by randomly selecting from the column's
    non-missing values with replacement. This preserves the original distribution of
    values and works for any column type (numeric, categorical, object).
    
    The random_state parameter ensures reproducibility - the same random_state will
    always produce the same imputed values for the same input data.
    
    Args:
        df: The DataFrame to impute (will not be modified)
        column: The target column to impute missing values in
        random_state: Random seed for reproducibility (default: 42)
    
    Returns:
        Tuple of (transformed_df, rows_affected) where:
            - transformed_df: New DataFrame with imputed values
            - rows_affected: Number of missing values that were filled
    
    Raises:
        ValueError: If all values in the column are missing
    
    Validates:
        - Requirement 1A.6: Random sample with replacement from non-missing values
        - Uses random_state for reproducibility
        - Works for any column type
    
    Examples:
        >>> df = pd.DataFrame({
        ...     'category': ['A', 'B', 'A', np.nan, 'C', np.nan, 'B']
        ... })
        >>> result_df, affected = _apply_random_sample(df, 'category', random_state=42)
        >>> result_df['category'].isna().sum()
        0
        >>> # Missing values filled with random samples from ['A', 'B', 'A', 'C', 'B']
        
        >>> df = pd.DataFrame({
        ...     'value': [10, 20, np.nan, 30, np.nan]
        ... })
        >>> result_df, affected = _apply_random_sample(df, 'value', random_state=42)
        >>> result_df['value'].isna().sum()
        0
    """
    # Create a copy to avoid modifying the original DataFrame
    df_copy = df.copy()
    
    # Get the series for the target column
    series = df_copy[column]
    
    # Extract valid (non-missing) values
    valid_values = series[series.notna()]
    
    # Check if there are any valid values to sample from
    if len(valid_values) == 0:
        raise ValueError(
            f"Cannot apply random_sample: all values in column '{column}' are missing"
        )
    
    # Count missing values before imputation
    missing_count = _convert_to_native_types(series.isna().sum())
    
    # If no missing values, return early
    if missing_count == 0:
        return df_copy, 0
    
    # Create a mask for missing values
    missing_mask = series.isna()
    
    # Set random seed for reproducibility
    np.random.seed(random_state)
    
    # Sample with replacement from valid values
    # Number of samples = number of missing values
    sampled_values = np.random.choice(
        valid_values.values,  # Use .values to get numpy array
        size=missing_count,
        replace=True
    )
    
    # Fill missing values with sampled values
    # Use loc to ensure we're setting values at the correct positions
    df_copy.loc[missing_mask, column] = sampled_values
    
    # Return the transformed DataFrame and count of affected rows
    return df_copy, missing_count
