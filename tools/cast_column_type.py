"""
Column Type Casting Tools for MCP Server.

This module provides functionality to cast columns to specific data types.
It implements Phase 4 (Transformation) of the dataset analysis workflow.

Functions:
    cast_column_type: Cast specified columns to desired data types
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from pydantic import BaseModel, Field

from utils.state_manager import GlobalStateManager
from tools.discovery import load_dataset_metadata


class ColumnTypeSpec(BaseModel):
    column: str = Field(..., description="Name of the column to cast")
    dtype: str = Field(..., description="Target data type (e.g. 'int', 'float', 'datetime', 'category')")


class CastColumnTypeRequest(BaseModel):
    dataset_name: str = Field(..., description="Name of the dataset (e.g., 'data.csv').")
    columns: List[ColumnTypeSpec] = Field(..., description="List of columns to cast with their target types")


def cast_column_type(request: CastColumnTypeRequest) -> Dict[str, Any]:
    """
    Cast specified columns to desired data types.
    
    This function allows the LLM to transform column data types based on analysis needs.
    It supports common pandas data types and includes validation and error handling.
    
    Supported data types:
        - int, int8, int16, int32, int64: Integer types
        - float, float16, float32, float64: Floating point types
        - str, string, object: String/text types
        - bool, boolean: Boolean types
        - datetime, datetime64: Datetime types
        - category: Categorical type (for optimization)
    
    Args:
        request: CastColumnTypeRequest containing dataset_name and columns specification.
        
    Returns:
        Dictionary with structure:
        {
            "success": bool,
            "columns_cast": List[Dict],  # Successfully cast columns with details
            "errors": List[Dict],         # Failed casts with error messages
            "total_columns_processed": int,
            "remaining_rows": int
        }
    """
    dataset_name = request.dataset_name
    columns_specs = request.columns
    
    manager = GlobalStateManager()
    
    # Ensure dataset is loaded
    if manager.get_dataset_name() != dataset_name:
        try:
            load_dataset_metadata(dataset_name)
        except Exception as e:
            return {"error": f"Failed to load dataset: {str(e)}"}
            
    df = manager.get_data()
    if df is None:
        return {"error": "Dataset loaded but DataFrame is None."}
    
    # Create a copy to avoid mutating original data
    df_cast = df.copy()
    
    # Track results
    columns_cast = []
    errors = []
    
    # Process each column casting request
    for spec in columns_specs:
        column = spec.column
        target_dtype = spec.dtype.lower().strip()
        
        # Validate column exists
        if column not in df_cast.columns:
            errors.append({
                "column": column,
                "target_dtype": target_dtype,
                "error": f"Column '{column}' not found in dataset."
            })
            continue
        
        # Get original dtype for reporting
        original_dtype = str(df_cast[column].dtype)
        
        try:
            # Normalize dtype names and perform casting
            if target_dtype in ["int", "int64"]:
                df_cast[column] = pd.to_numeric(df_cast[column], errors='coerce').astype('Int64')
            elif target_dtype in ["int8"]:
                df_cast[column] = pd.to_numeric(df_cast[column], errors='coerce').astype('Int8')
            elif target_dtype in ["int16"]:
                df_cast[column] = pd.to_numeric(df_cast[column], errors='coerce').astype('Int16')
            elif target_dtype in ["int32"]:
                df_cast[column] = pd.to_numeric(df_cast[column], errors='coerce').astype('Int32')
            elif target_dtype in ["float", "float64"]:
                df_cast[column] = pd.to_numeric(df_cast[column], errors='coerce').astype('float64')
            elif target_dtype in ["float16"]:
                df_cast[column] = pd.to_numeric(df_cast[column], errors='coerce').astype('float16')
            elif target_dtype in ["float32"]:
                df_cast[column] = pd.to_numeric(df_cast[column], errors='coerce').astype('float32')
            elif target_dtype in ["str", "string", "object"]:
                df_cast[column] = df_cast[column].astype(str)
            elif target_dtype in ["bool", "boolean"]:
                df_cast[column] = df_cast[column].astype(bool)
            elif target_dtype in ["datetime", "datetime64"]:
                df_cast[column] = pd.to_datetime(df_cast[column], errors='coerce')
            elif target_dtype == "category":
                df_cast[column] = df_cast[column].astype('category')
            else:
                errors.append({
                    "column": column,
                    "target_dtype": target_dtype,
                    "error": f"Unsupported data type '{target_dtype}'. See function docstring for supported types."
                })
                continue
            
            # Count NaN values introduced by coercion (for numeric/datetime conversions)
            null_count = df_cast[column].isnull().sum() - df[column].isnull().sum()
            
            new_dtype = str(df_cast[column].dtype)
            
            columns_cast.append({
                "column": column,
                "original_dtype": original_dtype,
                "new_dtype": new_dtype,
                "values_converted_to_null": int(null_count) if null_count > 0 else 0
            })
            
        except Exception as e:
            errors.append({
                "column": column,
                "target_dtype": target_dtype,
                "error": f"Casting failed: {str(e)}"
            })
            # Revert to original column if casting fails
            df_cast[column] = df[column]
    
    # Update global state only if at least one column was successfully cast
    if len(columns_cast) > 0:
        manager.load_data(df_cast, dataset_name)
    
    return {
        "success": len(columns_cast) > 0,
        "columns_cast": columns_cast,
        "errors": errors if len(errors) > 0 else None,
        "total_columns_processed": len(columns_cast),
        "remaining_rows": len(df_cast)
    }
