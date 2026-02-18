import pandas as pd
import numpy as np
from typing import Dict, Any
from pydantic import BaseModel, Field
from utils.state_manager import GlobalStateManager


class ValidateActionRequest(BaseModel):
    tool: str = Field(..., description="Name of the tool to validate")
    params: Dict[str, Any] = Field(..., description="Parameters for the tool")


class ValidateActionResponse(BaseModel):
    allowed: bool = Field(..., description="Whether the action is allowed")
    reason: str = Field(..., description="Reason for allowing or rejecting")
    estimated_memory_mb: float = Field(..., description="Estimated memory impact in MB")


def validate_action(request: ValidateActionRequest) -> ValidateActionResponse:
    """
    Dry-run mode for validating actions before execution.
    Estimates memory impact and rejects unsafe operations.
    
    Args:
        request: ValidateActionRequest containing tool name and parameters.
        
    Returns:
        ValidateActionResponse with allowed status, reason, and memory estimate.
    """
    manager = GlobalStateManager()
    df = manager.get_data()
    
    # If no dataset loaded, most operations are unsafe
    if df is None and request.tool not in ["list_datasets", "load_dataset_metadata"]:
        response = ValidateActionResponse(
            allowed=False,
            reason="No dataset loaded in memory. Load a dataset first.",
            estimated_memory_mb=0.0
        )
        # Log failed validation
        manager.log_action("validate_operation", {
            "target_tool": request.tool,
            "allowed": False,
            "reason": response.reason
        })
        return response
    
    tool = request.tool
    params = request.params
    
    # Calculate current memory usage if dataset exists
    current_memory_mb = 0.0
    if df is not None:
        current_memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    
    response = None

    # Validation logic per tool
    if tool == "load_dataset_metadata":
        estimated_memory_mb = 10.0  # Default estimate
        response = ValidateActionResponse(
            allowed=True,
            reason="Dataset loading is allowed",
            estimated_memory_mb=estimated_memory_mb
        )
    
    elif tool == "drop_columns":
        columns = params.get("columns", [])
        if not columns:
            response = ValidateActionResponse(
                allowed=False,
                reason="No columns specified for dropping",
                estimated_memory_mb=current_memory_mb
            )
        else:
            # Check if columns exist
            missing_cols = [col for col in columns if col not in df.columns]
            if missing_cols:
                response = ValidateActionResponse(
                    allowed=False,
                    reason=f"Columns not found: {missing_cols}",
                    estimated_memory_mb=current_memory_mb
                )
            else:
                remaining_cols = [col for col in df.columns if col not in columns]
                estimated_memory_mb = df[remaining_cols].memory_usage(deep=True).sum() / (1024 * 1024)
                response = ValidateActionResponse(
                    allowed=True,
                    reason=f"Dropping {len(columns)} column(s) is safe",
                    estimated_memory_mb=estimated_memory_mb
                )
    
    elif tool == "remove_outliers":
        column = params.get("column")
        method = params.get("method")
        
        if not column:
            response = ValidateActionResponse(
                allowed=False,
                reason="No column specified for outlier removal",
                estimated_memory_mb=current_memory_mb
            )
        elif column not in df.columns:
            response = ValidateActionResponse(
                allowed=False,
                reason=f"Column '{column}' not found in dataset",
                estimated_memory_mb=current_memory_mb
            )
        elif not pd.api.types.is_numeric_dtype(df[column]):
            response = ValidateActionResponse(
                allowed=False,
                reason=f"Column '{column}' is not numeric",
                estimated_memory_mb=current_memory_mb
            )
        
        # Estimate memory (typically removes 1-5% of rows)
        estimated_rows_removed = int(len(df) * 0.03)  # Conservative 3% estimate
        estimated_memory_mb = current_memory_mb * (1 - 0.03)
        
        return ValidateActionResponse(
            allowed=True,
            reason=f"Outlier removal on '{column}' using {method} is safe",
            estimated_memory_mb=estimated_memory_mb
        )
    
    elif tool == "handle_missing_values":
        column = params.get("column")
        strategy = params.get("strategy")
        
        if not column:
            return ValidateActionResponse(
                allowed=False,
                reason="No column specified for missing value handling",
                estimated_memory_mb=current_memory_mb
            )
        
        if column not in df.columns:
            return ValidateActionResponse(
                allowed=False,
                reason=f"Column '{column}' not found in dataset",
                estimated_memory_mb=current_memory_mb
            )
        
        missing_count = df[column].isnull().sum()
        if missing_count == 0:
            return ValidateActionResponse(
                allowed=True,
                reason=f"No missing values in '{column}', operation will have no effect",
                estimated_memory_mb=current_memory_mb
            )
        
        if strategy == "drop_rows":
            # Estimate memory after dropping rows
            estimated_memory_mb = current_memory_mb * (1 - missing_count / len(df))
        else:
            # Imputation doesn't change memory significantly
            estimated_memory_mb = current_memory_mb
        
        return ValidateActionResponse(
            allowed=True,
            reason=f"Handling missing values in '{column}' using {strategy} is safe",
            estimated_memory_mb=estimated_memory_mb
        )
    
    elif tool == "create_feature":
        name = params.get("name")
        expression = params.get("expression")
        
        if not name:
            return ValidateActionResponse(
                allowed=False,
                reason="No feature name specified",
                estimated_memory_mb=current_memory_mb
            )
        
        if not expression:
            return ValidateActionResponse(
                allowed=False,
                reason="No expression specified for feature creation",
                estimated_memory_mb=current_memory_mb
            )
        
        if name in df.columns:
            return ValidateActionResponse(
                allowed=False,
                reason=f"Feature '{name}' already exists in dataset",
                estimated_memory_mb=current_memory_mb
            )
        
        # Estimate memory increase (one additional column)
        # Assume average column size
        avg_col_size = current_memory_mb / len(df.columns) if len(df.columns) > 0 else 1.0
        estimated_memory_mb = current_memory_mb + avg_col_size
        
        return ValidateActionResponse(
            allowed=True,
            reason=f"Creating feature '{name}' is safe",
            estimated_memory_mb=estimated_memory_mb
        )
    
    elif tool == "train_test_split":
        test_size = params.get("test_size", 0.2)
        
        if test_size <= 0 or test_size >= 1:
            return ValidateActionResponse(
                allowed=False,
                reason="test_size must be between 0 and 1",
                estimated_memory_mb=current_memory_mb
            )
        
        # Memory doubles (train + test sets)
        estimated_memory_mb = current_memory_mb * 2
        
        return ValidateActionResponse(
            allowed=True,
            reason=f"Train-test split with test_size={test_size} is safe",
            estimated_memory_mb=estimated_memory_mb
        )
    
    elif tool in ["describe_dataset", "correlation_analysis", "detect_data_quality_issues"]:
        # Read-only operations are always safe
        return ValidateActionResponse(
            allowed=True,
            reason=f"{tool} is a read-only operation",
            estimated_memory_mb=current_memory_mb
        )
    
    elif tool == "drop_duplicates":
        # Estimate memory (typically removes 0-10% of rows)
        duplicate_count = df.duplicated().sum()
        estimated_memory_mb = current_memory_mb * (1 - duplicate_count / len(df))
        
        return ValidateActionResponse(
            allowed=True,
            reason=f"Dropping duplicates is safe (estimated {duplicate_count} duplicates)",
            estimated_memory_mb=estimated_memory_mb
        )
    
    elif tool in [
        "normalize_categorical_text",
        "harmonize_categorical_values",
        "cluster_similar_categories",
    ]:
        column = params.get("column")
        if not column:
            return ValidateActionResponse(
                allowed=False,
                reason="No column specified for normalization",
                estimated_memory_mb=current_memory_mb
            )
        if column not in df.columns:
            return ValidateActionResponse(
                allowed=False,
                reason=f"Column '{column}' not found in dataset",
                estimated_memory_mb=current_memory_mb
            )
        if pd.api.types.is_numeric_dtype(df[column]):
            return ValidateActionResponse(
                allowed=False,
                reason=f"Column '{column}' is numeric, normalization requires string/categorical columns",
                estimated_memory_mb=current_memory_mb
            )
        return ValidateActionResponse(
            allowed=True,
            reason=f"{tool} on '{column}' is safe (in-place string transform)",
            estimated_memory_mb=current_memory_mb
        )
    
    elif tool == "ml_prepare_categorical":
        column = params.get("column")
        method = params.get("method", "deduplicate")
        if not column:
            return ValidateActionResponse(
                allowed=False,
                reason="No column specified for ML preparation",
                estimated_memory_mb=current_memory_mb
            )
        if column not in df.columns:
            return ValidateActionResponse(
                allowed=False,
                reason=f"Column '{column}' not found in dataset",
                estimated_memory_mb=current_memory_mb
            )
        if method == "gap_encoder":
            n_components = params.get("n_components", 10)
            avg_col_size = current_memory_mb / len(df.columns) if len(df.columns) > 0 else 1.0
            estimated_memory_mb = current_memory_mb + avg_col_size * n_components
        else:
            estimated_memory_mb = current_memory_mb
        return ValidateActionResponse(
            allowed=True,
            reason=f"ml_prepare_categorical ({method}) on '{column}' is safe",
            estimated_memory_mb=estimated_memory_mb
        )
    
    else:
        # Unknown tool - be conservative
        return ValidateActionResponse(
            allowed=False,
            reason=f"Unknown tool '{tool}'. Cannot validate safety.",
            estimated_memory_mb=current_memory_mb
        )

