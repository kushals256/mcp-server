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
        else:
            estimated_memory_mb = current_memory_mb * (1 - 0.03)
            response = ValidateActionResponse(
                allowed=True,
                reason=f"Outlier removal on '{column}' using {method} is safe",
                estimated_memory_mb=estimated_memory_mb
            )
    
    elif tool == "create_feature":
        name = params.get("name")
        expression = params.get("expression")
        
        if not name:
            response = ValidateActionResponse(allowed=False, reason="No feature name specified", estimated_memory_mb=current_memory_mb)
        elif not expression:
            response = ValidateActionResponse(allowed=False, reason="No expression specified", estimated_memory_mb=current_memory_mb)
        elif name in df.columns:
            response = ValidateActionResponse(allowed=False, reason=f"Feature '{name}' already exists", estimated_memory_mb=current_memory_mb)
        else:
            avg_col_size = current_memory_mb / len(df.columns) if len(df.columns) > 0 else 1.0
            estimated_memory_mb = current_memory_mb + avg_col_size
            response = ValidateActionResponse(allowed=True, reason=f"Creating feature '{name}' is safe", estimated_memory_mb=estimated_memory_mb)
    
    elif tool == "train_test_split":
        test_size = params.get("test_size", 0.2)
        if test_size <= 0 or test_size >= 1:
            response = ValidateActionResponse(allowed=False, reason="test_size must be between 0 and 1", estimated_memory_mb=current_memory_mb)
        else:
            estimated_memory_mb = current_memory_mb * 2
            response = ValidateActionResponse(allowed=True, reason=f"Train-test split with test_size={test_size} is safe", estimated_memory_mb=estimated_memory_mb)
    
    elif tool in ["describe_dataset", "correlation_analysis", "detect_data_quality_issues"]:
        response = ValidateActionResponse(allowed=True, reason=f"{tool} is a read-only operation", estimated_memory_mb=current_memory_mb)
    
    elif tool == "drop_duplicates" or tool == "remove_duplicates":
        duplicate_count = df.duplicated().sum()
        estimated_memory_mb = current_memory_mb * (1 - duplicate_count / len(df))
        response = ValidateActionResponse(
            allowed=True,
            reason=f"Dropping duplicates is safe (estimated {duplicate_count} duplicates)",
            estimated_memory_mb=estimated_memory_mb
        )
    
    else:
        response = ValidateActionResponse(allowed=False, reason=f"Unknown tool '{tool}'. Cannot validate safety.", estimated_memory_mb=current_memory_mb)

    # Log the result of the validation
    manager.log_action("validate_operation", {
        "target_tool": tool,
        "allowed": response.allowed,
        "reason": response.reason,
        "estimated_memory_mb": round(response.estimated_memory_mb, 2)
    })

    return response