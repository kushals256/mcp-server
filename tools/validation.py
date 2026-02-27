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
    if df is None and request.tool not in ["list_datasets", "load_dataset_metadata", "peek_dataset_metadata", "load_dataset"]:
        response = ValidateActionResponse(
            allowed=False,
            reason="No dataset loaded in memory. Load a dataset first.",
            estimated_memory_mb=0.0
        )
        # Log failed validation
        manager.log_action("validate_action", {
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
    

    # Validation logic per tool
    if tool in ("load_dataset_metadata", "peek_dataset_metadata", "load_dataset"):
        if tool == "peek_dataset_metadata":
            estimated_memory_mb = 0.0
        else:
            estimated_memory_mb = 10.0
        return ValidateActionResponse(
            allowed=True,
            reason=f"{tool} is safe" + (" (read-only, no state change)" if tool == "peek_dataset_metadata" else ""),
            estimated_memory_mb=estimated_memory_mb
        )

    elif tool == "list_datasets":
        return ValidateActionResponse(
            allowed=True,
            reason="list_datasets is a read-only filesystem scan",
            estimated_memory_mb=0.0
        )
    
    elif tool == "drop_columns":
        columns = params.get("columns", [])
        if not columns:
            return ValidateActionResponse(
                allowed=False,
                reason="No columns specified for dropping",
                estimated_memory_mb=current_memory_mb
            )
        else:
            # Check if columns exist
            missing_cols = [col for col in columns if col not in df.columns]
            if missing_cols:
                return ValidateActionResponse(
                    allowed=False,
                    reason=f"Columns not found: {missing_cols}",
                    estimated_memory_mb=current_memory_mb
                )
            else:
                remaining_cols = [col for col in df.columns if col not in columns]
                estimated_memory_mb = df[remaining_cols].memory_usage(deep=True).sum() / (1024 * 1024)
                return ValidateActionResponse(
                    allowed=True,
                    reason=f"Dropping {len(columns)} column(s) is safe",
                    estimated_memory_mb=estimated_memory_mb
                )
    
    elif tool == "remove_outliers":
        column = params.get("column")
        method = params.get("method")
        
        if not column:
            return ValidateActionResponse(
                allowed=False,
                reason="No column specified for outlier removal",
                estimated_memory_mb=current_memory_mb
            )
        elif column not in df.columns:
            return ValidateActionResponse(
                allowed=False,
                reason=f"Column '{column}' not found in dataset",
                estimated_memory_mb=current_memory_mb
            )
        elif not pd.api.types.is_numeric_dtype(df[column]):
            return ValidateActionResponse(
                allowed=False,
                reason=f"Column '{column}' is not numeric",
                estimated_memory_mb=current_memory_mb
            )
        else:
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
    
    elif tool in ["describe_dataset", "correlation_analysis", "detect_data_quality_issues", "generate_preprocessing_report"]:
        # Read-only operations are always safe
        return ValidateActionResponse(
            allowed=True,
            reason=f"{tool} is a read-only operation",
            estimated_memory_mb=current_memory_mb
        )

    elif tool == "drop_duplicate_rows":
        # Estimate memory (typically removes 0-10% of rows)
        subset_columns = params.get("subset_columns")
        duplicate_count = df.duplicated(subset=subset_columns).sum()
        estimated_memory_mb = current_memory_mb * (1 - duplicate_count / len(df)) if len(df) > 0 else 0.0

        return ValidateActionResponse(
            allowed=True,
            reason=f"Dropping duplicates is safe (estimated {duplicate_count} duplicates)",
            estimated_memory_mb=estimated_memory_mb
        )

    elif tool == "cast_column_type":
        columns = params.get("columns", [])
        if not columns:
            return ValidateActionResponse(
                allowed=False,
                reason="No columns specified for type casting",
                estimated_memory_mb=current_memory_mb
            )
        missing = [c["column"] for c in columns if isinstance(c, dict) and c.get("column") not in df.columns]
        if missing:
            return ValidateActionResponse(
                allowed=False,
                reason=f"Columns not found in dataset: {missing}",
                estimated_memory_mb=current_memory_mb
            )
        return ValidateActionResponse(
            allowed=True,
            reason=f"Casting {len(columns)} column(s) is safe (memory impact is negligible)",
            estimated_memory_mb=current_memory_mb
        )

    elif tool in ("save_processed_dataset", "export_pipeline_config"):
        # Persistence operations don't change in-memory state
        return ValidateActionResponse(
            allowed=True,
            reason=f"{tool} writes to disk only — no memory impact",
            estimated_memory_mb=current_memory_mb
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
    
    elif tool == "encode_categorical_feature":
        column = params.get("column")
        method = params.get("method")
        target_column = params.get("target_column")
        ordinal_mapping = params.get("ordinal_mapping")
        n_components = params.get("n_components", 8)

        supported_methods = [
            "one_hot", "label", "ordinal", "frequency",
            "target", "binary", "hashing", "leave_one_out",
        ]

        # --- Basic validation ---
        if not column:
            return ValidateActionResponse(
                allowed=False,
                reason="No column specified for encoding",
                estimated_memory_mb=current_memory_mb
            )
        if column not in df.columns:
            return ValidateActionResponse(
                allowed=False,
                reason=f"Column '{column}' not found in dataset",
                estimated_memory_mb=current_memory_mb
            )
        if pd.api.types.is_numeric_dtype(df[column]) and not isinstance(
            df[column].dtype, pd.CategoricalDtype
        ):
            return ValidateActionResponse(
                allowed=False,
                reason=f"Column '{column}' is numeric (dtype={df[column].dtype}). Encoding requires categorical/string columns.",
                estimated_memory_mb=current_memory_mb
            )
        if method and method not in supported_methods:
            return ValidateActionResponse(
                allowed=False,
                reason=f"Unknown encoding method '{method}'. Supported: {supported_methods}",
                estimated_memory_mb=current_memory_mb
            )

        # --- Method-specific parameter checks ---
        if method == "ordinal" and ordinal_mapping is None:
            return ValidateActionResponse(
                allowed=False,
                reason="ordinal_mapping is required for method='ordinal'.",
                estimated_memory_mb=current_memory_mb
            )
        if method in ("target", "leave_one_out"):
            if target_column is None:
                return ValidateActionResponse(
                    allowed=False,
                    reason=f"target_column is required for method='{method}'.",
                    estimated_memory_mb=current_memory_mb
                )
            if target_column not in df.columns:
                return ValidateActionResponse(
                    allowed=False,
                    reason=f"Target column '{target_column}' not found in dataset.",
                    estimated_memory_mb=current_memory_mb
                )

        # --- Cardinality & memory estimation ---
        cardinality = int(df[column].nunique(dropna=True))
        total_rows = len(df)
        avg_col_size = current_memory_mb / len(df.columns) if len(df.columns) > 0 else 0.01

        # Estimate new columns per method
        if method == "one_hot":
            new_cols = cardinality
            # Mirror the encoder's hard block at ONE_HOT_BLOCK_CARDINALITY (100)
            if cardinality > 100:
                return ValidateActionResponse(
                    allowed=False,
                    reason=(
                        f"One-hot encoding '{column}' would create {cardinality} columns "
                        f"(exceeds limit of 100). Estimated memory: "
                        f"{round(current_memory_mb + avg_col_size * cardinality, 2)} MB. "
                        "Consider 'binary', 'hashing', or 'frequency' encoding instead."
                    ),
                    estimated_memory_mb=round(current_memory_mb + avg_col_size * cardinality, 2)
                )
            elif cardinality > 20:
                # Warn zone (ONE_HOT_MAX_CARDINALITY = 20)
                estimated_memory_mb = round(current_memory_mb + avg_col_size * cardinality, 2)
                return ValidateActionResponse(
                    allowed=True,
                    reason=(
                        f"One-hot encoding '{column}' will create {cardinality} columns "
                        f"(above recommended max of 20). Estimated memory: {estimated_memory_mb} MB. "
                        "Sparse representation recommended. Consider 'binary' encoding for fewer columns."
                    ),
                    estimated_memory_mb=estimated_memory_mb
                )
        elif method == "binary":
            import math
            new_cols = max(1, math.ceil(math.log2(cardinality + 1)))
        elif method == "hashing":
            new_cols = n_components
        else:
            # label, ordinal, frequency, target, leave_one_out → 1 new column
            new_cols = 1

        estimated_memory_mb = round(current_memory_mb + avg_col_size * new_cols, 2)

        # Leakage warning for target encoding without train/test split
        warnings = []
        if method == "target":
            warnings.append("Target encoding has leakage risk if applied before train/test split.")
        if method == "leave_one_out":
            warnings.append("Leave-one-out encoding is leakage-safe but computationally expensive on large datasets.")

        reason = (
            f"Encoding '{column}' (cardinality={cardinality}) with '{method}' will create "
            f"{new_cols} new column(s). Estimated memory: {estimated_memory_mb} MB."
        )
        if warnings:
            reason += " Warnings: " + " ".join(warnings)

        return ValidateActionResponse(
            allowed=True,
            reason=reason,
            estimated_memory_mb=estimated_memory_mb
        )

    elif tool in ["list_versions", "diff_versions"]:
        # Read-only versioning operations
        return ValidateActionResponse(
            allowed=True,
            reason=f"{tool} is a read-only operation",
            estimated_memory_mb=current_memory_mb
        )

    elif tool == "rollback_version":
        version = params.get("version")
        if version is None:
            return ValidateActionResponse(
                allowed=False,
                reason="No version specified for rollback",
                estimated_memory_mb=current_memory_mb
            )
        return ValidateActionResponse(
            allowed=True,
            reason=f"Rolling back to version {version} is safe (creates a new version entry)",
            estimated_memory_mb=current_memory_mb
        )

    else:
        # Unknown tool - be conservative
        return ValidateActionResponse(
            allowed=False,
            reason=f"Unknown tool '{tool}'. Cannot validate safety.",
            estimated_memory_mb=current_memory_mb
        )

