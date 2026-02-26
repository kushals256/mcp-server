from typing import Dict, Any, List
from pydantic import BaseModel, Field
from utils.state_manager import GlobalStateManager

class GenerateReportResponse(BaseModel):
    steps: List[str] = Field(..., description="List of preprocessing steps performed")

def generate_preprocessing_report() -> GenerateReportResponse:
    manager = GlobalStateManager()
    history = manager.get_history()
    
    steps = []
    
    if not history:
        return GenerateReportResponse(steps=["No actions recorded in this session."])
        
    for i, entry in enumerate(history, 1):
        tool = entry.get("tool")
        params = entry.get("params", {})
        
        description = f"Step {i}: Unknown Action ({tool})"
        
        if tool == "load_data":
            name = params.get("dataset_name", "unknown")
            description = f"Step {i}: Loaded dataset '{name}'."
            
        elif tool == "drop_duplicate_rows":
            removed = params.get("rows_removed", 0)
            subset = params.get("subset_columns")
            desc_sub = f"subset={subset}" if subset else "all columns"
            description = f"Step {i}: Removed {removed} duplicate rows ({desc_sub})."
            
        elif tool == "drop_columns":
            cols = params.get("columns_dropped", params.get("columns", []))
            description = f"Step {i}: Dropped {len(cols)} columns: {', '.join(str(c) for c in cols)}."
            
        elif tool in ("remove_outliers", "remove_data_outliers"):
            col = params.get("column", "?")
            method = params.get("method", "?")
            removed = params.get("rows_removed", "?")
            description = f"Step {i}: Removed {removed} outliers from '{col}' using {method}."
            
        elif tool == "create_feature":
            name = params.get("name")
            expr = params.get("expression")
            description = f"Step {i}: Created feature '{name}' = {expr}."
            
        elif tool in ("encode_categorical", "encode_categorical_feature"):
            col = params.get("column")
            method = params.get("method")
            n_new = params.get("new_columns_count", "?")
            description = f"Step {i}: Encoded '{col}' using {method} ({n_new} new columns)."
            
        elif tool in ("cast_column_type", "change_column_types"):
            cols = params.get("columns", [])
            if cols and isinstance(cols[0], dict):
                details = ", ".join([f"{c.get('column')}->{c.get('new_dtype')}" for c in cols])
            else:
                details = str(cols)
            description = f"Step {i}: Cast column types: {details}."
            
        elif tool == "train_test_split":
            tr = params.get("train_rows")
            te = params.get("test_rows")
            ts = params.get("test_size")
            description = f"Step {i}: Split data (Test size: {ts}). Train: {tr} rows, Test: {te} rows."
            
        elif tool == "detect_data_quality_issues":
             description = f"Step {i}: Ran data quality check."

        elif tool == "handle_missing_values":
            col = params.get("column", "?")
            strategy = params.get("strategy", "?")
            affected = params.get("rows_affected", "?")
            description = f"Step {i}: Handled missing values in '{col}' using {strategy} ({affected} rows affected)."

        elif tool == "normalize_categorical_text":
            col = params.get("column", "?")
            ops = params.get("operations", [])
            description = f"Step {i}: Normalized text in '{col}' ({', '.join(ops) if ops else 'default ops'})."

        elif tool == "harmonize_categorical_values":
            col = params.get("column", "?")
            count = params.get("canonical_count", "?")
            description = f"Step {i}: Harmonized '{col}' to {count} canonical values."

        elif tool == "cluster_similar_categories":
            col = params.get("column", "?")
            threshold = params.get("threshold", "?")
            description = f"Step {i}: Clustered similar categories in '{col}' (threshold={threshold})."

        elif tool == "ml_prepare_categorical":
            col = params.get("column", "?")
            method = params.get("method", "?")
            description = f"Step {i}: ML-prepared '{col}' using {method}."

        elif tool == "rollback":
            target_v = params.get("rolled_back_to", "?")
            new_v = params.get("new_version", "?")
            description = f"Step {i}: Rolled back to version {target_v} (now version {new_v})."

        elif tool in ("validate_action", "validate_operation"):
            target_tool = params.get("target_tool", "?")
            allowed = params.get("allowed", "?")
            description = f"Step {i}: Validated '{target_tool}' — allowed={allowed}."

        steps.append(description)
        
    return GenerateReportResponse(steps=steps)