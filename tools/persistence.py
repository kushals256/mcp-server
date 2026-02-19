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
            cols = params.get("columns", [])
            description = f"Step {i}: Dropped {len(cols)} columns: {', '.join(cols)}."
            
        elif tool == "remove_data_outliers" or tool == "remove_outliers":
            col = params.get("column", "?")
            method = params.get("method", "?")
            description = f"Step {i}: Removed outliers from '{col}' using {method}."
            
        elif tool == "create_feature":
            name = params.get("name")
            expr = params.get("expression")
            description = f"Step {i}: Created feature '{name}' = {expr}."
            
        elif tool == "encode_categorical" or tool == "encode_column":
            col = params.get("column")
            method = params.get("method")
            n_new = params.get("new_columns_count", "?")
            description = f"Step {i}: Encoded '{col}' using {method} ({n_new} new columns)."
            
        elif tool == "change_column_types":
            cols = params.get("columns", [])
            # Handle list of dicts for pretty printing
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

        steps.append(description)
        
    return GenerateReportResponse(steps=steps)