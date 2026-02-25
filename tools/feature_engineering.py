import pandas as pd
import numpy as np
from typing import Dict, Any
from pydantic import BaseModel, Field
from utils.state_manager import GlobalStateManager


class CreateFeatureRequest(BaseModel):
    name: str = Field(..., description="Name of the new feature")
    expression: str = Field(..., description="Expression to create the feature")


def create_feature(request: CreateFeatureRequest) -> Dict[str, Any]:
    """
    Create a new feature in the dataset using a pandas expression.
    
    The expression can reference existing columns and use pandas/numpy operations.
    
    Examples:
    - Simple arithmetic: "df['Age'] * 2"
    - Conditional: "df['Age'].apply(lambda x: 'Adult' if x >= 18 else 'Minor')"
    - Multiple columns: "df['Fare'] / df['Pclass']"
    - String operations: "df['Name'].str.split(',').str[0]"
    - Date operations: "pd.to_datetime(df['Date']).dt.year"
    
    Args:
        request: CreateFeatureRequest containing feature name and expression.
        
    Returns:
        Dictionary containing:
            - feature_name: Name of the created feature
            - rows_affected: Number of rows in the dataset
            - dtype: Data type of the created feature
            - sample_values: Sample values from the new feature
            - error: Error message if operation failed
    """
    manager = GlobalStateManager()
    df = manager.get_data()
    
    if df is None:
        return {"error": "No dataset loaded in memory. Please load a dataset first."}
    
    feature_name = request.name
    expression = request.expression
    
    # Validate feature name doesn't already exist
    if feature_name in df.columns:
        return {"error": f"Feature '{feature_name}' already exists in dataset. Choose a different name."}
    
    # Validate expression is not empty
    if not expression.strip():
        return {"error": "Expression cannot be empty."}
    
    # Create a copy to avoid mutating original data
    df_copy = df.copy()
    
    try:
        # Create a safe evaluation context with df, pd, and np
        eval_context = {
            'df': df_copy,
            'pd': pd,
            'np': np
        }
        
        # Evaluate the expression
        new_feature = eval(expression, eval_context)
        
        # Handle different return types
        if isinstance(new_feature, pd.Series):
            if len(new_feature) != len(df_copy):
                return {"error": f"Expression returned a Series with {len(new_feature)} rows, but dataset has {len(df_copy)} rows."}
            df_copy[feature_name] = new_feature
        elif isinstance(new_feature, (list, np.ndarray)):
            if len(new_feature) != len(df_copy):
                return {"error": f"Expression returned {len(new_feature)} values, but dataset has {len(df_copy)} rows."}
            df_copy[feature_name] = new_feature
        elif np.isscalar(new_feature):
            # Broadcast scalar to all rows
            df_copy[feature_name] = new_feature
        else:
            return {"error": f"Expression returned unsupported type: {type(new_feature)}. Expected Series, list, array, or scalar."}
        
        # Update state
        manager.load_data(df_copy, manager.get_dataset_name())
        manager.log_action("create_feature", {
            "name": feature_name,
            "expression": expression
        })
        
        # Get sample values (first 5 non-null values)
        sample_values = df_copy[feature_name].dropna().head(5).tolist()
        
        return {
            "feature_name": feature_name,
            "rows_affected": len(df_copy),
            "dtype": str(df_copy[feature_name].dtype),
            "sample_values": sample_values
        }
        
    except SyntaxError as e:
        return {"error": f"Invalid expression syntax: {str(e)}"}
    except NameError as e:
        return {"error": f"Invalid column or function reference: {str(e)}"}
    except Exception as e:
        return {"error": f"Error creating feature: {str(e)}"}