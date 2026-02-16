import pandas as pd
import numpy as np
from typing import Dict, Any
from pydantic import BaseModel, Field
from utils.state_manager import GlobalStateManager
from tools.discovery import load_dataset_metadata


class CreateFeatureRequest(BaseModel):
    name: str = Field(..., description="Name of the new feature")
    expression: str = Field(..., description="Expression to create the feature")


class CreateFeatureResponse(BaseModel):
    feature_name: str = Field(..., description="Name of the created feature")
    rows_affected: int = Field(..., description="Number of rows in the dataset")
    dtype: str = Field(..., description="Data type of the created feature")
    sample_values: list = Field(..., description="Sample values from the new feature")


def create_feature(request: CreateFeatureRequest) -> CreateFeatureResponse:
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
        CreateFeatureResponse with feature name and metadata.
    """
    manager = GlobalStateManager()
    df = manager.get_data()
    
    if df is None:
        raise ValueError("No dataset loaded in memory. Please load a dataset first.")
    
    feature_name = request.name
    expression = request.expression
    
    # Validate feature name doesn't already exist
    if feature_name in df.columns:
        raise ValueError(f"Feature '{feature_name}' already exists in dataset. Choose a different name.")
    
    # Validate expression is not empty
    if not expression.strip():
        raise ValueError("Expression cannot be empty.")
    
    try:
        # Create a safe evaluation context with df, pd, and np
        eval_context = {
            'df': df,
            'pd': pd,
            'np': np
        }
        
        # Evaluate the expression
        new_feature = eval(expression, eval_context)
        
        # Handle different return types
        if isinstance(new_feature, pd.Series):
            # Direct series assignment
            if len(new_feature) != len(df):
                raise ValueError(f"Expression returned a Series with {len(new_feature)} rows, but dataset has {len(df)} rows.")
            df[feature_name] = new_feature
        elif isinstance(new_feature, (list, np.ndarray)):
            # Convert to series
            if len(new_feature) != len(df):
                raise ValueError(f"Expression returned {len(new_feature)} values, but dataset has {len(df)} rows.")
            df[feature_name] = new_feature
        elif np.isscalar(new_feature):
            # Broadcast scalar to all rows
            df[feature_name] = new_feature
        else:
            raise ValueError(f"Expression returned unsupported type: {type(new_feature)}. Expected Series, list, array, or scalar.")
        
        # Update state
        manager.load_data(df, manager.get_dataset_name())
        manager.log_action("create_feature", {
            "name": feature_name,
            "expression": expression
        })
        
        # Get sample values (first 5 non-null values)
        sample_values = df[feature_name].dropna().head(5).tolist()
        
        return CreateFeatureResponse(
            feature_name=feature_name,
            rows_affected=len(df),
            dtype=str(df[feature_name].dtype),
            sample_values=sample_values
        )
        
    except SyntaxError as e:
        raise ValueError(f"Invalid expression syntax: {str(e)}")
    except NameError as e:
        raise ValueError(f"Invalid column or function reference: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error creating feature: {str(e)}")
