import sys
import os
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock yaml module if not present
# Mock optional dependencies if not present
from unittest.mock import MagicMock
import sys

dependencies_to_mock = ["yaml", "scipy", "scipy.stats", "sklearn", "sklearn.ensemble", "sklearn.preprocessing", "matplotlib", "matplotlib.pyplot", "seaborn"]

for dep in dependencies_to_mock:
    try:
        __import__(dep)
    except ImportError:
        sys.modules[dep] = MagicMock()

from tools.cast_column_type import cast_column_type, CastColumnTypeRequest
from utils.state_manager import GlobalStateManager

def setup_test_data():
    """Create a dataframe with various mixed types for testing"""
    df = pd.DataFrame({
        'age_str': ['25', '30', '35', 'nan'],
        'price_str': ['10.5', '20.0', '15.75', 'invalid'],
        'date_str': ['2023-01-01', '2023-02-15', 'not_a_date', '2023-03-30'],
        'category_code': [1, 2, 1, 3],
        'boolean_num': [1, 0, 1, 0],
        'already_int': [10, 20, 30, 40]
    })
    return df

def test_cast_basic_numeric():
    """Test casting strings to numeric types"""
    df = setup_test_data()
    manager = GlobalStateManager()
    manager.load_data(df, "test_numeric.csv")
    
    # Cast strings to numbers
    result = cast_column_type(CastColumnTypeRequest(
        dataset_name="test_numeric.csv",
        columns=[
            {"column": "age_str", "dtype": "int"},
            {"column": "price_str", "dtype": "float"}
        ]
    ))
    
    assert result["success"] is True
    assert len(result["columns_cast"]) == 2
    
    # Verify data in manager
    new_df = manager.get_data()
    
    # Check age column (Int64 to handle potential NaNs)
    assert pd.api.types.is_integer_dtype(new_df["age_str"])
    assert new_df["age_str"].iloc[0] == 25
    assert pd.isna(new_df["age_str"].iloc[3])  # 'nan' string becomes NaN
    
    # Check price column
    assert pd.api.types.is_float_dtype(new_df["price_str"])
    assert new_df["price_str"].iloc[0] == 10.5
    assert pd.isna(new_df["price_str"].iloc[3])  # 'invalid' becomes NaN

def test_cast_datetime():
    """Test casting strings to datetime"""
    df = setup_test_data()
    manager = GlobalStateManager()
    manager.load_data(df, "test_datetime.csv")
    
    result = cast_column_type(CastColumnTypeRequest(
        dataset_name="test_datetime.csv",
        columns=[{"column": "date_str", "dtype": "datetime"}]
    ))
    
    assert result["success"] is True
    
    new_df = manager.get_data()
    assert pd.api.types.is_datetime64_any_dtype(new_df["date_str"])
    assert new_df["date_str"].iloc[0].year == 2023
    assert pd.isna(new_df["date_str"].iloc[2])  # 'not_a_date' becomes NaT

def test_cast_category():
    """Test casting to category"""
    df = setup_test_data()
    manager = GlobalStateManager()
    manager.load_data(df, "test_category.csv")
    
    result = cast_column_type(CastColumnTypeRequest(
        dataset_name="test_category.csv",
        columns=[{"column": "category_code", "dtype": "category"}]
    ))
    
    assert result["success"] is True
    
    new_df = manager.get_data()
    assert isinstance(new_df["category_code"].dtype, pd.CategoricalDtype)
    assert len(new_df["category_code"].cat.categories) == 3

def test_cast_boolean():
    """Test casting numbers to boolean"""
    df = setup_test_data()
    manager = GlobalStateManager()
    manager.load_data(df, "test_boolean.csv")
    
    result = cast_column_type(CastColumnTypeRequest(
        dataset_name="test_boolean.csv",
        columns=[{"column": "boolean_num", "dtype": "bool"}]
    ))
    
    assert result["success"] is True
    
    new_df = manager.get_data()
    assert pd.api.types.is_bool_dtype(new_df["boolean_num"])
    assert new_df["boolean_num"].iloc[0] == True
    assert new_df["boolean_num"].iloc[1] == False

def test_errors_and_validation():
    """Test error handling"""
    df = setup_test_data()
    manager = GlobalStateManager()
    manager.load_data(df, "test_errors.csv")
    
    # Test non-existent column
    result = cast_column_type(CastColumnTypeRequest(
        dataset_name="test_errors.csv",
        columns=[{"column": "non_existent", "dtype": "int"}]
    ))
    
    assert result["success"] is False
    assert result["errors"] is not None
    assert "not found" in result["errors"][0]["error"]
    
    # Test unsupported type
    result = cast_column_type(CastColumnTypeRequest(
        dataset_name="test_errors.csv",
        columns=[{"column": "already_int", "dtype": "unsupported_type"}]
    ))
    
    assert result["success"] is False
    assert "Unsupported data type" in result["errors"][0]["error"]
    
    # Test mixed success and failure
    result = cast_column_type(CastColumnTypeRequest(
        dataset_name="test_errors.csv",
        columns=[
            {"column": "age_str", "dtype": "int"},
            {"column": "non_existent", "dtype": "int"}
        ]
    ))
    
    assert result["success"] is True  # Partial success is considered success for the operation
    assert len(result["columns_cast"]) == 1
    assert len(result["errors"]) == 1

def run_tests():
    """Run tests manually if pytest is not available"""
    print("Running tests manually...")
    try:
        test_cast_basic_numeric()
        print("✅ test_cast_basic_numeric passed")
        test_cast_datetime()
        print("✅ test_cast_datetime passed")
        test_cast_category()
        print("✅ test_cast_category passed")
        test_cast_boolean()
        print("✅ test_cast_boolean passed")
        test_errors_and_validation()
        print("✅ test_errors_and_validation passed")
        print("\nAll tests passed successfully!")
    except AssertionError as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        import pytest
        sys.exit(pytest.main(["-v", __file__]))
    except ImportError:
        run_tests()
