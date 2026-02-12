import sys
import os
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.validation import validate_action, ValidateActionRequest
from utils.state_manager import GlobalStateManager


def setup_test_data():
    """Create and load sample dataset for testing."""
    manager = GlobalStateManager()
    manager.clear_state()
    
    df = pd.DataFrame({
        'Age': [25, 30, 35, 40, 45],
        'Fare': [10.5, 20.0, 30.5, 40.0, 50.5],
        'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'Survived': [1, 0, 1, 0, 1]
    })
    
    manager.load_data(df, "test_data.csv")
    return manager


def test_validate_action_no_dataset():
    """Test validation when no dataset is loaded."""
    print("\n1. Testing validation with no dataset loaded...")
    manager = GlobalStateManager()
    manager.clear_state()
    
    request = ValidateActionRequest(
        tool="remove_outliers",
        params={"column": "Age", "method": "zscore"}
    )
    
    response = validate_action(request)
    
    assert response.allowed is False, "Should not allow action without dataset"
    assert "No dataset loaded" in response.reason, "Should mention no dataset"
    assert response.estimated_memory_mb == 0.0, "Memory should be 0"
    print("✓ PASSED: Correctly rejects action when no dataset loaded")


def test_validate_action_drop_columns_success():
    """Test validation for dropping columns."""
    print("\n2. Testing validation for dropping columns...")
    manager = setup_test_data()
    
    request = ValidateActionRequest(
        tool="drop_columns",
        params={"columns": ["Name"]}
    )
    
    response = validate_action(request)
    
    assert response.allowed is True, "Should allow dropping existing column"
    assert "safe" in response.reason.lower(), "Should mention safe"
    assert response.estimated_memory_mb > 0, "Should estimate memory"
    print(f"✓ PASSED: Allows dropping column (estimated memory: {response.estimated_memory_mb:.2f} MB)")


def test_validate_action_drop_columns_missing():
    """Test validation for dropping non-existent columns."""
    print("\n3. Testing validation for dropping non-existent columns...")
    manager = setup_test_data()
    
    request = ValidateActionRequest(
        tool="drop_columns",
        params={"columns": ["NonExistent"]}
    )
    
    response = validate_action(request)
    
    assert response.allowed is False, "Should not allow dropping non-existent column"
    assert "not found" in response.reason, "Should mention column not found"
    print("✓ PASSED: Correctly rejects dropping non-existent column")


def test_validate_action_remove_outliers_success():
    """Test validation for removing outliers."""
    print("\n4. Testing validation for removing outliers...")
    manager = setup_test_data()
    
    request = ValidateActionRequest(
        tool="remove_outliers",
        params={"column": "Age", "method": "zscore"}
    )
    
    response = validate_action(request)
    
    assert response.allowed is True, "Should allow outlier removal on numeric column"
    assert "safe" in response.reason.lower(), "Should mention safe"
    assert response.estimated_memory_mb > 0, "Should estimate memory"
    print(f"✓ PASSED: Allows outlier removal (estimated memory: {response.estimated_memory_mb:.2f} MB)")


def test_validate_action_remove_outliers_non_numeric():
    """Test validation for removing outliers on non-numeric column."""
    print("\n5. Testing validation for removing outliers on non-numeric column...")
    manager = setup_test_data()
    
    request = ValidateActionRequest(
        tool="remove_outliers",
        params={"column": "Name", "method": "zscore"}
    )
    
    response = validate_action(request)
    
    assert response.allowed is False, "Should not allow outlier removal on non-numeric column"
    assert "not numeric" in response.reason, "Should mention not numeric"
    print("✓ PASSED: Correctly rejects outlier removal on non-numeric column")


def test_validate_action_create_feature_success():
    """Test validation for creating a new feature."""
    print("\n6. Testing validation for creating a new feature...")
    manager = setup_test_data()
    
    request = ValidateActionRequest(
        tool="create_feature",
        params={"name": "AgeDouble", "expression": "df['Age'] * 2"}
    )
    
    response = validate_action(request)
    
    assert response.allowed is True, "Should allow creating new feature"
    assert "safe" in response.reason.lower(), "Should mention safe"
    assert response.estimated_memory_mb > 0, "Should estimate memory"
    print(f"✓ PASSED: Allows feature creation (estimated memory: {response.estimated_memory_mb:.2f} MB)")


def test_validate_action_create_feature_exists():
    """Test validation for creating a feature that already exists."""
    print("\n7. Testing validation for creating existing feature...")
    manager = setup_test_data()
    
    request = ValidateActionRequest(
        tool="create_feature",
        params={"name": "Age", "expression": "df['Age'] * 2"}
    )
    
    response = validate_action(request)
    
    assert response.allowed is False, "Should not allow creating existing feature"
    assert "already exists" in response.reason, "Should mention already exists"
    print("✓ PASSED: Correctly rejects creating existing feature")


def test_validate_action_read_only_operations():
    """Test validation for read-only operations."""
    print("\n8. Testing validation for read-only operations...")
    manager = setup_test_data()
    
    read_only_tools = ["describe_dataset", "correlation_analysis", "detect_data_quality_issues"]
    
    for tool in read_only_tools:
        request = ValidateActionRequest(
            tool=tool,
            params={"dataset_name": "test_data.csv"}
        )
        
        response = validate_action(request)
        
        assert response.allowed is True, f"Should allow read-only tool {tool}"
        assert "read-only" in response.reason.lower(), "Should mention read-only"
    
    print(f"✓ PASSED: All {len(read_only_tools)} read-only operations allowed")


def test_validate_action_unknown_tool():
    """Test validation for unknown tool."""
    print("\n9. Testing validation for unknown tool...")
    manager = setup_test_data()
    
    request = ValidateActionRequest(
        tool="unknown_tool",
        params={}
    )
    
    response = validate_action(request)
    
    assert response.allowed is False, "Should not allow unknown tool"
    assert "Unknown tool" in response.reason, "Should mention unknown tool"
    print("✓ PASSED: Correctly rejects unknown tool")


if __name__ == "__main__":
    print("=" * 60)
    print("Running Validation Tool Tests")
    print("=" * 60)
    
    try:
        test_validate_action_no_dataset()
        test_validate_action_drop_columns_success()
        test_validate_action_drop_columns_missing()
        test_validate_action_remove_outliers_success()
        test_validate_action_remove_outliers_non_numeric()
        test_validate_action_create_feature_success()
        test_validate_action_create_feature_exists()
        test_validate_action_read_only_operations()
        test_validate_action_unknown_tool()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
