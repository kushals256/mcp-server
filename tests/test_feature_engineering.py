import sys
import os
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.feature_engineering import create_feature, CreateFeatureRequest
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


def test_create_feature_simple_arithmetic():
    """Test creating a feature with simple arithmetic."""
    print("\n1. Testing simple arithmetic feature creation...")
    manager = setup_test_data()
    
    request = CreateFeatureRequest(
        name="AgeDouble",
        expression="df['Age'] * 2"
    )
    
    response = create_feature(request)
    
    assert response.feature_name == "AgeDouble", "Feature name should match"
    assert response.rows_affected == 5, "Should affect 5 rows"
    assert len(response.sample_values) == 5, "Should have 5 sample values"
    
    # Verify the feature was created correctly
    df = manager.get_data()
    assert "AgeDouble" in df.columns, "Feature should be in dataframe"
    assert df["AgeDouble"].tolist() == [50, 60, 70, 80, 90], "Values should be doubled"
    print(f"✓ PASSED: Created feature with values {df['AgeDouble'].tolist()}")


def test_create_feature_multiple_columns():
    """Test creating a feature using multiple columns."""
    print("\n2. Testing feature creation with multiple columns...")
    manager = setup_test_data()
    
    request = CreateFeatureRequest(
        name="FarePerAge",
        expression="df['Fare'] / df['Age']"
    )
    
    response = create_feature(request)
    
    assert response.feature_name == "FarePerAge", "Feature name should match"
    assert response.rows_affected == 5, "Should affect 5 rows"
    
    # Verify the feature was created correctly
    df = manager.get_data()
    assert "FarePerAge" in df.columns, "Feature should be in dataframe"
    expected = 10.5 / 25
    actual = df["FarePerAge"].iloc[0]
    assert abs(actual - expected) < 0.001, f"First value should be {expected}"
    print(f"✓ PASSED: Created feature with first value {actual:.4f}")


def test_create_feature_conditional():
    """Test creating a feature with conditional logic."""
    print("\n3. Testing feature creation with conditional logic...")
    manager = setup_test_data()
    
    request = CreateFeatureRequest(
        name="AgeGroup",
        expression="df['Age'].apply(lambda x: 'Young' if x < 35 else 'Old')"
    )
    
    response = create_feature(request)
    
    assert response.feature_name == "AgeGroup", "Feature name should match"
    assert response.rows_affected == 5, "Should affect 5 rows"
    
    # Verify the feature was created correctly
    df = manager.get_data()
    assert "AgeGroup" in df.columns, "Feature should be in dataframe"
    expected = ['Young', 'Young', 'Old', 'Old', 'Old']
    assert df["AgeGroup"].tolist() == expected, f"Values should be {expected}"
    print(f"✓ PASSED: Created feature with values {df['AgeGroup'].tolist()}")


def test_create_feature_string_operations():
    """Test creating a feature with string operations."""
    print("\n4. Testing feature creation with string operations...")
    manager = setup_test_data()
    
    request = CreateFeatureRequest(
        name="NameLength",
        expression="df['Name'].str.len()"
    )
    
    response = create_feature(request)
    
    assert response.feature_name == "NameLength", "Feature name should match"
    assert response.rows_affected == 5, "Should affect 5 rows"
    
    # Verify the feature was created correctly
    df = manager.get_data()
    assert "NameLength" in df.columns, "Feature should be in dataframe"
    expected = [5, 3, 7, 5, 3]
    assert df["NameLength"].tolist() == expected, f"Values should be {expected}"
    print(f"✓ PASSED: Created feature with values {df['NameLength'].tolist()}")


def test_create_feature_numpy_operations():
    """Test creating a feature with numpy operations."""
    print("\n5. Testing feature creation with numpy operations...")
    manager = setup_test_data()
    
    request = CreateFeatureRequest(
        name="AgeSqrt",
        expression="np.sqrt(df['Age'])"
    )
    
    response = create_feature(request)
    
    assert response.feature_name == "AgeSqrt", "Feature name should match"
    assert response.rows_affected == 5, "Should affect 5 rows"
    
    # Verify the feature was created correctly
    df = manager.get_data()
    assert "AgeSqrt" in df.columns, "Feature should be in dataframe"
    expected = np.sqrt(25)
    actual = df["AgeSqrt"].iloc[0]
    assert abs(actual - expected) < 0.001, f"First value should be {expected}"
    print(f"✓ PASSED: Created feature with first value {actual:.4f}")


def test_create_feature_scalar_broadcast():
    """Test creating a feature with scalar value (broadcast to all rows)."""
    print("\n6. Testing feature creation with scalar broadcast...")
    manager = setup_test_data()
    
    request = CreateFeatureRequest(
        name="Constant",
        expression="100"
    )
    
    response = create_feature(request)
    
    assert response.feature_name == "Constant", "Feature name should match"
    assert response.rows_affected == 5, "Should affect 5 rows"
    
    # Verify the feature was created correctly
    df = manager.get_data()
    assert "Constant" in df.columns, "Feature should be in dataframe"
    expected = [100, 100, 100, 100, 100]
    assert df["Constant"].tolist() == expected, f"Values should be {expected}"
    print(f"✓ PASSED: Created feature with constant value 100")


def test_create_feature_already_exists():
    """Test creating a feature that already exists."""
    print("\n7. Testing feature creation with existing name...")
    manager = setup_test_data()
    
    request = CreateFeatureRequest(
        name="Age",
        expression="df['Age'] * 2"
    )
    
    try:
        create_feature(request)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "already exists" in str(e), "Should mention already exists"
        print("✓ PASSED: Correctly rejects creating existing feature")


def test_create_feature_no_dataset():
    """Test creating a feature when no dataset is loaded."""
    print("\n8. Testing feature creation with no dataset...")
    manager = GlobalStateManager()
    manager.clear_state()
    
    request = CreateFeatureRequest(
        name="NewFeature",
        expression="df['Age'] * 2"
    )
    
    try:
        create_feature(request)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "No dataset loaded" in str(e), "Should mention no dataset"
        print("✓ PASSED: Correctly rejects when no dataset loaded")


def test_create_feature_invalid_expression():
    """Test creating a feature with invalid expression."""
    print("\n9. Testing feature creation with invalid expression...")
    manager = setup_test_data()
    
    request = CreateFeatureRequest(
        name="Invalid",
        expression="df['NonExistent'] * 2"
    )
    
    try:
        create_feature(request)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Error creating feature" in str(e), "Should mention error"
        print("✓ PASSED: Correctly rejects invalid expression")


def test_create_feature_empty_expression():
    """Test creating a feature with empty expression."""
    print("\n10. Testing feature creation with empty expression...")
    manager = setup_test_data()
    
    request = CreateFeatureRequest(
        name="Empty",
        expression=""
    )
    
    try:
        create_feature(request)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "cannot be empty" in str(e), "Should mention empty"
        print("✓ PASSED: Correctly rejects empty expression")


def test_create_feature_logs_action():
    """Test that feature creation is logged in pipeline history."""
    print("\n11. Testing feature creation logging...")
    manager = setup_test_data()
    
    request = CreateFeatureRequest(
        name="LogTest",
        expression="df['Age'] + 10"
    )
    
    create_feature(request)
    
    history = manager.get_history()
    
    # Find the create_feature action in history
    create_actions = [h for h in history if h["tool"] == "create_feature"]
    assert len(create_actions) > 0, "Should have logged action"
    assert create_actions[-1]["params"]["name"] == "LogTest", "Should log correct name"
    print(f"✓ PASSED: Action logged in pipeline history")


if __name__ == "__main__":
    print("=" * 60)
    print("Running Feature Engineering Tool Tests")
    print("=" * 60)
    
    try:
        test_create_feature_simple_arithmetic()
        test_create_feature_multiple_columns()
        test_create_feature_conditional()
        test_create_feature_string_operations()
        test_create_feature_numpy_operations()
        test_create_feature_scalar_broadcast()
        test_create_feature_already_exists()
        test_create_feature_no_dataset()
        test_create_feature_invalid_expression()
        test_create_feature_empty_expression()
        test_create_feature_logs_action()
        
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
