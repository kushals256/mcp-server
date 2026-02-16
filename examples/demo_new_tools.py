"""
Demo script showing the usage of validate_action and create_feature tools.
"""
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from tools.validation import validate_action, ValidateActionRequest
from tools.feature_engineering import create_feature, CreateFeatureRequest
from utils.state_manager import GlobalStateManager


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def demo_validation():
    """Demonstrate the validate_action tool."""
    print_section("DEMO: validate_action Tool")
    
    # Setup test data
    manager = GlobalStateManager()
    manager.clear_state()
    
    df = pd.DataFrame({
        'Age': [25, 30, 35, 40, 45, 50, 55, 60],
        'Salary': [50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000],
        'Department': ['Sales', 'IT', 'HR', 'Sales', 'IT', 'HR', 'Sales', 'IT'],
        'Experience': [2, 5, 8, 10, 12, 15, 18, 20]
    })
    
    manager.load_data(df, "employees.csv")
    print(f"\nLoaded dataset with {len(df)} rows and {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")
    
    # Example 1: Validate a safe operation
    print("\n1. Validating outlier removal on numeric column...")
    request = ValidateActionRequest(
        tool="remove_outliers",
        params={"column": "Age", "method": "zscore", "threshold": 3.0}
    )
    response = validate_action(request)
    print(f"   Allowed: {response.allowed}")
    print(f"   Reason: {response.reason}")
    print(f"   Estimated Memory: {response.estimated_memory_mb:.4f} MB")
    
    # Example 2: Validate an unsafe operation
    print("\n2. Validating outlier removal on non-numeric column...")
    request = ValidateActionRequest(
        tool="remove_outliers",
        params={"column": "Department", "method": "zscore"}
    )
    response = validate_action(request)
    print(f"   Allowed: {response.allowed}")
    print(f"   Reason: {response.reason}")
    
    # Example 3: Validate feature creation
    print("\n3. Validating new feature creation...")
    request = ValidateActionRequest(
        tool="create_feature",
        params={"name": "SalaryPerYear", "expression": "df['Salary'] / df['Experience']"}
    )
    response = validate_action(request)
    print(f"   Allowed: {response.allowed}")
    print(f"   Reason: {response.reason}")
    print(f"   Estimated Memory: {response.estimated_memory_mb:.4f} MB")
    
    # Example 4: Validate duplicate feature creation
    print("\n4. Validating duplicate feature creation...")
    request = ValidateActionRequest(
        tool="create_feature",
        params={"name": "Age", "expression": "df['Age'] * 2"}
    )
    response = validate_action(request)
    print(f"   Allowed: {response.allowed}")
    print(f"   Reason: {response.reason}")


def demo_feature_engineering():
    """Demonstrate the create_feature tool."""
    print_section("DEMO: create_feature Tool")
    
    # Setup test data
    manager = GlobalStateManager()
    manager.clear_state()
    
    df = pd.DataFrame({
        'Age': [25, 30, 35, 40, 45, 50, 55, 60],
        'Salary': [50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000],
        'Department': ['Sales', 'IT', 'HR', 'Sales', 'IT', 'HR', 'Sales', 'IT'],
        'Experience': [2, 5, 8, 10, 12, 15, 18, 20]
    })
    
    manager.load_data(df, "employees.csv")
    print(f"\nLoaded dataset with {len(df)} rows and {len(df.columns)} columns")
    
    # Example 1: Simple arithmetic
    print("\n1. Creating feature with simple arithmetic...")
    request = CreateFeatureRequest(
        name="SalaryInK",
        expression="df['Salary'] / 1000"
    )
    response = create_feature(request)
    print(f"   Feature: {response.feature_name}")
    print(f"   Type: {response.dtype}")
    print(f"   Sample values: {response.sample_values}")
    
    # Example 2: Multiple columns
    print("\n2. Creating feature from multiple columns...")
    request = CreateFeatureRequest(
        name="SalaryPerYear",
        expression="df['Salary'] / df['Experience']"
    )
    response = create_feature(request)
    print(f"   Feature: {response.feature_name}")
    print(f"   Type: {response.dtype}")
    print(f"   Sample values: {[f'{v:.2f}' for v in response.sample_values]}")
    
    # Example 3: Conditional logic
    print("\n3. Creating feature with conditional logic...")
    request = CreateFeatureRequest(
        name="SeniorityLevel",
        expression="df['Experience'].apply(lambda x: 'Senior' if x >= 10 else 'Junior')"
    )
    response = create_feature(request)
    print(f"   Feature: {response.feature_name}")
    print(f"   Type: {response.dtype}")
    print(f"   Sample values: {response.sample_values}")
    
    # Example 4: String operations
    print("\n4. Creating feature with string operations...")
    request = CreateFeatureRequest(
        name="DeptLength",
        expression="df['Department'].str.len()"
    )
    response = create_feature(request)
    print(f"   Feature: {response.feature_name}")
    print(f"   Type: {response.dtype}")
    print(f"   Sample values: {response.sample_values}")
    
    # Example 5: Numpy operations
    print("\n5. Creating feature with numpy operations...")
    request = CreateFeatureRequest(
        name="LogSalary",
        expression="np.log10(df['Salary'])"
    )
    response = create_feature(request)
    print(f"   Feature: {response.feature_name}")
    print(f"   Type: {response.dtype}")
    print(f"   Sample values: {[f'{v:.4f}' for v in response.sample_values]}")
    
    # Show final dataset
    df = manager.get_data()
    print(f"\n6. Final dataset has {len(df.columns)} columns:")
    print(f"   {list(df.columns)}")


def demo_combined_workflow():
    """Demonstrate using both tools together."""
    print_section("DEMO: Combined Workflow (Validate + Create)")
    
    # Setup test data
    manager = GlobalStateManager()
    manager.clear_state()
    
    df = pd.DataFrame({
        'Height': [1.75, 1.80, 1.65, 1.70, 1.85, 1.60, 1.78, 1.82],
        'Weight': [70, 80, 60, 75, 90, 55, 78, 85],
        'Age': [25, 30, 35, 40, 45, 50, 55, 60]
    })
    
    manager.load_data(df, "health.csv")
    print(f"\nLoaded dataset with {len(df)} rows and {len(df.columns)} columns")
    
    # Define features to create
    features = [
        {
            "name": "BMI",
            "expression": "df['Weight'] / (df['Height'] ** 2)",
            "description": "Body Mass Index"
        },
        {
            "name": "BMI_Category",
            "expression": "df['BMI'].apply(lambda x: 'Underweight' if x < 18.5 else ('Normal' if x < 25 else ('Overweight' if x < 30 else 'Obese')))",
            "description": "BMI Category"
        },
        {
            "name": "AgeGroup",
            "expression": "df['Age'].apply(lambda x: 'Young' if x < 35 else ('Middle' if x < 55 else 'Senior'))",
            "description": "Age Group"
        }
    ]
    
    print("\nCreating features with validation...")
    
    for i, feature in enumerate(features, 1):
        print(f"\n{i}. {feature['description']} ({feature['name']})")
        
        # Step 1: Validate
        validate_request = ValidateActionRequest(
            tool="create_feature",
            params={"name": feature["name"], "expression": feature["expression"]}
        )
        validation = validate_action(validate_request)
        
        print(f"   Validation: {'✓ PASS' if validation.allowed else '✗ FAIL'}")
        print(f"   Reason: {validation.reason}")
        
        if validation.allowed:
            # Step 2: Create
            try:
                create_request = CreateFeatureRequest(
                    name=feature["name"],
                    expression=feature["expression"]
                )
                result = create_feature(create_request)
                print(f"   Created: {result.feature_name} ({result.dtype})")
                print(f"   Sample: {result.sample_values[:3]}")
            except Exception as e:
                print(f"   Error: {str(e)}")
        else:
            print(f"   Skipped due to validation failure")
    
    # Show final dataset
    df = manager.get_data()
    print(f"\n✓ Final dataset has {len(df.columns)} columns:")
    for col in df.columns:
        print(f"   - {col}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  MCP Tools Demo: validate_action & create_feature")
    print("=" * 60)
    
    try:
        demo_validation()
        demo_feature_engineering()
        demo_combined_workflow()
        
        print("\n" + "=" * 60)
        print("  ✓ Demo completed successfully!")
        print("=" * 60 + "\n")
    except Exception as e:
        print(f"\n✗ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
