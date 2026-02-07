import sys
import os
import json
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.eda import describe_dataset

def create_dummy_data():
    data = {
        'age': [25, 30, 35, 40, 45, 100, -5, 25, 30, 25],
        'score': [85.5, 90.0, 88.5, 92.0, 70.0, 95.0, 80.0, 85.0, 90.0, 85.0],
        'category': ['A', 'B', 'A', 'C', 'A', 'B', 'D', 'A', 'E', 'A'],
        'description': ['Short', 'Medium length', 'Very long description text here', 'Short', 'Medium', 'Longer text', 'S', 'M', 'L', 'XL']
    }
    df = pd.DataFrame(data)
    df.to_csv("data/test_eda.csv", index=False)
    print("Created dummy dataset: data/test_eda.csv")

def test_describe_dataset():
    print("\nTesting describe_dataset...")
    result = describe_dataset("test_eda.csv")
    
    # Check for errors
    if "error" in result:
        print(f"FAILED: {result['error']}")
        return

    # specific checks
    num_summary = result.get('numerical_summary')
    cat_summary = result.get('categorical_summary')
    
    # 1. Check Age (Numerical)
    age_stats = num_summary.get('age')
    print(f"Age Mean: {age_stats['mean']} (Expected ~41.5)")
    print(f"Age Negatives: {age_stats['quality_checks']['num_negatives']} (Expected 1)")
    
    # 2. Check Category (Categorical)
    cat_stats = cat_summary.get('category')
    print(f"Category Unique: {cat_stats['unique_count']} (Expected 5)")
    print(f"Category Top Freq: {cat_stats['top_frequencies']}")
    
    # 3. String Metadata
    desc_stats = cat_summary.get('description')
    print(f"Description Avg Length: {desc_stats['avg_string_length']}")

    print("\nFULL OUTPUT JSON:")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    create_dummy_data()
    test_describe_dataset()
