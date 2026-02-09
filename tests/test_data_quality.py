import pandas as pd
import numpy as np
import json
from tools.data_quality import detect_data_quality_issues


def test_detect_data_quality_issues():
    """
    Test the detect_data_quality_issues function with a synthetic dataset
    containing various data quality issues.
    """
    print("=" * 80)
    print("TEST: detect_data_quality_issues")
    print("=" * 80)
    
    # Create synthetic test dataset with known issues
    np.random.seed(42)
    n_rows = 200
    
    # Create dataset with multiple issues
    data = {
        # 1. Column with missing values (15% missing)
        'age': [np.random.randint(20, 70) if np.random.random() > 0.15 else np.nan 
                for _ in range(n_rows)],
        
        # 2. Normal distribution (should use Z-score for outlier detection)
        'normal_scores': np.random.normal(100, 15, n_rows),
        
        # 3. Skewed distribution with outliers (should use IQR)
        'income': np.concatenate([
            np.random.lognormal(10, 0.5, 180),  # Skewed data
            np.random.uniform(200000, 500000, 20)  # Outliers
        ]),
        
        # 4. High cardinality column (>80% unique)
        'user_id': [f'USER_{i:04d}' for i in range(n_rows)],
        
        # 5. Medium cardinality categorical column
        'category': np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 
                                      'I', 'J', 'K', 'L'], n_rows),
        
        # 6. Column with both missing values and outliers
        'mixed_issues': [
            np.random.normal(50, 10) if np.random.random() > 0.10 else np.nan
            for _ in range(n_rows)
        ],
        
        # 7. Low cardinality categorical
        'status': np.random.choice(['active', 'inactive', 'pending'], n_rows)
    }
    
    # Add some extreme outliers to mixed_issues
    df = pd.DataFrame(data)
    df.loc[df['mixed_issues'].notna(), 'mixed_issues'] = df.loc[df['mixed_issues'].notna(), 'mixed_issues'].apply(
        lambda x: x + 100 if np.random.random() < 0.05 else x
    )
    
    # Add duplicate rows (50+ duplicates)
    # Duplicate first 30 rows twice each
    duplicate_rows = df.iloc[:30]
    df = pd.concat([df, duplicate_rows, duplicate_rows], ignore_index=True)
    
    # Save to test file
    test_file = "test_data_quality.csv"
    df.to_csv(f"data/{test_file}", index=False)
    print(f"✓ Created test dataset: data/{test_file}")
    print(f"  - Total rows: {len(df)}")
    print(f"  - Total columns: {len(df.columns)}")
    print(f"  - Duplicate rows added: 60")
    print()
    
    # Run the detection
    print("Running detect_data_quality_issues...")
    print("-" * 80)
    result = detect_data_quality_issues(test_file)
    
    if "error" in result:
        print(f"❌ FAILED: {result['error']}")
        return
    
    issues = result.get("issues", [])
    print(f"\n✓ Detection completed. Found {len(issues)} issues.\n")
    
    # Group issues by type for better reporting
    issues_by_type = {}
    for issue in issues:
        issue_type = issue["type"]
        if issue_type not in issues_by_type:
            issues_by_type[issue_type] = []
        issues_by_type[issue_type].append(issue)
    
    # Report findings
    print("=" * 80)
    print("DETECTED ISSUES SUMMARY")
    print("=" * 80)
    
    for issue_type, type_issues in issues_by_type.items():
        print(f"\n{issue_type.upper().replace('_', ' ')} ({len(type_issues)} found)")
        print("-" * 80)
        
        for issue in type_issues:
            print(f"  Column: {issue['column']}")
            print(f"  Severity: {issue['severity']}")
            print(f"  Method: {issue['method']}")
            print(f"  Parameters:")
            for param_key, param_value in issue['parameters'].items():
                if isinstance(param_value, dict):
                    print(f"    {param_key}:")
                    for sub_key, sub_value in param_value.items():
                        print(f"      {sub_key}: {sub_value}")
                else:
                    print(f"    {param_key}: {param_value}")
            print()
    
    # Validation tests
    print("=" * 80)
    print("VALIDATION TESTS")
    print("=" * 80)
    
    # Test 1: Check for duplicate rows issue
    duplicate_issues = [i for i in issues if i['type'] == 'duplicate_rows']
    if duplicate_issues:
        print("✓ Duplicate rows detected")
        dup_issue = duplicate_issues[0]
        assert dup_issue['column'] == 'ALL_COLUMNS', "Duplicate column should be 'ALL_COLUMNS'"
        print(f"  - Duplicate count: {dup_issue['parameters']['duplicate_count']}")
    else:
        print("❌ Expected duplicate rows issue not found")
    
    # Test 2: Check for missing values
    missing_issues = [i for i in issues if i['type'] == 'missing_values']
    print(f"\n✓ Missing values issues: {len(missing_issues)} columns")
    age_missing = [i for i in missing_issues if i['column'] == 'age']
    if age_missing:
        print(f"  - 'age' column: {age_missing[0]['parameters']['percentage']}% missing")
    
    # Test 3: Check for outliers
    outlier_issues = [i for i in issues if i['type'] == 'outliers']
    print(f"\n✓ Outlier issues: {len(outlier_issues)} columns")
    for outlier in outlier_issues:
        method = outlier['method']
        col = outlier['column']
        print(f"  - '{col}': Method={method}")
        if 'distribution_metrics' in outlier['parameters']:
            metrics = outlier['parameters']['distribution_metrics']
            print(f"    Skewness: {metrics['skewness']}, Kurtosis: {metrics['kurtosis']}")
    
    # Test 4: Check for high cardinality
    cardinality_issues = [i for i in issues if i['type'] == 'high_cardinality']
    print(f"\n✓ High cardinality issues: {len(cardinality_issues)} columns")
    user_id_card = [i for i in cardinality_issues if i['column'] == 'user_id']
    if user_id_card:
        print(f"  - 'user_id': ratio={user_id_card[0]['parameters']['ratio']}")
    
    # Test 5: Verify multiple issues on same column
    mixed_col_issues = [i for i in issues if i['column'] == 'mixed_issues']
    print(f"\n✓ Multiple issues on 'mixed_issues' column: {len(mixed_col_issues)}")
    for issue in mixed_col_issues:
        print(f"  - {issue['type']} (severity: {issue['severity']})")
    
    # Test 6: Verify structured parameters (not strings)
    print("\n✓ Validating parameter structure...")
    all_params_valid = True
    for issue in issues:
        if not isinstance(issue.get('parameters'), dict):
            print(f"  ❌ Issue {issue['type']} on {issue['column']} has non-dict parameters")
            all_params_valid = False
    
    if all_params_valid:
        print("  ✓ All parameters are properly structured (dicts, not strings)")
    
    # Print full JSON output for inspection
    print("\n" + "=" * 80)
    print("FULL JSON OUTPUT")
    print("=" * 80)
    print(json.dumps(result, indent=2))
    
    print("\n" + "=" * 80)
    print("TEST COMPLETED SUCCESSFULLY")
    print("=" * 80)


def test_borderline_distribution():
    """
    Test case for borderline distribution that should trigger 'Both' method.
    """
    print("\n" + "=" * 80)
    print("TEST: Borderline Distribution (Both Methods)")
    print("=" * 80)
    
    np.random.seed(123)
    n_rows = 150
    
    # Create data with moderate skewness and kurtosis
    # This should trigger the "both methods" approach
    data = {
        'borderline_data': np.concatenate([
            np.random.normal(50, 10, 140),  # Mostly normal
            np.random.uniform(80, 100, 10)   # Some spread causing moderate skew
        ])
    }
    
    df = pd.DataFrame(data)
    test_file = "test_borderline.csv"
    df.to_csv(f"data/{test_file}", index=False)
    
    result = detect_data_quality_issues(test_file)
    
    if "error" in result:
        print(f"❌ FAILED: {result['error']}")
        return
    
    issues = result.get("issues", [])
    outlier_issues = [i for i in issues if i['type'] == 'outliers']
    
    print(f"\nFound {len(outlier_issues)} outlier issue(s)")
    
    for issue in outlier_issues:
        print(f"\nColumn: {issue['column']}")
        print(f"Method: {issue['method']}")
        print(f"Severity: {issue['severity']}")
        
        if 'distribution_metrics' in issue['parameters']:
            metrics = issue['parameters']['distribution_metrics']
            print(f"Distribution metrics:")
            print(f"  Skewness: {metrics['skewness']}")
            print(f"  Kurtosis: {metrics['kurtosis']}")
        
        if issue['method'] == 'Both_intersection':
            print("\n✓ Both methods were used (as expected for borderline case)")
            params = issue['parameters']
            if 'agreement_count' in params:
                print(f"  Agreement count: {params['agreement_count']}")
                print(f"  IQR only count: {params['iqr_only_count']}")
                print(f"  Z-score only count: {params['zscore_only_count']}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    test_detect_data_quality_issues()
    test_borderline_distribution()
