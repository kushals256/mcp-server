import sys
import os
import json
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.eda import correlation_analysis

def create_correlated_data():
    # Create dataset with known relationships
    np.random.seed(42)
    n = 100
    
    # 1. Num-Num: Perfect linear
    x = np.linspace(0, 10, n)
    y = 2 * x + 1 # Perfect correlation
    
    # 2. Cat-Cat: Strong association
    # Group A usually buys Product X, Group B buys Product Y
    group = ['A'] * 50 + ['B'] * 50
    product = ['X'] * 45 + ['Y'] * 5 + ['X'] * 5 + ['Y'] * 45
    
    # 3. Num-Cat: Strong separation (high correlation ratio)
    # Category determines value range
    # Cat 1: mean 10, Cat 2: mean 100
    cat_vals = ['C1'] * 50 + ['C2'] * 50
    num_vals = np.concatenate([np.random.normal(10, 1, 50), np.random.normal(100, 1, 50)])
    
    df = pd.DataFrame({
        'x': x,
        'y': y, # Num-Num target
        'group': group,
        'product': product, # Cat-Cat target
        'category': cat_vals,
        'value': num_vals # Num-Cat target
    })
    
    df.to_csv("data/test_corr.csv", index=False)
    print("Created test_corr.csv")

def test_correlation():
    print("\nRunning correlation_analysis on test_corr.csv...")
    results = correlation_analysis("test_corr.csv")
    
    if "error" in results:
        print(f"FAILED: {results['error']}")
        return

    print("\n--- Numerical Correlations (Expected x-y ~ 1.0) ---")
    for r in results.get('numerical_correlations', []):
        if (r['feature_1'] == 'x' and r['feature_2'] == 'y') or \
           (r['feature_1'] == 'y' and r['feature_2'] == 'x'):
            print(f"x vs y: {r['value']} (Abs: {r['abs_value']})")

    print("\n--- Categorical Correlations (Expected group-product to be high) ---")
    for r in results.get('categorical_correlations', []):
        if set([r['feature_1'], r['feature_2']]) == set(['group', 'product']):
            print(f"group vs product (Cramer's V): {r['value']} (MI: {r.get('mutual_info')})")

    print("\n--- Num-Cat Correlations (Expected category-value to be high) ---")
    for r in results.get('numerical_categorical_correlations', []):
        if (r['feature_1'] == 'value' and r['feature_2'] == 'category'):
            print(f"value vs category (Eta): {r['value']} (ANOVA p: {r.get('anova_p_val')})")

if __name__ == "__main__":
    create_correlated_data()
    test_correlation()
