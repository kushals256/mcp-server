import sys
import os
import json

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from tools.discovery import load_dataset_metadata
from tools.remove_outliers import remove_outliers
from tools.normalize_categorical import normalize_categorical_text
from tools.encode_categorical import encode_categorical_feature
from tools.eda import correlation_analysis, CorrelationAnalysisRequest
from tools.train_test_split import train_test_split
from tools.harmonize_categorical import harmonize_categorical_values
from tools.feature_engineering import create_feature, CreateFeatureRequest
from tools.cast_column_type import cast_column_type, CastColumnTypeRequest, ColumnTypeSpec
from tools.save_dataset import save_processed_dataset, SaveDatasetRequest
from tools.cluster_categorical import cluster_similar_categories, ClusterCategoricalRequest

def run_test_case(name, func):
    result = []
    result.append(f"### {name}")
    try:
        res = func()
        if hasattr(res, 'model_dump'):
            res = res.model_dump()
        result.append(f"**Result:**\n```json\n{json.dumps(res, indent=2, default=str)}\n```")
    except Exception as e:
        result.append(f"**Exception Raised:**\n```text\n{type(e).__name__}: {str(e)}\n```")
    return "\n".join(result)

def main():
    report = ["# Segment 3: Invalid Parameters & Edge Cases Test Results\n"]
    dataset_name = "Titanic-Dataset.csv"
    
    report.append("## Pre-requisite: Load Dataset")
    report.append(run_test_case("Load Titanic Dataset", lambda: load_dataset_metadata(dataset_name)))

    report.append("\n## 3.1 Non-Existent Columns")
    report.append(run_test_case("Remove outliers from NonExistentColumn", lambda: remove_outliers(dataset_name, "NonExistentColumn", "iqr")))
    report.append(run_test_case("Normalize numeric Age", lambda: normalize_categorical_text(dataset_name, "Age")))
    report.append(run_test_case("One-hot encode FakeColumn", lambda: encode_categorical_feature(dataset_name, "FakeColumn", "onehot")))

    report.append("\n## 3.2 Invalid Methods")
    report.append(run_test_case("Outliers -> method=invalid", lambda: remove_outliers(dataset_name, "Fare", "invalid")))
    
    report.append(run_test_case("Correlation -> unknown method", 
        lambda: correlation_analysis(CorrelationAnalysisRequest(dataset_name=dataset_name, method="unknown method"))))
    
    report.append(run_test_case("Encoding -> super_encode", lambda: encode_categorical_feature(dataset_name, "Sex", "super_encode")))
    
    report.append(run_test_case("Clustering -> nonexistent scorer", 
        lambda: cluster_similar_categories(ClusterCategoricalRequest(dataset_name=dataset_name, column="Name", method="nonexistent scorer"))))

    report.append("\n## 3.3 Split Edge Cases")
    report.append(run_test_case("Split test_size=1.5", lambda: train_test_split(test_size=1.5)))
    report.append(run_test_case("Split test_size=0.0", lambda: train_test_split(test_size=0.0)))
    report.append(run_test_case("Split stratify_by=Age (contains NaN)", lambda: train_test_split(test_size=0.2, stratify="Age")))

    report.append("\n## 3.4 Empty Synonym Map")
    report.append(run_test_case("Harmonize Embarked with: {}", lambda: harmonize_categorical_values(dataset_name, "Embarked", {})))

    report.append("\n## 3.5 Feature Collision")
    report.append(run_test_case("Create Feature: Age = Age * 2", 
        lambda: create_feature(CreateFeatureRequest(name="Age", expression="df['Age'] * 2"))))

    report.append("\n## 3.6 Dangerous Expressions")
    report.append(run_test_case("Dangerous: import os; os.system('ls')", 
        lambda: create_feature(CreateFeatureRequest(name="Dangerous1", expression="import os; os.system('ls')"))))
    
    report.append(run_test_case("Dangerous: __import__('subprocess')", 
        lambda: create_feature(CreateFeatureRequest(name="Dangerous2", expression="__import__('subprocess').call(['rm','-rf','/'])"))))

    report.append("\n## 3.7 Type Mismatch")
    report.append(run_test_case("Cast Name -> int", 
        lambda: cast_column_type(CastColumnTypeRequest(dataset_name=dataset_name, columns=[ColumnTypeSpec(column="Name", type="int")]))))
    
    report.append(run_test_case("Cast Age -> bool", 
        lambda: cast_column_type(CastColumnTypeRequest(dataset_name=dataset_name, columns=[ColumnTypeSpec(column="Age", type="bool")]))))

    report.append("\n## 3.8 Save Overwrite")
    report.append(run_test_case("Attempt to overwrite Titanic-Dataset.csv", 
        lambda: save_processed_dataset(SaveDatasetRequest(dataset_name=dataset_name, output_filename="Titanic-Dataset.csv", overwrite=False))))

    report.append("\n## 3.9 Unsupported Format")
    report.append(run_test_case("Save as .xlsx", 
        lambda: save_processed_dataset(SaveDatasetRequest(dataset_name=dataset_name, output_filename="titanic_cleaned.xlsx", overwrite=True))))

    report_path = os.path.join(os.path.dirname(__file__), "Edge_Cases_Report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(report))
    print(f"Report saved to {report_path}")

if __name__ == "__main__":
    main()
