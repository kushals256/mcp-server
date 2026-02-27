"""
Tools package for Dataset Analysis MCP Server.

This package contains all MCP tool implementations organized by phase:
- Phase 1 (Discovery): Dataset listing and metadata loading
- Phase 2 (Persistence): Saving processed data and pipeline configs
- Phase 3 (Analysis): EDA and data quality detection
- Phase 4 (Transformation): Outlier removal and data cleaning
- Phase 4.5 (Normalization): Categorical text normalization pipeline
- Phase 5 (Feature Engineering): Feature creation
- Phase 6 (Validation & Safety): Pre-execution validation
- Phase 6.5 (Versioning): Version snapshots, rollback, diff
"""

from .discovery import list_datasets, load_dataset_metadata, peek_dataset_metadata, load_dataset
from .save_dataset import save_processed_dataset, export_pipeline_config
from .eda import describe_dataset, correlation_analysis
from .data_quality import detect_data_quality_issues
from .remove_outliers import remove_outliers
from .cast_column_type import cast_column_type
from .drop_columns import drop_columns
from .cleaning import drop_duplicate_rows
from .handle_missing_values import handle_missing_values
from .encode_categorical import encode_categorical_feature
from .train_test_split import train_test_split
from .normalize_categorical import normalize_categorical_text
from .harmonize_categorical import harmonize_categorical_values
from .cluster_categorical import cluster_similar_categories
from .ml_prepare_categorical import ml_prepare_categorical
from .feature_engineering import create_feature
from .extract_features import extract_features
from .reduce_features import reduce_features
from .remove_features import remove_features
from .validation import validate_action
from .persistence import generate_preprocessing_report
from .versioning import list_versions, rollback_version, diff_versions

__all__ = [
    # Phase 1: Discovery
    "list_datasets",
    "load_dataset_metadata",
    "peek_dataset_metadata",
    "load_dataset",
    # Phase 2: Persistence
    "save_processed_dataset",
    "export_pipeline_config",
    # Phase 3: Analysis
    "describe_dataset",
    "correlation_analysis",
    "detect_data_quality_issues",
    # Phase 4: Transformation
    "remove_outliers",
    "cast_column_type",
    "drop_columns",
    "drop_duplicate_rows",
    "handle_missing_values",
    "encode_categorical_feature",
    "train_test_split",
    # Phase 4.5: Normalization
    "normalize_categorical_text",
    "harmonize_categorical_values",
    "cluster_similar_categories",
    "ml_prepare_categorical",
    # Phase 5: Feature Engineering
    "create_feature",
    "extract_features",
    "reduce_features",
    "remove_features",
    # Phase 6: Validation & Safety
    "validate_action",
    "generate_preprocessing_report",
    # Phase 6.5: Versioning & Audit Trail
    "list_versions",
    "rollback_version",
    "diff_versions",
]
