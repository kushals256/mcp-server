"""
Tools package for Dataset Analysis MCP Server.

This package contains all MCP tool implementations organized by phase:
- Phase 1 (Discovery): Dataset listing and metadata loading
- Phase 2 (Persistence): Saving processed data and pipeline configs
- Phase 3 (Analysis): EDA and data quality detection
- Phase 4 (Transformation): Outlier removal and data cleaning
- Phase 4.5 (Normalization): Categorical text normalization pipeline
"""

from .discovery import list_datasets, load_dataset_metadata
from .save_dataset import save_processed_dataset, export_pipeline_config
from .eda import describe_dataset, correlation_analysis
from .data_quality import detect_data_quality_issues
from .remove_outliers import remove_outliers
from .cast_column_type import cast_column_type
from .encode_categorical import encode_categorical_feature
from .normalize_categorical import normalize_categorical_text
from .harmonize_categorical import harmonize_categorical_values
from .cluster_categorical import cluster_similar_categories
from .ml_prepare_categorical import ml_prepare_categorical

__all__ = [
    # Phase 1: Discovery
    "list_datasets",
    "load_dataset_metadata",
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
    "encode_categorical_feature",
    # Phase 4.5: Normalization
    "normalize_categorical_text",
    "harmonize_categorical_values",
    "cluster_similar_categories",
    "ml_prepare_categorical",
]

