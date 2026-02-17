"""
Dataset Analysis MCP Server - Main Entry Point.

This is the entry point for the MCP server. It initializes the FastMCP server
and registers all tool functions organized by workflow phase.

Workflow Phases:
    Phase 1: Discovery - List and load datasets
    Phase 2: Persistence - Save results and export configs
    Phase 3: Analysis - EDA and data quality detection
    Phase 4: Transformation - Data cleaning and outlier removal
"""

from mcp.server.fastmcp import FastMCP

from config import SERVER_NAME
from tools.discovery import list_datasets, load_dataset_metadata
from tools.save_dataset import save_processed_dataset, export_pipeline_config
from tools.eda import describe_dataset, correlation_analysis
from tools.data_quality import detect_data_quality_issues
from tools.remove_outliers import remove_outliers
from tools.cast_column_type import cast_column_type
from tools.encode_categorical import encode_categorical_feature
from tools.train_test_split import train_test_split
from tools.cleaning import drop_duplicate_rows
from tools.normalize_categorical import normalize_categorical_text
from tools.harmonize_categorical import harmonize_categorical_values
from tools.cluster_categorical import cluster_similar_categories
from tools.ml_prepare_categorical import ml_prepare_categorical
from tools.feature_engineering import create_feature
from tools.validation import validate_action

# ============================================================================
# Initialize MCP Server
# ============================================================================

mcp = FastMCP(SERVER_NAME)


# ============================================================================
# Register Tools (by workflow phase)
# ============================================================================

# Phase 1: Discovery
# List available datasets and load metadata into global state
mcp.tool()(list_datasets)
mcp.tool()(load_dataset_metadata)

# Phase 2: Persistence
# Save processed data and export pipeline configurations
mcp.tool()(save_processed_dataset)
mcp.tool()(export_pipeline_config)

# Phase 3: Analysis
# Perform exploratory data analysis and detect quality issues
mcp.tool()(describe_dataset)
mcp.tool()(detect_data_quality_issues)
mcp.tool()(correlation_analysis)

# Phase 4: Transformation
# Clean data and remove outliers
# Phase 4: Transformation
# Clean data and remove outliers
mcp.tool()(drop_duplicate_rows)
mcp.tool()(remove_outliers)
mcp.tool()(cast_column_type)
mcp.tool()(encode_categorical_feature)
mcp.tool()(train_test_split)

# Phase 4.5: Categorical Normalization
# Surface cleanup, synonym mapping, fuzzy clustering, ML prep
mcp.tool()(normalize_categorical_text)
mcp.tool()(harmonize_categorical_values)
mcp.tool()(cluster_similar_categories)
mcp.tool()(ml_prepare_categorical)

# Register Phase 5 Tools (Feature Engineering)
mcp.tool()(create_feature)

# Register Phase 6 Tools (Validation & Safety)
mcp.tool()(validate_action)

if __name__ == "__main__":
    mcp.run()

