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
from tools.persistence import save_processed_dataset, export_pipeline_config
from tools.eda import describe_dataset, correlation_analysis
from tools.data_quality import detect_data_quality_issues
from tools.remove_outliers import remove_outliers

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
mcp.tool()(remove_outliers)


if __name__ == "__main__":
    mcp.run()

