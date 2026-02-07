from mcp.server.fastmcp import FastMCP
from tools.discovery import list_datasets, load_dataset_metadata
from tools.persistence import save_processed_dataset, export_pipeline_config

# Initialize FastMCP server
mcp = FastMCP("Dataset Analysis MCP")

# Register Phase 1 Tools
mcp.tool()(list_datasets)
mcp.tool()(load_dataset_metadata)

# Register Phase 2 Tools
mcp.tool()(save_processed_dataset)
mcp.tool()(export_pipeline_config)

# Register Phase 3 Tools
from tools.eda import describe_dataset, correlation_analysis
mcp.tool()(describe_dataset)
mcp.tool()(correlation_analysis)

# Register Phase 4 Tools (Data Quality)
from tools.data_quality import detect_data_quality_issues
mcp.tool()(detect_data_quality_issues)

if __name__ == "__main__":
    mcp.run()
