# Dataset Analysis MCP Server

This is a Model Context Protocol (MCP) server designed for dataset analysis. It implements a stateful workflow, allowing users to load datasets, perform operations, and save results or pipeline configurations.

## Getting Started

### Prerequisites

- Python 3.10 or higher
- `uv` (recommended) or `pip`

### Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd mcp-server
    ```

2.  **Set up the environment**:
    It is recommended to use `uv` for dependency management, but standard `pip` works as well.

    ```bash
    # Using uv (creates .venv automatically)
    uv sync

    # Using pip
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt  # If requirements.txt exists
    # OR install from pyproject.toml
    pip install .
    ```

3.  **Run the Server**:
    The server uses `fastmcp`.

    ```bash
    # Ensure virtual environment is activated
    source .venv/bin/activate
    
    # Run the server
    python main.py
    ```

## Testing and Verification

To verify the server is working correctly, you can use the MCP Inspector (coming soon/standard MCP tooling) or create simple client scripts.

### Automated Verification
The project previously contained verification scripts (`verify_phase1.py`, etc.) which have been removed to keep the codebase clean. 

To test locally:
1. Ensure `data/` directory exists or let the tools create it.
2. Use an MCP client (like Claude Desktop or the MCP Inspector) to connect to the server.
3. Execute the `list_datasets` tool to check connectivity.

## Development Guide

### Project Structure

- `main.py`: Entry point for the MCP server.
- `tools/`: Contains the MCP tool definitions.
    - `discovery.py`: Tools for listing and loading datasets.
    - `persistence.py`: Tools for saving results and exporting pipelines.
- `utils/`: Utility modules.
    - `state_manager.py`: Singleton class managing the in-memory state of the current dataset.
- `data/`: Directory for storing input/output datasets (gitignored).

### Stateful Architecture

This server uses a **Global State Manager** (`utils.state_manager.GlobalStateManager`) to maintain context across tool calls. This avoids passing large datasets between the client and server.

#### How to Add a New Tool

1.  **Import the State Manager**:
    ```python
    from utils.state_manager import GlobalStateManager
    ```

2.  **Access State**:
    Inside your tool function, instantiate the manager (it's a singleton) and retrieve the data.
    ```python
    def my_analysis_tool():
        manager = GlobalStateManager()
        df = manager.get_data()
        
        if df is None:
            return "Error: No dataset loaded."
            
        # Perform analysis on df
        result = df.describe()
        
        # Log the action (optional but recommended for pipeline export)
        manager.log_action("my_analysis_tool", {"param": "value"})
        
        return result.to_markdown()
    ```

3.  **Register the Tool**:
    Add your new tool to `main.py` using the `mcp.tool()` decorator.

### workflow

1.  **Discovery**: usage of `list_datasets` and `load_dataset_metadata` to find and load data into the global state.
2.  **Analysis**: (Future tools) Perform operations on the loaded `DataFrame`.
3.  **Persistence**: Use `save_processed_dataset` to save the modified state or `export_pipeline_config` to save the sequence of operations.
