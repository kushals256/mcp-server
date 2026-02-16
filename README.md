# Dataset Analysis MCP Server

A powerful Model Context Protocol (MCP) server for comprehensive dataset analysis. This server implements a stateful workflow for loading, analyzing, transforming, and saving datasets.

## Features

✨ **Dataset Discovery**: List and load CSV/JSON datasets with automatic metadata extraction  
📊 **Exploratory Data Analysis**: Comprehensive statistical summaries and correlation analysis  
🔍 **Data Quality Detection**: Automated detection of missing values, outliers, duplicates, and high cardinality  
🧹 **Data Cleaning**: Remove outliers using statistical methods (Z-score, IQR)  
💾 **Persistence**: Save processed datasets and export reproducible pipeline configurations  
🔄 **Stateful Architecture**: Maintains dataset context across multiple tool calls

## Getting Started

### Prerequisites

- Python 3.10 or higher
- `uv` (recommended) or `pip`

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository_url>
   cd mcp-server
   ```

2. **Set up the environment**:
   
   Using `uv` (recommended):
   ```bash
   uv sync
   ```
   
   Using `pip`:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Run the Server**:
   ```bash
   # Ensure virtual environment is activated
   source .venv/bin/activate
   
   # Run the server
   python main.py
   ```

## Project Structure

```
mcp-server/
├── config.py              # Centralized configuration and constants
├── main.py                # Server entry point
├── tools/                 # MCP tool implementations
│   ├── __init__.py        # Package exports
│   ├── discovery.py       # Phase 1: Dataset listing and loading
│   ├── persistence.py     # Phase 2: Saving data and configs
│   ├── eda.py             # Phase 3: Statistical analysis
│   ├── data_quality.py    # Phase 3: Quality issue detection
│   └── remove_outliers.py # Phase 4: Data cleaning
├── utils/                 # Utility modules
│   ├── __init__.py        # Package exports
│   └── state_manager.py   # Global state management (singleton)
├── tests/                 # Unit tests
│   ├── test_correlation.py
│   ├── test_data_quality.py
│   ├── test_eda.py
│   └── test_remove_outliers.py
├── data/                  # Input/output datasets (gitignored)
├── pyproject.toml         # Project metadata and dependencies
└── requirements.txt       # Pip-compatible requirements file
```

## Workflow

The server implements a four-phase workflow:

1. **Discovery**: Use `list_datasets` and `load_dataset_metadata` to find and load data into global state
2. **Analysis**: Perform EDA with `describe_dataset`, `correlation_analysis`, and `detect_data_quality_issues`
3. **Transformation**: Clean data using `remove_outliers`
4. **Persistence**: Save results with `save_processed_dataset` or export the pipeline with `export_pipeline_config`

## Development Guide

### Stateful Architecture

This server uses a **GlobalStateManager** (`utils.state_manager.GlobalStateManager`) singleton to maintain context across tool calls, avoiding the need to pass large datasets between client and server.

### Adding a New Tool

1. **Import the State Manager**:
   ```python
   from utils.state_manager import GlobalStateManager
   from config import DATA_DIR  # Use centralized config
   ```

2. **Access State**:
   ```python
   def my_analysis_tool():
       manager = GlobalStateManager()
       df = manager.get_data()
       
       if df is None:
           return {"error": "No dataset loaded"}
           
       # Perform analysis on df
       result = df.describe()
       
       # Log the action (optional but recommended)
       manager.log_action("my_analysis_tool", {"param": "value"})
       
       return result.to_dict()
   ```

3. **Register the Tool**:
   Add your tool to `main.py`:
   ```python
   from tools.my_module import my_analysis_tool
   mcp.tool()(my_analysis_tool)
   ```

### Configuration

All configuration constants are centralized in `config.py`:
- Data directory paths
- Statistical thresholds
- Severity levels
- Default parameters

This eliminates code duplication and provides a single source of truth for all settings.

## Testing

Run the test suite:

```bash
# Activate virtual environment
source .venv/bin/activate

# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_eda.py -v
```

## Troubleshooting

### Server won't start
- Ensure virtual environment is activated: `source .venv/bin/activate`
- Verify all dependencies are installed: `pip list`
- Check Python version: `python --version` (must be 3.10+)

### Dataset not found
- Place CSV/JSON files in the `data/` directory
- Use `list_datasets` tool to verify files are detected
- Check file permissions

### Import errors
- Ensure `__init__.py` files exist in `tools/` and `utils/` directories
- Verify you're running from the project root directory

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]