# Dataset Analysis MCP Server

A powerful Model Context Protocol (MCP) server for comprehensive dataset analysis. This server implements a stateful workflow for loading, analyzing, transforming, normalizing, encoding, and saving datasets.

## Features

✨ **Dataset Discovery** — List and load CSV/JSON datasets with automatic metadata extraction  
📊 **Exploratory Data Analysis** — Statistical summaries and correlation analysis (Pearson, Spearman, Kendall)  
🔍 **Data Quality Detection** — Missing values, outliers, duplicates, and high cardinality detection  
🧹 **Data Cleaning** — Remove outliers (Z-score, IQR), cast column types  
🧠 **Categorical Normalization** — 4-layer pipeline: surface cleanup → synonym mapping → fuzzy clustering → ML prep  
🏷️ **Categorical Encoding** — 8 methods from one-hot to target/leave-one-out encoding  
⚙️ **Feature Engineering** — Create derived features from expressions  
✂️ **Train/Test Split** — Stratified splitting with immutability guarantees  
🛡️ **Validation & Safety** — Dry-run mode to estimate memory impact before operations  
💾 **Persistence** — Save processed datasets and export reproducible pipeline configs  
🔄 **Stateful Architecture** — Maintains dataset context across multiple tool calls

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
   source .venv/bin/activate
   python main.py
   ```

## Project Structure

```
mcp-server/
├── config.py                          # Centralized configuration and constants
├── main.py                            # Server entry point and tool registration
├── tools/                             # MCP tool implementations
│   ├── __init__.py                    # Package exports
│   ├── discovery.py                   # Phase 1: Dataset listing and loading
│   ├── save_dataset.py                # Phase 2: Saving data and pipeline configs
│   ├── eda.py                         # Phase 3: EDA and statistical analysis
│   ├── data_quality.py                # Phase 3: Quality issue detection
│   ├── remove_outliers.py             # Phase 4: Outlier removal
│   ├── cast_column_type.py            # Phase 4: Column type casting
│   ├── normalize_categorical.py       # Phase 4.5: Surface text normalization
│   ├── harmonize_categorical.py       # Phase 4.5: Synonym → canonical mapping
│   ├── cluster_categorical.py         # Phase 4.5: Fuzzy string clustering
│   ├── ml_prepare_categorical.py      # Phase 4.5: ML-aware preparation (skrub)
│   ├── encode_categorical.py          # Phase 4: Categorical encoding (8 methods)
│   ├── train_test_split.py            # Phase 4: Train/test splitting
│   ├── feature_engineering.py         # Phase 5: Feature creation
│   └── validation.py                  # Phase 6: Dry-run validation
├── utils/
│   └── state_manager.py               # Global state management (singleton)
├── tests/                             # Unit tests (100+ tests)
│   ├── test_normalize_categorical.py
│   ├── test_harmonize_categorical.py
│   ├── test_cluster_categorical.py
│   ├── test_ml_prepare_categorical.py
│   ├── test_encode_categorical.py
│   ├── test_data_quality.py
│   ├── test_eda.py
│   ├── test_correlation.py
│   ├── test_remove_outliers.py
│   ├── test_cast_column_type.py
│   ├── test_train_test_split.py
│   ├── test_feature_engineering.py
│   ├── test_save_dataset.py
│   ├── test_validation.py
│   └── test_mcp_integration.py
├── data/                              # Input/output datasets
├── pyproject.toml                     # Project metadata and dependencies
└── requirements.txt                   # Pip-compatible requirements
```

## Workflow

The server implements a multi-phase workflow:

### Phase 1 — Discovery
| Tool | Description |
|---|---|
| `list_datasets` | List CSV/JSON files in the data directory |
| `load_dataset_metadata` | Load a dataset into global state with metadata |

### Phase 2 — Persistence
| Tool | Description |
|---|---|
| `save_processed_dataset` | Save the current dataset to CSV/JSON/Parquet |
| `export_pipeline_config` | Export all logged operations as a reproducible config |

### Phase 3 — Analysis
| Tool | Description |
|---|---|
| `describe_dataset` | Statistical summary (mean, std, quantiles, etc.) |
| `correlation_analysis` | Pearson / Spearman / Kendall correlation matrix |
| `detect_data_quality_issues` | Detect missing values, outliers, duplicates, high cardinality |

### Phase 4 — Transformation
| Tool | Description |
|---|---|
| `remove_outliers` | Remove outliers using Z-score or IQR methods |
| `cast_column_type` | Cast a column to a different dtype |
| `encode_categorical_feature` | Encode categories (one-hot, label, ordinal, frequency, target, binary, hashing, leave-one-out) |
| `train_test_split` | Stratified train/test split with immutability |

### Phase 4.5 — Categorical Normalization

A 4-layer pipeline that runs **before** encoding to clean and standardize categorical values:

```
normalize_categorical_text        (Layer 1 — surface cleanup)
        ↓
harmonize_categorical_values      (Layer 2 — synonym mapping)
        ↓
cluster_similar_categories        (Layer 3 — fuzzy grouping)
        ↓
ml_prepare_categorical            (Layer 4 — ML-aware, optional)
        ↓
encode_categorical_feature        (encoding)
```

| Tool | Library | Purpose |
|---|---|---|
| `normalize_categorical_text` | unicodedata, clean-text, text-unidecode | Unicode normalization, accent stripping, case folding |
| `harmonize_categorical_values` | flashtext | Deterministic synonym → canonical mapping (substring-safe) |
| `cluster_similar_categories` | rapidfuzz | Fuzzy typo clustering with deterministic tie-breaking |
| `ml_prepare_categorical` | skrub (lazy-imported) | Auto-deduplication or GapEncoder for high-cardinality data |

### Phase 5 — Feature Engineering
| Tool | Description |
|---|---|
| `create_feature` | Create a derived column from a Python expression |

### Phase 6 — Validation & Safety
| Tool | Description |
|---|---|
| `validate_action` | Dry-run mode: estimate memory impact and check safety before execution |

## Development Guide

### Stateful Architecture

The server uses a **GlobalStateManager** singleton (`utils/state_manager.py`) to maintain dataset context across tool calls, avoiding the need to pass DataFrames between client and server.

### Adding a New Tool

1. **Create a tool function** in `tools/`:
   ```python
   from utils.state_manager import GlobalStateManager

   def my_tool(dataset_name: str, column: str) -> Dict[str, Any]:
       manager = GlobalStateManager()
       df = manager.get_data()
       if df is None:
           return {"error": "No dataset loaded"}
       # ... do work ...
       manager.load_data(df_modified, dataset_name)
       manager.log_action("my_tool", {"column": column})
       return {"result": "..."}
   ```

2. **Register** in `main.py`:
   ```python
   from tools.my_module import my_tool
   mcp.tool()(my_tool)
   ```

3. **Add exports** in `tools/__init__.py` and validation rules in `tools/validation.py`.

### Configuration

All constants live in `config.py`:
- Directory paths and server settings
- Statistical thresholds (outlier, missing values, cardinality)
- Encoding parameters (one-hot limits, hashing defaults)
- Normalization parameters (fuzzy threshold, max comparisons)

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run normalization tests only
python -m pytest tests/test_normalize_categorical.py tests/test_harmonize_categorical.py tests/test_cluster_categorical.py tests/test_ml_prepare_categorical.py -v

# Run a specific test file
python -m pytest tests/test_encode_categorical.py -v
```

## Troubleshooting

### Server won't start
- Ensure virtual environment is activated: `source .venv/bin/activate`
- Verify all dependencies: `uv sync` or `pip install -r requirements.txt`
- Check Python version: `python --version` (must be 3.10+)

### Dataset not found
- Place CSV/JSON files in the `data/` directory
- Use `list_datasets` to verify files are detected

### Import errors
- Ensure `__init__.py` files exist in `tools/` and `utils/`
- Run from the project root directory

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]