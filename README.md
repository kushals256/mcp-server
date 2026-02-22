<p align="center">
  <h1 align="center">🧙‍♂️ Dataset Analysis MCP Server</h1>
  <p align="center">
    <strong>A stateful Model Context Protocol server that turns LLMs into data scientists.</strong>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
    <img src="https://img.shields.io/badge/MCP-1.26+-00C7B7?style=for-the-badge" />
    <img src="https://img.shields.io/badge/tools-22-blueviolet?style=for-the-badge" />
    <img src="https://img.shields.io/badge/tests-18_suites-success?style=for-the-badge" />
  </p>
</p>

---

Load a CSV. Ask your LLM to clean it, find outliers, normalize categories, encode features, split for ML, and export a reproducible pipeline config — all through natural language.

The server maintains **in-memory state** across tool calls, so the LLM doesn't need to pass DataFrames back and forth. Just talk to your data.

---

## ⚡ Quick Start

```bash
# Clone & setup
git clone <repository_url> && cd mcp-server
uv sync  # or: pip install -r requirements.txt

# Run
source .venv/bin/activate
python main.py
```

### Claude Desktop Config

```json
{
  "mcpServers": {
    "dataset-analysis": {
      "command": "/path/to/.venv/bin/python",
      "args": ["/path/to/mcp-server/main.py"]
    }
  }
}
```

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        LLM (Claude, etc.)                        │
└─────────────────────────────┬────────────────────────────────────┘
                              │ MCP Protocol
┌─────────────────────────────▼────────────────────────────────────┐
│                      FastMCP Server (main.py)                    │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              GlobalStateManager (Singleton)                 │ │
│  │  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌────────────┐  │ │
│  │  │ DataFrame│  │ Test Set  │  │ Pipeline │  │Transformers│  │ │
│  │  │ (active) │  │ (hidden)  │  │ History  │  │ (fitted)   │  │ │
│  │  └──────────┘  └───────────┘  └──────────┘  └────────────┘  │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Phase 1        Phase 3         Phase 4          Phase 4.5       │
│  ┌─────────┐   ┌───────────┐   ┌────────────┐   ┌───────────┐    │
│  │Discovery│──▶│ Analysis  │──▶│  Transform │──▶│ Normalize │    │
│  └─────────┘   └───────────┘   └────────────┘   └─────┬─────┘    │
│                                                       │          │
│  Phase 5         Phase 6         Phase 2              │          │
│  ┌──────────┐   ┌───────────┐   ┌────────────┐        │          │
│  │ Feature  │◀──│ Validate  │   │  Persist   │◀───────┘          │
│  │  Eng.    │   │  (Dry-run)│   │  (Save)    │                   │
│  └──────────┘   └───────────┘   └────────────┘                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ All 22 Tools

### Phase 1 — Discovery

| Tool | Description |
|:-----|:------------|
| `list_datasets` | Scan `data/` for CSV & JSON files |
| `load_dataset_metadata` | Load a file into memory + return metadata |
| `peek_dataset_metadata` | Read-only inspection — **no state mutation** |

### Phase 2 — Persistence

| Tool | Description |
|:-----|:------------|
| `save_processed_dataset` | Save to CSV / JSON / Parquet (train or test split) |
| `export_pipeline_config` | Export all operations as reproducible JSON / YAML |
| `generate_preprocessing_report` | Generate a summary report of all transformations |

### Phase 3 — Analysis

| Tool | Description |
|:-----|:------------|
| `describe_dataset` | Statistical summary (mean, std, quantiles, value counts) |
| `correlation_analysis` | Pearson · Spearman · Kendall · Cramér's V · Eta |
| `detect_data_quality_issues` | Missing values · outliers · duplicates · high cardinality · **zero-variance** |

### Phase 4 — Transformation

| Tool | Description |
|:-----|:------------|
| `drop_duplicate_rows` | Remove exact-match duplicate rows |
| `handle_missing_values` | Impute or drop missing values (multiple strategies) |
| `remove_outliers` | Z-score · IQR · Modified Z · Isolation Forest · LOF |
| `cast_column_type` | Cast columns to int, float, str, bool, datetime, category |
| `drop_columns` | Drop columns with **duplicate-creation warnings** for identity columns |
| `encode_categorical_feature` | 8 methods: one-hot · label · ordinal · frequency · target · binary · hashing · leave-one-out |
| `train_test_split` | Stratified splitting with immutability guarantees |

### Phase 4.5 — Categorical Normalization Pipeline

A 4-layer pipeline that runs **before encoding** to clean messy categorical data:

```
  ① normalize_categorical_text        surface cleanup (unicode, accents, casing)
              ↓
  ② harmonize_categorical_values      synonym → canonical mapping (FlashText)
              ↓
  ③ cluster_similar_categories        fuzzy typo clustering (RapidFuzz)
              ↓
  ④ ml_prepare_categorical            ML-aware dedup / GapEncoder (skrub)
```

| Layer | Tool | Engine |
|:-----:|:-----|:-------|
| 1 | `normalize_categorical_text` | `unicodedata` · `clean-text` · `text-unidecode` |
| 2 | `harmonize_categorical_values` | `flashtext` (Aho-Corasick) |
| 3 | `cluster_similar_categories` | `rapidfuzz` |
| 4 | `ml_prepare_categorical` | `skrub` (lazy-loaded) |

### Phase 5 — Feature Engineering

| Tool | Description |
|:-----|:------------|
| `create_feature` | Create derived columns from Python/pandas expressions |

### Phase 6 — Validation & Safety

| Tool | Description |
|:-----|:------------|
| `validate_action` | Dry-run any tool — get memory estimates and risk flags before executing |

---

## 🔒 Safety Features

| Feature | How |
|:--------|:----|
| **Dry-run validation** | `validate_action` estimates memory, flags leakage risk, blocks unsafe cardinality |
| **Source file protection** | `save_processed_dataset` blocks overwriting the loaded source file |
| **Identity column warnings** | `drop_columns` warns when dropping ID columns would create duplicates |
| **Split immutability** | `train_test_split` prevents re-splitting; test set is hidden from training operations |
| **Cardinality guards** | One-hot encoding blocked at >100 unique values, warned at >20 |
| **Zero-variance detection** | `detect_data_quality_issues` flags constant and near-constant columns |
| **NaN preservation** | All encoders explicitly preserve NaN rows through transformations |

---

## 📂 Project Structure

```
mcp-server/
├── main.py                             # Server entry point — registers all 22 tools
├── config.py                           # Centralized thresholds & constants
├── tools/                              # Tool implementations (23 files)
│   ├── discovery.py                    #   list, load, peek datasets
│   ├── save_dataset.py                 #   save data + export pipeline
│   ├── persistence.py                  #   preprocessing reports
│   ├── eda.py                          #   describe + correlation
│   ├── data_quality.py                 #   quality issue detection
│   ├── cleaning.py                     #   drop duplicates
│   ├── handle_missing_values.py        #   missing value strategies
│   ├── remove_outliers.py              #   5 outlier removal methods
│   ├── cast_column_type.py             #   type casting
│   ├── drop_columns.py                 #   column dropping + warnings
│   ├── encode_categorical.py           #   8 encoding methods
│   ├── train_test_split.py             #   stratified splitting
│   ├── normalize_categorical.py        #   Layer 1: surface cleanup
│   ├── harmonize_categorical.py        #   Layer 2: synonym mapping
│   ├── cluster_categorical.py          #   Layer 3: fuzzy clustering
│   ├── ml_prepare_categorical.py       #   Layer 4: ML-aware prep
│   ├── feature_engineering.py          #   expression-based features
│   └── validation.py                   #   dry-run safety checks
├── utils/
│   └── state_manager.py                # GlobalStateManager singleton
├── tests/                              # 18 test suites
├── data/                               # Input/output datasets
├── pyproject.toml                      # Dependencies & build config
└── requirements.txt                    # Pip-compatible deps
```

---

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run a specific suite
python -m pytest tests/test_encode_categorical.py -v

# Run normalization pipeline tests
python -m pytest tests/test_normalize_categorical.py \
                 tests/test_harmonize_categorical.py \
                 tests/test_cluster_categorical.py \
                 tests/test_ml_prepare_categorical.py -v

# With coverage
python -m pytest tests/ --cov=tools --cov-report=term-missing
```

---

## 🧑‍💻 Adding a New Tool

```python
# 1. Create tools/my_tool.py
from utils.state_manager import GlobalStateManager

def my_tool(dataset_name: str, column: str) -> dict:
    manager = GlobalStateManager()
    df = manager.get_data()
    if df is None:
        return {"error": "No dataset loaded"}

    # ... transform df ...

    manager.load_data(df_modified, dataset_name, reset_split=False)
    manager.log_action("my_tool", {"column": column})
    return {"result": "done"}
```

```python
# 2. Register in main.py
from tools.my_tool import my_tool
mcp.tool()(my_tool)
```

```python
# 3. Add to validate_action (tools/validation.py)
elif tool == "my_tool":
    return ValidateActionResponse(allowed=True, reason="...", estimated_memory_mb=current_memory_mb)
```

---

## ⚙️ Configuration

All thresholds live in `config.py`:

| Category | Key Constants |
|:---------|:-------------|
| **Outlier Detection** | `DEFAULT_ZSCORE_THRESHOLD=3.0`, `DEFAULT_IQR_MULTIPLIER=1.5` |
| **Encoding Limits** | `ONE_HOT_MAX_CARDINALITY=20`, `ONE_HOT_BLOCK_CARDINALITY=100` |
| **Fuzzy Matching** | `FUZZY_SCORE_THRESHOLD=85`, `FUZZY_MAX_COMPARISONS=1000` |
| **Missing Values** | Low/Medium/High severity thresholds |
| **Quality Detection** | Cardinality ratios, skewness/kurtosis bounds |

---

## 📦 Dependencies

| Package | Purpose |
|:--------|:--------|
| `mcp` | Model Context Protocol server framework |
| `pandas` | DataFrame operations |
| `scikit-learn` | Outlier detection, label encoding, train/test split |
| `category-encoders` | Target, binary, hashing, leave-one-out encoding |
| `rapidfuzz` | Fuzzy string matching for category clustering |
| `flashtext` | Aho-Corasick keyword replacement for synonym mapping |
| `skrub` | ML-aware deduplication and GapEncoder |
| `clean-text` | Unicode fixing, control char stripping |
| `scipy` | Statistical tests for correlation analysis |

---

## 🛠️ Troubleshooting

<details>
<summary><strong>Server won't start</strong></summary>

```bash
source .venv/bin/activate
python --version  # Must be 3.10+
uv sync           # or pip install -r requirements.txt
```
</details>

<details>
<summary><strong>Dataset not found</strong></summary>

Place CSV/JSON files in the `data/` directory, then use `list_datasets` to verify.
</details>

<details>
<summary><strong>"Unknown tool" in validate_action</strong></summary>

Ensure the tool name matches exactly — use the registered function name, not aliases.
All 22 tools are covered in `validate_action` as of the latest version.
</details>

<details>
<summary><strong>Import errors</strong></summary>

Run from the project root. Ensure `__init__.py` exists in `tools/` and `utils/`.
</details>

---

<p align="center">
  Built with ❤️ using <a href="https://modelcontextprotocol.io">Model Context Protocol</a>
</p>