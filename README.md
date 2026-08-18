<p align="center">
  <img src="brand/source/prism-logo.png" alt="Prism logo" width="120" />
  <h1 align="center">Prism</h1>
  <p align="center"><strong>Dataset Analysis MCP for Mac</strong></p>
  <p align="center">
    <strong>A stateful Model Context Protocol server that turns LLMs into data scientists.</strong>
  </p>
  <p align="center">
    <img src="https://img.shields.io/pypi/v/dataset-analysis-mcp?style=for-the-badge" />
    <img src="https://img.shields.io/badge/python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
    <img src="https://img.shields.io/badge/MCP-1.26+-00C7B7?style=for-the-badge" />
    <img src="https://img.shields.io/badge/tools-29-blueviolet?style=for-the-badge" />
    <img src="https://img.shields.io/badge/tests-18_suites-success?style=for-the-badge" />
  </p>
</p>

---

Load a CSV. Ask your LLM to clean it, find outliers, normalize categories, encode features, split for ML, and export a reproducible pipeline config — all through natural language.

The server maintains **in-memory state** across tool calls, so the LLM doesn't need to pass DataFrames back and forth. Just talk to your data.

---

## ⚡ Quick Start (macOS)

**Website:** [kushals256.github.io/mcp-server](https://kushals256.github.io/mcp-server/) · **Download:** [Latest release](https://github.com/kushals256/mcp-server/releases/latest)

```bash
# Recommended
pip install dataset-analysis-mcp
dataset-analysis-mcp-setup
dataset-analysis-mcp-doctor
```

Then quit Claude Desktop (Cmd+Q), reopen it, and try:

```text
Load ~/datasets/sample_sales.csv and run a data quality check
```

### Menu bar companion

Download the **DMG** for the native Mac menu bar app. When active, a chart icon appears in your menu bar — click it to disable the server, open your data folder, or run a health check.

The pip setup wizard configures Claude only; it does not include the menu bar app bundle.

### Claude Desktop config (auto-written by setup)

```json
{
  "mcpServers": {
    "dataset-analysis": {
      "command": "dataset-analysis-mcp",
      "args": [],
      "env": { "MCP_DATA_DIR": "~/datasets" }
    }
  }
}
```

If you use `uv` instead of pip, `uvx dataset-analysis-mcp` also works.

### Developer setup

```bash
git clone <repository_url> && cd mcp-server
uv sync
source .venv/bin/activate
python main.py
```

### Releasing

1. Bump `version` in `pyproject.toml`
2. Commit, tag, and push: `git tag v0.1.2 && git push origin main --tags`
3. The **Release** GitHub Action builds the DMG, uploads assets, and publishes to PyPI

For PyPI trusted publishing, add a pending publisher for workflow `release.yml` at [pypi.org/manage/account/publishing/](https://pypi.org/manage/account/publishing/).

---

## 📥 Loading Datasets

There are two ways to load data into the server:

### Option 1: Load from anywhere (recommended)

Use `load_dataset` to load files from **any location** on your machine — no copying required:

```
User: "Analyze the file at ~/Downloads/sales.csv"
→ AI calls load_dataset("~/Downloads/sales.csv")
```

Supported path formats:
- **Absolute**: `/Users/me/data/sales.csv`
- **Home shorthand**: `~/Downloads/sales.csv`
- **Relative** (from CWD): `../data/sales.csv`

Supported file types: `.csv`, `.json`, `.parquet`, `.xlsx`

### Option 2: Use the `data/` directory

Place files in the `data/` folder and use the classic workflow:

```
1. list_datasets()           → see what's available
2. load_dataset_metadata("sales.csv")  → load into memory
```

By default, `data/` is resolved relative to where you start the server. You can override this with the `MCP_DATA_DIR` environment variable (see [Configuration](#️-configuration)).

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

## 🛠️ All 29 Tools

### Phase 1 — Discovery

| Tool | Description |
|:-----|:------------|
| `list_datasets` | Scan `data/` for CSV & JSON files |
| `load_dataset_metadata` | Load a file from `data/` into memory + return metadata |
| `peek_dataset_metadata` | Read-only inspection — **no state mutation** |
| `load_dataset` | Load a dataset from **any path** on your machine (absolute, `~`, relative) |

### Phase 2 — Persistence

| Tool | Description |
|:-----|:------------|
| `save_processed_dataset` | Save to CSV / JSON / Parquet (train or test split) |
| `export_pipeline_config` | Export all operations as reproducible JSON / YAML |

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
| `extract_features` | Extract numeric features from text or datetime columns |
| `reduce_features` | Dimensionality reduction (PCA, feature selection) |
| `remove_features` | Drop engineered or redundant feature columns |
| `generate_preprocessing_report` | Generate a summary report of all transformations |

### Phase 6 — Validation & Safety

| Tool | Description |
|:-----|:------------|
| `validate_action` | Dry-run any tool — get memory estimates and risk flags before executing |

### Phase 7 — Versioning

| Tool | Description |
|:-----|:------------|
| `list_versions` | List saved dataset versions in memory |
| `rollback_version` | Restore a previous dataset version |
| `diff_versions` | Compare two versions side by side |

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
├── main.py                             # Server entry point — registers all 29 tools
├── config.py                           # Centralized thresholds & constants
├── tools/                              # Tool implementations
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
│   ├── versioning.py                   #   list, rollback, diff versions
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

### Data Directory (`MCP_DATA_DIR`)

The `data/` directory used by `list_datasets`, `load_dataset_metadata`, and `save_processed_dataset` defaults to `./data/` relative to your **current working directory**. Override it with an environment variable:

**CLI:**
```bash
export MCP_DATA_DIR=/Users/me/my-datasets
python main.py
```

**Claude Desktop:**
```json
{
  "mcpServers": {
    "dataset-analysis": {
      "command": "/path/to/.venv/bin/python",
      "args": ["/path/to/mcp-server/main.py"],
      "env": { "MCP_DATA_DIR": "/Users/me/my-datasets" }
    }
  }
}
```

> **Note:** `load_dataset` accepts full file paths directly, so it works regardless of `MCP_DATA_DIR`.

### Algorithm Thresholds

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

- **Quickest fix:** Use `load_dataset("~/path/to/your/file.csv")` to load from any location.
- **Using `data/` folder:** Make sure you're running the server from the directory that contains `data/`, or set `MCP_DATA_DIR` to point to the right place.
- **Verify:** Run `list_datasets()` to see what the server can find.
</details>

<details>
<summary><strong>"Unknown tool" in validate_action</strong></summary>

Ensure the tool name matches exactly — use the registered function name, not aliases.
All 29 tools are covered in `validate_action` as of the latest version.
</details>

<details>
<summary><strong>Import errors</strong></summary>

Run from the project root. Ensure `__init__.py` exists in `tools/` and `utils/`.
</details>

---

<p align="center">
  Built with ❤️ using <a href="https://modelcontextprotocol.io">Model Context Protocol</a>
</p>