# New MCP Tools Documentation

This document describes the two new tools added to the Dataset Analysis MCP server: `validate_action` and `create_feature`.

## 1. validate_action

### Purpose
Provides a dry-run validation mode for operations before execution. Estimates memory impact and rejects unsafe operations.

### Input Schema
```json
{
  "type": "object",
  "required": ["tool", "params"],
  "properties": {
    "tool": { "type": "string" },
    "params": { "type": "object" }
  }
}
```

### Output Schema
```json
{
  "type": "object",
  "required": ["allowed"],
  "properties": {
    "allowed": { "type": "boolean" },
    "reason": { "type": "string" },
    "estimated_memory_mb": { "type": "number" }
  }
}
```

### Features
- **Memory Estimation**: Calculates estimated memory usage after operation
- **Safety Checks**: Validates parameters and data types before execution
- **Column Validation**: Ensures columns exist before operations
- **Type Checking**: Verifies numeric columns for numeric operations
- **Read-Only Detection**: Identifies safe read-only operations

### Supported Tool Validations
- `load_dataset_metadata` - Always allowed
- `drop_columns` - Validates columns exist
- `remove_outliers` - Validates column exists and is numeric
- `handle_missing_values` - Validates column exists and has missing values
- `create_feature` - Validates feature name doesn't exist
- `train_test_split` - Validates test_size parameter
- `describe_dataset`, `correlation_analysis`, `detect_data_quality_issues` - Read-only operations
- `drop_duplicates` - Estimates duplicate count

### Example Usage

```python
from tools.validation import validate_action, ValidateActionRequest

# Validate outlier removal
request = ValidateActionRequest(
    tool="remove_outliers",
    params={"column": "Age", "method": "zscore", "threshold": 3.0}
)

response = validate_action(request)
print(f"Allowed: {response.allowed}")
print(f"Reason: {response.reason}")
print(f"Estimated Memory: {response.estimated_memory_mb:.2f} MB")
```

### Error Handling
- Returns `allowed=False` when no dataset is loaded (except for discovery tools)
- Returns `allowed=False` for invalid column names
- Returns `allowed=False` for type mismatches (e.g., outlier removal on text columns)
- Returns `allowed=False` for unknown tools

---

## 2. create_feature

### Purpose
Creates new features in the dataset using pandas expressions. Supports arithmetic operations, conditional logic, string operations, and numpy functions.

### Input Schema
```json
{
  "type": "object",
  "required": ["name", "expression"],
  "properties": {
    "name": { "type": "string" },
    "expression": { "type": "string" }
  }
}
```

### Output Schema
```json
{
  "type": "object",
  "required": ["feature_name"],
  "properties": {
    "feature_name": { "type": "string" },
    "rows_affected": { "type": "integer" },
    "dtype": { "type": "string" },
    "sample_values": { "type": "array" }
  }
}
```

### Features
- **Expression Evaluation**: Safely evaluates pandas/numpy expressions
- **Multiple Column Support**: Can reference multiple columns in expressions
- **Type Flexibility**: Supports Series, lists, arrays, and scalar values
- **Automatic Broadcasting**: Scalar values are broadcast to all rows
- **Pipeline Logging**: Logs feature creation in pipeline history
- **Sample Preview**: Returns sample values from created feature

### Expression Context
The expression is evaluated with access to:
- `df` - The current DataFrame
- `pd` - pandas library
- `np` - numpy library

### Example Expressions

#### Simple Arithmetic
```python
request = CreateFeatureRequest(
    name="AgeDouble",
    expression="df['Age'] * 2"
)
```

#### Multiple Columns
```python
request = CreateFeatureRequest(
    name="FarePerAge",
    expression="df['Fare'] / df['Age']"
)
```

#### Conditional Logic
```python
request = CreateFeatureRequest(
    name="AgeGroup",
    expression="df['Age'].apply(lambda x: 'Adult' if x >= 18 else 'Minor')"
)
```

#### String Operations
```python
request = CreateFeatureRequest(
    name="FirstName",
    expression="df['Name'].str.split(',').str[0]"
)
```

#### Numpy Functions
```python
request = CreateFeatureRequest(
    name="LogAge",
    expression="np.log(df['Age'])"
)
```

#### Scalar Broadcast
```python
request = CreateFeatureRequest(
    name="Constant",
    expression="100"
)
```

### Error Handling
- Raises `ValueError` if no dataset is loaded
- Raises `ValueError` if feature name already exists
- Raises `ValueError` if expression is empty
- Raises `ValueError` for syntax errors in expression
- Raises `ValueError` for invalid column references
- Raises `ValueError` if expression returns wrong number of rows

### Safety Considerations
- Expression evaluation uses `eval()` with controlled context
- Only `df`, `pd`, and `np` are available in evaluation context
- Validates output length matches dataset length
- Prevents overwriting existing columns

---

## Testing

Both tools have comprehensive test suites:

### Run Validation Tests
```bash
source .venv/bin/activate
python tests/test_validation.py
```

### Run Feature Engineering Tests
```bash
source .venv/bin/activate
python tests/test_feature_engineering.py
```

### Run Integration Tests
```bash
source .venv/bin/activate
python tests/test_mcp_integration.py
```

---

## Integration with MCP Server

Both tools are registered in `main.py`:

```python
# Register Phase 5 Tools (Feature Engineering)
from tools.feature_engineering import create_feature
mcp.tool()(create_feature)

# Register Phase 6 Tools (Validation & Safety)
from tools.validation import validate_action
mcp.tool()(validate_action)
```

---

## Best Practices

### Using validate_action
1. Always validate operations before execution in production
2. Check `estimated_memory_mb` to prevent out-of-memory errors
3. Use validation results to provide user feedback
4. Consider validation for all destructive operations

### Using create_feature
1. Use descriptive feature names
2. Test expressions on small datasets first
3. Handle potential division by zero in expressions
4. Consider using `.fillna()` for missing values in expressions
5. Validate feature creation with `validate_action` first

### Example Workflow
```python
# 1. Validate the operation
validate_request = ValidateActionRequest(
    tool="create_feature",
    params={"name": "BMI", "expression": "df['Weight'] / (df['Height'] ** 2)"}
)
validation = validate_action(validate_request)

if validation.allowed:
    # 2. Create the feature
    feature_request = CreateFeatureRequest(
        name="BMI",
        expression="df['Weight'] / (df['Height'] ** 2)"
    )
    result = create_feature(feature_request)
    print(f"Created feature: {result.feature_name}")
else:
    print(f"Validation failed: {validation.reason}")
```

---

## Future Enhancements

### validate_action
- Add more sophisticated memory estimation algorithms
- Support validation for batch operations
- Add cost estimation for cloud deployments
- Implement rollback simulation

### create_feature
- Add support for window functions
- Implement feature templates/presets
- Add automatic feature type inference
- Support for time-series feature engineering
- Integration with feature stores
