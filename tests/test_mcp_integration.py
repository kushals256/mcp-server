"""
MCP Integration Tests.

Verifies that the MCP server initializes correctly and all expected tools
are importable and callable.
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_mcp_server_initialization():
    """Test that MCP server initializes correctly with all tools."""
    from main import mcp

    assert mcp is not None, "MCP server should be initialized"
    assert mcp.name == "Prism", "Server name should match"


def test_all_tools_registered():
    """Test that all 29 expected tools are importable and callable."""

    from tools.discovery import list_datasets, load_dataset_metadata, peek_dataset_metadata, load_dataset
    from tools.save_dataset import save_processed_dataset, export_pipeline_config
    from tools.eda import describe_dataset, correlation_analysis
    from tools.data_quality import detect_data_quality_issues
    from tools.remove_outliers import remove_outliers
    from tools.cast_column_type import cast_column_type
    from tools.drop_columns import drop_columns
    from tools.encode_categorical import encode_categorical_feature
    from tools.train_test_split import train_test_split
    from tools.cleaning import drop_duplicate_rows
    from tools.handle_missing_values import handle_missing_values
    from tools.normalize_categorical import normalize_categorical_text
    from tools.harmonize_categorical import harmonize_categorical_values
    from tools.cluster_categorical import cluster_similar_categories
    from tools.ml_prepare_categorical import ml_prepare_categorical
    from tools.feature_engineering import create_feature
    from tools.extract_features import extract_features
    from tools.reduce_features import reduce_features
    from tools.remove_features import remove_features
    from tools.validation import validate_action
    from tools.persistence import generate_preprocessing_report
    from tools.versioning import list_versions, rollback_version, diff_versions

    tools_map = {
        # Phase 1: Discovery
        "list_datasets": list_datasets,
        "load_dataset_metadata": load_dataset_metadata,
        "peek_dataset_metadata": peek_dataset_metadata,
        "load_dataset": load_dataset,
        # Phase 2: Persistence
        "save_processed_dataset": save_processed_dataset,
        "export_pipeline_config": export_pipeline_config,
        # Phase 3: Analysis
        "describe_dataset": describe_dataset,
        "correlation_analysis": correlation_analysis,
        "detect_data_quality_issues": detect_data_quality_issues,
        # Phase 4: Transformation
        "drop_duplicate_rows": drop_duplicate_rows,
        "handle_missing_values": handle_missing_values,
        "remove_outliers": remove_outliers,
        "cast_column_type": cast_column_type,
        "drop_columns": drop_columns,
        "encode_categorical_feature": encode_categorical_feature,
        "train_test_split": train_test_split,
        # Phase 4.5: Categorical Normalization
        "normalize_categorical_text": normalize_categorical_text,
        "harmonize_categorical_values": harmonize_categorical_values,
        "cluster_similar_categories": cluster_similar_categories,
        "ml_prepare_categorical": ml_prepare_categorical,
        # Phase 5: Feature Engineering
        "create_feature": create_feature,
        "extract_features": extract_features,
        "reduce_features": reduce_features,
        "remove_features": remove_features,
        "generate_preprocessing_report": generate_preprocessing_report,
        # Phase 6: Validation & Safety
        "validate_action": validate_action,
        # Phase 6.5: Versioning
        "list_versions": list_versions,
        "rollback_version": rollback_version,
        "diff_versions": diff_versions,
    }

    assert len(tools_map) == 29, f"Expected 29 tools, got {len(tools_map)}"

    for tool_name, func in tools_map.items():
        assert callable(func), f"Tool '{tool_name}' should be callable"


if __name__ == "__main__":
    print("=" * 60)
    print("Running MCP Integration Tests")
    print("=" * 60)

    try:
        test_mcp_server_initialization()
        print("✓ PASSED: MCP server initialized successfully")

        test_all_tools_registered()
        print("✓ PASSED: All 27 tools are defined and callable")

        print("\n" + "=" * 60)
        print("✓ ALL INTEGRATION TESTS PASSED!")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
