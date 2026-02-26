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
    assert mcp.name == "Dataset Analysis MCP", "Server name should match"


def test_all_tools_registered():
    """Test that all expected tools are importable and callable."""

    expected_tools = {
        "list_datasets": "tools.discovery",
        "load_dataset_metadata": "tools.discovery",
        "save_processed_dataset": "tools.save_dataset",
        "export_pipeline_config": "tools.save_dataset",
        "describe_dataset": "tools.eda",
        "correlation_analysis": "tools.eda",
        "detect_data_quality_issues": "tools.data_quality",
        "remove_outliers": "tools.remove_outliers",
        "create_feature": "tools.feature_engineering",
        "validate_action": "tools.validation",
        "list_versions": "tools.versioning",
        "rollback_version": "tools.versioning",
        "diff_versions": "tools.versioning",
    }

    from tools.discovery import list_datasets, load_dataset_metadata
    from tools.save_dataset import save_processed_dataset, export_pipeline_config
    from tools.eda import describe_dataset, correlation_analysis
    from tools.data_quality import detect_data_quality_issues
    from tools.remove_outliers import remove_outliers
    from tools.feature_engineering import create_feature
    from tools.validation import validate_action
    from tools.versioning import list_versions, rollback_version, diff_versions

    tools_map = {
        "list_datasets": list_datasets,
        "load_dataset_metadata": load_dataset_metadata,
        "save_processed_dataset": save_processed_dataset,
        "export_pipeline_config": export_pipeline_config,
        "describe_dataset": describe_dataset,
        "correlation_analysis": correlation_analysis,
        "detect_data_quality_issues": detect_data_quality_issues,
        "remove_outliers": remove_outliers,
        "create_feature": create_feature,
        "validate_action": validate_action,
        "list_versions": list_versions,
        "rollback_version": rollback_version,
        "diff_versions": diff_versions,
    }

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
        print("✓ PASSED: All tools are defined and callable")

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
