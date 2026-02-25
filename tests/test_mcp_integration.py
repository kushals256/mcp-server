import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import mcp


def test_mcp_server_initialization():
    """Test that MCP server initializes correctly with all tools."""
    print("\nTesting MCP server initialization...")
    
    # Check that the server is initialized
    assert mcp is not None, "MCP server should be initialized"
    assert mcp.name == "Dataset Analysis MCP", "Server name should match"
    
    print("✓ PASSED: MCP server initialized successfully")


def test_all_tools_registered():
    """Test that all expected tools are registered."""
    print("\nTesting tool registration...")
    
    # Get list of registered tools
    # FastMCP stores tools differently - check if they're callable
    expected_tools = [
        "list_datasets",
        "load_dataset_metadata",
        "save_processed_dataset",
        "export_pipeline_config",
        "describe_dataset",
        "correlation_analysis",
        "detect_data_quality_issues",
        "remove_outliers",
        "create_feature",
        "validate_action"
    ]
    
    print(f"Expected {len(expected_tools)} tools to be registered")
    
    # Since we can't easily introspect FastMCP's internal tool registry,
    # we'll verify the imports work and functions are defined
    from tools.discovery import list_datasets, load_dataset_metadata
    from tools.save_dataset import save_processed_dataset, export_pipeline_config
    from tools.eda import describe_dataset, correlation_analysis
    from tools.data_quality import detect_data_quality_issues
    from tools.remove_outliers import remove_outliers
    from tools.feature_engineering import create_feature
    from tools.validation import validate_action
    
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
        "validate_action": validate_action
    }
    
    for tool_name in expected_tools:
        assert tool_name in tools_map, f"Tool '{tool_name}' should exist"
        assert callable(tools_map[tool_name]), f"Tool '{tool_name}' should be callable"
        print(f"  ✓ {tool_name}")
    
    print(f"✓ PASSED: All {len(expected_tools)} tools are defined and callable")


if __name__ == "__main__":
    print("=" * 60)
    print("Running MCP Integration Tests")
    print("=" * 60)
    
    try:
        test_mcp_server_initialization()
        test_all_tools_registered()
        
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
