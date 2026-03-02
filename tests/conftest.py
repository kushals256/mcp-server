"""
Pytest configuration for MCP Server tests.

Provides an autouse fixture that resets the GlobalStateManager singleton
before every test function, preventing state contamination between tests.
"""

import sys
import os
import pytest

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset_analysis_mcp.utils.state_manager import GlobalStateManager


@pytest.fixture(autouse=True)
def reset_global_state():
    """Reset the GlobalStateManager singleton before each test.
    
    This prevents state contamination between tests caused by the
    singleton pattern. Without this, tests that load data into the
    manager can affect subsequent tests that expect a clean state.
    """
    manager = GlobalStateManager()
    manager.clear_state()
    yield
    # Teardown: also clear after the test for good measure
    manager.clear_state()
