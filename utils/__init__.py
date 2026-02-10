"""
Utility modules for Dataset Analysis MCP Server.

This package contains utility classes and functions:
- GlobalStateManager: Singleton for managing in-memory dataset state
"""

from .state_manager import GlobalStateManager

__all__ = [
    "GlobalStateManager",
]
