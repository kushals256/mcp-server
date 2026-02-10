"""
Global State Management for MCP Server.

This module provides a singleton class that manages the in-memory state of the
current dataset and pipeline history. This allows tools to share data without
passing large DataFrames between client and server.

Classes:
    GlobalStateManager: Singleton for managing dataset state and operation history
"""

import pandas as pd
from typing import Optional, List, Dict, Any
import copy


class GlobalStateManager:
    """
    Singleton class for managing the global state of the MCP server.
    
    This class maintains:
        - Current dataset in memory (pandas DataFrame)
        - Dataset name/metadata
        - Pipeline history (all operations performed)
    
    The singleton pattern ensures that all tools access the same state instance,
    enabling stateful workflow across multiple tool calls.
    
    Usage:
        manager = GlobalStateManager()  # Always returns the same instance
        manager.load_data(df, "data.csv")
        df = manager.get_data()
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GlobalStateManager, cls).__new__(cls)
            cls._instance.initialize()
        return cls._instance
    
    def initialize(self):
        self._current_df: Optional[pd.DataFrame] = None
        self._current_dataset_name: Optional[str] = None
        self._pipeline_history: List[Dict[str, Any]] = []

    def load_data(self, df: pd.DataFrame, name: str):
        """
        Load a dataframe into memory and log the action.
        
        Args:
            df: Pandas DataFrame to store
            name: Name/identifier for the dataset (e.g., filename)
        """
        self._current_df = df
        self._current_dataset_name = name
        self.log_action("load_data", {"dataset_name": name})

    def get_data(self) -> Optional[pd.DataFrame]:
        """
        Get the current dataframe from memory.
        
        Returns:
            The current DataFrame if loaded, None otherwise
        """
        return self._current_df
    
    def get_dataset_name(self) -> Optional[str]:
        return self._current_dataset_name

    def log_action(self, tool: str, params: Dict[str, Any]):
        """
        Log an action to the pipeline history.
        
        Args:
            tool: Name of the tool/operation
            params: Dictionary of parameters used in the operation
        """
        self._pipeline_history.append({
            "tool": tool,
            "params": params
        })

    def get_history(self) -> List[Dict[str, Any]]:
        return self._pipeline_history

    def clear_state(self):
        """Clear all state and reset to initial values."""
        self.initialize()
