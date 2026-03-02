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

from dataset_analysis_mcp.config import MAX_DATASET_VERSIONS
from dataset_analysis_mcp.utils.version_manager import VersionManager

class GlobalStateManager:
    """
    Singleton class for managing the global state of the MCP server.
    
    This class maintains:
        - Current dataset in memory (pandas DataFrame)
        - Dataset name/metadata
        - Pipeline history (all operations performed)
    
    The singleton pattern ensures that all tools access the same state instance,
    enabling stateful workflow across multiple tool calls.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GlobalStateManager, cls).__new__(cls)
            cls._instance.initialize()
        return cls._instance
    
    def initialize(self):
        self._current_df: Optional[pd.DataFrame] = None
        self._test_df: Optional[pd.DataFrame] = None
        self._current_dataset_name: Optional[str] = None
        self._pipeline_history: List[Dict[str, Any]] = []
        self._transformers: Dict[str, Any] = {}
        
        # Split state
        self._is_split: bool = False
        
        # Version management
        self._version_manager = VersionManager(max_versions=MAX_DATASET_VERSIONS)

    def load_data(self, df: pd.DataFrame, name: str, preserve_split: bool = False, reset_split: bool = True):
        """
        Load a NEW dataframe into memory from disk.
        Logs a 'load_data' event and creates a version snapshot.
        """
        # Store a defensive COPY of the input DataFrame
        self._current_df = df.copy()
        self._current_dataset_name = name
        
        # If we are loading entirely new data, we usually wipe the test set
        if not preserve_split and reset_split:
            self._test_df = None
            self._is_split = False
        
        # Auto-snapshot on load
        self._version_manager.snapshot(
            df=self._current_df,
            tool="load_data",
            params={"dataset_name": name}
        )
            
        self.log_action("load_data", {"dataset_name": name})

    def update_data(self, df: pd.DataFrame, tool_name: str = "unknown", tool_params: Optional[Dict[str, Any]] = None):
        """
        Update the CURRENT training dataframe in memory (e.g., after cleaning).
        Does NOT log a 'load_data' event (the tool calling this should log its own action).
        Creates a version snapshot automatically.
        
        Args:
            df: The updated DataFrame.
            tool_name: Name of the tool performing the update (for audit trail).
            tool_params: Parameters passed to the tool (for audit trail).
        """
        self._current_df = df.copy()
        
        # Auto-snapshot on update
        self._version_manager.snapshot(
            df=self._current_df,
            tool=tool_name,
            params=tool_params or {}
        )

    def set_split_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame, metadata: Dict[str, Any]):
        """Store split datasets (Train and Test). Creates a version snapshot of the training set."""
        if self._is_split:
            raise ValueError("Dataset is already split. Please reload the dataset to start a fresh split.")
            
        # Store COPIES to ensure immutability
        self._current_df = train_df.copy()
        self._test_df = test_df.copy()
        self._is_split = True
        
        # Auto-snapshot the training set
        self._version_manager.snapshot(
            df=self._current_df,
            tool="train_test_split",
            params=metadata
        )
        
        self.log_action("train_test_split", metadata)

    def is_split(self) -> bool:
        """Check if the dataset has been split into train/test."""
        return self._is_split

    def get_data(self) -> Optional[pd.DataFrame]:
        """
        Get the training/main dataset.
        Returns a COPY to prevent accidental mutation of internal state.
        """
        if self._current_df is not None:
            return self._current_df.copy()
        return None
    
    def get_test_data(self) -> Optional[pd.DataFrame]:
        """
        Get the testing dataset.
        Returns a COPY to prevent accidental mutation.
        """
        if self._test_df is not None:
            return self._test_df.copy()
        return None
    
    def get_dataset_name(self) -> Optional[str]:
        return self._current_dataset_name

    def log_action(self, tool: str, params: Dict[str, Any]):
        """Log an action to the pipeline history for the final report."""
        self._pipeline_history.append({
            "tool": tool,
            "params": params
        })

    def get_history(self) -> List[Dict[str, Any]]:
        return self._pipeline_history

    # =========================================================================
    # Test Set Management (Leakage Prevention)
    # =========================================================================

    def update_test_data(self, new_test_df: pd.DataFrame):
        """
        Update the hidden test dataset with a transformed version.
        
        CRITICAL SAFETY CHECKS:
        1. Must preserve row count.
        2. Must match the schema of the CURRENT training data (columns, order).
        """
        if not self._is_split:
            raise ValueError("Values cannot be updated: Dataset is not split.")
            
        if self._current_df is None:
             raise ValueError("Training data is missing. Cannot validate schema.")

        # 1. Schema Consistency Check
        train_cols = list(self._current_df.columns)
        test_cols = list(new_test_df.columns)
        
        if train_cols != test_cols:
            train_set = set(train_cols)
            test_set = set(test_cols)
            missing = train_set - test_set
            extra = test_set - train_set
            
            error_msg = "Schema Mismatch! Train and Test columns must be identical."
            if missing: error_msg += f" Missing in Test: {missing}."
            if extra: error_msg += f" Extra in Test: {extra}."
            if not missing and not extra: error_msg += " Column order mismatch."
            
            raise ValueError(error_msg)

        # 3. Update State
        self._test_df = new_test_df.copy()

    # =========================================================================
    # Transformer Persistence & Versioning
    # =========================================================================

    def _get_next_version(self, base_name: str) -> str:
        """
        Generate a unique name for a transformer using auto-incrementing version.
        Example: 'pca' -> 'pca_v1', 'pca_v2'
        """
        if not hasattr(self, "_transformers"):
            self._transformers = {}
            
        version = 1
        while True:
            name = f"{base_name}_v{version}"
            if name not in self._transformers:
                return name
            version += 1

    def save_transformer(self, base_name: str, transformer: Any, columns_in: Optional[List[str]] = None) -> str:
        """Save an ML transformer (like PCA or StandardScaler) for later use."""
        if not hasattr(self, "_transformers"):
            self._transformers = {}
            
        unique_name = self._get_next_version(base_name)
        
        self._transformers[unique_name] = {
            "model": transformer,
            "columns_in": columns_in,
            "version": unique_name.split("_v")[-1]
        }
        
        # Log minimal info to avoid clutter
        self.log_action("save_transformer", {"name": unique_name, "type": type(transformer).__name__})
        return unique_name

    def get_transformer(self, name: str) -> Optional[Any]:
        """Retrieve a saved ML transformer."""
        if not hasattr(self, "_transformers") or name not in self._transformers:
            return None
        return self._transformers[name]["model"]

    def list_transformers(self) -> Dict[str, str]:
        """List all stored transformers and their types."""
        if not hasattr(self, "_transformers"):
            return {}
        return {k: type(v["model"]).__name__ for k, v in self._transformers.items()}

    @property
    def versions(self) -> VersionManager:
        """Access the version manager for listing, diffing, and rollback."""
        return self._version_manager

    def clear_state(self):
        """Completely reset the state manager, including all version history."""
        self.initialize()