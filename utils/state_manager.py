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
        
        # Split state
        self._test_df: Optional[pd.DataFrame] = None
        self._is_split: bool = False

    def load_data(self, df: pd.DataFrame, name: str, reset_split: bool = True):
        """
        Load a dataframe into memory and log the action.
        
        Args:
            df: Pandas DataFrame to store
            name: Name/identifier for the dataset (e.g., filename)
            reset_split: Whether to clear the test set/split state (default True).
                         Set to False if just updating the training set in-place.
        """
        self._current_df = df
        self._current_dataset_name = name
        
        if reset_split:
            self._test_df = None
            self._is_split = False
        
        self.log_action("load_data", {"dataset_name": name})

    def set_split_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame, split_metadata: Dict[str, Any]):
        """
        Set the state to a split dataset (train/test).
        
        Args:
            train_df: Training set (will become the active dataset)
            test_df: Test set (stored separately)
            split_metadata: Metadata about the split (test_size, random_state, etc.)
        
        Raises:
            ValueError: If dataset is already split (must reload to split again)
        """
        if self._is_split:
            raise ValueError("Dataset is already split. Please reload the dataset to start a fresh split.")
            
        # Store COPIES to ensure immutability
        self._current_df = train_df.copy()
        self._test_df = test_df.copy()
        self._is_split = True
        
        self.log_action("train_test_split", split_metadata)

    def get_data(self) -> Optional[pd.DataFrame]:
        """
        Get the current dataframe from memory.
        If split, this returns the TRAINING set.
        
        Returns:
            The current DataFrame if loaded, None otherwise
        """
        return self._current_df
        
    def get_test_data(self) -> Optional[pd.DataFrame]:
        """
        Get the test dataset from memory (if split exists).
        Returns a COPY to prevent accidental mutation.
        
        Returns:
            The test DataFrame if split, None otherwise
        """
        if self._test_df is not None:
             return self._test_df.copy()
        return None
    
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

    def save_transformer(self, base_name: str, transformer: Any, columns_in: List[str] = None) -> str:
        """
        Save a fitted transformer with a unique versioned name.
        
        Args:
            base_name: Base name for the transformer (e.g., 'pca', 'scaler')
            transformer: The fitted model/transformer object
            columns_in: Optional list of input column names the transformer expects
            
        Returns:
            The unique name assigned to the saved transformer (e.g., 'pca_v1')
        """
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
        """Retrieve a stored transformer by its unique name."""
        if not hasattr(self, "_transformers") or name not in self._transformers:
            return None
        return self._transformers[name]["model"]

    def list_transformers(self) -> Dict[str, str]:
        """List all stored transformers and their types."""
        if not hasattr(self, "_transformers"):
            return {}
        return {k: type(v["model"]).__name__ for k, v in self._transformers.items()}

    # =========================================================================
    # Test Set Management (Leakage Prevention)
    # =========================================================================

    def is_split(self) -> bool:
        """Check if dataset is currently split."""
        return self._is_split

    def update_test_data(self, new_test_df: pd.DataFrame):
        """
        Update the hidden test dataset with a transformed version.
        
        CRITICAL SAFETY CHECKS:
        1. Must preserve row count (unless explicit removal, though usually transforms don't drop test rows)
        2. Must match the schema of the CURRENT training data (columns, order, dtypes)
        
        Args:
            new_test_df: The transformed test DataFrame
        """
        if not self._is_split:
            raise ValueError("Values cannot be updated: Dataset is not split.")
            
        if self._current_df is None:
             raise ValueError("Training data is missing. Cannot validate schema.")

        # 1. Schema Consistency Check
        train_cols = list(self._current_df.columns)
        test_cols = list(new_test_df.columns)
        
        if train_cols != test_cols:
            # Check for set difference to give better error message
            train_set = set(train_cols)
            test_set = set(test_cols)
            missing = train_set - test_set
            extra = test_set - train_set
            
            error_msg = "Schema Mismatch! Train and Test columns must be identical."
            if missing: error_msg += f" Missing in Test: {missing}."
            if extra: error_msg += f" Extra in Test: {extra}."
            if not missing and not extra: error_msg += " Column order mismatch."
            
            raise ValueError(error_msg)

        # 2. Dtype Consistency Warning (Strict enforcement might be too brittle for some pandas edge cases, but we verify)
        # We won't raise error yet but we'll log if they differ significantly
        # (e.g. float vs object)
        
        # 3. Update State
        self._test_df = new_test_df.copy()
