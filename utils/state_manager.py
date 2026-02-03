import pandas as pd
from typing import Optional, List, Dict, Any
import copy

class GlobalStateManager:
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
        """Load a dataframe into memory."""
        self._current_df = df
        self._current_dataset_name = name
        self.log_action("load_data", {"dataset_name": name})

    def get_data(self) -> Optional[pd.DataFrame]:
        """Get the current dataframe."""
        return self._current_df
    
    def get_dataset_name(self) -> Optional[str]:
        return self._current_dataset_name

    def log_action(self, tool: str, params: Dict[str, Any]):
        """Log an action to the pipeline history."""
        self._pipeline_history.append({
            "tool": tool,
            "params": params
        })

    def get_history(self) -> List[Dict[str, Any]]:
        return self._pipeline_history

    def clear_state(self):
        self.initialize()
