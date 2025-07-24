from abc import ABC, abstractmethod
from typing import Tuple, Optional
from dataclasses import dataclass
from datetime import time
import pandas as pd
import numpy as np

@dataclass
class ModelConfig:
    name: str
    version: str

class Model(ABC):
    """
    Abstract base class for building, saving, loading, and testing models.
    """

    @abstractmethod
    def _build(self, data_train: pd.DataFrame) -> None:
        """
        Train the model using the training data.
        """
        pass    

    @abstractmethod
    def save_model(self, folder_path: str, prefix: Optional[str] = None):
        """
        Save the model to disk.
        """
        pass

    @abstractmethod
    def load_model(self, folder_path: str, prefix: Optional[str] = None):
        """
        Load the model from disk.
        """
        pass

    
    @abstractmethod
    def forecast(self, prior_data: pd.DataFrame, curr_time: Optional[time] = None) -> pd.DataFrame:
        """
        Predict future values based on prior data.
        """
        pass


    @abstractmethod
    def test(self, data_test: pd.DataFrame, forecast_issuance_times: list) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate the model on test data or perform diagnostic checks.
        """
        pass
