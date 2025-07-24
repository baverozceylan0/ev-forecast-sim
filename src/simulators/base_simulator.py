from abc import ABC, abstractmethod
from typing import Tuple, Optional
from dataclasses import dataclass
from datetime import time
import pandas as pd
import numpy as np


class Simulator(ABC):
    """
    Abstract base class for building, saving, loading, and testing models.
    """

    @abstractmethod
    def initilize(self) -> None:
        pass

    @abstractmethod
    def step(self, curr_time: time, df_agg_timeseries: pd.DataFrame, df_usr_sessions: pd.DataFrame, active_session_info: pd.DataFrame) -> None:
        pass

    def publish_results(self, output_dir: str) -> None:
        pass
