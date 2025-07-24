import logging
from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Dict, Type

from src.common.utils import validate_ev_charging_dataframe

logger = logging.getLogger(__name__)


class CleaningStrategy(ABC):
    """Abstract base class for cleaning strategies."""

    def clean(self, df_input: pd.DataFrame) -> pd.DataFrame:
        """Cleans the dataset and validates structure and types."""

        df_output = self._clean(df_input)
        return validate_ev_charging_dataframe(df_output)

    @abstractmethod
    def _clean(self, df_input: pd.DataFrame) -> pd.DataFrame:
        pass


class NoCleaning(CleaningStrategy):
    """A no-op cleaning strategy that returns the data unchanged."""

    def _clean(self, df_input: pd.DataFrame) -> pd.DataFrame:
        return df_input


class ASRDataCleaning(CleaningStrategy): 
    """Cleans the ASR dataset by dropping sessions before February 2022"""

    def _clean(self, df_input: pd.DataFrame) -> pd.DataFrame:
        cutoff_date = pd.to_datetime("2022-02-01")
        total_sessions = len(df_input)

        df_out = df_input[df_input["start_datetime"] >= cutoff_date].copy()
        dropped_pct = 100 * (total_sessions - len(df_out)) / total_sessions
        logger.info(f"[ASR Cleaning] Dropped {dropped_pct:.1f}% of sessions: only keeping sessions after Feb 2022.")
        
        return df_out
    
class DropWeekendSessions(CleaningStrategy):
    """Removes rows where start_datetime falls on a weekend."""

    def _clean(self, df_input: pd.DataFrame) -> pd.DataFrame:
        total_sessions = len(df_input)

        df_out = df_input[df_input["start_datetime"].dt.weekday < 5].copy()
        dropped_pct = 100 * (total_sessions - len(df_out)) / total_sessions
        logger.info(f"[DropWeekend] Dropped {dropped_pct:.1f}% of sessions: only keeping sessions falls on the weekdays.")
        
        return df_out

# ----- Composite Strategy -----
class CompositeCleaningStrategy(CleaningStrategy):
    """Applies a list of cleaning strategies in sequence."""

    def __init__(self, strategies: List[CleaningStrategy]):
        self.strategies = strategies

    def _clean(self, df_input: pd.DataFrame) -> pd.DataFrame:
        df = df_input
        for strategy in self.strategies:
            df = strategy._clean(df)
        return df
    
# ----- Context -----
class DataCleaner:

    CLEANING_STRATEGY_REGISTRY: Dict[str, Type] = {
    "NoCleaning": NoCleaning,
    "ASRDataCleaning": ASRDataCleaning,
    "DropWeekendSessions": DropWeekendSessions,
    }
    
    def __init__(self):
        self.strategy = NoCleaning()

    def set_strategy(self, strategy_names: List[str]):
        strategies = []

        for name in strategy_names:
            if name not in self.CLEANING_STRATEGY_REGISTRY:
                raise ValueError(f"Unknown cleaning strategy: {name}")
            strategies.append(self.CLEANING_STRATEGY_REGISTRY[name]())

        self.strategy = CompositeCleaningStrategy(strategies)     

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:        
            return self.strategy.clean(df_input=df)
    