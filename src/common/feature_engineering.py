import logging
from abc import ABC, abstractmethod
from typing import Dict, Type, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class FeatureEngineeringStrategy(ABC):
    """Abstract base class for feature engineering strategies."""

    def apply_transformation(self, df_input: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Base class for all feature engineering strategies.
        Enforces:
        - Input and output each contain exactly one of 'timestamp' or 'start_datetime'
        - That column must be of datetime dtype
        """

        # Enforce: input must contain exactly one of 'timestamp' or 'start_datetime'
        time_cols_in = [col for col in ["timestamp", "start_datetime"] if col in df_input.columns]
        if len(time_cols_in) != 1:
            raise ValueError(f"Input must contain exactly one of 'timestamp' or 'start_datetime'. Found: {time_cols_in}")

        df_out = self._apply_transformation(df_input, **kwargs)

        # Enforce: output must also contain exactly one of 'timestamp' or 'start_datetime'
        time_cols_out = [col for col in ["timestamp", "start_datetime"] if col in df_out.columns]
        if len(time_cols_out) != 1:
            raise ValueError(f"Output must contain exactly one of 'timestamp' or 'start_datetime'. Found: {time_cols_out}")
        
        col_out = time_cols_out[0]
        if not pd.api.types.is_datetime64_any_dtype(df_out[col_out]):
            raise TypeError(f"Column '{col_out}' in output must be datetime dtype, but got: {df_out[col_out].dtype}")
                
        return df_out

    @abstractmethod
    def _apply_transformation(self, df_input: pd.DataFrame) -> pd.DataFrame:
        pass

    
class EnrichSessions(FeatureEngineeringStrategy):
    """Adds month, day, and time info to session-based data."""
    def _apply_transformation(self, df_input: pd.DataFrame) -> pd.DataFrame: 
        time_col = "timestamp" if "timestamp" in df_input.columns else "start_datetime"

        df_input["month"] = df_input[time_col].dt.month_name()
        df_input["month_enum"] = df_input[time_col].dt.month
        df_input["day_of_week"] = df_input[time_col].dt.day_name()
        df_input["day_of_week_enum"] = df_input[time_col].dt.weekday
        df_input["date"] = df_input[time_col].dt.date
        df_input["start_time"] = df_input[time_col].dt.time
        if "end_datetime" in df_input.columns:
            df_input["end_time"] = df_input["end_datetime"].dt.time

        return df_input 


class SessionsToEvents(FeatureEngineeringStrategy): 
    """Transforms start/end sessions into arrival/departure events"""

    def _apply_transformation(self, df_input: pd.DataFrame) -> pd.DataFrame:
        required_features = ['start_datetime','end_datetime','total_energy']
        missing = [f for f in required_features if f not in df_input.columns]
        if missing:
            raise ValueError(f"Missing required feature(s): {missing}")   

        if 'start_time' not in df_input.columns:
            df_input["start_time"] = df_input["start_datetime"].dt.time
        if 'end_time' not in df_input.columns:
            df_input["end_time"] = df_input["end_datetime"].dt.time

        core_features = ['start_datetime', 'start_time', 'end_datetime', 'end_time', 'total_energy'] 
        derived_features = ['time_of_day', 'timestamp', 'event_label']                    
        auxiliary_features = [f for f in df_input.columns.tolist() if (f not in core_features) and (f not in derived_features)]            
              
        # Create 'arrival' rows with NaN energy
        df_start = df_input[['start_datetime', 'start_time']+ auxiliary_features].copy()
        df_start = df_start.rename(columns={'start_datetime': 'timestamp'})         
        df_start = df_start.rename(columns={'start_time': 'time_of_day'})       
        df_start['event_label'] = 'arrival'
        df_start['total_energy'] = float('nan')

        # Create 'end' rows with energy filled
        df_end = df_input[['end_datetime', 'end_time','total_energy']+ auxiliary_features].copy()
        df_end = df_end.rename(columns={'end_datetime': 'timestamp'})
        df_end = df_end.rename(columns={'end_time': 'time_of_day'})
        df_end['event_label'] = 'departure'

        # Combine and sort
        df_out = pd.concat([df_start, df_end], ignore_index=True).sort_values('timestamp').reset_index(drop=True)

        return df_out
    
class EventsToTimeseries(FeatureEngineeringStrategy): 
    """Turns event data into 15-minute binned time series."""

    def _apply_transformation(self, df_input: pd.DataFrame) -> pd.DataFrame:
        if 'timestamp' not in df_input.columns:
            raise ValueError("Missing 'timestamp' column.") 
        
        df_input["date"] = df_input['timestamp'].dt.date
        df_input = df_input.set_index('timestamp')

        df_input['ev_count_delta'] = df_input['event_label'].map({'arrival': 1, 'departure': -1})     

        # Define the full time range (00:00 to 23:45) — will use this per day
        time_bins = pd.date_range("00:00", "23:45", freq="15min").time        
        resampled_days = []

        for date, group in df_input.groupby(df_input['date']):

            # Create a full datetime index for the day
            full_index = pd.to_datetime([f"{date} {t}" for t in time_bins])            

            # Resample
            tmp = group[['ev_count_delta']].resample('15min').sum().reindex(full_index, fill_value=0)

            # Add date and cumulative values
            tmp['date'] = date
            tmp['cum_ev_count'] = tmp['ev_count_delta'].cumsum()
            tmp['total_energy'] = group[['total_energy']].resample('15min').sum().reindex(full_index, fill_value=0).cumsum()

            resampled_days.append(tmp)

        # Concatenate all daily results
        df_combined = pd.concat(resampled_days).reset_index().rename(columns={'index': 'timestamp'})
        df_combined['time_interval_index'] = df_combined.groupby('date').cumcount()
        df_combined['time_interval_normalized'] = df_combined['time_interval_index'] / 95

        if 'day_of_week' in df_input.columns:
            df_combined['day_of_week'] = pd.to_datetime(df_combined['date']).dt.dayofweek

        if 'month' in df_input.columns:
            df_combined['month'] = pd.to_datetime(df_combined['date']).dt.month

        df_combined = df_combined.sort_values(['date', 'timestamp'])   

        return df_combined

class CircularEncodeMonth(FeatureEngineeringStrategy): 
    """Encodes month cyclically using sine/cosine."""

    def _apply_transformation(self, df_input: pd.DataFrame) -> pd.DataFrame:
        if 'month' not in df_input.columns:
            raise ValueError("Missing 'month' column.")               
        df_input['month_sin'] = np.sin(2 * np.pi * df_input['month'] / 12)
        df_input['month_cos'] = np.cos(2 * np.pi * df_input['month'] / 12)
        return df_input.drop(columns=["month"]).copy()
    
class OneHotEncodeDayOfWeek(FeatureEngineeringStrategy): 
    """One-hot encodes the day_of_week column."""

    def _apply_transformation(self, df_input: pd.DataFrame) -> pd.DataFrame:
        if 'day_of_week' not in df_input.columns:
            raise ValueError("Missing 'day_of_week' column.")
       
        df_input = pd.get_dummies(df_input, columns=['day_of_week'], prefix='day_of_week', drop_first=False)
        return df_input.copy()

    

# ----- Context -----
class FeatureEngineer:

    FEATURE_ENGINEERING_STRATEGY_REGISTRY: Dict[str, Type] = {
    'EnrichSessions': EnrichSessions,
    'SessionsToEvents': SessionsToEvents,
    'EventsToTimeseries': EventsToTimeseries,
    'CircularEncodeMonth': CircularEncodeMonth,
    'OneHotEncodeDayOfWeek': OneHotEncodeDayOfWeek
    }

    def __init__(self, logging_off: bool = False):
        self.logging_off = logging_off
        self.strategy: Optional[FeatureEngineeringStrategy] = None
        self._available_strategies = {
            'EnrichSessions': EnrichSessions(),
            'SessionsToEvents': SessionsToEvents(),
            'EventsToTimeseries': EventsToTimeseries(),
            'CircularEncodeMonth': CircularEncodeMonth(),
            'OneHotEncodeDayOfWeek': OneHotEncodeDayOfWeek()
        }

    def set_strategy(self, strategy: str):
        if strategy not in self.FEATURE_ENGINEERING_STRATEGY_REGISTRY:
            raise ValueError(f"Unknown strategy '{strategy}'. Available: {list(self.FEATURE_ENGINEERING_STRATEGY_REGISTRY.keys())}")
        self.strategy = self._available_strategies[strategy]
        if not self.logging_off:    
            logger.info(f"Set feature engineering strategy: {strategy}")

    def apply_transformation(self, df: pd.DataFrame, **kwargs):
        if not self.strategy:
            logger.warning("No strategy set. Returning input data unchanged.")
            return df
        return self.strategy.apply_transformation(df, **kwargs)
    
        
    