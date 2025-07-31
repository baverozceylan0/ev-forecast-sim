import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional

from src.models.base_model import Model, ModelConfig

from datetime import datetime, time, timedelta
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

@dataclass
class BayesianModelConfig(ModelConfig):
    lambda_weight: float
    min_sessions: int
    percentile: float


class BayesianModelBuilding(Model):
    def __init__(self, model_config: BayesianModelConfig):
        self.model_config = model_config    
        self.historical_data: Optional[pd.DataFrame] = None


    def save_model(self, folder_path: str, prefix: Optional[str] = None):
        """
        Save the historical data used by the model as a parquet file.
        """
        os.makedirs(folder_path, exist_ok=True)
        f_name = "historical_data.parquet" if prefix == None else f"{prefix}_historical_data.parquet"
        file_path = os.path.join(folder_path, f_name)
        
        try:
            if self.historical_data is None:
                raise ValueError("No historical data available to save.")

            self.historical_data.to_parquet(file_path)
            logger.info(f"Model data successfully saved to: {file_path}")

        except Exception as e:
            logger.error(f"Failed to save model data to '{file_path}': {e}")
            raise


    def load_model(self, folder_path: str, prefix: Optional[str] = None): 
        """
        Load the historical data used by the model from a parquet file.
        """
        f_name = "historical_data.parquet" if prefix == None else f"{prefix}_historical_data.parquet"
        file_path = os.path.join(folder_path, f_name)

        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"[Model Load] File does not exist: {file_path}")
        
        try:
            self.historical_data = pd.read_parquet(file_path)
            logger.info(f"Model data loaded from: {file_path}")
        except Exception as e:
            logger.error(f"Failed to load model data from {file_path}: {e}")
            raise 
    

    def _build(self, data_train: pd.DataFrame) -> None:
        """
        Stores the training set.
        """
        required_features = ['date', 'EV_id_x', 'total_energy', 'day_of_week', 'start_time', 'end_time']
        missing = [f for f in required_features if f not in data_train.columns]
        if missing:
            raise ValueError(f"Missing required feature(s): {missing}")         

        historical_data = data_train[required_features].copy()
        self.historical_data = historical_data.sort_values(['EV_id_x']) 

        logger.debug(f"Historical data: shape{self.historical_data.shape} -- columns{self.historical_data.columns.to_list()}")

    def forecast(self, prior_data: pd.DataFrame, curr_time: Optional[time] = None) -> pd.DataFrame:
        required_features = ['EV_id_x', 'start_time', 'est_end_time', 'est_total_energy']
        missing = [f for f in required_features if f not in prior_data.columns]
        if missing:
            raise ValueError(f"Missing required feature(s): {missing}")
        if curr_time == None:
            raise ValueError(f"Current time information is missing.") 
        prior_data = prior_data.copy()
        for idx, row in prior_data.iterrows():
            try:
                est_end_time, est_total_energy = self.estimate_departure_and_energy(arrival_time=row['start_time'], EV_id_x=row['EV_id_x'], curr_time=curr_time)
                prior_data.at[idx, 'est_end_time'] = est_end_time
                prior_data.at[idx, 'est_total_energy'] = est_total_energy
            except Exception as e:
                # If estimation fails, return NaNs
                prior_data.at[idx,'est_end_time'] = pd.NaT
                prior_data.at[idx,'est_total_energy'] = np.nan
                logger.debug(f"[Bayesian Estimator] Estimation failed: {e}")

        return prior_data          
        

    def test(self, data_test: pd.DataFrame, forecast_issuance_times: list) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        required_features = ['date', 'EV_id_x', 'total_energy', 'day_of_week', 'start_time', 'end_time']
        missing = [f for f in required_features if f not in data_test.columns]
        if missing:
            raise ValueError(f"Missing required feature(s): {missing}") 
        
        data_test.sort_values(["date", "start_time"]).index

        unique_dates = sorted(data_test["date"].unique())
        date_to_idx = {d: i for i, d in enumerate(unique_dates)}

        results = []

        # Forecast loop over days
        for date in unique_dates:
            date_idx = date_to_idx[date]
            data_current_day = data_test[data_test['date'] == date]

            for forecast_issuance_time_index, forecast_issuance_time in enumerate(forecast_issuance_times):

                mask = (data_current_day['start_time'] < forecast_issuance_time) & (data_current_day['end_time'] >= forecast_issuance_time)
                prior = data_current_day[mask][required_features].copy()
                prior['est_end_time'] = pd.NaT
                prior['est_total_energy'] = np.nan          

                # Add forecast_issuance_time as a column
                prior['forecast_issuance_time'] = forecast_issuance_time

                prior = self.forecast(prior)
                
                results.append(
                    prior[['start_time', 'forecast_issuance_time', 'est_end_time', 'end_time', 'est_total_energy', 'total_energy']].copy()
                )

        results_df = pd.concat(results, ignore_index=True)

        return results_df

                

    def estimate_departure_and_energy(self, arrival_time: time, EV_id_x: int, curr_time: time) -> Tuple[time, float]:
        if self.historical_data is None or self.historical_data.empty:
            raise ValueError("Historical data is either missing or empty. First build the model!")        

        arrival_minutes = arrival_time.hour * 60 + arrival_time.minute
        def to_minutes(t): return t.hour * 60 + t.minute

        mask = (self.historical_data['EV_id_x'] == EV_id_x) & (self.historical_data['end_time'] >= curr_time)
        user_history = self.historical_data[mask]

        if len(user_history) >= self.model_config.min_sessions:
            df = user_history.copy()            
        else:
            mask = (self.historical_data['end_time'] >= curr_time)
            df = self.historical_data[mask].copy()

        df['arrival_min_diff'] = df['start_time'].apply( lambda t: abs(to_minutes(t) - arrival_minutes) )
        df['end_time_min'] = df['end_time'].apply( lambda t: to_minutes(t) )
        
        weights: np.ndarray = np.exp(-self.model_config.lambda_weight * df['arrival_min_diff'].to_numpy())
        end_time: float = self.weighted_percentile(df['end_time_min'].to_numpy(), weights, 100-self.model_config.percentile)
        total_energy: float = self.weighted_percentile(df['total_energy'].to_numpy(), weights, self.model_config.percentile)

        def minutes_to_time(minutes: float) -> time:
            base = datetime.combine(datetime.today(), time(0, 0))
            return (base + timedelta(minutes=minutes)).time()

        return minutes_to_time(end_time), total_energy

    def weighted_percentile(self,
        values: np.ndarray,
        weights: np.ndarray,
        percentile: float
    ) -> float:
        """
        Compute the weighted percentile of `values` with `weights`.
        percentile: float in [0, 100]
        """
        sorted_idx = np.argsort(values)
        sorted_values = values[sorted_idx]
        sorted_weights = weights[sorted_idx]

        cumulative_weight = np.cumsum(sorted_weights)
        normalized_cum_weight = cumulative_weight / cumulative_weight[-1]

        return np.interp(percentile / 100.0, normalized_cum_weight, sorted_values)