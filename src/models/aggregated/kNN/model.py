import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional

import mlflow
from mlflow.models import infer_signature
from src.models.base_model import Model, ModelConfig
from src.common.utils import split_by_date_grouped

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


@dataclass
class kNNModelConfig(ModelConfig):
    weight_sigma: float = 5
    use_month: bool = False



class kNNModelBuilding(Model):
    def __init__(self, model_config: kNNModelConfig):
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
    

    def _build(self, data_train: pd.DataFrame) -> Tuple[float, float]:
        """
        Stores the training set.
        """
        required_features = ['timestamp','cum_ev_count', 'total_energy']
        missing = [f for f in required_features if f not in data_train.columns]
        if missing:
            raise ValueError(f"Missing required feature(s): {missing}")        

        logger.info(f"Splitting the train data by date with validation_size={self.model_config.validation_size}, random_state={self.model_config.random_state}")
        df_train, df_validation = split_by_date_grouped(data_train, date_col='date', test_size=self.model_config.validation_size, random_state=self.model_config.random_state)

        logger.info(f"Train data shape: {df_train.shape} | Validation data shape: {df_validation.shape}")
 

        historical_data = df_train[required_features].copy()

        # Ensure 'timestamp' is datetime
        historical_data['timestamp'] = pd.to_datetime(historical_data['timestamp'])

        # Create date-based helper columns
        historical_data['date'] = historical_data['timestamp'].dt.date
        historical_data['day_of_week'] = historical_data['timestamp'].dt.day_name()
        historical_data['month'] = historical_data['timestamp'].dt.month_name()
        historical_data['time'] = historical_data['timestamp'].dt.strftime('%H:%M')
        self.historical_data = historical_data.sort_values(['date', 'timestamp']) 

        logger.debug(f"Historical data: shape{self.historical_data.shape} -- columns{self.historical_data.columns.to_list()}") 

        # Define forecast issuance times
        forecast_times = [dt.time() for dt in pd.date_range(start="00:00", end="23:45", freq="15min").tolist()]

        real_ev, real_energy, forecast_ev, forecast_energy = self.test(df_validation, forecast_times)

        rmse_ev = np.sqrt(np.mean((real_ev[:, None, :] - forecast_ev) ** 2))
        rmse_energy = np.sqrt(np.mean((real_energy[:, None, :] - forecast_energy) ** 2))

        return rmse_ev, rmse_energy
       
        
    def forecast(self, prior_data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict future values based on prior data.
        """
        required_features = ['timestamp','cum_ev_count', 'total_energy']
        missing = [f for f in required_features if f not in prior_data.columns]
        if missing:
            raise ValueError(f"Missing required feature(s): {missing}")      
        
        # Check that the first timestamp is 00:00
        if prior_data['timestamp'].iloc[0].time() != pd.Timestamp("00:00").time():
            raise ValueError("First timestamp is not 00:00.")
        
        # Check that time deltas between rows are 15 minutes
        deltas = prior_data['timestamp'].diff().dropna()
        expected_delta = pd.Timedelta(minutes=15)

        if not all(deltas == expected_delta):
            raise ValueError("Timestamps are not monotonically increasing by 15 minutes.")
        
        dow = prior_data['timestamp'].iloc[0].day_name()
        month = prior_data['timestamp'].iloc[0].month_name() if self.model_config.use_month else None            
        
        # --- Historical Pattern Extraction ---            
        hist_matrix_cum_ev_count = self.timeseries_to_day_matrix('cum_ev_count', dow, month)
        hist_matrix_total_energy = self.timeseries_to_day_matrix('total_energy', dow, month)

        y_ev_prior = prior_data['cum_ev_count'].to_numpy()
        y_energy_prior = prior_data['total_energy'].to_numpy()


        # Determine the date from your existing data
        date_str = prior_data['timestamp'].dt.date.iloc[0]  # e.g., 2024-09-12

        # Generate the full day's 15-minute timestamps
        full_range = pd.date_range(start=f"{date_str} 00:00", end=f"{date_str} 23:45", freq="15min")

        # Merge with existing data
        completed = pd.DataFrame({'timestamp': full_range})
        completed = completed.merge(prior_data, on='timestamp', how='left')

        completed['cum_ev_count'].iloc[prior_data.shape[0]:] = self.forecast_weighted_average(y_ev_prior, hist_matrix_cum_ev_count)
        completed['total_energy'].iloc[prior_data.shape[0]:] = self.forecast_weighted_average(y_energy_prior, hist_matrix_total_energy)

        return completed
            

    def test(self, data_test: pd.DataFrame, forecast_issuance_times: list) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Forecasts the rest of the day at multiple issuance times using test data.

        Args:
            data_test: DataFrame with 'timestamp', 'cum_ev_count', 'total_energy' columns.
            forecast_issuance_times: List of datetime.time objects indicating forecast issuance times.
                    
        Returns:
            - y_real_ev_matrix: Actual EV count per 15min slot (days × 96)
            - y_real_energy_matrix: Actual total energy per 15min slot (days × 96)
            - forecast_ev: Forecasted EV counts (days × issuance times × 96)
            - forecast_energy: Forecasted energy (days × issuance times × 96)
            - forecast_issuance_times: Array of issuance times
        """

        required_features = ['timestamp','cum_ev_count', 'total_energy']
        missing = [f for f in required_features if f not in data_test.columns]
        if missing:
            raise ValueError(f"Missing required feature(s): {missing}")      
        
        self.validate_daily_time_coverage(data_test)

        # Add date column and sort
        data_test['date'] = data_test['timestamp'].dt.date
        data_test.sort_values(["date", "timestamp"]).index

        # Prepare real values matrix
        y_real_ev_matrix = data_test['cum_ev_count'].to_numpy().reshape(-1, 96)
        y_real_energy_matrix = data_test['total_energy'].to_numpy().reshape(-1, 96)
        
        unique_dates = sorted(data_test["date"].unique())
        date_to_idx = {d: i for i, d in enumerate(unique_dates)}

        time_bins = pd.date_range("00:00", "23:45", freq="15min").time
        time_to_idx = {t: i for i, t in enumerate(time_bins)}

        # Initialize storage arrays
        n_days, n_issuance, n_time_slots = len(unique_dates), len(forecast_issuance_times), 96
        forecast_ev = np.full((n_days, n_issuance, n_time_slots), np.nan)
        forecast_energy = np.full((n_days, n_issuance, n_time_slots), np.nan)

        # Forecast loop over days
        for date in unique_dates:
            date_idx = date_to_idx[date]
            data_current_day = data_test[data_test['date'] == date]
            dow = data_current_day['timestamp'].iloc[0].day_name()
            month = data_current_day['timestamp'].iloc[0].month_name() if self.model_config.use_month else None
            
            # --- Historical Pattern Extraction ---            
            hist_matrix_cum_ev_count = self.timeseries_to_day_matrix('cum_ev_count', dow, month)
            hist_matrix_total_energy = self.timeseries_to_day_matrix('total_energy', dow, month)
            
            for forecast_issuance_time_index, forecast_issuance_time in enumerate(forecast_issuance_times):  
                cutoff_idx = time_to_idx[forecast_issuance_time]

                y_ev_prior = data_current_day.iloc[:cutoff_idx]['cum_ev_count'].to_numpy()
                y_energy_prior = data_current_day.iloc[:cutoff_idx]['total_energy'].to_numpy()

                forecast_ev[date_idx,forecast_issuance_time_index,:cutoff_idx] = y_ev_prior
                forecast_ev[date_idx,forecast_issuance_time_index,cutoff_idx:] = self.forecast_weighted_average(y_ev_prior, hist_matrix_cum_ev_count)

                forecast_energy[date_idx,forecast_issuance_time_index,:cutoff_idx] = y_energy_prior
                forecast_energy[date_idx,forecast_issuance_time_index,cutoff_idx:] = self.forecast_weighted_average(y_energy_prior, hist_matrix_total_energy)
        
        return y_real_ev_matrix, y_real_energy_matrix, forecast_ev, forecast_energy


    def forecast_weighted_average(self, prior: np.ndarray, historical: np.ndarray) -> np.ndarray:
        """
        Forecasts the rest of the day using a Gaussian-kernel-weighted average over historical data.
        
        Args:
            prior (np.ndarray): Array of shape (T_obs,) representing prior values for the current day.
            historical (np.ndarray): Array of shape (N_days, T_total), where each row is a full day's time series.
            sigma (float): Standard deviation for the Gaussian kernel.
        
        Returns:
            np.ndarray: Forecasted values of shape (T_total - T_obs,)
        """
        T_obs = prior.shape[0]
        sigma = self.model_config.weight_sigma
        # Check dimensions
        if historical.shape[1] <= T_obs:
            raise ValueError(f"Historical data must be longer than the prior time window: historical.shape={historical.shape[1]}, T_obs={T_obs}")

        # Slice historical data up to the current time
        historical_prior = historical[:, :T_obs]

        # Compute distances and weights
        dists = np.linalg.norm(historical_prior - prior, axis=1)
        weights = np.exp(-dists**2 / (2 * sigma**2))
        weights_sum = weights.sum()
        if weights_sum == 0 or np.isnan(weights_sum):
            # Fallback to uniform weights if no valid weights
            weights = np.ones_like(weights) / len(weights)
        else:
            weights /= weights_sum

        # Forecast: weighted average of remaining time steps
        forecast = np.average(historical[:, T_obs:], axis=0, weights=weights)

        if np.isnan(forecast).any():
            if np.isnan(historical[:, T_obs:]).any(): 
                print("{np.isnan(historical[:, T_obs:]}")

            if np.isnan(weights).any(): 
                print("weights")

            raise ValueError("[kNNModelBuilding - forecast_weighted_average]⚠️ There are NaNs in forecast.") 
        
        return forecast
    
    
    def validate_daily_time_coverage(self, df: pd.DataFrame, time_col: str = 'timestamp') -> bool:
        """
        Verifies that each unique day in the timestamp column has exactly 96 entries 
        (i.e., full 15-minute intervals in a 24-hour day).
        
        Args:
            df: Input DataFrame with a datetime column.
            time_col: Name of the timestamp column.
            
        Returns:
            bool: True if all days have 96 entries, False otherwise.
            
        Raises:
            ValueError if any date does not have exactly 96 entries.
        """
        if time_col not in df.columns:
            raise ValueError(f"Column '{time_col}' not found in DataFrame.")
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
            try:
                df[time_col] = pd.to_datetime(df[time_col])
            except Exception:
                raise TypeError(f"Could not convert '{time_col}' to datetime.")
        
        # Extract date and count entries per day
        df['date'] = df[time_col].dt.date
        counts = df.groupby('date').size().reset_index(name='entry_count')

        # Check if all days have 96 entries
        invalid_days = counts[counts['entry_count'] != 96]
        if not invalid_days.empty:
            raise ValueError(
                f"The following days do not have 96 entries:\n{invalid_days.to_string(index=False)}"
            )
        
        # Check uniform frequency
        grouped = df.sort_values(time_col).groupby("date")
        for date, group in grouped:
            time_deltas = group[time_col].diff().dropna()
            if not all(time_deltas == pd.Timedelta(minutes=15)):
                raise ValueError(f"Timestamps in partial_df must have consistent 15mis intervals.")

        return True


    def timeseries_to_day_matrix(self,
        value_col: str,
        day_of_week: str,
        month: Optional[str] = None
    ) -> np.ndarray:
        """
        Converts a filtered timeseries dataframe to a 2D numpy array of shape (num_days, num_intervals_per_day).
        
        Args:
            df (pd.DataFrame): DataFrame containing at least 'timestamp', 'date', and the target `value_col`.
            value_col (str): Column name of the time series value (e.g., 'cum_ev_count', 'total_energy').
            day_of_week (str): Name of the day to filter on (e.g., 'Monday').
            month (Optional[str]): Optional. Name of the month to filter on (e.g., 'February').
        
        Returns:
            np.ndarray: A matrix of shape (num_days, num_time_intervals_per_day).
        """
        if self.historical_data is None or self.historical_data.empty:
            raise ValueError("Historical data is either missing or empty. First build the model!")
        
        # Filter
        mask = self.historical_data['day_of_week'] == day_of_week
        if month:
            mask &= self.historical_data['month'] == month

        df_filtered = self.historical_data[mask].copy()

        if df_filtered.empty:
            filter_msg = f"{day_of_week}s"
            if month:
                filter_msg += f" in {month}"
            raise ValueError(f"No data found for {filter_msg}.")

        # Pivot by date and time
        pivot_df = df_filtered.pivot(index='date', columns='time', values=value_col)
        pivot_df = pivot_df.sort_index(axis=1)

        return pivot_df.to_numpy()
            

    def split_features_targets_and_timedata(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits the input DataFrame into features, target variables, and timedata columns.

        Returns:
            - X: Features (all columns excluding targets and timedata)
            - y: Target variables (['cum_ev_count', 'total_energy'])
            - t: Timedata (['timestamp', 'date'])
        """
        target_cols = ['cum_ev_count', 'total_energy']
        timedata_cols = ['timestamp', 'date']

        # Check that required columns exist
        missing = [col for col in target_cols + timedata_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        feature_cols = sorted(
            [col for col in df.columns if col not in target_cols + timedata_cols]
        )

        x = df[feature_cols]
        y = df[target_cols]
        t = df[timedata_cols]

        return x, y, t



 

