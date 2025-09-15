import os
import logging
from typing import Tuple, Optional
from dataclasses import dataclass,asdict
import pandas as pd
import numpy as np

import mlflow
from mlflow.models import infer_signature
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from src.common.utils import add_lag_features
from src.models.base_model import Model

from src.models.base_model import Model, ModelConfig

logger = logging.getLogger(__name__)   


@dataclass
class XGBRegressorModelConfig(ModelConfig):
    n_lags: int
    n_estimators: int
    max_depth: int = 5
    learning_rate: float = 0.02


class XGBRegressorModelBuilding(Model):
    def __init__(self, model_config: XGBRegressorModelConfig):
        self.model_config = model_config    
        params = dict(
            objective="reg:squarederror",
            # booster="dart",
            # rate_drop=0.15,
            # skip_drop=0.0,
            # sample_type="uniform",
            # normalize_type="tree",
            n_estimators=self.model_config.n_estimators,
            learning_rate=self.model_config.learning_rate,
            max_depth=self.model_config.max_depth,
        )        

        self.model_ev = xgb.XGBRegressor(**params)
        self.model_energy = xgb.XGBRegressor(**params)

    def save_model(self, folder_path: str, prefix: Optional[str] = None):
        """
        Save the two xgboost models (ev and energy) as json files.
        """
        os.makedirs(folder_path, exist_ok=True)
        f_name_ev = "model_ev.json" if prefix == None else f"{prefix}_model_ev.json"
        f_name_energy = "model_energy.json" if prefix == None else f"{prefix}_model_energy.json"
        file_path_ev = os.path.join(folder_path, f_name_ev)
        file_path_energy = os.path.join(folder_path, f_name_energy)

        try:
            self.model_ev.save_model(file_path_ev)
            self.model_energy.save_model(file_path_energy)
            logger.info(f"Model data successfully saved to: {folder_path}")

        except Exception as e:
            logger.error(f"Failed to save model data to '{folder_path}': {e}")
            raise


    def load_model(self, folder_path: str, prefix: Optional[str] = None):   
        """
        Load the two xgboost models (ev and energy) form the folder.
        """   
        os.makedirs(folder_path, exist_ok=True)
        f_name_ev = "model_ev.json" if prefix == None else f"{prefix}_model_ev.json"
        f_name_energy = "model_energy.json" if prefix == None else f"{prefix}_model_energy.json"
        file_path_ev = os.path.join(folder_path, f_name_ev)
        file_path_energy = os.path.join(folder_path, f_name_energy)

        if not os.path.isfile(file_path_ev):
            raise FileNotFoundError(f"[Model Load] File does not exist: {file_path_ev}")
        if not os.path.isfile(file_path_energy):
            raise FileNotFoundError(f"[Model Load] File does not exist: {file_path_energy}")
        
        try:
            self.model_ev.load_model(file_path_ev)
            self.model_energy.load_model(file_path_energy)
            logger.info(f"Model data loaded from: {folder_path}")
        except Exception as e:
            logger.error(f"Failed to load model data from {folder_path}: {e}")
            raise 
    
    def get_input_and_target_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        preprocessed_data_train = add_lag_features(
            df=data,
            columns=['cum_ev_count', 'total_energy'],
            n_lags=self.model_config.n_lags,
            fillna_value=0
        )

        # Define input features (exclude target columns)
        target_cols = ['cum_ev_count', 'total_energy']
        feature_cols = preprocessed_data_train.columns.difference(['timestamp', 'date'] + target_cols).tolist()

        x = preprocessed_data_train[feature_cols]
        y = preprocessed_data_train[target_cols]  # shape (N, 2)  
        t = preprocessed_data_train[['timestamp', 'date']]

        return x, y, t  
    
    def _build(self, data_train: pd.DataFrame) -> Tuple[float, float]:
        """
        Train two XGBoost models for cum_ev_count and total_energy. Log metrics and models to MLflow.
        """         
        x_train, y_train, t_train = self.get_input_and_target_data(data_train)        
        logger.info(f"Input Features: {x_train.columns.tolist()}")
        logger.info(f"Target Features: {y_train.columns.tolist()}")

        # Split into training and validation 
        x_tr, x_val, y_tr, y_val = train_test_split(
            x_train, y_train, test_size=self.model_config.validation_size, random_state= self.model_config.random_state)
        logger.debug("Train shape: x=%s, y=%s", x_tr.shape, y_tr.shape)
        logger.debug("Validation shape: x=%s, y=%s", x_val.shape, y_val.shape)

        # Train separate models
        y_tr_ev, y_val_ev = y_tr['cum_ev_count'], y_val['cum_ev_count']
        y_tr_energy, y_val_energy = y_tr['total_energy'], y_val['total_energy']

        self.model_ev.fit(
            x_tr, y_tr_ev,
            eval_set=[(x_tr, y_tr_ev), (x_val, y_val_ev)],
            verbose=False
        )
        results_ev = self.model_ev.evals_result() 
        for step, (tr, va) in enumerate(zip(results_ev["validation_0"]["rmse"],
                                            results_ev["validation_1"]["rmse"])):
            mlflow.log_metric("ev-train-rmse", tr, step=step)
            mlflow.log_metric("ev-valid-rmse", va, step=step)
        
        
        self.model_energy.fit(
            x_tr, y_tr_energy,
            eval_set=[(x_tr, y_tr_energy), (x_val, y_val_energy)],
            verbose=False
        )
        results_energy = self.model_energy.evals_result() 
        for step, (tr, va) in enumerate(zip(results_energy["validation_0"]["rmse"],
                                            results_energy["validation_1"]["rmse"])):
            mlflow.log_metric("energy-train-rmse", tr, step=step)
            mlflow.log_metric("energy-valid-rmse", va, step=step)

        # Evaluate
        y_pred_ev = self.model_ev.predict(x_val)
        y_pred_energy = self.model_energy.predict(x_val)        

        rmse_ev = np.sqrt(mean_squared_error(y_val_ev, y_pred_ev))
        rmse_energy = np.sqrt(mean_squared_error(y_val_energy, y_pred_energy))

        logger.info("Validation RMSE — cum_ev_count: %.4f", rmse_ev)
        logger.info("Validation RMSE — total_energy: %.4f", rmse_energy)

        # Log models to MLflow
        x_sample = x_tr.head(100)
        signature_ev = infer_signature(x_sample, self.model_ev.predict(x_sample))
        signature_energy = infer_signature(x_sample, self.model_energy.predict(x_sample))

        mlflow.xgboost.log_model(
            xgb_model=self.model_ev,
            name="model_ev",
            signature=signature_ev,
            input_example=x_sample[:5], 
            model_format="json",
        )

        mlflow.xgboost.log_model(
            xgb_model=self.model_energy,
            name="model_energy",
            signature=signature_energy,
            input_example=x_sample[:5], 
            model_format="json",
        )

        return rmse_ev, rmse_energy

    
    def forecast(self, prior_data: pd.DataFrame) -> pd.DataFrame:
        x_test, y_test, t_test = self.get_input_and_target_data(prior_data) 


    
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
        """
        x_test, y_test, t_test = self.get_input_and_target_data(data_test)        

        # Split target into two series
        y_test_ev = y_test['cum_ev_count']
        y_test_energy = y_test['total_energy']

        sort_idx = t_test.sort_values(["date", "timestamp"]).index
        t_test = t_test.loc[sort_idx]
        y_test_ev = y_test_ev.loc[sort_idx]
        y_test_energy = y_test_energy.loc[sort_idx]

        y_real_ev_matrix = y_test_ev.to_numpy().reshape(-1, 96)
        y_real_energy_matrix = y_test_energy.to_numpy().reshape(-1, 96)


        t_test["time"] = t_test["timestamp"].dt.time

        time_bins = pd.date_range("00:00", "23:45", freq="15min").time
        time_to_idx = {t: i for i, t in enumerate(time_bins)}

        unique_dates = sorted(t_test["date"].unique())
        date_to_idx = {d: i for i, d in enumerate(unique_dates)}

        # Count number of entries per day
        counts_per_day = t_test.groupby("date").size()

        # Check if all days have 96 time slots
        all_days_complete = (counts_per_day == 96).all()

        # Print result
        if not all_days_complete:
            incomplete_days = counts_per_day[counts_per_day != 96]
            raise ValueError(f"Some days do not have exactly 96 time slots:\n{incomplete_days}")

        # Initialize storage arrays
        n_forecast_issuance_times = len(forecast_issuance_times)
        n_unique_days = t_test["timestamp"].dt.date.nunique()
        n_unique_time_slots = t_test["timestamp"].dt.time.nunique()
        forecast_ev = np.full((n_unique_days, n_forecast_issuance_times, n_unique_time_slots), np.nan)
        forecast_energy = np.full((n_unique_days, n_forecast_issuance_times, n_unique_time_slots), np.nan)


        for forecast_issuance_time_index, forecast_issuance_time in enumerate(forecast_issuance_times): 
            #print(f"forecast_issuance_time: ({forecast_issuance_time},{forecast_issuance_time_index},{time_to_idx[forecast_issuance_time]})")
            forecast_ev[:,forecast_issuance_time_index,:time_to_idx[forecast_issuance_time]] = y_real_ev_matrix[:,:time_to_idx[forecast_issuance_time]]
            forecast_energy[:,forecast_issuance_time_index,:time_to_idx[forecast_issuance_time]] = y_real_energy_matrix[:,:time_to_idx[forecast_issuance_time]]


            x_init = x_test[t_test["time"] == forecast_issuance_time]
            x = x_init.copy()
            for t in range(time_to_idx[forecast_issuance_time],96):
                y_pred_ev_at_time_t = self.model_ev.predict(x)
                y_pred_energy_at_time_t = self.model_energy.predict(x)  
                
                forecast_ev[:,forecast_issuance_time_index,t] = y_pred_ev_at_time_t
                forecast_energy[:,forecast_issuance_time_index,t] = y_pred_energy_at_time_t   
             
                for i in range(self.model_config.n_lags, 1, -1):  
                    x[f'cum_ev_count_lag_{i}'] = x[f'cum_ev_count_lag_{i-1}']
                    x[f'total_energy_lag_{i}'] = x[f'total_energy_lag_{i-1}']

                x[f'cum_ev_count_lag_1'] = y_pred_ev_at_time_t
                x[f'total_energy_lag_1'] = y_pred_energy_at_time_t

                if 'time_interval_index' in x.columns.to_list():
                    x['time_interval_index'] = t

                if 'time_interval_normalized' in x.columns.to_list():
                    x['time_interval_normalized'] = t / 95

        return y_real_ev_matrix, y_real_energy_matrix, forecast_ev, forecast_energy
                

        
       
    
