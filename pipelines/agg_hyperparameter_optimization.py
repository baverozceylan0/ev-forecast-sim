import os
from pathlib import Path
import yaml
from dataclasses import asdict, dataclass
from typing import List, Optional, Any

from metaflow import FlowSpec, step, Parameter
import mlflow
from datetime import time, datetime, timedelta
from typing import cast
import optuna


import pandas as pd
import numpy as np
from datetime import time

from src.configs_lib import PipelineConfig
from src.common.load_data import DataLoaderFactory
from src.common.clean_data import DataCleaner
from src.common.feature_engineering import FeatureEngineer
from src.common.utils import split_by_date_grouped, log_parameters_mlflow, filter_and_order_columns, add_session_id

from pipelines.common_steps import data_loading_step, feature_engineering_step, data_cleaning_step, data_splitter_step

from src.models.base_model import ModelConfig, Model

from src.models.registry import MODEL_REGISTRY
from src.common.utils import setup_logger
from src.constants import MLFLOW_TRACKING_URI

from omegaconf import MISSING, OmegaConf, DictConfig, ListConfig
from hydra.utils import instantiate
import hydra
from src.configs_lib import BaseConfig, PipelineConfig
import logging

from src.simulators.edf import EDF

LOGGER_LVL = "INFO"

class AggHyperparameterOptimization(FlowSpec):

    log_file: str = Parameter('log_file', default="main.log", type=str)
    config_file: str = Parameter('config_file')
    results_folder: str = Parameter('results_folder')
    search_space_file: str = Parameter('search_space_file')
    
    @step
    def start(self):
        """
        Start the flow.
        Load the configuration files.
        """
        setup_logger(log_file=self.log_file, level=LOGGER_LVL)
        logger = logging.getLogger(__name__)

        logger.info("Logger INFO")
        logger.debug("Logger DEBUG")

        self.config = OmegaConf.load(self.config_file)
        self.pipeline_config: PipelineConfig = instantiate(self.config.pipeline) 
        logger.info(f"Pipeline Configuration: {self.pipeline_config}")

        if OmegaConf.select(self.config, "model_agg") is None:
            err = "'model_agg' is null in the configuration file. Please specify the aggregated model."
            logger.error(err)
            raise ValueError(err)
    
        _model = self.config.model_agg
        if "_target_" not in _model.keys():
            err = f"Missing '_target_' in config section: 'model_agg'"
            logger.error(err)
            raise ValueError(err)
        self.model_config: ModelConfig = instantiate(_model)
        logger.info(f"Using model_agg: {self.model_config}")    

        self.key = 'agg'

        if not isinstance(self.pipeline_config.feature_engineering_strategy_steps, DictConfig):
            err = f"feature_engineering_strategy_steps must be a DictConfig."
            logger.error(err)
            raise ValueError(err)
        if not isinstance(self.pipeline_config.feature_engineering_selected_features, DictConfig):
            err = f"selected_features must be a DictConfig."
            logger.error(err)
            raise ValueError(err)

        if self.key not in self.pipeline_config.feature_engineering_strategy_steps:
            err = f"Missing '{self.key}' in feature_engineering_strategy_steps."
            logger.error(err)
            raise ValueError(err)
        if self.key not in self.pipeline_config.feature_engineering_selected_features:
            err = f"Missing '{self.key}' in feature_engineering_selected_features."
            logger.error(err)
            raise ValueError(err)
        
       
        self.next(self.data_loading)


    @step
    def data_loading(self) -> None:
        """
        Load raw data using the appropriate DataLoader based on the dataset_id
        defined in the pipeline configuration.
        """
        self.df_raw: pd.DataFrame = data_loading_step(self.pipeline_config.dataset_id, log_file=self.log_file)

        self.next(self.data_cleaning)


    @step
    def data_cleaning(self) -> None:
        """
        Clean the raw dataset using the strategy list specified in the pipeline config.
        If the strategy list is empty, skip cleaning.
        """
        self.df_cleaned = data_cleaning_step(self.df_raw, self.pipeline_config.data_cleaning_strategy_steps, log_file=self.log_file)

        self.next(self.data_splitter)


    @step
    def data_splitter(self):
        """
        Split the selected features into training and test sets using date-based grouping.
        """
        self.df_train, self.df_test = data_splitter_step(self.df_cleaned, self.pipeline_config.test_size, self.pipeline_config.random_state, log_file=self.log_file)
        
        self.next(self.feature_engineering)
    

    @step
    def feature_engineering(self):
        """
        Apply feature engineering using the strategy list specified in the pipeline config.
        If the strategy list is empty, skip feature engineering.
        """      
        always_include=['timestamp','date']
        self.df_train_feature_engineered, self.df_test_feature_engineered = feature_engineering_step(self.df_train, self.df_test, self.key, 
                                                                                                     self.pipeline_config.feature_engineering_strategy_steps[self.key], self.pipeline_config.feature_engineering_selected_features[self.key], always_include,
                                                                                                     self.log_file)               

        self.next(self.hyperparameter_optimization)


    @step
    def hyperparameter_optimization(self):
        """
        Load the model defined in the pipeline config.
        Logs parameters and metrics to MLflow and saves the trained model.
        """
        setup_logger(log_file=self.log_file, level=LOGGER_LVL)
        logger = logging.getLogger(__name__)        

        def suggest_from_yaml(trial, space: dict) -> dict:
            """Return a dict of {param_name: suggested_value} from a YAML-defined space."""
            suggested = {}
            for name, spec in space.items():
                ptype = spec.get("type")
                if ptype == "int":
                    low, high = spec["low"], spec["high"]
                    step = spec.get("step")
                    log = spec.get("log", False)
                    if step is not None:
                        val = trial.suggest_int(name, low, high, step=step, log=log)
                    else:
                        val = trial.suggest_int(name, low, high, log=log)
                elif ptype == "float":
                    low, high = float(spec["low"]), float(spec["high"])
                    step = spec.get("step")
                    log = spec.get("log", False)
                    if step is not None:
                        val = trial.suggest_float(name, low, high, step=float(step), log=log)
                    else:
                        val = trial.suggest_float(name, low, high, log=log)
                elif ptype == "categorical":
                    choices = spec["choices"]
                    val = trial.suggest_categorical(name, choices)
                else:
                    raise ValueError(f"Unsupported search space type for '{name}': {ptype}")
                suggested[name] = val
            return suggested

        def objective(trial):

            # Start an MLflow run to log the model training process
            if not mlflow.active_run():
                mlflow.start_run()  # Start a new MLflow run if there isn't one active                   

            params = suggest_from_yaml(trial, search_space)
            for k, v in params.items():
                setattr(self.model_config, k, v)

            _plain_dict = OmegaConf.to_container(self.config.pipeline, resolve=True)
            _plain_dict['pipeline_name'] = _plain_dict.pop('name')
            log_parameters_mlflow(params=_plain_dict)

            _plain_dict = asdict(self.model_config)          
            _plain_dict['model_name'] = _plain_dict.pop('name')
            log_parameters_mlflow(params=_plain_dict)

            try:
                # Load model
                builder_cls = MODEL_REGISTRY[self.key][self.model_config.name]["model_class"]
                model_builder: Model = builder_cls(self.model_config)
                model_builder._build(self.df_train_feature_engineered)

                # Define forecast issuance times
                forecast_times = [dt.time() for dt in pd.date_range(start="00:00", end="23:45", freq="15min").tolist()]

                # Run testing
                real_ev, real_energy, forecast_ev, forecast_energy = model_builder.test(self.df_test_feature_engineered, forecast_times)

                rmse_ev = np.sqrt(np.mean((real_ev[:, None, :] - forecast_ev) ** 2))
                rmse_energy = np.sqrt(np.mean((real_energy[:, None, :] - forecast_energy) ** 2))

                logger.info(f"rmse_ev: {rmse_ev}; rmse_energy: {rmse_energy}")


                # Log parameters manually
                mlflow.log_metric("rmse_cum_ev_count", rmse_ev)
                mlflow.log_metric("rmse_total_energy", rmse_energy)
                
            except Exception as e:
                logging.error(f"Error during model training: {e}")
                raise e
            
            finally:
                # End the MLflow run
                mlflow.end_run()

            return rmse_ev, rmse_energy

        with open(self.search_space_file, "r") as f:
            search_space_cfg = yaml.safe_load(f)
        optuna_cfg = search_space_cfg.get("optuna", {})
        search_space = search_space_cfg.get("search_space", {})

        n_trials = int(optuna_cfg.get("n_trials", 100))
        timeout = optuna_cfg.get("timeout", 1000)
        sampler = optuna.samplers.TPESampler(seed=search_space_cfg.get("seed"))

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(f"optuna_{self.model_config.name}_experiment")
    
        logger.info("Optimization step starts")

        study = optuna.create_study(directions=["minimize", "minimize"], sampler=sampler)
        study.optimize(objective, n_trials=n_trials, timeout=timeout)

        mlflow.end_run()                

        self.next(self.end)    

    @step
    def end(self):
        print("END!")

if __name__ == '__main__':
    AggHyperparameterOptimization()
