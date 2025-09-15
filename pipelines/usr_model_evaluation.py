import os
from pathlib import Path
import yaml
from dataclasses import asdict, dataclass
from typing import List, Optional, Any

from metaflow import FlowSpec, step, Parameter
import mlflow
from datetime import time, datetime, timedelta
from typing import cast


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

class UsrModelEvaluationPipeline(FlowSpec):

    log_file: str = Parameter('log_file', default="main.log", type=str)
    config_file: str = Parameter('config_file')
    results_folder: str = Parameter('results_folder')
    
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

        if OmegaConf.select(self.config, "model_usr") is None:
            err = "'model_usr' is null in the configuration file. Please specify the user model."
            logger.error(err)
            raise ValueError(err)
    
        _model = self.config.model_usr
        if "_target_" not in _model.keys():
            err = f"Missing '_target_' in config section: 'model_usr'"
            logger.error(err)
            raise ValueError(err)
        self.model_config: ModelConfig = instantiate(_model)
        logger.info(f"Using model_usr: {self.model_config}")    

        self.key = 'usr'

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

        self.next(self.model_building)

    @step
    def model_building(self):
        """
        Load the model defined in the pipeline config.
        Logs parameters and metrics to MLflow and saves the trained model.
        """
        setup_logger(log_file=self.log_file, level=LOGGER_LVL)
        logger = logging.getLogger(__name__)
        
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(f"model_training_{self.model_config.name}")
        with mlflow.start_run():

            # Model Folder is "models/<key>/<model_name>/<version_name>"    
            model_folder = os.path.join('models', self.key, f"{self.model_config.name}_{self.model_config.version}")
            logger.debug(f"Model folder  ({self.key}): {model_folder}")

            try:
                # Build and train model
                builder_cls = MODEL_REGISTRY[self.key][self.model_config.name]["model_class"]
                model_builder: Model = builder_cls(self.model_config)
                model_builder._build(self.df_train_feature_engineered)
            
                logger.info(f"Model training completed for: {self.key}-{self.model_config.name}-{self.model_config.version}")
                
                os.makedirs(model_folder, exist_ok=True)                
                model_builder.save_model(model_folder, prefix="model_evaluation")                   

                with open(os.path.join(model_folder, "model_config.yaml"), "w") as f:
                    _plain_dict = asdict(self.model_config)          
                    _plain_dict['model_name'] = _plain_dict.pop('name')
                    log_parameters_mlflow(params=_plain_dict)
                    yaml.safe_dump(_plain_dict, f, sort_keys=False) 

            except Exception as e:
                logger.error(f"Model building failed: {e}")
                raise

            with open(os.path.join(model_folder, "pipeline_config.yaml"), "w") as f:
                _plain_dict = OmegaConf.to_container(self.config.pipeline, resolve=True)
                _plain_dict['pipeline_name'] = _plain_dict.pop('name')
                log_parameters_mlflow(params=_plain_dict)
                yaml.safe_dump(_plain_dict, f, sort_keys=False)

        mlflow.end_run()                

        self.next(self.model_testing)    

    @step
    def model_testing(self):
        """
        Load and test the trained model for each day in the test data. Save KPIs.
        """
        setup_logger(log_file=self.log_file, level=LOGGER_LVL)
        logger = logging.getLogger(__name__)

        # Load model
        model_folder = os.path.join('models', self.key, f"{self.model_config.name}_{self.model_config.version}")
        builder_cls = MODEL_REGISTRY[self.key][self.model_config.name]["model_class"]
        model_builder: Model = builder_cls(self.model_config)
        model_builder.load_model(model_folder, prefix="model_evaluation")

        # Define forecast issuance times
        forecast_times = [dt.time() for dt in pd.date_range(start="00:00", end="23:45", freq="15min").tolist()]

        # Run testing
        results: pd.DataFrame = model_builder.test(self.df_test_feature_engineered, forecast_times)

        logger.debug(f"Forecast Issuance Times: {forecast_times}")

        # Prepare output directory
        results_folder = os.path.join(model_folder,"results")
        os.makedirs(results_folder, exist_ok=True)

        np.save(os.path.join(results_folder, 'forecast_issuance_times.npy'), np.array(forecast_times))
        results.to_csv(os.path.join(results_folder, 'results.csv'))
        results.to_parquet(os.path.join(results_folder, 'results.parquet'))

        logger.info(f"Forecast results saved to: {results_folder}")

        self.next(self.end)

    
    @step
    def end(self):
        print("END!")

if __name__ == '__main__':
    UsrModelEvaluationPipeline()
