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
from src.simulators.focs import Optimizer
from src.simulators.uncontrolled import Uncontrolled
from src.simulators.pp import OA_benchmark
from src.simulators.avr import AVR_benchmark
from src.simulators.lyncs import LYNCS
from src.simulators.llyncs import lLYNCS
from src.simulators.almightyoracle import Oracle_benchmark

LOGGER_LVL = logging.DEBUG

class ModelEvaluationPipeline(FlowSpec):

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

        self.config = OmegaConf.load(self.config_file)
        self.pipeline_config: PipelineConfig = instantiate(self.config.pipeline) 
        logger.info(f"Pipeline Configuration: {self.pipeline_config}")

        self.model_agg_skip_flag = (OmegaConf.select(self.config, "model_agg") is None)
        self.model_usr_skip_flag = (OmegaConf.select(self.config, "model_usr") is None)

        self.model_skip_flags = {'agg': (OmegaConf.select(self.config, "model_agg") is None), 'usr': (OmegaConf.select(self.config, "model_usr") is None)}
        if self.model_skip_flags['agg'] and self.model_skip_flags['usr']:
            err = "Both model_agg and model_usr are null. Please specify at least one model."
            logger.error(err)
            raise ValueError(err)
    
        self.model_config: dict[str, Optional[ModelConfig]] = {'agg': None, 'usr': None}
        for key in ['agg', 'usr']:
            if not self.model_skip_flags[key]:
                _model = self.config.model_agg if key == 'agg' else self.config.model_usr
                if "_target_" not in _model.keys():
                    err = f"Missing '_target_' in config section: 'model_{key}'"
                    logger.error(err)
                    raise ValueError(err)
                self.model_config[key] = instantiate(_model)
                logger.info(f"Using model_{key}: {self.model_config[key]}")     
            else:
                logger.debug(f"model_{key} is not give!")     
       
        self.next(self.data_loading_step)


    @step
    def data_loading_step(self) -> None:
        """
        Load raw data using the appropriate DataLoader based on the dataset_id
        defined in the pipeline configuration.
        """
        setup_logger(log_file=self.log_file, level=LOGGER_LVL)
        logger = logging.getLogger(__name__)


        try:
            loader = DataLoaderFactory.get_loader(
                self.pipeline_config.dataset_id,
                force_download=False
            )
            df_raw: pd.DataFrame = loader.load()

            if df_raw is None or df_raw.empty:
                raise ValueError(f"Loaded data is empty for dataset: {self.pipeline_config.dataset_id}")

            self.df_raw = df_raw
            logger.info(f"Loaded raw data with shape: {df_raw.shape}")  
            
        except Exception as e:
            err = f"Failed to load data for dataset '{self.pipeline_config.dataset_id}': {e}"
            logger.error(err)
            raise

        self.next(self.data_cleaning_step)


    @step
    def data_cleaning_step(self) -> None:
        """
        Clean the raw dataset using the strategy list specified in the pipeline config.
        If the strategy list is empty, skip cleaning.
        """
        setup_logger(log_file=self.log_file, level=LOGGER_LVL)
        logger = logging.getLogger(__name__)

        try:
            strategy_steps = self.pipeline_config.data_cleaning_strategy_steps

            if not strategy_steps:
                logger.debug("No data cleaning strategies provided. Skipping cleaning step.")
                self.df_cleaned = self.df_raw.copy()
            else:

                cleaner = DataCleaner()
                cleaner.set_strategy(strategy_steps)

                self.df_cleaned: pd.DataFrame = cleaner.clean(self.df_raw)

                if self.df_cleaned is None or self.df_cleaned.empty:
                    err = f"Data cleaning produced empty output for steps: {strategy_steps}"
                    logger.error(err)  
                    raise ValueError(err)

            logger.info(f"Cleaned data shape: {self.df_cleaned.shape}")            

        except Exception as e:
            err = f"Failed to clean data:'{self.pipeline_config.data_cleaning_strategy_steps}': {e}"
            logger.error(err)
            raise

        self.next(self.data_splitter_step)


    @step
    def data_splitter_step(self):
        """
        Split the selected features into training and test sets using date-based grouping.
        """
        setup_logger(log_file=self.log_file, level=LOGGER_LVL)
        logger = logging.getLogger(__name__)

        test_size = self.pipeline_config.test_size
        random_state = self.pipeline_config.random_state

        logger.info(f"Splitting data by date with test_size={test_size}, random_state={random_state}")

        self.df_cleaned['date'] = self.df_cleaned['start_datetime'].dt.date
        self.df_train, self.df_test = split_by_date_grouped(self.df_cleaned, date_col='date', test_size=test_size, random_state=random_state)

        logger.info(f"Train data shape: {self.df_train.shape} | Test data shape: {self.df_test.shape}")

        self.next(self.feature_engineering_step)
    

    @step
    def feature_engineering_step(self):
        """
        Apply feature engineering using the strategy list specified in the pipeline config.
        If the strategy list is empty, skip feature engineering.
        """      
        setup_logger(log_file=self.log_file, level=LOGGER_LVL)
        logger = logging.getLogger(__name__)

        def apply_feature_engineering_steps(strategy_steps, df_input: pd.DataFrame) -> pd.DataFrame:
            if not strategy_steps:
                logger.debug("No feature engineering strategies provided. Skipping feature engineering step.")
                df_feature_engineered: pd.DataFrame = df_input.copy()
            else:
                engineer = FeatureEngineer() 
                df_feature_engineered: pd.DataFrame = df_input.copy()
                for s in strategy_steps:
                    engineer.set_strategy(s)
                    df_feature_engineered: pd.DataFrame = engineer.apply_transformation(df_feature_engineered)
                    logger.debug(f"Features after feature engineering step'{df_feature_engineered.columns.to_list()}':")

                if df_feature_engineered is None:
                    err = f"Feature engineering produced empty output for dataset: {strategy_steps}"
                    logger.error(err)
                    raise ValueError(err)
                
            return df_feature_engineered
    
        strategy_steps = self.pipeline_config.feature_engineering_strategy_steps
        selected_features = self.pipeline_config.feature_engineering_selected_features
        if not isinstance(strategy_steps, DictConfig):
            raise ValueError("feature_engineering_strategy_steps must be a DictConfig.")
        if not isinstance(selected_features, DictConfig):
            raise ValueError("selected_features must be a DictConfig.")
        
        self.df_train_feature_engineered: dict[str, Optional[pd.DataFrame]] = {'agg': None, 'usr': None}
        self.df_test_feature_engineered: dict[str, Optional[pd.DataFrame]] = {'agg': None, 'usr': None}
        always_include = {'agg': ['timestamp','date'], 'usr': ['start_datetime','date']}
        for key in ['agg', 'usr']:
            if not self.model_skip_flags[key]:      
                if key not in strategy_steps:
                    err = f"Missing '{key}' in feature_engineering_strategy_steps."
                    logger.error(err)
                    raise ValueError(err)
                if key not in selected_features:
                    err = f"Missing '{key}' in feature_engineering_selected_features."
                    logger.error(err)
                    raise ValueError(err)
                
                if not isinstance(strategy_steps[key], ListConfig):
                    err = f"strategy_steps['{key}'] must be a list."
                    logger.error(err)
                    raise ValueError(err)
                if not isinstance(selected_features[key], ListConfig):
                    err = f"selected_features['{key}'] must be a list."
                    logger.error(err)
                    raise ValueError(err)

                if not all(isinstance(step, str) for step in strategy_steps[key]):
                    err = f"All entries in strategy_steps['{key}'] must be strings."
                    logger.error(err)
                    raise ValueError(err)
                if not all(isinstance(feature, str) for feature in selected_features[key]):
                    err = f"All entries in selected_features['{key}'] must be strings."
                    logger.error(err)
                    raise ValueError(err)
                
                 
                self.df_train_feature_engineered[key] = apply_feature_engineering_steps(strategy_steps[key], self.df_train)
                self.df_test_feature_engineered[key] = apply_feature_engineering_steps(strategy_steps[key], self.df_test)     

                self.df_train_feature_engineered[key] = filter_and_order_columns(self.df_train_feature_engineered[key], selected_features[key], always_include[key])
                self.df_test_feature_engineered[key] = filter_and_order_columns(self.df_test_feature_engineered[key], selected_features[key], always_include[key])

                logger.debug(f"Selected Features ({key}): {self.df_test_feature_engineered[key].columns.tolist()}")
                logger.info(f"Data shape after feature engineering: {key}-train: {self.df_train_feature_engineered[key].shape} | {key}-test: {self.df_test_feature_engineered[key].shape}")


        self.next(self.model_building_step)

    @step
    def model_building_step(self):
        """
        Load the model defined in the pipeline config.
        Logs parameters and metrics to MLflow and saves the trained model.
        """
        setup_logger(log_file=self.log_file, level=LOGGER_LVL)
        logger = logging.getLogger(__name__)
        
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        with mlflow.start_run():
            log_parameters_mlflow(params=asdict(self.pipeline_config))
            for key in ['agg', 'usr']:
                if not self.model_skip_flags[key]:     
                    with mlflow.start_run(nested=True):
                        try:
                            # Log all configs
                            log_parameters_mlflow(params=asdict(self.model_config[key]))

                            # Build and train model
                            print(self.model_config[key])
                            builder_cls = MODEL_REGISTRY[key][self.model_config[key].name]["model_class"]
                            model_builder: Model = builder_cls(self.model_config[key])
                            model_builder._build(self.df_train_feature_engineered[key])
                        
                            logger.info(f"Model training completed for: {key}-{self.model_config[key].name}-{self.model_config[key].version}")        

                            # Save model to "models/aggreagated/<model_name>/<version_name>"      
                            model_folder = os.path.join('models', key, f"{self.model_config[key].name}_{self.model_config[key].version}")
                            os.makedirs(model_folder, exist_ok=True)
                            logger.debug(f"Model folder  ({key}): {model_folder}")
                            model_builder.save_model(model_folder, prefix="model_simulation")   

                            with open(os.path.join(model_folder, "pipeline_config.yaml"), "w") as f:
                                _plain_dict = OmegaConf.to_container(self.config.pipeline, resolve=True)
                                yaml.safe_dump(_plain_dict, f, sort_keys=False)

                            with open(os.path.join(model_folder, "model_config.yaml"), "w") as f:
                                _model_config = self.config.model_agg if key == 'agg' else self.config.model_usr
                                _plain_dict = OmegaConf.to_container(_model_config, resolve=True)
                                yaml.safe_dump(_plain_dict, f, sort_keys=False)

                        except Exception as e:
                            logger.error(f"Model building failed: {e}")
                            raise
            
        mlflow.end_run()

        self.next(self.simulation_step)    

    @step
    def simulation_step(self):   
        """
        Load the trained model and run simulation for each day in the test data. Save KPIs.
        """
        setup_logger(log_file=self.log_file, level=LOGGER_LVL)
        logger = logging.getLogger(__name__)

        # TODO: Replace hardcoded EDF() instantiation with a dynamic simulator loader.
        #       Later, we will generalize this by reading the simulator name and source file path 
        #       from the pipeline config file and instantiating it via a Simulator abstract class.
        # simulator = EDF() 
        # simulator = Optimizer()            
        # simulator = LYNCS()            
        # simulator = lLYNCS()            
        # simulator = OA_benchmark()
        simulator = AVR_benchmark()
        # simulator = Uncontrolled()
        # simulator = Oracle_benchmark()

        # Load model
        key = 'agg' 
        model_folder = os.path.join('models', key, f"{self.model_config[key].name}_{self.model_config[key].version}")
        builder_cls = MODEL_REGISTRY[key][self.model_config[key].name]["model_class"]
        model_builder_agg: Model = builder_cls(self.model_config[key])
        model_builder_agg.load_model(model_folder, prefix="model_simulation")

        key = 'usr'
        model_folder = os.path.join('models', key, f"{self.model_config[key].name}_{self.model_config[key].version}")
        builder_cls = MODEL_REGISTRY[key][self.model_config[key].name]["model_class"]
        model_builder_usr: Model = builder_cls(self.model_config[key])
        model_builder_usr.load_model(model_folder, prefix="model_simulation")   

        dates_agg = sorted(self.df_test_feature_engineered['agg']["date"].unique()) # type: ignore
        dates_usr = sorted(self.df_test_feature_engineered['usr']["date"].unique()) # type: ignore

        if dates_agg == dates_usr:
            unique_dates = dates_agg
            date_to_idx = {d: i for i, d in enumerate(unique_dates)}
        else:
            raise ValueError("Date mismatch between aggregated and individual data.")
        
        self.df_test_feature_engineered['usr'] = add_session_id(self.df_test_feature_engineered['usr'])
        
        # Loop over days
        for date in unique_dates:
            try:
                date_idx = date_to_idx[date]
                logger.info(f"Simulating Date: {date} ({date_idx+1}/{len(unique_dates)})")

                curr_day_agg_real = self.df_test_feature_engineered['agg'][self.df_test_feature_engineered['agg']['date'] == date] # type: ignore
                curr_day_usr_real = self.df_test_feature_engineered['usr'][self.df_test_feature_engineered['usr']['date'] == date] # type: ignore             

                # Sort and extract unique time slots (e.g., 15-minute intervals)
                time_slots = curr_day_agg_real['timestamp'].dt.time

                # Define forecast issuance times
                time_slots = [dt.time() for dt in pd.date_range(start="00:00", end="23:45", freq="15min")]
                
                time_bins = [dt.time() for dt in pd.date_range(start="00:00", end="23:45", freq="15min")]
                time_to_idx = {t: i for i, t in enumerate(time_bins)}

                #Initilize the simulator for the day
                simulator.initilize()

                # Loop over time slots
                log_every = 16
                for time_slot in time_slots:            
                    time_idx = time_to_idx[time_slot]
                    if time_idx == 0:
                        continue  # Skip first time slot
                    if (time_idx+1) % log_every == 0:
                        logger.info(f"Progress: {time_idx+1}/{len(time_slots)} ({(time_idx+1)/len(time_slots):.1%})")
                    
                    curr_time = time_slot
                    next_time = curr_time if curr_time == time(23, 45) else (datetime.combine(datetime.today(), curr_time) + timedelta(minutes=15)).time()                

                    # Prepare scheduler_view         
                    curr_day_usr_prior = curr_day_usr_real[curr_day_usr_real['start_time'] < curr_time][['date', 'EV_id_x', 'total_energy', 'start_time', 'end_time','session_id']]
                    mask = (curr_day_usr_prior['end_time'] >= curr_time)
                    curr_day_usr_prior.loc[mask, 'end_time'] = pd.NaT 
                    curr_day_usr_prior.loc[mask, 'total_energy'] = np.nan                
                
                    if curr_day_usr_prior.shape[0] > 0:                            
                        mask = curr_day_usr_prior['end_time'].isna()
                        _forecast = curr_day_usr_prior[mask].copy()
                        _forecast['est_end_time'] = pd.NaT
                        _forecast['est_total_energy'] = np.nan
                        _forecast = model_builder_usr.forecast(_forecast, curr_time)
                        
                        curr_day_usr_prior.loc[mask, 'end_time'] = _forecast['est_end_time']
                        curr_day_usr_prior.loc[mask, 'total_energy'] = _forecast['est_total_energy']

                    scheduler_view = curr_day_usr_prior                 
                    scheduler_view['start_datetime'] = pd.to_datetime(scheduler_view['date'].astype(str) + ' ' + scheduler_view['start_time'].astype(str))
                    scheduler_view['end_datetime'] = pd.to_datetime(scheduler_view['date'].astype(str) + ' ' + scheduler_view['end_time'].astype(str), format='mixed')
                    scheduler_view=scheduler_view[['EV_id_x','start_datetime','end_datetime','total_energy','session_id']]
                    if scheduler_view.isnull().values.any():
                        raise ValueError("⚠️ There are NaNs in scheduler_view.")
                    

                    #prepare forecast_agg
                    curr_day_agg_prior = curr_day_agg_real.iloc[:time_idx].copy()

                    forecast_agg = model_builder_agg.forecast(curr_day_agg_prior[['timestamp','cum_ev_count', 'total_energy']])
                    if forecast_agg.isnull().values.any():
                        logger.debug(f"forecast_agg: {forecast_agg}")
                        raise ValueError("⚠️ There are NaNs in forecast_agg.")
                    

                    # Prepare active_session_info
                    active_session_info = curr_day_usr_real[(curr_day_usr_real['start_time'] < next_time) & (curr_day_usr_real['end_time'] >= curr_time)][['date', 'EV_id_x', 'total_energy', 'start_time', 'end_time', 'session_id']]
                    if active_session_info.isnull().values.any():
                        raise ValueError("⚠️ There are NaNs in active_session_info.")        
                    
                    if not isinstance(curr_time, time):
                        raise TypeError(f"Expected curr_time to be datetime.time, got {type(curr_time)} instead.")

                    simulator.step(curr_time, forecast_agg, scheduler_view, active_session_info)

                simulator.publish_results(self.results_folder)    
            except Exception as e:
                err = f"Failed to simulate day: {date} : {e}"
                logger.error(err)
                

        # read in parquets and combine - globalmetrics
        globalmets = []
        prefix = simulator.identifier # e.g., 'focs', 'llyncs' or 'oa'.
        logger.debug('start compiling results of all days')
        for date in date_range:
            try:
                file = os.path.join(self.results_folder,"{}_{}_globalmetrics.parquet".format(date,prefix))
                globalmets += [pd.read_parquet(file)]
            except:
                logger.debug('[WARNING]: could not find parquet for date {}.'.format(date))
        try:
            globalmets = pd.concat(globalmets)
            file = os.path.join(self.results_folder,"{}_globalmetrics.csv".format(prefix))
            globalmets.to_csv(file)
            file = os.path.join(self.results_folder,"{}_globalmetrics.parquet".format(prefix))
            globalmets.to_parquet(file)
            logger.debug('compiled results saved successfully')
        except:
            if len(globalmets) == 0:
                logger.error('Nothing to concat')
            else:
                logger.error('Something went wrong when saving global results for all days.')


        # read in parquets and combine - qosqoe per job
        jobmets = []
        logger.debug('start compiling results per job')
        for date in date_range:
            try:
                file = os.path.join(self.results_folder,"{}_{}_qosqoe.parquet".format(date,prefix))
                jobmets += [pd.read_parquet(file)]
            except:
                logger.debug('[WARNING]: could not find parquet for date {}.'.format(date))
        try:
            jobmets = pd.concat(jobmets)
            jobmets['id'] = jobmets['ids'].apply(lambda x: x.split('-')[0][1:])
            file = os.path.join(self.results_folder,"{}_qosqoe.csv".format(prefix))
            jobmets.to_csv(file)
            file = os.path.join(self.results_folder,"{}_qosqoe.parquet".format(prefix))
            jobmets.to_parquet(file)
            for id in jobmets['id'].unique():
                file = os.path.join(self.results_folder,"{}_qosqoe_{}.csv".format(prefix, id))
                jobmets[jobmets['id']==id].to_csv(file)
                file = os.path.join(self.results_folder,"{}_qosqoe_{}.parquet".format(prefix, id))
                jobmets[jobmets['id']==id].to_parquet(file)
            logger.debug('compiled results saved successfully')
        except:
            if len(globalmets) == 0:
                logger.error('Nothing to concat')
            else:
                logger.error('Something went wrong when saving per job results')

        self.next(self.end)    

    @step
    def end(self):
        print("END!")

if __name__ == '__main__':
    ModelEvaluationPipeline()
