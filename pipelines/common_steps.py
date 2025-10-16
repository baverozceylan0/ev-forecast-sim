import os
from pathlib import Path
import yaml
from dataclasses import asdict, dataclass
from typing import List, Optional, Any, Tuple

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

from src.constants import Paths


def data_loading_step(dataset_id: str, 
                      log_file: str) -> pd.DataFrame:
    
    setup_logger(log_file=log_file)
    logger = logging.getLogger(__name__)

    try:
        df_raw: pd.DataFrame = DataLoaderFactory.get_loader(dataset_id).load() 
        logger.info(f"Loaded raw data with shape: {df_raw.shape}")  
    except Exception as e:
        err = f"Failed to load data for dataset '{dataset_id}': {e}"
        logger.error(err)
        raise

    return df_raw


def data_cleaning_step(df_input: pd.DataFrame, strategy_steps: List[str], 
                       log_file: str) -> pd.DataFrame:

    setup_logger(log_file=log_file)
    logger = logging.getLogger(__name__)

    try:
        if not strategy_steps:
            logger.debug("No data cleaning strategies provided. Skipping cleaning step.")
            df_cleaned = df_input.copy()
        else:
            cleaner = DataCleaner()
            cleaner.set_strategy(strategy_steps)
            df_cleaned: pd.DataFrame = cleaner.clean(df_input)
        logger.info(f"Cleaned data shape: {df_cleaned.shape}")  
    except Exception as e:
        err = f"Failed to clean data:'{strategy_steps}': {e}"
        logger.error(err)
        raise

    return df_cleaned


def data_splitter_step(df_input: pd.DataFrame, test_size: float, random_state: int,
                       log_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    setup_logger(log_file=log_file)
    logger = logging.getLogger(__name__)

    logger.info(f"Splitting data by date with test_size={test_size}, random_state={random_state}")

    df_input['date'] = df_input['start_datetime'].dt.date
    df_train, df_test = split_by_date_grouped(df_input, date_col='date', test_size=test_size, random_state=random_state)

    logger.info(f"Train data shape: {df_train.shape} | Test data shape: {df_test.shape}")
    # path = Path.cwd() / "data" / "sample_set"
    # os.makedirs(path, exist_ok=True)
    # df_train.to_parquet(path / "train.parquet")
    # df_test.to_parquet(path / "test.parquet")
    return df_train, df_test


def feature_engineering_step(df_train: pd.DataFrame, df_test: pd.DataFrame,
                             key: str, strategy_steps: List[str], selected_features: List[str], always_include: List[str], 
                             log_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
       
    setup_logger(log_file=log_file)
    logger = logging.getLogger(__name__)

    if key not in ['agg', 'usr']:
        err = f"key must be 'agg' or 'usr'"
        logger.error(err)
        raise ValueError(err)

    if not isinstance(strategy_steps, ListConfig):
        err = f"feature_engineering_strategy_steps must be a list."
        logger.error(err)
        raise ValueError(err)
    if not isinstance(selected_features, ListConfig):
        err = f"selected_features must be a list."
        logger.error(err)
        raise ValueError(err)
    
    if not all(isinstance(step, str) for step in strategy_steps):
        err = f"All entries in strategy_steps['{key}'] must be strings."
        logger.error(err)
        raise ValueError(err)
    if not all(isinstance(feature, str) for feature in selected_features):
        err = f"All entries in selected_features['{key}'] must be strings."
        logger.error(err)
        raise ValueError(err)

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
            
    df_train_feature_engineered: pd.DataFrame = apply_feature_engineering_steps(strategy_steps, df_train)
    df_test_feature_engineered: pd.DataFrame = apply_feature_engineering_steps(strategy_steps, df_test)     

    df_train_feature_engineered = filter_and_order_columns(df_train_feature_engineered, selected_features, always_include)
    df_test_feature_engineered = filter_and_order_columns(df_test_feature_engineered, selected_features, always_include)

    logger.debug(f"Selected Features ({key}): {df_test_feature_engineered.columns.tolist()}")
    logger.info(f"Data shape after feature engineering: {key}-train: {df_train_feature_engineered.shape} | {key}-test: {df_test_feature_engineered.shape}")

    return df_train_feature_engineered, df_test_feature_engineered
