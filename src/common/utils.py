import pandas as pd
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Dict, Any

import mlflow
import logging

import os
import sys
import yaml
from typing import Any, Optional, Type
from datetime import datetime
from pathlib import Path 

def setup_logger(name: str = "app", log_file: str = "run.log") -> logging.Logger:
    """
    Sets up a logger that prints to both the console and a file.
    Ensures consistent logging across files and subprocesses.
    """
    logger = logging.getLogger(name)

    if logger.hasHandlers():
        return logger  # prevent duplicate handlers

    logger.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(log_file, mode='a')
    fh.setLevel(logging.DEBUG)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger

def load_and_instantiate_config(file_path: str, config_class: Type, config_name: Optional[str] = "Config") -> Any:
    """
    Load a YAML file and instantiate a config class from its contents.

    Args:
        file_path (str): Path to the YAML file.
        config_class (Type): Class to instantiate using parsed YAML as kwargs.
        config_name (str): Optional label for clearer error messages.

    Returns:
        An instance of `config_class`.

    Raises:
        FileNotFoundError: If path doesn't exist.
        ValueError: If path isn't a YAML file or contents aren't dict-like.
        yaml.YAMLError: If the YAML content is malformed.
        TypeError: If the config class can't be instantiated with the given keys.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{config_name} path does not exist:\n → {file_path}")

    if not os.path.isfile(file_path):
        raise ValueError(f"{config_name} path is not a file:\n → {file_path}")

    if not (file_path.endswith(".yaml") or file_path.endswith(".yml")):
        raise ValueError(f"{config_name} must be a .yaml or .yml file:\n → {file_path}")

    try:
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Failed to parse {config_name} file:\n → {file_path}\nYAML error: {e}")

    if not isinstance(data, dict):
        raise ValueError(f"{config_name} must contain a dictionary of parameters, but got: {type(data).__name__}")

    try:
        return config_class(**data)
    except TypeError as e:
        raise TypeError(f"Failed to instantiate {config_class.__name__} from {config_name}:\n → {file_path}\nError: {e}")


def validate_ev_charging_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate that the dataframe has all required columns and correct data types
    for EV charging session data.

    Args:
        df (pd.DataFrame): The input DataFrame to validate.

    Returns:
        pd.DataFrame: A validated and type-corrected copy of the input DataFrame.

    Raises:
        ValueError: If required columns are missing.
        TypeError: If column data types are incorrect.
    """
    required_columns = ["EV_id_x", "start_datetime", "end_datetime", "total_energy"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Convert to correct datetime types
    df["start_datetime"] = pd.to_datetime(df["start_datetime"], errors="coerce")
    df["end_datetime"] = pd.to_datetime(df["end_datetime"], errors="coerce")

    # Validate types
    if not pd.api.types.is_float_dtype(df["total_energy"]):
        raise TypeError("Column 'total_energy' must be of float type.")
    if not pd.api.types.is_string_dtype(df["EV_id_x"]):
        raise TypeError("Column 'EV_id_x' must be of string type.")

    return df


def split_by_date_grouped(
    df: pd.DataFrame,
    date_col: str = 'date',
    test_size: float = 0.2,
    random_state: int = 1337
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits a DataFrame into train and test day-groups.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - date_col (str): Column containing date values (will be converted to datetime).
    - test_size (float): Fraction of days to assign to test.
    - random_state (int): Random seed for reproducibility.

    Returns:
    - train_days (pd.DataFrame): DataFrame contains data for one training days.
    - test_days (pd.DataFrame): DataFrame contains data for one testing days.
    """

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    unique_dates = df[date_col].drop_duplicates()

    # Split the dates into train/test
    train_dates, test_dates = train_test_split(
        unique_dates,
        test_size=test_size,
        random_state=random_state
    )

    # Use Boolean indexing to select rows
    train_df = df[df[date_col].isin(train_dates)]
    test_df = df[df[date_col].isin(test_dates)]

    return train_df, test_df


def add_lag_features(df, columns, n_lags, group_by='date', fillna_value=None) -> pd.DataFrame:
    """
    Add lag features for specified columns grouped by a key (e.g., 'date').

    Parameters:
        df (pd.DataFrame): DataFrame with time series data.
        columns (list): Columns to lag.
        n_lags (int): Number of lags to create.
        group_by (str): Column to group by before shifting (e.g. 'date').

    Returns:
        pd.DataFrame: Original DataFrame with new lag columns.
    """
    df = df.sort_values([group_by, 'timestamp']).copy()

    for col in columns:
        for lag in range(1, n_lags + 1):
            lag_col = f'{col}_lag_{lag}'
            df[lag_col] = df.groupby(group_by)[col].shift(lag)
            if fillna_value is not None:
                df[lag_col] = df[lag_col].fillna(fillna_value)

    return df


def log_parameters_mlflow(params: Dict[str, Any], prefix: Optional[str] = None) -> None:
    """
    Logs parameters to MLflow with optional prefix for namespacing.

    Args:
        params: Dictionary of parameters to log.
        prefix: Optional prefix for parameter keys (e.g. "model.").
    """
    for key, value in params.items():
        full_key = f"{prefix}.{key}" if prefix else key

        if isinstance(value, (str, int, bool)):
            mlflow.log_param(full_key, value)

        elif isinstance(value, float):
            mlflow.log_param(full_key, f"{value:.6f}")

        elif isinstance(value, list):
            mlflow.log_param(full_key, ",".join(map(str, value)))

        elif isinstance(value, pd.DataFrame):
            mlflow.log_param(f"{full_key}_rows", value.shape[0])
            mlflow.log_param(f"{full_key}_cols", value.shape[1])

        elif value is None:
            mlflow.log_param(full_key, "None")

        else:
            mlflow.log_param(full_key, str(value))


def filter_and_order_columns(
    df: pd.DataFrame, 
    prefixes: List[str], 
    always_include: List[str]
) -> pd.DataFrame:
    """
    Filters and orders columns in a DataFrame based on given prefixes and always-include list.

    Args:
        df: The input DataFrame.
        prefixes: List of column name prefixes to include.
        always_include: List of column names that should always be included (and appear first).

    Returns:
        A filtered DataFrame with selected and ordered columns.
    """
    # Sort prefixes by length to prioritize longer, more specific ones
    prefixes = sorted(prefixes, key=len, reverse=True)

    def matches_prefix(col: str) -> bool:
        return any(col == p or col.startswith(p + "_") for p in prefixes)
    
    selected_cols = [
        col for col in df.columns
        if col in always_include or matches_prefix(col)
    ]

    # Ensure always_include columns appear first in order
    ordered_cols = [col for col in always_include if col in selected_cols]
    ordered_cols += [col for col in selected_cols if col not in always_include]

    return df[ordered_cols]


def add_session_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a session_id column to an EV session dataset.
    Format: U<user_id>-<YYYYMMDD>-<counter>
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with 'EV_id_x' and 'start_datetime' columns.
        
    Returns
    -------
    pd.DataFrame
        Dataframe with an additional 'session_id' column.
    """

    date_format = "%Y%m%d"

    df = df.copy()
    df['_date_helper'] = df['start_datetime'].dt.strftime(date_format)
    
    # Sort a copy for counting
    sorted_idx = df.sort_values(by=['EV_id_x', '_date_helper', 'start_datetime']).index
    session_counts = (
        df.loc[sorted_idx]
        .groupby(['EV_id_x', '_date_helper'])
        .cumcount() + 1
    )

    # Create session_id for the sorted order
    session_ids = [
        f"U{row['EV_id_x']}-{row['_date_helper']}-{count}"
        for row, count in zip(df.loc[sorted_idx].to_dict('records'), session_counts)
    ]

    # Map back to original order
    df.loc[sorted_idx, 'session_id'] = session_ids

    # Drop helper column
    df.drop(columns='_date_helper', inplace=True)

    
    return df


def to_time_or_nat(x):
    try: 
        return datetime.strptime(str(x), "%H:%M").time()
    except Exception:
        try:
            return datetime.strptime(str(x), "%H:%M:%S").time()
        except Exception:
            try:
                return datetime.strptime(str(x), "%H:%M:%S.%f").time()
            except:
                return pd.NaT  # or None