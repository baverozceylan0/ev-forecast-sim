import pandas as pd
import numpy as np
import datetime

def _convert_time_to_minutes(time_series: pd.Series) -> pd.Series:
    """Convert datetime.time values to minutes since midnight."""
    return time_series.apply(lambda t: t.hour * 60 + t.minute + t.second / 60)

def _format_minutes_to_hhmm(minutes: float) -> str:
    """Convert minutes since midnight to HH:MM string."""
    h = int(minutes) // 60
    m = int(minutes) % 60
    return f"{h:02}:{m:02}"

def _format_time_axis(label_step):
    """
    Generate time bins and corresponding tick positions and labels 
    for a 24-hour day divided into 15-minute intervals.

    Parameters:
    -----------
    label_step : int
        Controls how many x-axis labels are shown (e.g., every 4th label).

    Returns:
    --------
    bins : np.ndarray
        The bin edges for histogramming (in minutes since midnight).
    ticks : list[int]
        The x-axis tick positions to show (sparse, depending on label_step).
    labels : list[str]
        The formatted labels in HH:MM format for selected tick positions.
    """
    bins = np.arange(0, 1441, 15) # 15-minute intervals
    labels = [f"{h:02}:{m:02}" for h in range(24) for m in range(0, 60, 15)] # 15-minute intervals
    labels.append("23:59")
    tick_label_pairs = list(zip(bins, labels))
    ticks = [tick for i, (tick, _) in enumerate(tick_label_pairs) if i % label_step == 0]
    labels = [label for i, (_, label) in enumerate(tick_label_pairs) if i % label_step == 0]
    return bins, ticks, labels

def add_enriched_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add commonly used time-derived columns to a DataFrame with 'start_datetime' and 'end_datetime'.

    Columns added:
    - 'month': Full month name (e.g., "January")
    - 'day_of_week': Full day name (e.g., "Monday")
    - 'day_of_week_enum': Integer 0-6 (Monday=0)
    - 'duration_minutes': Difference between end and start in minutes
    - 'date': Date part of start_datetime (YYYY-MM-DD)
    - 'start_time': Time part of start_datetime (HH:MM:SS)
    - 'end_time': Time part of end_datetime (HH:MM:SS)

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame with 'start_datetime' and 'end_datetime' columns.

    Returns:
    --------
    pd.DataFrame
        A copy of the DataFrame with new columns added.
    """
    required_columns = ["EV_id_x", "start_datetime", "end_datetime", "total_energy"]
    if not all(col in df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in df.columns]
        raise ValueError(f"Missing columns in dataframe: {missing}")

    df = df.copy()
    df["start_datetime"] = pd.to_datetime(df["start_datetime"], errors='coerce')
    df["end_datetime"] = pd.to_datetime(df["end_datetime"], errors='coerce')

    df["month"] = df["start_datetime"].dt.month_name()
    df["day_of_week"] = df["start_datetime"].dt.day_name()
    df["day_of_week_enum"] = df["start_datetime"].dt.weekday
    df["duration_minutes"] = (df["end_datetime"] - df["start_datetime"]).dt.total_seconds() / 60
    df["date"] = df["start_datetime"].dt.date
    df["start_time"] = df["start_datetime"].dt.time
    df["end_time"] = df["end_datetime"].dt.time

    return df

def get_daily_session_stats(df: pd.DataFrame, date_column: str = "date") -> pd.DataFrame:
    """
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe containing at least the datetime column.
    datetime_column : str
        Name of the date column to group by (default: 'date').

    Returns:
    --------
    pd.DataFrame
        A dataframe indexed by date with columns indicating
        daily stats.
    """
    if date_column not in df.columns:
        raise ValueError(f"'{date_column}' column not found in dataframe.")
    
    df = df.copy()

    # Create helper columns
    df["start_date"] = df["start_datetime"].dt.date
    df["start_minute"] = df["start_datetime"].dt.hour * 60 + df["start_datetime"].dt.minute
    df["end_minute"] = df["end_datetime"].dt.hour * 60 + df["end_datetime"].dt.minute

    # Group by day and compute statistics
    grouped = df.groupby("date").agg(
        day_of_week=("start_datetime", lambda x: x.dt.day_name().iloc[0]),
        day_of_week_enum=("start_datetime", lambda x: x.dt.weekday.iloc[0]), 
        month=("start_datetime", lambda x: x.dt.month_name().iloc[0]),
        num_sessions=("start_datetime", "count"),  
        avg_start_minute=("start_minute", "mean"),
        avg_end_minute=("end_minute", "mean"),
        avg_duration_minutes=("duration_minutes", "mean"),
        avg_total_energy=("total_energy", "mean")
    )

    # Convert avg_start_minute to datetime.time object
    grouped["avg_start_time"] = pd.to_timedelta(grouped["avg_start_minute"], unit="m").apply(lambda x: (datetime.datetime.min + x).time())

    # Convert avg_end_minute to datetime.time object
    grouped["avg_end_time"] = pd.to_timedelta(grouped["avg_end_minute"], unit="m").apply(lambda x: (datetime.datetime.min + x).time())


    # Drop internal numeric average column
    return grouped.drop(columns=["avg_start_minute","avg_end_minute"])
