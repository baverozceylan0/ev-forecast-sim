import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

from src.plots.plot_tools import _convert_time_to_minutes, _format_minutes_to_hhmm, _format_time_axis

# ----- Strategy Interface -----
class PlotStrategy(ABC):
    @abstractmethod
    def plot(self, df: pd.DataFrame, **kwargs):
        pass   


# ----- Concrete Strategies -----
# Plot number of sessions grouped by frequency
class PlotNumberOfSessions(PlotStrategy):
    def __init__(self, frequency: str = "M", max_xticks: int = 20):
        """
        Parameters:
        -----------
        frequency : str
            Pandas offset alias: "D" (day), "W" (week), "M" (month), etc.
        max_xticks : int
            Maximum number of x-axis tick labels to show, to keep the plot readable.
        """
        self.frequency = frequency
        self.max_xticks = max_xticks

    def plot(self, df: pd.DataFrame, datetime_column: str = "start_datetime", ax: plt.Axes = None) -> plt.Figure:
        """
        Groups sessions by a time frequency and plots the number of EV arrivals.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing at least the datetime column (e.g., 'start_datetime')
        datetime_column : str
            Name of the datetime column to group by (default: 'start_datetime')
        ax : matplotlib.axes.Axes, optional
            Optional matplotlib Axes object. If provided, the plot will be drawn on this axis.
            Otherwise, a new figure and axis will be created.
        """
        if datetime_column not in df.columns:
            raise ValueError(f"Column '{datetime_column}' not found in dataframe")
        
        # Create a copy
        df = df.copy()        
        df["date"] = df[datetime_column].dt.date

        # Group by the chosen frequency
        grouped = df.groupby(pd.Grouper(key=datetime_column, freq=self.frequency)).size()

        # Format x-axis labels based on the frequency
        if self.frequency == "M":
            labels = grouped.index.to_period("M").astype(str)
        else:
            labels = grouped.index.strftime("%Y-%m-%d")

        # Plot
        fig, ax = plt.subplots(figsize=(10, 5)) if ax is None else (ax.figure, ax)

        ax.bar(labels, grouped.values)
        ax.set_xlabel("Time")
        ax.set_ylabel("Number of Arrivals")
        ax.set_title(f"Number of EV Arrivals per {self.frequency}")
        ax.grid(True)

        # Format x-ticks (sparse, aligned to end)
        stride = max(1, len(labels) // self.max_xticks)
        tick_indices = np.arange(len(labels) - 1, -1, -stride)[::-1]
        ax.set_xticks(tick_indices)
        ax.set_xticklabels([labels[i] for i in tick_indices], rotation=45, ha='right')

        fig.tight_layout()

        # Return the figure
        return fig
        
  
class PlotAverageSessionTimes(PlotStrategy):
    def __init__(self, frequency: str = "W"):
        """
        Parameters:
        -----------
        frequency : str
            Pandas offset alias (e.g., 'D' for daily, 'W' for weekly, 'M' for monthly)
        """
        self.frequency = frequency

    def plot(self, df: pd.DataFrame, show_duration: bool = True) -> plt.Figure:
        required_cols = {"start_datetime", "end_datetime"}
        if not required_cols.issubset(df.columns):
            raise ValueError("Missing required datetime columns.")

        # Create a copy
        df = df.copy()

        # Compute minute-of-day and duration
        df["start_minute"] = _convert_time_to_minutes(df["start_datetime"])
        df["end_minute"] = _convert_time_to_minutes(df["end_datetime"])
        if show_duration:
            df["duration_minutes"] = (df["end_datetime"] - df["start_datetime"]).dt.total_seconds() / 60


        # Grouping aggregation logic
        agg_dict = {
            "start_minute": ["mean", lambda x: x.quantile(0.05), lambda x: x.quantile(0.95)],
            "end_minute": ["mean", lambda x: x.quantile(0.05), lambda x: x.quantile(0.95)],
        }
        if show_duration:
            agg_dict["duration_minutes"] = ["mean", lambda x: x.quantile(0.05), lambda x: x.quantile(0.95)]

        grouped = (
            df.set_index("start_datetime")
            .groupby(pd.Grouper(freq=self.frequency))
            .agg(agg_dict)
            .dropna()
        )

        # Rename columns
        col_names = ["start_mean", "start_p5", "start_p95",
                     "end_mean", "end_p5", "end_p95"]
        if show_duration:
            col_names += ["dur_mean", "dur_p5", "dur_p95"]
        grouped.columns = col_names

        # Time labels
        time_labels = grouped.index.strftime("%Y-%m-%d")
        xticks = np.arange(0, len(time_labels), max(1, len(time_labels) // 20))

        y_axis_time_interval = 60 # minutes
        # Time bins and labels
        y_time_bins = np.arange(0, 24 * 60 + y_axis_time_interval, y_axis_time_interval)
        y_time_labels = [f"{h:02}:{m:02}" for h in range(24) for m in range(0, 60, y_axis_time_interval)]
        y_tick_label_pairs = list(zip(y_time_bins, y_time_labels))
        y_label_step = 2
        y_ticks = [tick for i, (tick, _) in enumerate(y_tick_label_pairs) if i % y_label_step == 0]
        y_labels = [label for i, (_, label) in enumerate(y_tick_label_pairs) if i % y_label_step == 0]

        # Plot
        fig, ax = plt.subplots(figsize=(12, 5))

        # Helper for repeated logic
        def plot_with_band(mean_col, p5_col, p95_col, label, color):
            ax.plot(time_labels, grouped[mean_col], label=label, color=color)
            ax.fill_between(time_labels, grouped[p5_col], grouped[p95_col], alpha=0.2, color=color)

        plot_with_band("start_mean", "start_p5", "start_p95", "Avg. Start Time", "tab:blue")
        plot_with_band("end_mean", "end_p5", "end_p95", "Avg. End Time", "tab:orange")
        if show_duration:
            plot_with_band("dur_mean", "dur_p5", "dur_p95", "Avg. Duration", "tab:green")

        ax.set_xlabel("Time")
        ax.set_ylabel("Minutes")
        ax.set_title("")  # leave to be set externally
        ax.set_xticks(xticks)
        ax.set_xticklabels([time_labels[i] for i in xticks], rotation=45, ha='right')
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        return fig
    

class PlotUserFrequencyDistribution(PlotStrategy):
    def __init__(self, log_scale: bool = False):
        """
        Parameters:
        -----------
        
        log_scale : bool
            Whether to use logarithmic scale on the y-axis (useful for heavy-tailed distributions).
        """
        self.log_scale = log_scale

    def plot(self, df: pd.DataFrame, UUID_column: str = "EV_id_x", top_n: int = 100) -> plt.Figure:
        """
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing a column with UUIDs.
        UUID_column : str
            Name of the column containing UUIDs.
        top_n : int
            Number of most frequent UUIDs to display in the bar chart.
            If -1, all UUIDs will be shown. 
        """
        if UUID_column not in df.columns:
            raise ValueError(f"'{UUID_column}' column not found in dataframe.")
        
        UUID_counts = df[UUID_column].value_counts()
        total_UUIDs = len(UUID_counts)

        fig, ax = plt.subplots(figsize=(10, 5))

        if top_n > 100 or top_n == -1:
            # Plot all users: sorted frequency curve
            if top_n == -1:
                ax.plot(UUID_counts.values, marker='', linestyle='-', color='steelblue')
            else:
                ax.plot(UUID_counts.head(top_n).values, marker='', linestyle='-', color='steelblue')
            ax.set_xlabel("User Rank (sorted by activity)")
        else:
            # Bar chart for top-N
            top_counts = UUID_counts.head(top_n)
            top_counts.plot(kind="bar", ax=ax, color="skyblue", edgecolor="black")
            ax.set_xlabel("User ID")
            
        ax.set_ylabel("Number of Sessions")
        if top_n == -1:
            ax.set_title("User Activity Distribution (All Users)")
        else:
            ax.set_title(f"Top {top_n} Most Active Users")

        # Axis scaling and formatting
        if self.log_scale:
            ax.set_yscale("log")
            ax.set_ylabel("Log(Number of Sessions)")

        ax.grid(True, axis='y')
        fig.tight_layout()

        summary_table = pd.DataFrame()
        
        stats = UUID_counts.head(top_n).describe(percentiles=[0.25, 0.5, 0.75 , 0.95])
        summary_table = stats[["min", "25%", "50%", "75%", "95%", "max", "mean"]]
        summary_table.loc["std"] = UUID_counts.head(top_n).std()
        summary_table.loc["ROI_ratio"] = sum(UUID_counts.head(top_n))/sum(UUID_counts)

        return fig, summary_table


# ----- Context -----
class Plotter:
    def __init__(self, strategy: PlotStrategy):
        self.strategy = strategy

    def set_strategy(self, strategy: PlotStrategy):
        self.strategy = strategy

    def plot(self, df: pd.DataFrame, **kwargs):
        if self.strategy is None or self.strategy is PlotStrategy:
            raise ValueError("No concrete PlotStrategy has been set.")
        return self.strategy.plot(df, **kwargs)