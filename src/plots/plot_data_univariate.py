import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

from src.plots.plot_tools import _convert_time_to_minutes, _format_minutes_to_hhmm, _format_time_axis

from src.plots.plot_data_basic import PlotStrategy

class PlotCategoricalDistribution(PlotStrategy):
    def __init__(self, sort_by_count: bool = True):
        """
        Parameters:
        -----------
        sort_by_count : bool
            Whether to sort categories by descending count.
        """
        self.sort_by_count = sort_by_count

    def plot(self, df: pd.DataFrame, column: str, ax: plt.Axes = None) -> plt.Figure:
        """
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe containing the data to be plotted.
        column : str
            Name of the column to be plotted.
        ax : matplotlib.axes.Axes, optional
            Optional matplotlib Axes object. If provided, the plot will be drawn on this axis.
            Otherwise, a new figure and axis will be created.
        """
        if column not in df.columns:
            raise ValueError(f"'{column}' column not found in dataframe.")        

        # Handle known chronological orderings
        if column == "month":
            ordered_categories = [
                "January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November", "December"
            ]
            df[column] = pd.Categorical(df[column], categories=ordered_categories, ordered=True)
        elif column == "weekday" or column == "day_of_week":
            ordered_categories = [
                "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
            ]
            df[column] = pd.Categorical(df[column], categories=ordered_categories, ordered=True)

        # Count values
        counts = df[column].value_counts()
        if self.sort_by_count:
            counts = counts.sort_values(ascending=False).sort_index()

        # Create figure/axes if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))
        else:
            fig = ax.figure
        # Plot   
        counts.plot(kind='bar', ax=ax, color="teal", edgecolor="black")
        ax.set_ylabel("Count")
        ax.set_xlabel(column)
        ax.set_xticklabels(counts.index, rotation=45, ha='right')

        ax.set_title(f"Distribution of '{column}'")
        ax.grid(True, axis='y')
        fig.tight_layout()
        return fig
    

class PlotNumericalDistribution(PlotStrategy):
    def __init__(self, kde: bool = False, label_step: int = 8): 
        """
        Parameters:
        -----------
        kde : bool
            Whether to overlay a Kernel Density Estimate (KDE).
        label_step : int
            Controls sparsity of x-axis labels if time values are used.
        """
        self.kde = kde
        self.label_step = label_step    
    
    def plot(self, df: pd.DataFrame, column: str, bins: int = 30, ax: plt.Axes = None, time_of_day_settings_flag: bool = False) -> plt.Figure:
        """
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe containing the data to be plotted.
        column : str
            Name of the column to plot. Can be numeric (e.g., 'duration_minutes', 'total_energy')
            or time-of-day (e.g., datetime.time values like 'start_time').
        bins : int, optional
            Number of bins to use in the histogram (if time_of_day_settings_flag is False).
        ax : matplotlib.axes.Axes, optional
            Optional matplotlib Axes object. If provided, the plot will be drawn on this axis.
            Otherwise, a new figure and axis will be created.
        time_of_day_settings_flag : bool, optional
            If True, assumes the data represents time-of-day and applies special axis formatting:
            - X-axis labels will be in HH:MM format over the whole day.
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The resulting matplotlib Figure object for further customization or saving.
        """
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in dataframe.")

        data = df[column]

        # Determine if it's a time column
        try :
            data = _convert_time_to_minutes(data)
            is_time_data = True
        except AttributeError:
            if np.issubdtype(data.dtype, np.number):
                is_time_data = False
            else:
                raise TypeError(f"Column '{column}' must be numeric or datetime.time.")

        # Plot
        fig, ax = plt.subplots(figsize=(6, 4)) if ax is None else (ax.figure, ax)

        if time_of_day_settings_flag:
            # Time-of-day histogram settings
            time_bins, ticks, labels = _format_time_axis(self.label_step)
            bins = time_bins

        sns.histplot(data, ax=ax, bins=bins, edgecolor="black", alpha=0.6, label="Histogram", kde=self.kde)        

        if time_of_day_settings_flag:
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_xlim([0, 1440])
        if self.kde:
            ax.legend()

        # Labels
        ax.set_title(f"Distribution of {column.replace('_', ' ').title()}")
        ax.set_ylabel("Count")
        ax.grid(True)

        if is_time_data:
            ax.set_xlabel("Time of Day")
        else:
            ax.set_xlabel(column.replace('_', ' ').title())

        fig.tight_layout()
        return fig


