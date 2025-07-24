import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from src.plots.plot_tools import _convert_time_to_minutes, _format_minutes_to_hhmm, _format_time_axis

from src.plots.plot_data_basic import PlotStrategy

# This strategy analyzes the relationship between two numerical features using scatter plots.
class PlotNumericalVsNumericalScatter(PlotStrategy):
    def __init__(self, label_step: int = 8):
        self.label_step = label_step    

    def plot(self, df: pd.DataFrame, feature1: str, feature2: str, ax: plt.Axes = None, time_of_day_settings_flag: tuple[bool, bool] = (False, False), gridsize: int = 40) -> plt.Figure:
        """
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe containing the data to be plotted.
        feature1 : str
            Name of the column to be plotted on the x-axis.
        feature2 : str
            Name of the column to be plotted on the y-axis.
        ax : matplotlib.axes.Axes, optional
            Optional matplotlib Axes object. If provided, the plot will be drawn on this axis.
            Otherwise, a new figure and axis will be created.
        time_of_day_settings_flag : tuple of bool (x_flag, y_flag), optional
            If True, assumes the data represents time-of-day and applies special axis formatting:
            - X-axis labels will be in HH:MM format over the whole day.
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The resulting matplotlib Figure object for further customization or saving.
        """
        if feature1 not in df.columns:
            raise ValueError(f"Column '{feature1}' not found in dataframe.")
        if feature2 not in df.columns:
            raise ValueError(f"Column '{feature2}' not found in dataframe.")
        
        data_x = df[feature1]
        data_y = df[feature2]

        # Determine if feature1's a time column
        try :
            data_x = _convert_time_to_minutes(df[feature1])
            is_time_data = True
        except AttributeError:
            if np.issubdtype(df[feature1].dtype, np.number):
                is_time_data = False
            else:
                raise TypeError(f"Column '{feature1}' must be numeric or datetime.time.")
            
        # Determine if feature2's a time column
        try :
            data_y = _convert_time_to_minutes(df[feature2])
            is_time_data = True
        except AttributeError:
            if np.issubdtype(df[feature2].dtype, np.number):
                is_time_data = False
            else:
                raise TypeError(f"Column '{feature2}' must be numeric or datetime.time.")
        

        # Plot
        fig, ax = plt.subplots(figsize=(6, 4)) if ax is None else (ax.figure, ax)

        if any(time_of_day_settings_flag):
            # Time-of-day histogram settings
            time_bins, ticks, labels = _format_time_axis(self.label_step)
            bins = time_bins

        ax.hexbin(data_x, data_y,
            gridsize=gridsize,
            cmap="viridis",
            mincnt=1,
        )

        if time_of_day_settings_flag[0]:
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_xlim([0, 1440])

        if time_of_day_settings_flag[1]:
            ax.set_yticks(ticks)
            ax.set_yticklabels(labels, ha='right')
            ax.set_ylim([0, 1440])

        ax.set_title(f"{feature1.replace('_', ' ').title()} vs {feature2.replace('_', ' ').title()}")
        ax.set_xlabel(f"{feature1.replace('_', ' ').title()}")
        ax.set_ylabel(f"{feature2.replace('_', ' ').title()}")

        # Return the figure
        return fig

# This strategy analyzes the relationship between a categorical and a numerical features using box plots.
class PlotCategoricalVsNumericalBox(PlotStrategy):
    def __init__(self, label_step: int = 8):
        self.label_step = label_step  

    def plot(self, df: pd.DataFrame, categorical_col: str, numerical_col: str, ax: plt.Axes = None, time_of_day_settings_flag: bool = False, gridsize: int = 40) -> plt.Figure:           
        """
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe containing the data to be plotted.
        categorical_col : str
            Name of the categorical column to be plotted on the x-axis.
        feature2 : str
            Name of the numerical  column to be plotted on the y-axis.
        ax : matplotlib.axes.Axes, optional
            Optional matplotlib Axes object. If provided, the plot will be drawn on this axis.
            Otherwise, a new figure and axis will be created.
        time_of_day_settings_flag : bool, optional
            If True, assumes the numerical_col data represents time-of-day and applies special axis formatting:
            - Y-axis labels will be in HH:MM format over the whole day.
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The resulting matplotlib Figure object for further customization or saving.
        """
        if categorical_col not in df.columns:
            raise ValueError(f"Column '{categorical_col}' not found in dataframe.")
        if numerical_col not in df.columns:
            raise ValueError(f"Column '{numerical_col}' not found in dataframe.")        

        # Handle known chronological orderings for categorical_col
        if categorical_col == "month":
            ordered_categories = [
                "January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November", "December"
            ]
            df[categorical_col] = pd.Categorical(df[categorical_col], categories=ordered_categories, ordered=True)
        elif categorical_col == "weekday" or categorical_col == "day_of_week":
            ordered_categories = [
                "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
            ]
            df[categorical_col] = pd.Categorical(df[categorical_col], categories=ordered_categories, ordered=True)

        data_x = df[categorical_col]
        data_y = df[numerical_col]

        # Determine if numerical_col's a time column
        try :
            data_y = _convert_time_to_minutes(df[numerical_col])
            is_time_data = True
        except AttributeError:
            if np.issubdtype(df[numerical_col].dtype, np.number):
                is_time_data = False
            else:
                raise TypeError(f"Column '{numerical_col}' must be numeric or datetime.time.")
        
        # Plot
        fig, ax = plt.subplots(figsize=(6, 4)) if ax is None else (ax.figure, ax)
        
        if time_of_day_settings_flag:
            # Time-of-day histogram settings
            time_bins, ticks, labels = _format_time_axis(self.label_step)
            bins = time_bins

        sns.boxplot(
            x=data_x,
            y=data_y,
            ax=ax,
            palette="Set3",
            hue=data_x  # prevents unnecessary legends
        )

        if time_of_day_settings_flag:
            ax.set_yticks(ticks)
            ax.set_yticklabels(labels, ha='right')
            ax.set_ylim([0, 1440])

        # Rotate x-axis labels
        ax.tick_params(axis='x', rotation=45)

        ax.set_title(f"{categorical_col.replace('_', ' ').title()} vs {numerical_col.replace('_', ' ').title()}")
        ax.set_xlabel(f"{categorical_col.replace('_', ' ').title()}")
        ax.set_ylabel(f"{numerical_col.replace('_', ' ').title()}")
        ax.grid(True, axis='y')

        # Return the figure
        return fig



# This strategy analyzes the relationship between two numerical features using scatter plots.
class PlotCorrHeatmap(PlotStrategy):
    def __init__(self, heatmap_corr_method: str = "pearson"):
        """
        Parameters:
        -----------
        pairplot_cols : list or None
            List of column names to include in the seaborn pairplot.
            If None, all numeric columns will be used.
        """
        self.heatmap_corr_method = heatmap_corr_method

    def plot(self, df: pd.DataFrame, features: list[str], ax: plt.Axes = None) -> plt.Figure:
        """
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe containing the data to be plotted.
        features : list of str
            List of column names to include in the plots.
        ax : matplotlib.axes.Axes, optional
            Optional matplotlib Axes object. If provided, the plot will be drawn on this axis.
            Otherwise, a new figure and axis will be created.
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The resulting matplotlib Figure object for further customization or saving.
        """
        if not all(col in df.columns for col in features):
            missing = [col for col in features if col not in df.columns]
            raise ValueError(f"Missing columns in dataframe: {missing}")
        
        data = df[features].copy()
        # Determine if features are time columns
        for feature in features:
            try :
                data[feature] = _convert_time_to_minutes(data[feature])
                is_time_data = True
            except AttributeError:
                if np.issubdtype(data[feature].dtype, np.number):
                    is_time_data = False
                else:
                    raise TypeError(f"Column '{feature}' must be numeric or datetime.time.")
                
        
        corr = data.corr(method=self.heatmap_corr_method)

        # Plot
        fig, ax = plt.subplots(figsize=(6, 4)) if ax is None else (ax.figure, ax)

        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True, ax=ax)
        
        ax.set_title(f"Correlation Heatmap ({self.heatmap_corr_method.capitalize()})")

        # Return the figure
        return fig