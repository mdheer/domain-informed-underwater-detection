import matplotlib.pyplot as plt
import pandas as pd
import os
from typing import Tuple

from src.tools.enums import DataSplit


class SubPlot:
    """
    A utility class for managing and customizing individual subplots in a larger plot figure.

    This class is designed to streamline the process of adding plots and their attributes to a subplot within a matplotlib figure.

    Attributes:
        obj (matplotlib.axes.Axes): The Axes object representing the subplot in a matplotlib figure.
    """

    def __init__(self, obj) -> None:
        """
        Initializes the SubPlot with a matplotlib Axes object.

        Parameters:
            obj (matplotlib.axes.Axes): The Axes object that this SubPlot class will manage.
        """
        self.obj = obj

    def add_plot(self, x_data, y_data, colour, label):
        """
        Adds a plot to the subplot with the provided data and styling.

        Parameters:
            x_data: The data for the x-axis.
            y_data: The data for the y-axis.
            colour (str): The color of the plot line.
            label (str): The label for the plot, used in the legend.
        """
        self.obj.plot(x_data, y_data, color=colour, label=label)

    def add_attributes(self, title, xlabel, ylabel, ymin, ymax):
        """
        Sets various attributes for the subplot such as title, labels, and y-axis limits.

        Parameters:
            title (str): The title of the subplot.
            xlabel (str): The label for the x-axis.
            ylabel (str): The label for the y-axis.
            ymin (float): The minimum value for the y-axis.
            ymax (float): The maximum value for the y-axis.
        """
        self.obj.set_title(title)
        self.obj.set_xlabel(xlabel)
        self.obj.set_ylabel(ylabel)
        self.obj.set_ylim(ymin=ymin, ymax=ymax)
        self.obj.legend()


class Plotting:
    """
    A class for creating complex plots with multiple subplots from an input dictionary.

    This class takes an input dictionary containing data for multiple plots and automatically creates a figure with subplots. Each subplot visualizes different aspects of the data as specified in the 'subplot_specs' attribute.

    Attributes:
        input_dict (dict): A dictionary containing data to be plotted.
        subplot_specs (dict): Specifications for each subplot, including data keys, titles, axis labels, and plot locations.

    Methods:
        plot_results: Creates and displays the plot with the specified subplots and data.
    """

    def __init__(self, input_dict: dict) -> None:
        """
        Initializes the Plotting class with the input dictionary and predefined subplot specifications.

        Parameters:
            input_dict (dict): The input dictionary containing the data to be plotted in various subplots.
        """
        self.input_dict = input_dict
        self.subplot_specs = {
            "y": {
                "data_key": "y",
                "title": "Vertical position vs time",
                "xlabel": "time [s]",
                "ylabel": "y [m]",
                "ylim": (-1, 1),
                "plot_location": [0, 0],
            },
            "vy": {
                "data_key": "vy",
                "title": "Vertical velocity vs time",
                "xlabel": "time [s]",
                "ylabel": "vy [m/s]",
                "ylim": (-1, 1),
                "plot_location": [0, 1],
            },
            "z": {
                "data_key": "z",
                "title": "Horizontal position vs time",
                "xlabel": "time [s]",
                "ylabel": "z [m]",
                "ylim": (-2, 2),
                "plot_location": [1, 0],
            },
            "vz": {
                "data_key": "vz",
                "title": "Horizontal velocity vs time",
                "xlabel": "time [s]",
                "ylabel": "vz [m/s]",
                "ylim": (-2, 2),
                "plot_location": [1, 1],
            },
        }

        self.plot_results()

    def plot_results(self):
        """
        Creates and displays a figure with multiple subplots based on the input data and subplot specifications.

        This method iterates through the input data and uses the 'SubPlot' class to add plots and their attributes to each subplot in the figure. The method organizes the plots according to the predefined subplot specifications and displays the final figure with a title and a tight layout.
        """

        fig, ax = plt.subplots(2, 2)

        for data_point in self.input_dict.keys():
            for subplot_name in self.input_dict[data_point]:
                subplot_specs = self.subplot_specs[subplot_name]
                subplot_data = self.input_dict[data_point][subplot_name]

                plot_loc = subplot_specs["plot_location"]
                subplot = SubPlot(ax[plot_loc[0], plot_loc[1]])

                for curve in subplot_data:
                    subplot.add_plot(
                        x_data=curve["x-axis"],
                        y_data=curve["y-axis"],
                        colour=curve["colour"],
                        label=curve["type"],
                    )

                subplot.add_attributes(
                    subplot_specs["title"],
                    subplot_specs["xlabel"],
                    subplot_specs["ylabel"],
                    subplot_specs["ylim"][0],
                    subplot_specs["ylim"][1],
                )

        fig.suptitle(
            f"Comparison various input data ({len(self.input_dict.values())} runs)"
        )
        plt.tight_layout()
        plt.show()


class PlotTrainingResults:
    """
    A class for creating and saving various types of plots to visualize training results.

    This class takes a dictionary of training results and generates plots to visualize metrics like losses and accuracies. It supports plotting metrics separately for each data split (train, validate, test) and combined plots for comparison.

    Attributes:
        results (dict): A dictionary containing training results.
        folder_path (str): Path to the folder where the generated graphs will be saved.
        train_df (pd.DataFrame): DataFrame containing training data.
        validate_df (pd.DataFrame): DataFrame containing validation data.
        test_df (pd.DataFrame): DataFrame containing test data.
        data_len (int): The length of the DataFrame used for plotting.

    Methods:
        prep_data: Prepares DataFrames from the results dictionary.
        plot_metric: Plots specified metrics for given data splits.
        plot_separate_metric: Plots metrics separately for a single data split.
        plot_combined_metric: Plots metrics for multiple data splits in a combined manner.
    """

    def __init__(self, results_dict: dict, log_path: str, test_epoch: int) -> None:
        """
        Initializes the PlotTrainingResults with the results dictionary, log path, and test epoch.

        Parameters:
            results_dict (dict): The dictionary containing the results of training, validation, and testing.
            log_path (str): The path to the directory where the plots should be saved.
            test_epoch (int): The epoch number identified as the best during testing.
        """

        self.results = results_dict
        self.folder_path = os.path.join(log_path, "graphs")
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)

        self.train_df, self.validate_df, self.test_df = self.prep_data(test_epoch)
        self.data_len = len(self.train_df)

    def prep_data(
        self, test_epoch: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Prepares DataFrames from the results dictionary for each data split.

        The method converts the results for each data split into separate DataFrames and adjusts the index for the test DataFrame to align with other data splits.

        Parameters:
            test_epoch (int): The epoch number identified as the best during testing.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing the DataFrames for training, validation, and test data.
        """

        train_df = pd.DataFrame(self.results.get(DataSplit.TRAIN.value, {}))
        validate_df = pd.DataFrame(self.results.get(DataSplit.VALIDATE.value, {}))
        test_df = pd.DataFrame(self.results.get(DataSplit.TEST.value, {}))

        # Create a list of new index values
        new_index = [
            test_epoch - 1
        ]  # test_epoch returns the epoch +1, this equalizes it with the others

        # Assign the new index to the DataFrame
        test_df.index = new_index

        return train_df, validate_df, test_df

    def plot_metric(
        self,
        data_splits: DataSplit,
        filename: str,
        title: str,
        metric_names: list,
        colours: list,
    ) -> None:
        """
        Plots specified metrics for the given data splits.

        This method creates a plot for each specified metric across the provided data splits. It customizes the plot's appearance based on the data split type and metric.

        Parameters:
            data_splits (List[DataSplit]): A list of DataSplits (e.g., TRAIN, VALIDATE, TEST) to include in the plot.
            filename (str): The filename to save the plot.
            title (str): The title of the plot.
            metric_names (List[str]): List of metric names to plot.
            colours (List[str]): List of colours to use for each metric.
        """
        plt.figure(figsize=(12, 8))
        plt.xticks(range(0, self.data_len + 1))
        plt.title(f"{title} vs Epoch")
        plt.xlabel("Epoch [-]")
        plt.ylabel(f"{title} [-]")

        for data_split in data_splits:
            marker = "o"
            linestyle = "-"
            marker_size = 4
            df = getattr(self, f"{data_split.value.lower()}_df")

            if len(data_splits) > 1 and data_split == DataSplit.VALIDATE:
                linestyle = ":"  # dashed line for validation in combined graph

            if data_split == DataSplit.TEST:
                marker = "*"
                linestyle = "-."
                marker_size = 15

            for metric, colour in zip(metric_names, colours):
                plt.plot(
                    df.index
                    + 1,  # to display the first one as epoch_1 and not epoch_0.
                    df[metric],
                    label=f"{data_split.value} {metric}",
                    linestyle=linestyle,
                    marker=marker,
                    markersize=marker_size,
                    color=colour,
                )

        plt.legend()
        plt.savefig(os.path.join(self.folder_path, filename), dpi=400)
        plt.close()

    def plot_separate_metric(
        self, data_split: DataSplit, metric_names: list, colours: list, title: str
    ) -> None:
        """
        Plots metrics separately for a single data split.

        This method is a wrapper for 'plot_metric', specifically for plotting metrics for only one data split. It generates a filename based on the data split and title.

        Parameters:
            data_split (DataSplit): The DataSplit (e.g., TRAIN, VALIDATE, TEST) for which the metrics are to be plotted.
            metric_names (List[str]): List of metric names to plot.
            colours (List[str]): List of colours to use for each metric.
            title (str): The title of the plot.
        """

        filename = f"{data_split.value}_{title.lower()}.png"
        self.plot_metric([data_split], filename, title, metric_names, colours)

    def plot_combined_metric(
        self, data_splits: DataSplit, metric_names: list, colours: list, title: str
    ) -> None:
        """
        Plots metrics for multiple data splits in a combined manner.

        This method creates a combined plot for the specified metrics across multiple data splits. It generates a filename indicating that it's a combined plot.

        Parameters:
            data_splits (List[DataSplit]): A list of DataSplits (e.g., TRAIN, VALIDATE, TEST) to include in the combined plot.
            metric_names (List[str]): List of metric names to plot.
            colours (List[str]): List of colours to use for each metric.
            title (str): The title of the plot.
        """

        filename = f"combined_{title.lower()}.png"
        self.plot_metric(data_splits, filename, title, metric_names, colours)
