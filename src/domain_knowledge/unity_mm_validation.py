import sys

import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
import os

from enum import Enum
from src.domain_knowledge.mathematical_model import MathematicalModel
from typing import Dict, List


class StatisticalParameters(Enum):
    """
    Enumeration for different types of statistical parameters.

    This enumeration defines various statistical parameters used in data analysis, such as error measurements and accuracy.

    Members:
        RMSE: Root Mean Squared Error, a standard way to measure the error of a model in predicting quantitative data.
        NRMSE_MAXMIN: Normalized Root Mean Squared Error based on the maximum and minimum values, used for error normalization.
        NRMSE_STD: Normalized Root Mean Squared Error based on the standard deviation, useful for comparing the performance of different models or systems.
        ACCURACY: Accuracy, typically used to measure the correctness of a model, especially in classification tasks.
    """

    RMSE = "Root mean squared error"
    NRMSE_MAXMIN = (
        "Normalized root mean squared error based on the maximum/minumum values"
    )
    NRMSE_STD = "Normalized root mean squared error based on the standard deviation"
    ACCURACY = "Accuracy"


class UnityData:
    """
    A class for processing and analyzing data extracted from Unity simulations.

    This class provides functionalities to create a DataFrame from Unity's JSON data, and to extract specific information like the start position and the duration of the simulation.

    Attributes:
        df (pd.DataFrame): The DataFrame containing position and velocity data from Unity.

    Methods:
        __init__: Initializes the UnityData instance with JSON data.
        _create_df_from_json: Creates a DataFrame from the provided JSON data.
        get_start_position: Retrieves the starting position and velocity from the DataFrame.
        get_duration: Calculates the duration of the simulation based on the time data.
    """

    def __init__(self, data: dict) -> None:
        """
        Initializes the UnityData instance with data from Unity's JSON output.

        Parameters:
            data (dict): The JSON data exported from Unity containing position, velocity, and frame time.
        """

        self.df = self._create_df_from_json(data)

    def _create_df_from_json(self, data: dict) -> pd.DataFrame:
        """
        Creates a DataFrame from Unity's JSON data.

        Parameters:
            data (dict): The JSON data exported from Unity.

        Returns:
            pd.DataFrame: A DataFrame containing position and velocity data along with frame time.
        """

        df_position = pd.DataFrame(data["position"])
        df_velocity = pd.DataFrame(data["velocity"])

        df_velocity.columns = ["vx", "vy", "vz"]

        df_time = pd.DataFrame(data["frameTime"], columns=["t"])

        combined_df = pd.concat([df_position, df_velocity, df_time], axis=1)

        return combined_df

    def get_start_position(self) -> dict:
        """
        Retrieves the starting position and velocity of the object from the DataFrame.

        Returns:
            dict: A dictionary containing the initial position (y_0, z_0) and velocity (vy_0, vz_0) values.
        """

        return {
            "y_0": self.df["y"][0],
            "vy_0": self.df["vy"][0],
            "z_0": self.df["z"][0],
            "vz_0": self.df["vz"][0],
        }

    def get_duration(self) -> float:
        """
        Calculates the duration of the Unity simulation.

        Returns:
            float: The maximum value of the time column in the DataFrame, representing the simulation duration.
        """

        return self.df["t"].max()


class SubPlot:
    """
    A utility class for managing individual subplots within a larger figure.

    This class provides an interface to add plots and attributes (like titles and labels) to a specific subplot within a Matplotlib figure.

    Attributes:
        obj (matplotlib.axes.Axes): The Axes object representing the subplot.

    Methods:
        add_plot: Adds a plot to the subplot.
        add_attributes: Sets attributes like title, labels, and axis limits for the subplot.
    """

    def __init__(self, obj: plt.Axes) -> None:
        """
        Initializes the SubPlot with a given Axes object.

        Parameters:
            obj (plt.Axes): A Matplotlib Axes object representing the subplot.
        """

        self.obj = obj

    def add_plot(
        self, x_data: List[float], y_data: List[float], colour: str, label: str
    ) -> None:
        """
        Adds a plot to the subplot.

        Parameters:
            x_data (List[float]): The x-axis data for the plot.
            y_data (List[float]): The y-axis data for the plot.
            colour (str): The color of the plot.
            label (str): The label for the plot, used in the legend.
        """
        self.obj.plot(x_data, y_data, color=colour, label=label)

    def add_attributes(
        self, title: str, xlabel: str, ylabel: str, ymin: float, ymax: float
    ) -> None:
        """
        Sets various attributes like title, labels, and y-axis limits for the subplot.

        Parameters:
            title (str): The title of the subplot.
            xlabel (str): The label for the x-axis.
            ylabel (str): The label for the y-axis.
            ymin (float): The minimum limit for the y-axis.
            ymax (float): The maximum limit for the y-axis.
        """

        self.obj.set_title(title)
        self.obj.set_xlabel(xlabel)
        self.obj.set_ylabel(ylabel)
        self.obj.set_ylim(ymin=ymin, ymax=ymax)
        self.obj.legend()


class Plotting:
    """
    A class for creating and displaying plots comparing Unity simulation data with mathematical models.

    This class takes lists of DataFrames containing Unity simulation data and mathematical model outputs, and creates plots to visualize and compare these datasets.

    Attributes:
        unity_df_lst (List[pd.DataFrame]): List of DataFrames containing Unity simulation data.
        mm_df_lst (List[pd.DataFrame]): List of DataFrames containing mathematical model outputs.
        mm_df_lst_ideal (List[pd.DataFrame]): List of DataFrames containing ideal mathematical model outputs.

    Methods:
        plot_results: Creates and displays a series of subplots comparing simulation and model data.
        plot_results_separate: Creates and saves individual plots for each comparison aspect.
    """

    def __init__(
        self,
        unity_data_df_lst: List[pd.DataFrame],
        mathematical_model_df_lst: List[pd.DataFrame],
        ideal_mm_model: List[pd.DataFrame],
    ) -> None:
        """
        Initializes the Plotting class with lists of DataFrames.

        Parameters:
            unity_data_df_lst (List[pd.DataFrame]): List of DataFrames with Unity simulation data.
            mathematical_model_df_lst (List[pd.DataFrame]): List of DataFrames with mathematical model data.
            ideal_mm_model (List[pd.DataFrame]): List of DataFrames with ideal mathematical model data.
        """

        self.unity_df_lst = unity_data_df_lst
        self.mm_df_lst = mathematical_model_df_lst
        self.mm_df_lst_ideal = ideal_mm_model

        self.plot_results()

    def plot_results(self) -> None:
        """
        Creates and displays subplots for the comparison of Unity simulation data with mathematical model outputs.
        """

        sim_col = "green"
        mm_col = "blue"
        mm_ideal_col = "red"

        fig, ax = plt.subplots(2, 2)  # Creates a 2x2 grid of subplots

        for run_number, (sim_df, mm_df, mm_df_ideal) in enumerate(
            zip(self.unity_df_lst, self.mm_df_lst, self.mm_df_lst_ideal)
        ):
            sub_plot_1 = SubPlot(ax[0, 0])
            sub_plot_1.add_plot(
                sim_df["t"], sim_df["y"], sim_col, f"y_sim run {run_number}"
            )
            sub_plot_1.add_plot(
                mm_df["t"], mm_df["y"], mm_col, f"y_mm run {run_number}"
            )
            sub_plot_1.add_plot(
                mm_df_ideal["t"],
                mm_df_ideal["y"],
                mm_ideal_col,
                f"y_mm ideal run {run_number}",
            )
            sub_plot_1.add_attributes(
                "Vertical position vs time", "time [s]", "y [m]", -1, 1
            )

            sub_plot_2 = SubPlot(ax[0, 1])
            sub_plot_2.add_plot(
                sim_df["t"], sim_df["vy"], sim_col, f"vy_sim run {run_number}"
            )
            sub_plot_2.add_plot(
                mm_df["t"], mm_df["vy"], mm_col, f"vy_mm run {run_number}"
            )
            sub_plot_2.add_plot(
                mm_df_ideal["t"],
                mm_df_ideal["vy"],
                mm_ideal_col,
                f"vy_mm ideal run {run_number}",
            )
            sub_plot_2.add_attributes(
                "Vertical velocity vs time", "time [s]", "vy [m/s]", -1, 1
            )

            sub_plot_3 = SubPlot(ax[1, 0])
            sub_plot_3.add_plot(
                sim_df["t"], sim_df["z"], sim_col, f"z_sim run {run_number}"
            )
            sub_plot_3.add_plot(
                mm_df["t"], mm_df["z"], mm_col, f"z_mm run {run_number}"
            )
            sub_plot_3.add_plot(
                mm_df_ideal["t"],
                mm_df_ideal["z"],
                mm_ideal_col,
                f"z_mm ideal run {run_number}",
            )
            sub_plot_3.add_attributes(
                "Horizontal position vs time", "time [s]", "z [m]", -2, 2
            )

            sub_plot_4 = SubPlot(ax[1, 1])
            sub_plot_4.add_plot(
                sim_df["t"], sim_df["vz"], sim_col, f"vz_sim run {run_number}"
            )
            sub_plot_4.add_plot(
                mm_df["t"], mm_df["vz"], mm_col, f"vz_mm run {run_number}"
            )
            sub_plot_4.add_plot(
                mm_df_ideal["t"],
                mm_df_ideal["vz"],
                mm_ideal_col,
                f"vz_mm ideal run {run_number}",
            )
            sub_plot_4.add_attributes(
                "Horizontal velocity vs time", "time [s]", "vz [m/s]", -2, 2
            )

        fig.suptitle(
            f"Comparison simulation vs mathematical model ({len(self.unity_df_lst)} runs)"
        )
        plt.show()

    def plot_results_separate(self) -> None:
        """
        Creates and saves separate plots for each aspect of the Unity simulation and mathematical model comparison.
        """
        sim_col = "green"
        mm_col = "blue"

        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()
        fig3, ax3 = plt.subplots()
        fig4, ax4 = plt.subplots()
        fig5, ax5 = plt.subplots()
        fig6, ax6 = plt.subplots()

        for run_number, (sim_df, mm_df) in enumerate(
            zip(self.unity_df_lst, self.mm_df_lst)
        ):
            sub_plot_1 = SubPlot(ax1)
            sub_plot_1.add_plot(
                sim_df["t"], sim_df["y"], sim_col, f"y_sim run {run_number}"
            )
            sub_plot_1.add_plot(
                mm_df["t"], mm_df["y"], mm_col, f"y_mm run {run_number}"
            )
            sub_plot_1.add_attributes("Vertical position vs time", "time [s]", "y [m]")

            sub_plot_2 = SubPlot(ax2)
            sub_plot_2.add_plot(
                sim_df["t"], sim_df["vy"], sim_col, f"vy_sim run {run_number}"
            )
            sub_plot_2.add_plot(
                mm_df["t"], mm_df["vy"], mm_col, f"vy_mm run {run_number}"
            )
            sub_plot_2.add_attributes(
                "Vertical velocity vs time", "time [s]", "vy [m/s]"
            )

            sub_plot_3 = SubPlot(ax3)
            sub_plot_3.add_plot(
                sim_df["t"], sim_df["z"], sim_col, f"z_sim run {run_number}"
            )
            sub_plot_3.add_plot(
                mm_df["t"], mm_df["z"], mm_col, f"z_mm run {run_number}"
            )
            sub_plot_3.add_attributes(
                "Horizontal position vs time", "time [s]", "z [m]"
            )

            sub_plot_4 = SubPlot(ax4)
            sub_plot_4.add_plot(
                sim_df["t"], sim_df["vz"], sim_col, f"vz_sim run {run_number}"
            )
            sub_plot_4.add_plot(
                mm_df["t"], mm_df["vz"], mm_col, f"vz_mm run {run_number}"
            )
            sub_plot_4.add_attributes(
                "Horizontal velocity vs time", "time [s]", "vz [m/s]"
            )

        fig1.savefig(
            r"C:\Users\mathi\Documents\Files\University of Technology Delft\Master\MSc 2\Thesis\Code\thesis_mathieu\model_identification\results3\graph1.png",
            dpi=300,
            format="png",
        )
        fig2.savefig(
            r"C:\Users\mathi\Documents\Files\University of Technology Delft\Master\MSc 2\Thesis\Code\thesis_mathieu\model_identification\results3\graph2.png",
            dpi=300,
            format="png",
        )
        fig3.savefig(
            r"C:\Users\mathi\Documents\Files\University of Technology Delft\Master\MSc 2\Thesis\Code\thesis_mathieu\model_identification\results3\graph3.png",
            dpi=300,
            format="png",
        )
        fig4.savefig(
            r"C:\Users\mathi\Documents\Files\University of Technology Delft\Master\MSc 2\Thesis\Code\thesis_mathieu\model_identification\results3\graph4.png",
            dpi=300,
            format="png",
        )


class StatisticalAnalysis:
    """
    A class for conducting statistical analysis on simulation and mathematical model data.

    This class provides methods for merging simulation and model data, computing errors, and evaluating the performance of models.

    Attributes:
        sim_df_lst (List[pd.DataFrame]): List of dataframes from the Unity simulator.
        mm_df_lst (List[pd.DataFrame]): List of dataframes from the mathematical model.
        keys (List[str]): List of column keys for analysis.

    Methods:
        merge_two_df: Merges two dataframes (simulation and mathematical model) for comparison.
        _compute_error: Computes specified error between true and predicted values.
        compute_error_for_all_runs: Computes errors for all simulation runs and averages them.
    """

    def __init__(
        self, sim_df_lst: List[pd.DataFrame], mm_df_lst: List[pd.DataFrame]
    ) -> None:
        """
        Initializes the StatisticalAnalysis with lists of simulation and model dataframes.

        Parameters:
            sim_df_lst (List[pd.DataFrame]): List of dataframes from the Unity simulator.
            mm_df_lst (List[pd.DataFrame]): List of dataframes from the mathematical model.
        """

        self.sim_df_lst = sim_df_lst
        self.mm_df_lst = mm_df_lst

        self.keys = ["y", "vy", "z", "vz"]

    def merge_two_df(self, df_sim, df_mm):
        """
        Merges two dataframes together and uses the timescale of the second dataframe (should correspond to the mathematical model)

        Parameters:
            df_sim (pd.DataFrame):  dataframe from Unity simulation
            df_mm (pd.DataFrame):   dataframe from the mathematical model

        Return:
            - merged_df (pd.DataFrame):  the dataframe contains columns with a prefix 'sim_' corresponding to the columns of the simulator dataframe and
                                      columns with the prefix 'mm_' referring to the columns of the mathematical model. The merged df contains both
                                      timestamps from the sim and mm and is interpolated. So the values can directly be compared.
        """

        # Copy and rename the columns
        # df_sim = df_sim[['y', 'z', 'vy', 'vz', 't']].copy()
        # df_sim.rename(columns={'y': 'sim_y', 'z': 'sim_z', 'vy': 'sim_vy', 'vz': 'sim_vz'}, inplace=True)
        # Replace the above two lines with the following ones after recording rotation signals
        df_sim = df_sim[["y", "z", "vy", "vz", "t"]].copy()
        df_sim.rename(
            columns={
                "y": "sim_y",
                "z": "sim_z",
                "vy": "sim_vy",
                "vz": "sim_vz",
            },
            inplace=True,
        )

        # Copy and rename the columns
        # df_mm = df_mm[['y', 'z', 'vy', 'vz', 't']].copy()
        # df_mm.rename(columns={'y': 'mm_y', 'z': 'mm_z', 'vy': 'mm_vy', 'vz': 'mm_vz'}, inplace=True)
        # Replace the above two lines with the following ones after recording rotation signals
        df_mm = df_mm[["y", "z", "vy", "vz", "t"]].copy()
        df_mm.rename(
            columns={
                "y": "mm_y",
                "z": "mm_z",
                "vy": "mm_vy",
                "vz": "mm_vz",
            },
            inplace=True,
        )

        # Set 't' as the index in both dataframes
        df_sim.set_index("t", inplace=True)
        df_mm.set_index("t", inplace=True)

        # Find the minimum of the maximum 't' in both original dataframes
        min_max_t = min(df_sim.index.max(), df_mm.index.max())

        # Create a common index
        common_index = df_sim.index.union(df_mm.index)

        # Reindex and interpolate both dataframes to the common index
        df_sim = df_sim.reindex(common_index).interpolate(method="linear")
        df_mm = df_mm.reindex(common_index).interpolate(method="linear")

        # Merge the dataframes on the index
        merged_df = pd.merge(df_sim, df_mm, left_index=True, right_index=True)

        # Remove the rows with 't' greater than the minimum of the maximum 't'. This is implemented due to a mismatch in length of
        # the unity simulator and the mathematical model (simulator is 12.2 sec, mathematical model only for 10sec)
        merged_df = merged_df[merged_df.index <= min_max_t]

        # Reset the index to get 't' back as a column
        merged_df.reset_index(inplace=True)

        return merged_df

    def _compute_error(
        self, y_true: np.ndarray, y_pred: np.ndarray, error_type: StatisticalParameters
    ) -> float:
        """
        Computes the specified error type between true and predicted values.

        Parameters:
            y_true (np.ndarray): The true values.
            y_pred (np.ndarray): The predicted values.
            error_type (StatisticalParameters): The type of error to compute.

        Returns:
            float: The computed error value.
        """

        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        if error_type == StatisticalParameters.RMSE:
            return rmse

        elif error_type == StatisticalParameters.NRMSE_MAXMIN:
            min_val = np.min(y_true)
            max_val = np.max(y_true)
            return (rmse / (max_val - min_val)) * 100

        elif error_type == StatisticalParameters.NRMSE_STD:
            std_dev = np.std(y_true)

            return (rmse / std_dev) * 100

        elif error_type == StatisticalParameters.ACCURACY:
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            return 100 - mape

    def compute_error_for_all_runs(
        self, error_type: StatisticalParameters
    ) -> pd.DataFrame:
        """
        Computes and averages the specified error for all simulation runs.

        Parameters:
            error_type (StatisticalParameters): The type of error to compute.

        Returns:
            pd.DataFrame: DataFrame containing the average error values for each key.
        """

        rmse_all_runs = pd.DataFrame(columns=self.keys)

        for sim_df, mm_df in zip(self.sim_df_lst, self.mm_df_lst):
            merged_df = self.merge_two_df(sim_df, mm_df)  # Merging dataframes

            mse_dict = {}
            for col in self.keys:
                rmse = self._compute_error(
                    merged_df[f"sim_{col}"],
                    merged_df[f"mm_{col}"],
                    error_type=error_type,
                )  # Computing RMSE
                mse_dict[col] = rmse

            # Replacing deprecated append method with pandas.concat
            mse_df = pd.DataFrame([mse_dict])
            rmse_all_runs = pd.concat([rmse_all_runs, mse_df], ignore_index=True)

        return rmse_all_runs.mean()


class ConfigLoader:
    """
    A class for loading configuration settings from a JSON file.

    Provides functionality to read and parse configuration settings used in simulations or models.

    Attributes:
        config_file_path (str): Path to the JSON configuration file.
        config (dict): Dictionary containing loaded configuration settings.

    Methods:
        load_config: Loads and parses the JSON configuration file.
    """

    def __init__(self, config_file_path: str) -> None:
        """
        Initializes the ConfigLoader with the path to a JSON configuration file.

        Parameters:
            config_file_path (str): Path to the JSON configuration file.
        """

        self.config_file_path = config_file_path

    def load_config(self) -> Dict:
        """
        Loads and parses the JSON configuration file.

        Returns:
            Dict: Dictionary containing the configuration settings.
        """

        with open(self.config_file_path, "r") as file:
            self.config = json.load(file)

        return self.config


if __name__ == "__main__":
    unity_all_runs = []
    mm_all_runs = []
    ideal_runs = []

    main_path = r"C:\Users\mathi\Documents\Files\University of Technology Delft\Master\MSc 2\Thesis\Code\thesis_mathieu\unity_data/parsed_data.json"
    n_decimals = 10
    n_samples = 8

    config_data = ConfigLoader(main_path).load_config()

    models = ["Plastic", "Fish"]

    i = 0
    for file in config_data.keys():
        unity_data = UnityData(config_data[file]["unity_annotations"])
        start_pos = unity_data.get_start_position()
        duration = unity_data.get_duration()

        original_model_type = config_data[file]["unity_settings"]["objectType"]

        # for model in models:
        #     print(
        #         f"Running {model} mathematical model for {original_model_type}paramaters ({file}...)"
        #     )

        model_pb = MathematicalModel(
            config=config_data[file]["unity_settings"],
            init_states=start_pos,
            tstart=0,
            tend=duration,
            tsample=0.1,
            n_decimals=n_decimals,
            model_type=original_model_type,
            sim_traj=unity_data.df,
            use_extracted_param=True,
        )

        model_pb_ideal = MathematicalModel(
            config=config_data[file]["unity_settings"],
            init_states=start_pos,
            tstart=0,
            tend=duration,
            tsample=0.1,
            n_decimals=n_decimals,
            model_type=original_model_type,
            sim_traj=unity_data.df,
            use_extracted_param=False,
        )

        solutions_df = model_pb.solve()
        solutions_df_ideal = model_pb_ideal.solve()

        ideal_runs.append(solutions_df_ideal)
        unity_all_runs.append(unity_data.df)
        mm_all_runs.append(solutions_df)

        i += 1

        if i == n_samples:
            break

    plotting = Plotting(unity_all_runs, mm_all_runs, ideal_runs)
    stat_analyis = StatisticalAnalysis(unity_all_runs, mm_all_runs)

    rmse_values = stat_analyis.compute_error_for_all_runs(StatisticalParameters.RMSE)
    print(
        f"Average Root Mean Squared Error (RMSE) for all runs in percentage: \n{rmse_values}"
    )
