import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

import os
import json
import pandas as pd
from datetime import datetime
from typing import Union, Tuple, List, Optional
from enum import Enum


class InvertedSimilarity(Enum):
    """
    Enumerates types of inverted similarity metrics.

    This Enum classifies different methods of calculating inverted similarity, which are typically used in contexts
    where lower values indicate higher similarity. It is commonly utilized in machine learning algorithms, data analysis,
    or any scenario where measuring dissimilarity in a quantitative manner is required.

    Members:
        AVERAGE_MSE: Represents the default Mean Squared Error (MSE). It is a measure of the average squared difference
                     between the estimated values and the actual value. Lower MSE indicates closer similarity.
        WEIGHTED_MSE: Signifies a weighted version of Mean Squared Error. It applies weights to different components
                      of the MSE, giving priority to certain aspects over others, depending on the specific application.
    """

    AVERAGE_MSE = "default_mse"
    WEIGHTED_MSE = "weighted_mse"


class Similarity(Enum):
    """
    Enumerates types of similarity metrics.

    This Enum classifies various methods of calculating similarity, often used in statistical analysis, machine learning
    model evaluation, and data comparison tasks. These metrics are designed to quantify the degree of similarity between
    two data sets or variables.

    Members:
        PEARSON_CORR: Represents the Pearson correlation coefficient. It measures the linear correlation between two
                      variables, with a value range of -1 (total negative linear correlation) to 1 (total positive linear correlation).
        SPEARMAN_CORR: Signifies the Spearman's rank correlation coefficient. It assesses how well the relationship between
                       two variables can be described using a monotonic function, considering the ranks of values rather than the raw data.
    """

    PEARSON_CORR = "weighted_pearson_correlation"
    SPEARMAN_CORR = "weighted_spearman_correlation"


class DataProcessor:
    """
    A class designed for processing and analyzing data from a given file path, particularly in the context of machine learning model evaluations.

    This class is equipped to handle tasks such as data loading, transformation, and specific analyses like calculating accuracy per epoch and parsing data for similarity evaluations.

    Attributes:
        file_path (str): Path to the data file. Determined automatically if last_run is True.
        data (pd.DataFrame or dict): Data loaded from the file, in various formats including DataFrame.
        similarity (Union[Similarity, InvertedSimilarity]): Type of similarity metric used in data analysis.
        data_index (List[str]): List of index labels from the DataFrame, usually representing epochs or batches.
        parsed_df (pd.DataFrame): DataFrame with processed data, including similarity calculations and other metrics.

    Methods:
        __init__: Initializes the DataProcessor with options for last run detection, similarity metric, and file path.
        get_last_run_path: Retrieves the path of the latest data file based on naming conventions.
        load_data: Loads data from the specified file path.
        to_dataframe: Converts the loaded data into a Pandas DataFrame.
        get_accuracy_per_epoch: Calculates and returns accuracy per epoch.
        parse_df: Processes and analyzes data, applying similarity calculations.
        parse_data_to_dict: Converts a subset of data into a dictionary format.
    """

    def __init__(
        self,
        last_run=bool,
        similarity=Union[Similarity, InvertedSimilarity],
        file_path: str = "",
    ):
        """
        Initializes the DataProcessor instance.

        Parameters:
            last_run (bool): If True, automatically finds the path of the last run.
            similarity (Union[Similarity, InvertedSimilarity]): The type of similarity metric to be used in data processing.
            file_path (str, optional): The path of the data file, used if last_run is False. Defaults to an empty string.

        Raises:
            FileNotFoundError: If the specified file path or the last run path is not found.
        """
        if last_run:
            try:
                self.file_path = self.get_last_run_path(".\\training_logs\\default")
            except Exception as e:
                raise FileNotFoundError(f"Error finding last run path: {e}")
        else:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File path {file_path} does not exist.")
            self.file_path = file_path

        self.data = None
        self.similarity = similarity
        self.load_data()
        self.to_dataframe()
        self.data_index = self.data.index.to_list()
        self.parsed_df = self.parse_df()

    def get_last_run_path(self, folder_path: str) -> str:
        """
        Retrieves the path of the last run based on the file naming convention in a specified folder.

        Parameters:
            folder_path (str): The directory path to search for the latest data file.

        Returns:
            str: The path of the most recent run file.

        Raises:
            FileNotFoundError: If no suitable files are found in the specified folder.
            ValueError: If file names in the folder don't follow the expected date-time format.
        """
        try:
            str_lst = os.listdir(folder_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Folder {folder_path} not found.")

        if not str_lst:
            raise FileNotFoundError(f"No files found in {folder_path}.")

        date_lst = []
        for name in str_lst:
            try:
                date_lst.append(datetime.strptime(name, "%Y%m%d-%H%M%S"))
            except ValueError:
                continue  # Skip files not following the date format

        if not date_lst:
            raise ValueError(
                f"No files in {folder_path} with the expected date format."
            )

        sorted_dates = sorted(date_lst, reverse=True)
        latest_filename = sorted_dates[0].strftime("%Y%m%d-%H%M%S")

        latest_folder_path = os.path.join(folder_path, latest_filename)
        lst_files = os.listdir(latest_folder_path)

        if not lst_files:
            raise FileNotFoundError(f"No files found in {latest_folder_path}.")

        return os.path.join(latest_folder_path, lst_files[0])

    def load_data(self) -> None:
        """
        Loads data from the file path specified in the instance.

        Parameters:
            None

        Raises:
            FileNotFoundError: If the file to load data from is not found.
            json.JSONDecodeError: If the file content is not valid JSON.
        """
        try:
            with open(self.file_path, "r") as file:
                self.data = json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"File {self.file_path} not found.")
        except json.JSONDecodeError:
            raise json.JSONDecodeError(
                f"File {self.file_path} does not contain valid JSON."
            )

    def to_dataframe(self) -> None:
        """
        Converts the loaded data into a Pandas DataFrame for easier analysis and processing.

        Parameters:
            None

        Raises:
            ValueError: If the data is not in a format convertible to a DataFrame.
        """
        try:
            self.data = pd.DataFrame(self.data)
            self.data = self.data.T
        except ValueError as e:
            raise ValueError(f"Error converting data to DataFrame: {e}")

    def get_accuracy_per_epoch(self) -> Tuple[List[int], List[float]]:
        """
        Extracts and returns the accuracy per epoch from the processed data.

        Parameters:
            None

        Returns:
            Tuple[List[int], List[float]]: A tuple containing epochs and their corresponding accuracy values.
        """

        x = self.data_index
        y = self.data["accuracy"]
        return x, y

    def parse_df(self) -> pd.DataFrame:
        """
        Processes the loaded data, applying similarity calculations and preparing it for detailed analysis.

        Parameters:
            None

        Returns:
            pd.DataFrame: A DataFrame containing the parsed and processed data, including similarity calculations.
        """

        first_epoch_data = self.data.loc["epoch_1"]
        sub_idx_lst = first_epoch_data.index

        data = []

        for batch_idx in sub_idx_lst:
            if "batch" in batch_idx:
                kys = first_epoch_data[batch_idx].keys()
                for data_point_idx in kys:
                    if "data_point" in data_point_idx:
                        data.append(
                            self.parse_data_to_dict(
                                first_epoch_data[batch_idx][data_point_idx]
                            )
                        )

        parsed_df = pd.DataFrame(data)

        if isinstance(self.similarity, InvertedSimilarity):
            parsed_df["similarity_plastic"] = 1 / parsed_df["similarity_plastic"]
            parsed_df["similarity_fish"] = 1 / parsed_df["similarity_fish"]

        def check_math_model_accuracy(plastic_similarity, fish_similarity, label):
            if plastic_similarity > fish_similarity:
                if label == 0:
                    return True
                else:
                    return False

            elif fish_similarity > plastic_similarity:
                if label == 1:
                    return True
                else:
                    return False
            else:
                print("fish and plastic have an equal mse")

        parsed_df["matching"] = parsed_df.apply(
            lambda x: check_math_model_accuracy(
                x["similarity_plastic"], x["similarity_fish"], x["label"]
            ),
            axis=1,
        )

        return parsed_df

    def parse_data_to_dict(self, sub_data: dict) -> dict:
        """
        Converts a specific subset of the data into a dictionary format, focusing on key metrics and calculations.

        Parameters:
            sub_data (dict): The subset of data to be processed.

        Returns:
            dict: A dictionary containing processed metrics and values derived from the subset of data.
        """

        dict = {}
        prob = sub_data["softmax_teacher"]

        dict["softmax_teacher_plastic"] = prob[0]
        dict["softmax_teacher_fish"] = prob[1]

        dict["similarity_plastic"] = sub_data["similarities"]["plastic"][
            self.similarity.value
        ]
        dict["similarity_fish"] = sub_data["similarities"]["fish"][
            self.similarity.value
        ]

        dict["mse_plastic"] = sub_data["similarities"]["plastic"][self.similarity.value]
        dict["mse_fish"] = sub_data["similarities"]["fish"][self.similarity.value]

        comp_plastic = sub_data["parameter_estimation"]["plastic"]["comparison"]
        comp_fish = sub_data["parameter_estimation"]["fish"]["comparison"]

        dict["param_estm_plastic"] = sum(comp_plastic.values()) / len(comp_plastic)
        dict["param_estm_fish"] = sum(comp_fish.values()) / len(comp_fish)

        dict["label"] = sub_data["labels"]

        dict["student_output_plastic"] = sub_data["student_outputs"][0]
        dict["student_output_fish"] = sub_data["student_outputs"][1]

        dict["teacher_output_plastic"] = sub_data["teacher_outputs"][0]
        dict["teacher_output_fish"] = sub_data["teacher_outputs"][1]

        return dict


class Plotter:
    """
    A class designed to facilitate various types of data plotting.

    This class supports different plotting styles, such as scatter plots and line plots. It provides a simple interface to plot data with customizable plot attributes.

    Methods:
        plot: Creates a plot based on the specified type and data.
    """

    def plot(
        self,
        plot_type: str,
        x,
        y,
        z: Optional[Union[list, None]] = None,
        title: str = "Scatter Plot",
        xlabel: str = "X-Axis",
        ylabel: str = "Y-Axis",
    ) -> None:
        """
        Generates a plot of the given type with the provided data.

        Depending on the plot type, this method creates either a scatter plot or a line plot. It also supports an optional third dataset.

        Parameters:
            plot_type (str): Type of plot to generate ('scatter' or 'line').
            x: Data for the X-axis.
            y: Primary data for the Y-axis.
            z (Optional[Union[list, None]]): Optional secondary data for the Y-axis. Default is None.
            title (str): Title of the plot. Default is "Scatter Plot".
            xlabel (str): Label for the X-axis. Default is "X-Axis".
            ylabel (str): Label for the Y-axis. Default is "Y-Axis".

        Returns:
            None: This method shows the plot but does not return any value.
        """

        if plot_type == "scatter":
            plt.scatter(x, y)
            if z is not None:
                plt.scatter(x, z)

        elif plot_type == "line":
            plt.plot(x, y)
            if z is not None:
                plt.plot(x, z)

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()


def main():
    """
    The main function to demonstrate the data processing and plotting using the DataProcessor and Plotter classes.

    This function processes data, calculates certain metrics, and visualizes the results. It demonstrates how to filter data, calculate ratios, and create plots to analyze the 'matching' data and the ratio of various metrics.

    It shows the use of the DataProcessor for parsing and processing data, followed by using the Plotter class to create specific plots for analysis.
    """

    data_processor = DataProcessor(
        last_run=True, similarity=InvertedSimilarity.WEIGHTED_MSE
    )
    # Count of True values in 'match' column

    df = data_processor.parsed_df
    df = df.loc[(df["mse_plastic"] - df["mse_fish"]).abs() > 0.018]
    true_count = df["matching"].sum()

    # Total number of rows
    total_rows = len(df)  # or df.shape[0]

    # Calculate the ratio
    ratio = true_count / total_rows

    print("Ratio of True values:", ratio)
    pd.set_option("display.max_rows", None)

    plt.plot(
        df.index,
        df["similarity_plastic"] / df["similarity_fish"],
        label="similarity plastic/fish",
    )
    plt.plot(
        df.index,
        df["softmax_teacher_plastic"] / df["softmax_teacher_fish"],
        label="softmax_teacher plastic/fish",
    )
    plt.ylim(top=3, bottom=0)
    plt.xlabel("number of datapoints [-]")
    plt.ylabel("Ratio plastic/fish [-]")
    plt.legend()
    plt.title("Ratio plastic/fish for the similarity and softmax teacher")
    plt.show()


if __name__ == "__main__":
    main()
