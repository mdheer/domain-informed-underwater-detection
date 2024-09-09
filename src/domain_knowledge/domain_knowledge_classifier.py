import os
import pandas as pd
import math
import torch
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import json
import numpy as np
from typing import Union, List, Tuple, Dict
import numpy as np
import pprint
from dtaidistance import dtw
from frechetdist import frdist
import sys
import similaritymeasures

from src.tools.general_functions import sync_timestamps, get_midpoints

from src.domain_knowledge.mathematical_model import (
    MathematicalModel,
    ParameterEstimation,
)

from src.tools.enums import (
    ClassName,
    DataVariant,
    Similarity,
    InvertedSimilarity,
)
from scipy.spatial.distance import directed_hausdorff


class Metrics:
    """
    A class designed for calculating various statistical metrics between columns of a DataFrame.

    This class provides functionalities to calculate metrics such as Mean Squared Error (MSE), accuracy, Median Absolute Deviation (MAD), Hausdorff distance, Pearson and Spearman correlation between specified pairs of columns.

    Attributes:
        df (pd.DataFrame): The DataFrame containing the data to analyze.
        mm_columns (dict): A dictionary mapping the column names for one set of data (e.g., mathematical model outputs).
        traj_columns (dict): A dictionary mapping the column names for another set of data (e.g., trajectory data).

    Methods:
        __init__: Initializes the Metrics class with a DataFrame and column mappings.
        calculate_mse: Calculates Mean Squared Error (MSE) for each pair of specified columns.
        calculate_accuracy: Calculates accuracy for each pair of specified columns based on MSE.
        calculate_mad: Calculates Median Absolute Deviation (MAD) for each pair of specified columns.
        calculate_discrete_hausdorff: Calculates the discrete Hausdorff distance for each pair of specified columns.
        calculate_pearson_correlation: Calculates Pearson correlation coefficient for each pair of specified columns.
        calculate_spearman_correlation: Calculates Spearman's rank correlation coefficient for each pair of specified columns.
    """

    def __init__(self, df: pd.DataFrame, mm_columns: dict, traj_columns: dict) -> None:
        """
        Initializes the Metrics class with a DataFrame and column mappings.

        Parameters:
            df (pd.DataFrame): The DataFrame containing the data for metric calculation.
            mm_columns (dict): Dictionary mapping column names for the first set of data.
            traj_columns (dict): Dictionary mapping column names for the second set of data.
        """
        self.df = df
        self.mm_columns = mm_columns
        self.traj_columns = traj_columns

    def calculate_mse(self) -> dict:
        """
        Calculates the Mean Squared Error (MSE) for each pair of specified columns.

        Returns:
            dict: A dictionary with keys as column pairs and values as the corresponding MSE.
        """
        mse = {}
        for i in self.mm_columns.keys():
            self.df[f"mse_{i}"] = (
                self.df[self.mm_columns[i]] - self.df[self.traj_columns[i]]
            ) ** 2
            mse[i] = self.df[f"mse_{i}"].mean()

        return mse

    def calculate_variance(self) -> dict:
        var = {}
        for i in self.mm_columns.keys():
            self.df[f"var_{i}"] = (
                (self.df[self.mm_columns[i]] - self.df[self.mm_columns[i]].mean())
                - (self.df[self.traj_columns[i]] - self.df[self.traj_columns[i]].mean())
            ) ** 2
            var[i] = self.df[f"var_{i}"].mean()

        return var

    def calculate_accuracy(self) -> dict:
        """
        Calculates the accuracy for each pair of specified columns based on MSE.

        Returns:
            dict: A dictionary with keys as column pairs and values as the corresponding accuracy.
        """
        mse = {}
        accuracy = {}
        for i in self.mm_columns.keys():
            self.df[f"mse_{i}"] = (
                self.df[self.mm_columns[i]] - self.df[self.traj_columns[i]]
            ) ** 2

            self.df[f"accuracy_{i}"] = 1 - abs(
                np.sqrt(self.df[f"mse_{i}"]) / self.df[self.traj_columns[i]]
            )

            # mse[i] = self.df[f"mse_{i}"].mean() / self.df[f"mse_{i}"].max()
            mse[i] = self.df[f"mse_{i}"].mean()
            accuracy[i] = self.df[f"accuracy_{i}"].mean()

        return accuracy

    def calculate_mad(self) -> dict:
        """
        Calculates the Median Absolute Deviation (MAD) for each pair of specified columns.

        Returns:
            dict: A dictionary with keys as column pairs and values as the corresponding MAD.
        """
        mad = {}
        for i in self.mm_columns.keys():
            # Calculate the absolute deviations from the median
            self.df[f"dev_{i}"] = abs(
                self.df[self.mm_columns[i]] - self.df[self.traj_columns[i]].median()
            )

            # Calculate the median of these absolute deviations
            mad[i] = self.df[f"dev_{i}"].median()

        return mad

    def calculate_discrete_hausdorff(self) -> dict:
        """
        Calculates the discrete Hausdorff distance for each pair of specified columns.

        Returns:
            dict: A dictionary with keys as column pairs and values as the corresponding Hausdorff distance.
        """
        hausdorff = {}
        t = list(self.df.index)

        for i in self.mm_columns.keys():
            mm_data = self.df[self.mm_columns[i]].to_list()
            sim_data = self.df[self.traj_columns[i]].to_list()

            u = []
            v = []
            for idx in range(len(t)):
                u.append([t[idx], mm_data[idx]])
                v.append([t[idx], sim_data[idx]])

            hausdorff[i] = max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0])

        return hausdorff

    def calculate_pearson_correlation(self) -> dict:
        """
        Calculates the Pearson correlation coefficient for each pair of specified columns.

        Returns:
            dict: A dictionary with keys as column pairs and values as the corresponding Pearson correlation coefficient.
        """
        pearson_correlation = {}
        for i in self.mm_columns.keys():
            mm_col = self.mm_columns[i]
            sim_col = self.traj_columns[i]
            correlation = self.df[mm_col].corr(self.df[sim_col])
            pearson_correlation[i] = correlation

        return pearson_correlation

    def calculate_spearman_correlation(self) -> dict:
        """
        Calculates the Spearman's rank correlation coefficient for each pair of specified columns.

        Returns:
            dict: A dictionary with keys as column pairs and values as the corresponding Spearman correlation coefficient.
                  If the p-value is not significant (< 0.05), the value is set to -1.1.
        """
        spearman_correlation = {}
        for i in self.mm_columns.keys():
            mm_col = self.mm_columns[i]
            sim_col = self.traj_columns[i]

            rho, p_value = spearmanr(self.df[mm_col], self.df[sim_col])

            if p_value < 0.05:
                spearman_correlation[i] = rho

            else:
                spearman_correlation[i] = -1.1
        return spearman_correlation

    def calculate_dynamic_time_warping(self) -> dict:
        dwt = {}
        for i in self.mm_columns.keys():
            dwt[i] = dtw.distance_fast(
                self.df[self.mm_columns[i]].to_numpy(),
                self.df[self.traj_columns[i]].to_numpy(),
            )

        return dwt

    def calculate_discrete_frechet_distance(self) -> dict:
        dfd = {}
        for i in self.mm_columns.keys():
            # Correctly resetting the index and naming it 't'
            df_reset = self.df.reset_index()
            df_reset.rename(columns={"index": "t"}, inplace=True)

            # Now, let's extract P and Q correctly
            sim = df_reset[["t", self.traj_columns[i]]].values.tolist()
            mm = df_reset[["t", self.mm_columns[i]]].values.tolist()

            dfd[i] = similaritymeasures.frechet_dist(
                sim,
                mm,
            )

        return dfd


class DomainKnowledgeClassifier:
    """
    A class for motion trajectory-based classification using domain knowledge.

    This class performs classification by comparing a target object's trajectory with the predicted trajectories from existing mathematical or simulation models.
    Based on the comparison error, it extracts class probabilities.

    Attributes:
        mm_input (ParameterEstimation): The input for the mathematical model.
        similarity_parameter (Union[str, int]): The parameter used to measure similarity.
        mm_model_folder (str): The directory containing the mathematical models.
        models (List[ClassName]): List of class names (e.g., PLASTIC, FISH) representing different models.
        traj_columns (dict): Mapping of trajectory data column names.
        mm_columns (dict): Mapping of mathematical model data column names.
        similarities (List[dict]): List to store similarity calculations for each model.
        param_estm (List[dict]): List to store parameter estimations for each model.
        param_bounds (dict): The bounds for the input parameters of the models.

    Methods:
        _data_present: Checks if the data is already present in the mathematical model runs.
        _calculate_mathematical_model: Calculates the output of the mathematical model for the given input.
        _visualize: Visualizes the trajectory and mathematical model data.
        _parse_similarity_to_logits: Converts similarity values to logits.
        _parse_similarity: Processes similarity values based on the specified similarity parameter.
        _normalize: Normalizes a value between a specified range.
        _assess_trajectories_similarity: Assesses the similarity between trajectory data and mathematical model outputs.
        get_teacher_output: Generates output that imitates a teacher network based on batched trajectory data.
    """

    def __init__(
        self,
        mm_input: ParameterEstimation,
        param_bounds: dict,
        mm_model_folder: str,
        similarity_parameter: Union[str, int],
    ):
        """
        Initializes the DomainKnowledgeClassifier with necessary settings and configurations.

        Parameters:
            mm_input (ParameterEstimation): Input for the mathematical model.
            param_bounds (Dict): Dictionary specifying the bounds for input parameters.
            mm_model_folder (str): Folder containing the mathematical models.
            similarity_parameter (Union[str, int]): Parameter used for similarity measurement.
        """
        self.mm_input = mm_input
        self.similarity_parameter = similarity_parameter

        # Different mathematical models
        self.mm_model_folder = mm_model_folder
        self.models = [ClassName.PLASTIC, ClassName.FISH]
        self.traj_columns = {
            "y": "sim_y",
            "z": "sim_z",
            "vy": "sim_vy",
            "vz": "sim_vz",
        }

        self.mm_columns = {
            "y": "mm_y",
            "z": "mm_z",
            "vy": "mm_vy",
            "vz": "mm_vz",
        }

        self.similarities = []
        self.param_estm = []

        self.param_bounds = param_bounds

    def _data_present(
        self,
        data_num: int,
        mm_runs: List[str],
        data_path: str,
        data_variant: DataVariant,
    ) -> Tuple[bool, Dict]:
        """
        Checks if the data for a given data number is already present in the mathematical model runs.

        Parameters:
            data_num (int): The data number to check.
            mm_runs (List[str]): A list of existing mathematical model runs.
            data_path (str): The path to the data file.
            data_variant (DataVariant): The type of neural network input.

        Returns:
            Tuple[bool, Dict]: A tuple where the first element is a boolean indicating if the data is present, and the second element is the data dictionary if present.
        """

        if data_num in mm_runs:
            with open(data_path, "r") as f:
                data = json.load(f)
            extr_mm_input = data["parameter_estimation"]
            extr_network_input = data["neural_network_data_input"]

            if (
                extr_mm_input == self.mm_input.value
                and extr_network_input == data_variant.value
            ):
                return True, data
            else:
                return False, None
        else:
            return False, None

    def _calculate_mathematical_model(
        self,
        input_parameters: Dict,
        model_type: ClassName,
        input_traj: pd.DataFrame,
        data_number: int,
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Calculates the output of the mathematical model for given input parameters and model type.

        Parameters:
            input_parameters (Dict): The parameters for the mathematical model.
            model_type (ClassName): The type of model to use.
            input_traj (pd.DataFrame): The input trajectory data.
            data_number (int): The data number for identification.

        Returns:
            Tuple[pd.DataFrame, Dict]: A tuple containing the mathematical model output as a DataFrame and parameter estimation as a dictionary.
        """
        saved_mm_runs = os.listdir(self.mm_model_folder)

        parsed_mm_runs = [run[:-5] for run in saved_mm_runs]
        data_number = f"{data_number}_{model_type.value}"
        data_path = os.path.join(self.mm_model_folder, data_number + ".json")
        data_variant = input_parameters["neural_network_input"]
        data_present, data_dict = self._data_present(
            data_number, parsed_mm_runs, data_path, data_variant
        )

        # If the data number is already in the data, return it
        if data_present:
            param_estm = data_dict["parameter_estimation"]

            output = pd.DataFrame(data_dict["mathematical_model"])

        else:
            # Calculate the output (assumed to be a DataFrame)
            mathematical_model = MathematicalModel(
                input_parameters["config"],
                input_parameters["init_states"],
                tend=input_parameters["config"]["tend"],
                model_type=model_type,
                sim_traj=input_traj,
                input_type=self.mm_input,
                param_bounds=self.param_bounds[model_type],
            )

            output = mathematical_model.solve()

            # Convert the DataFrame to a JSON format
            output_dict = output.to_dict()

            if self.mm_input != ParameterEstimation.OFF:
                param_estm = mathematical_model.parameter_estimation_dict
            else:
                param_estm = {}

            data_dict = {}
            data_dict["mathematical_model"] = output_dict
            data_dict["parameter_estimation"] = param_estm
            data_dict["parameter_estimation"] = self.mm_input.value
            data_dict["neural_network_data_input"] = data_variant.value

            # Save updated master JSON data
            with open(data_path, "w") as f:
                json.dump(data_dict, f, indent=2)

        return (output, param_estm)

    def _visualize(
        self,
        mm_df: pd.DataFrame,
        trajectory: Dict,
        frame_time: List[float],
        label: str,
        model_type: ClassName,
    ) -> None:
        """
        Visualizes the trajectory and mathematical model data.

        Parameters:
            mm_df (pd.DataFrame): The DataFrame containing the mathematical model's data.
            trajectory (Dict): The trajectory data.
            frame_time (List[float]): The frame times corresponding to the trajectory.
            label (str): The label of the model.
            model_type (ClassName): The type of model used.
        """

        trajectory_keys = list(trajectory.keys())

        # Create a subplot for each key in trajectory_keys
        fig, axes = plt.subplots(
            nrows=len(trajectory_keys), figsize=(8, 4 * len(trajectory_keys))
        )

        # If there's only one subplot, axes is not a list, so we wrap it in a list
        if len(trajectory_keys) == 1:
            axes = [axes]

        for ax, i in zip(axes, trajectory_keys):
            ax.plot(frame_time, trajectory[i], label=f"trajectory_{i}")
            ax.plot(mm_df["t"], mm_df[i], label=f"df_{i}")
            ax.legend()
            ax.set_title(
                f"Mathematical model: {model_type}, unity model: {label}, traj_type: {i}"
            )

        plt.tight_layout(pad=4.0)
        plt.show()

    def _parse_similarity_to_logits(self, similarity_dict: Dict) -> torch.Tensor:
        """
        Converts similarity values to logits, imitating a teacher network.

        Parameters:
            similarity_dict (Dict): The dictionary containing similarity values.

        Returns:
            torch.Tensor: The tensor of logits.
        """

        logits = []

        for model_type in self.models:
            similarity_value = similarity_dict[model_type.value]["similarity_value"]

            # The assumption is made that the mse is correlated to the confidence
            confidence = 1 - similarity_value

            # Converts the confidence into a logit such that it imitates a teacher network
            logit = math.log(confidence / (1 - confidence))

            logits.append(logit)

        return logits

    def _parse_similarity(self, similarity_dict: Dict) -> torch.Tensor:
        """
        Processes similarity values based on the specified similarity parameter.

        Parameters:
            similarity_dict (Dict): The dictionary containing similarity values.

        Returns:
            torch.Tensor: A tensor containing processed similarity outputs.
        """
        outputs = []

        for model_type in self.models:
            similarity_value = similarity_dict[model_type.value][
                self.similarity_parameter.value
            ]
            if isinstance(self.similarity_parameter, Similarity):
                simil = similarity_value
            elif isinstance(self.similarity_parameter, InvertedSimilarity):
                simil = 1 / similarity_value

            outputs.append(simil)

        return outputs

    def _parse_similarities(self, similarity_dict: Dict) -> Dict:
        """
        Processes similarity values and organizes them into a structured dictionary.
        The output dictionary maps model types to a list of dictionaries, where each dictionary
        represents a similarity type and its computed value.

        Parameters:
            similarity_dict (Dict): A dictionary containing similarity values for various models.

        Returns:
            Dict: A structured dictionary organizing similarity values by model and type.
        """
        # Initialize the result dictionary with keys but empty list of lists as values
        result_dict = {key: [] for key in similarity_dict[next(iter(similarity_dict))]}
        class_names = list(similarity_dict.keys())

        # Iterate through each similarity type
        for similarity_type in result_dict.keys():
            for class_name in class_names:
                # If the list for this similarity type is not initialized, do it now
                if len(result_dict[similarity_type]) < len(class_names):
                    result_dict[similarity_type].append([])

                # Get the raw value
                raw_value = similarity_dict[class_name][similarity_type]
                # Calculate adjusted value
                adjusted_value = self._calculate_similarity_value(
                    raw_value, similarity_type
                )

                # Append the adjusted value to the correct class list
                result_dict[similarity_type][class_names.index(class_name)].append(
                    adjusted_value
                )

        return result_dict

    def _calculate_similarity_value(self, value, similarity_parameter):
        """
        Calculates the similarity value based on the current similarity parameter setting.
        """

        def is_value_in_enum(value, enum):
            return value in (e.value for e in enum)

        if is_value_in_enum(similarity_parameter, Similarity):
            return value
        elif is_value_in_enum(similarity_parameter, InvertedSimilarity):
            return 1 / value
        else:
            raise ValueError(f"{similarity_parameter} unknown")

    def _normalize(self, value: float, min_value: float, max_value: float) -> float:
        """
        Normalizes a value to be between 0 and 1, based on specified minimum and maximum values.

        Parameters:
            value (float): The value to normalize.
            min_value (float): The minimum value for normalization.
            max_value (float): The maximum value for normalization.

        Returns:
            float: The normalized value.
        """
        return (value - min_value) / (max_value - min_value)

    def _assess_trajectories_similarity(self, df: pd.DataFrame) -> Dict:
        """
        Assesses the similarity between trajectory data and mathematical model outputs.

        Parameters:
            df (pd.DataFrame): The DataFrame containing trajectory and model data.

        Returns:
            Dict: A dictionary containing calculated similarities and other metrics.
        """

        def calculate_weighted_metric(metric):
            tot = (
                metric["y"] * 0.15
                + metric["vy"] * 0.15
                + metric["z"] * 0.35
                + metric["vz"] * 0.35
            )
            return tot

        metrics = Metrics(df, self.mm_columns, self.traj_columns)
        mse = metrics.calculate_mse()
        # hausdorff = metrics.calculate_discrete_hausdorff()
        # pearson_correlation = metrics.calculate_pearson_correlation()
        # spearman_correlation = metrics.calculate_spearman_correlation()
        # mad = metrics.calculate_mad()
        # dynamic_time_warping = metrics.calculate_dynamic_time_warping()
        # discret_frechet_distance = metrics.calculate_discrete_frechet_distance()
        # variance = metrics.calculate_variance()

        weighted_mse = calculate_weighted_metric(mse)
        # weighted_variance = calculate_weighted_metric(variance)
        # weighted_mad = calculate_weighted_metric(mad)
        # weighted_hausdorff = calculate_weighted_metric(hausdorff)
        # weighted_pearson = calculate_weighted_metric(pearson_correlation)
        # weighted_spearman = calculate_weighted_metric(spearman_correlation)
        # weighted_dtw = calculate_weighted_metric(dynamic_time_warping)
        # weighted_dfd = calculate_weighted_metric(discret_frechet_distance)

        return {
            "weighted_mse": weighted_mse,
            # "weighted_mad": weighted_mad,
            # "weighted_hausdorff": weighted_hausdorff,
            # "weighted_pearson_correlation": weighted_pearson,
            # "weighted_spearman_correlation": weighted_spearman,
            # "weighted_dtw": weighted_dtw,
            # "weighted_dfd": weighted_dfd,
            # "weighted_variance": weighted_variance,
        }

    def _merge_similarity_dicts(self, lst_of_dicts):
        # Initialize a merged dictionary with keys and empty lists
        merged_dict = {key: [] for key in lst_of_dicts[0]}

        for current_dict in lst_of_dicts:
            for key, pair_lists in current_dict.items():
                # For each similarity, extend the list with zipped pairs from the current iteration
                # Unpack the zipped pairs and convert them to lists before appending
                for pair in zip(*pair_lists):
                    merged_dict[key].append(list(pair))

        # The merged_dict now has the desired structure
        return merged_dict

    def get_teacher_output(
        self,
        batched_traj: List[Dict],
        batched_settings: List[Dict],
        input_trajects: List[Dict],
        labels: List[str],
        data_numbers: List[float],
    ) -> torch.Tensor:
        """
        Generates output that imitates a teacher network based on batched trajectory data.

        Parameters:
            batched_traj (List[Dict]): List of batched trajectory data.
            batched_settings (List[Dict]): List of batched settings data.
            input_trajects (List[Dict]): List of input trajectories.
            labels (List[str]): List of labels for the data.
            data_numbers (List[float]): List of data numbers.

        Returns:
            torch.Tensor: The tensor representing the teacher network's output.
        """

        batched_similarity = []
        batched_similarities = []
        self.df_list = []
        # Loop through each item in the batch
        for trajectory, settings, input_traj, label, data_number in zip(
            batched_traj, batched_settings, input_trajects, labels, data_numbers
        ):
            similarity_dict = {}
            param_estm_dict = {}
            df_dict = {}

            for model_type in self.models:
                # Calculate the mathematical model
                mm_model, param_estm = self._calculate_mathematical_model(
                    settings, model_type, input_traj, data_number
                )

                # Convert the trajectories to a Dataframe
                df = pd.DataFrame(trajectory)
                data_variant = settings["neural_network_input"]
                if data_variant == DataVariant.OPTICAL_FLOW:
                    original_time = settings["config"]["frameTime"]
                    df["t"] = get_midpoints(get_midpoints(original_time))

                else:
                    df["t"] = settings["config"]["frameTime"]

                # Merge the two dataframes together and reindex the timestamp
                merged_df = sync_timestamps(
                    df1=df,
                    new_col1=self.traj_columns,
                    df2=mm_model,
                    new_col2=self.mm_columns,
                )

                # Calculate the similarity
                similarity = self._assess_trajectories_similarity(merged_df)

                similarity_dict[model_type.value] = similarity
                param_estm_dict[model_type.value] = param_estm

                df_dict[model_type] = merged_df

            relative_similarities = self._parse_similarity(similarity_dict)
            parsed_similarities = self._parse_similarities(similarity_dict)

            batched_similarity.append(relative_similarities)
            batched_similarities.append(parsed_similarities)

            self.similarities.append(similarity_dict)
            self.param_estm.append(param_estm_dict)

            self.df_list.append(df_dict)

        self.merged_similarities = self._merge_similarity_dicts(batched_similarities)

        return torch.tensor(batched_similarity, dtype=torch.float64)
