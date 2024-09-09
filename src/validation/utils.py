import json
import pandas as pd


from src.tools.enums import DictKeys, ParameterEstimation
from src.domain_knowledge.mathematical_model import MathematicalModel
from src.tools.general_functions import parse_class_name, merge_lists, sync_timestamps
from src.domain_knowledge.domain_knowledge_classifier import Metrics


class Comparison:
    """
    Class for performing comparison between mathematical model output and ground truth data.

    Attributes:
        annotations (dict): Dictionary containing annotation data.
        settings (dict): Dictionary containing settings data.
        model_output (pd.DataFrame): DataFrame containing mathematical model output.
        ground_truth (pd.DataFrame): DataFrame containing ground truth data.
        merged_df (pd.DataFrame): Merged DataFrame of model output and ground truth data.
        accuracy (dict): Dictionary containing accuracy metrics for y, vy, z, and vz.

    Methods:
        _get_ground_truth_df(): Get the ground truth data as a DataFrame.
        _perform_mathematical_model_runs(): Perform mathematical model runs and return the output as a DataFrame.
    """

    def __init__(self, annotations: dict, settings: dict) -> None:
        """
        Initialize a Comparison object with annotation and settings data.

        Parameters:
            annotations (dict): Dictionary containing annotation data.
            settings (dict): Dictionary containing settings data.

        Returns:
            None
        """
        sim_col = {
            "y": "sim_y",
            "z": "sim_z",
            "vy": "sim_vy",
            "vz": "sim_vz",
        }

        mm_col = {
            "y": "mm_y",
            "z": "mm_z",
            "vy": "mm_vy",
            "vz": "mm_vz",
        }

        self.annotations = annotations
        self.settings = settings
        self.model_output = self._perform_mathematical_model_runs()
        self.ground_truth = self._get_ground_truth_df()
        self.merged_df = sync_timestamps(
            df1=self.ground_truth,
            new_col1=sim_col,
            df2=self.model_output,
            new_col2=mm_col,
        )

        metrics = Metrics(df=self.merged_df, mm_columns=mm_col, traj_columns=sim_col)
        self.accuracy = metrics.calculate_accuracy()
        if self.accuracy["y"] < 0 or self.accuracy["y"] > 1:
            print(f'y: {self.accuracy["y"]}')
        if self.accuracy["vy"] < 0 or self.accuracy["vy"] > 1:
            print(f'vy: {self.accuracy["vy"]}')
        if self.accuracy["z"] < 0 or self.accuracy["z"] > 1:
            print(f'z: {self.accuracy["z"]}')
        if self.accuracy["vz"] < 0 or self.accuracy["vz"] > 1:
            print(f'vz: {self.accuracy["vz"]}')

    def _get_ground_truth_df(self) -> pd.DataFrame:
        """
        Get the ground truth data as a DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing ground truth data.
        """
        dictie = merge_lists(
            pos_data=self.annotations[DictKeys.UNITY_POSITIONS.value],
            vel_data=self.annotations[DictKeys.UNITY_VELOCITIES.value],
        )
        t = self.annotations["frameTime"]
        df = pd.DataFrame(
            {
                "y": dictie["y"],
                "z": dictie["z"],
                "vy": dictie["vy"],
                "vz": dictie["vz"],
                "t": t,
            }
        )
        return df

    def _perform_mathematical_model_runs(self) -> pd.DataFrame:
        """
        Perform mathematical model runs and return the output as a DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing mathematical model output.
        """
        tend = self.annotations["frameTime"][-1]

        init_states = {
            "y_0": self.annotations[DictKeys.UNITY_POSITIONS.value][0]["y"],
            "z_0": self.annotations[DictKeys.UNITY_POSITIONS.value][0]["z"],
            "vy_0": self.annotations[DictKeys.UNITY_VELOCITIES.value][0]["y"],
            "vz_0": self.annotations[DictKeys.UNITY_VELOCITIES.value][0]["z"],
            "x_angular_0": self.annotations[DictKeys.UNITY_ANGLES.value][0]["z"],
            "vx_angular_0": self.annotations[DictKeys.UNITY_ANGULAR_VELOCITIES.value][
                0
            ]["x"],
        }

        mathematical_model = MathematicalModel(
            config=self.settings,
            init_states=init_states,
            model_type=parse_class_name(self.settings["objectType"]),
            tend=tend,
            input_type=ParameterEstimation.OFF,
        )
        output = mathematical_model.solve()

        return output


class Validation:
    """
    Class for performing validation on a set of data.

    Attributes:
        data (dict): Dictionary containing validation data.

    Methods:
        perform_validation(): Perform validation on the data.
        calculate_averages(dicts: list) -> dict: Calculate average values from a list of dictionaries.
    """

    def __init__(self, data_file_path: str) -> None:
        """
        Initialize a Validation object with a path to the data file.

        Parameters:
            data_file_path (str): Path to the data file containing validation data.

        Returns:
            None
        """
        self.data = self._extract_data(data_file_path)

    def _extract_data(self, filepath: str) -> dict:
        """
        Extract data from a JSON file.

        Parameters:
            filepath (str): Path to the JSON data file.

        Returns:
            dict: Dictionary containing the extracted data.
        """
        with open(filepath) as f:
            data = json.load(f)

        return data

    def perform_validation(self) -> None:
        """
        Perform validation on the data.

        Returns:
            None
        """
        keys = self.data.keys()

        accuracy_lst = []
        for key in keys:
            annotations = self.data[key][DictKeys.UNITY_ANNOTATIONS.value]
            settings = self.data[key][DictKeys.UNITY_SETTINGS.value]

            comp = Comparison(annotations, settings)
            accuracy_lst.append(comp.accuracy)

        print(self.calculate_averages(accuracy_lst))

    def calculate_averages(self, dicts: list) -> dict:
        """
        Calculate average values from a list of dictionaries.

        Parameters:
            dicts (list): List of dictionaries containing values to be averaged.

        Returns:
            dict: Dictionary containing the average values.
        """
        # Initialize sums to zero
        sum_y, sum_vy, sum_z, sum_vz = 0, 0, 0, 0
        n = len(dicts)  # Number of dictionaries

        # Iterate through each dictionary and accumulate the sums
        for d in dicts:
            sum_y += d.get("y", 0)
            sum_vy += d.get("vy", 0)
            sum_z += d.get("z", 0)
            sum_vz += d.get("vz", 0)

        # Calculate the averages
        avg_y = sum_y / n
        avg_vy = sum_vy / n
        avg_z = sum_z / n
        avg_vz = sum_vz / n

        avg = {"avg_y": avg_y, "avg_vy": avg_vy, "avg_z": avg_z, "avg_vz": avg_vz}

        return avg
