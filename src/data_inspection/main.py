import json
import pandas as pd

from src.data_inspection.plotting import Plotting
from src.domain_knowledge.mathematical_model import (
    MathematicalModel,
    ParameterEstimation,
)
from src.tools.enums import DataVariant, DictKeys, ClassName
from src.tools.general_functions import merge_lists, get_midpoints


class DataInspection:
    """
    A class designed for inspecting trajectories in a structured format.

    This class facilitates the visualization and analysis of trajectory data by generating structured dictionaries
    suitable for plotting. It supports various types of data including ground truth, optical flow data,
    Gaussian noise data, and mathematical model simulations.

    Attributes:
        data (dict): Data loaded from the specified path.
        subplot_names (list): Names of the subplots to be used in plotting.
        model_input (list): Parameters for the mathematical model simulations.
        param_bounds (dict): Bounds for the parameters of the mathematical model.
        plotting_dict (dict): Dictionary containing the structured data for plotting.

    Methods:
        generate_plotting_dict: Creates structured dictionaries for plotting.
        math_model_data_prep: Prepares the mathematical model data for plotting.
        combine_dicts: Combines multiple dictionaries into a single dictionary for plotting.
        data_prep: Prepares data for plotting by structuring it into a dictionary.

    Parameters:
        path_to_parsed_data (str): Path to the parsed_data.json file.
        mathematical_model_parameter_estimation (list, optional): Parameters for estimating the mathematical model.
        param_bounds (dict, optional): Bounds for the parameters of the mathematical model.
    """

    def __init__(
        self,
        path_to_parsed_data: str,
        mathematical_model_parameter_estimation: list = None,
        param_bounds: dict = None,
    ) -> None:
        """
        Initializes the DataInspection class and generates data for plotting.

        Parameters:
            path_to_parsed_data (str): Path to the parsed_data.json file.
            mathematical_model_parameter_estimation (list, optional): Parameters for estimating the mathematical model.
            param_bounds (dict, optional): Bounds for the parameters of the mathematical model.
        """
        with open(path_to_parsed_data, "r") as f:
            self.data = json.load(f)

        self.subplot_names = ["y", "vy", "z", "vz"]
        self.model_input = mathematical_model_parameter_estimation
        self.param_bounds = param_bounds

        self.plotting_dict = self.generate_plotting_dict()

    def generate_plotting_dict(self) -> dict:
        """
        Generates a dictionary for plotting the data in a structured manner.

        This method processes various types of data (ground truth, optical flow, Gaussian noise, and mathematical model data)
        and organizes them into a dictionary format suitable for plotting.

        Returns:
            dict: A dictionary containing the structured data for each run, ready for plotting.
        """
        plotting_dict = {}

        for run_name in self.data.keys():
            run_data = self.data[run_name]

            frame_time = run_data[DictKeys.UNITY_ANNOTATIONS.value]["frameTime"]
            processed_data = run_data[DictKeys.PROCESSED_DATA.value]

            # Prepare the data for the ground truth
            gt_data = self.data_prep(
                pos_data=run_data[DictKeys.UNITY_ANNOTATIONS.value][
                    DictKeys.UNITY_POSITIONS.value
                ],
                vel_data=run_data[DictKeys.UNITY_ANNOTATIONS.value][
                    DictKeys.UNITY_VELOCITIES.value
                ],
                frame_time=frame_time,
                col="blue",
                type_name="ground_truth (sim data)",
            )

            if processed_data[DataVariant.OPTICAL_FLOW.value] != {}:
                optical_flow_data = self.data_prep(
                    pos_data=processed_data[DataVariant.OPTICAL_FLOW.value][
                        DictKeys.OPTICAL_FLOW_POSITIONS.value
                    ],
                    vel_data=processed_data[DataVariant.OPTICAL_FLOW.value][
                        DictKeys.OPTICAL_FLOW_VELOCITIES.value
                    ],
                    frame_time=frame_time,
                    col="red",
                    type_name="optical_flow",
                )
            else:
                optical_flow_data = {}

            if processed_data[DataVariant.GAUSSIAN_NOISE.value] != {}:
                noisy_data = self.data_prep(
                    pos_data=processed_data[DataVariant.GAUSSIAN_NOISE.value][
                        DictKeys.NOISY_DATA_POSITIONS.value
                    ],
                    vel_data=processed_data[DataVariant.GAUSSIAN_NOISE.value][
                        DictKeys.NOISY_DATA_VELOCITIES.value
                    ],
                    frame_time=frame_time,
                    col="orange",
                    type_name="gaussian_noise",
                )
            else:
                noisy_data = {}

            if self.model_input != None:
                mathematical_model_data = []
                for idx, mathematical_run in enumerate(self.model_input):
                    mm_data = self.math_model_data_prep(
                        unity_settings=run_data[DictKeys.UNITY_SETTINGS.value],
                        unity_annotations=run_data[DictKeys.UNITY_ANNOTATIONS.value],
                        processed_data=processed_data,
                        col="green",
                        type_name=f"mm_model_{idx}",
                        parameter_input=mathematical_run,
                    )
                    mathematical_model_data.append(mm_data)

            else:
                mathematical_model_data = [{}]

            dicts = [gt_data, optical_flow_data, noisy_data]
            dicts += mathematical_model_data
            combined_dict = self.combine_dicts(dicts)

            plotting_dict[run_name] = combined_dict

            break

        return plotting_dict

    def math_model_data_prep(
        self,
        processed_data: dict,
        unity_settings: dict,
        unity_annotations: dict,
        col: str,
        type_name: str,
        parameter_input: DataVariant,
    ) -> dict:
        """
        Prepares the mathematical model data for plotting.

        This method extracts and processes data from the given inputs and structures it in a format suitable for plotting.
        It supports different types of input data, including optical flow data and Gaussian noise data.

        Parameters:
            processed_data (dict): The processed data dictionary.
            unity_settings (dict): Unity settings for the specific data point.
            unity_annotations (dict): Unity annotations for the specific data point.
            col (str): Color to be used for plotting.
            type_name (str): A label for the type of data, used in the graph.
            parameter_input (DataVariant): Specifies the type of parameter input for the mathematical model.

        Returns:
            dict: A dictionary containing processed and structured data for plotting.
        """
        dictie = {}
        for subplot_name in self.subplot_names:
            dictie[subplot_name] = []

        unity_data = UnityData(unity_annotations)
        start_pos = unity_data.get_start_position()
        duration = unity_data.get_duration()
        sim_traj = unity_data.df

        if parameter_input == DataVariant.OPTICAL_FLOW:
            optical_flow_data = processed_data[DataVariant.OPTICAL_FLOW.value]
            dictie = merge_lists(
                optical_flow_data[DictKeys.OPTICAL_FLOW_POSITIONS.value],
                optical_flow_data[DictKeys.OPTICAL_FLOW_VELOCITIES.value],
            )
            pos_times = get_midpoints(unity_annotations["frameTime"])
            vel_times = get_midpoints(pos_times)

            df_pos = pd.DataFrame({"y": dictie["y"], "z": dictie["z"], "t": pos_times})
            df_vel = pd.DataFrame(
                {"vy": dictie["vy"], "vz": dictie["vz"], "t": vel_times}
            )
            # Set 't' as the index for both DataFrames
            df_pos.set_index("t", inplace=True)
            df_vel.set_index("t", inplace=True)

            # Merge the two dataframes on the 't' index
            df_merged = df_pos.merge(
                df_vel, left_index=True, right_index=True, how="outer"
            )

            # First interpolate forward
            df_merged.interpolate(method="linear", inplace=True)

            # Then interpolate backward for any remaining NaNs
            df_merged.interpolate(
                method="linear", limit_direction="backward", inplace=True
            )
            sim_traj = df_merged
            mm_input = ParameterEstimation.PARAMETER_ESTIMATION

        elif parameter_input == DataVariant.GAUSSIAN_NOISE:
            noise_data = processed_data[DataVariant.GAUSSIAN_NOISE.value]
            dictie = merge_lists(
                noise_data[DictKeys.NOISY_DATA_POSITIONS.value],
                noise_data[DictKeys.NOISY_DATA_VELOCITIES.value],
            )
            t = unity_annotations["frameTime"]
            df = pd.DataFrame(
                {
                    "y": dictie["y"],
                    "z": dictie["z"],
                    "vy": dictie["vy"],
                    "vz": dictie["vz"],
                    "t": t,
                }
            )

            df.set_index("t", inplace=True)
            sim_traj = df
            mm_input = ParameterEstimation.PARAMETER_ESTIMATION

        elif parameter_input == None:
            mm_input = ParameterEstimation.OFF

        else:
            raise ValueError(
                f"The parameter input is not developed in this code yet: {parameter_input}"
            )

        if unity_settings["objectType"] == "Fish":
            model_type = ClassName.FISH
        else:
            model_type = ClassName.PLASTIC

        mathematical_model = MathematicalModel(
            config=unity_settings,
            init_states=start_pos,
            model_type=model_type,
            tend=duration,
            sim_traj=sim_traj,
            input_type=mm_input,
            param_bounds=self.param_bounds,
        )

        solutions_df = mathematical_model.solve()

        dictie["y"] = list(solutions_df["y"])
        dictie["z"] = list(solutions_df["z"])

        dictie["vy"] = list(solutions_df["vy"])
        dictie["vz"] = list(solutions_df["vz"])

        output_dict = {}
        for subplot_name in self.subplot_names:
            time_data = list(solutions_df["t"])

            output_dict[subplot_name] = {
                "type": type_name,
                "x-axis": time_data,
                "y-axis": dictie[subplot_name],
                "colour": col,
            }
        return output_dict

    def combine_dicts(self, data_list: list) -> dict:
        """
        Combines multiple dictionaries into a single large dictionary for plotting.

        This method takes a list of dictionaries, each representing different data sets (e.g., ground truth, optical flow),
        and combines them into a single dictionary. This combined dictionary is structured for ease of plotting.

        Parameters:
            data_list (list): A list of dictionaries to be combined.

        Returns:
            dict: A combined dictionary with structured data for plotting.
        """
        # Initialize an empty dictionary with the required keys and empty lists
        combined = {}
        for subplot_name in self.subplot_names:
            combined[subplot_name] = []

        # Loop through each dictionary in the data list
        for d in data_list:
            # Loop through each key-value pair in the dictionary
            for key, value in d.items():
                # Append the value to the corresponding key in the combined dictionary
                combined[key].append(value)

        return combined

    def data_prep(
        self, pos_data: dict, vel_data: dict, frame_time: list, col: str, type_name: str
    ) -> dict:
        """
        Prepares data for plotting by structuring it into a dictionary format.

        This method takes position and velocity data along with frame times, and structures them into a dictionary
        format that is suitable for plotting. It supports different types of data, such as optical flow.

        Parameters:
            pos_data (dict): Position data for plotting.
            vel_data (dict): Velocity data for plotting.
            frame_time (list): Timestamps of the frames.
            col (str): Color to be used for plotting.
            type_name (str): A label for the type of data, used in the graph.

        Returns:
            dict: A dictionary containing structured data for plotting.
        """
        dictie = merge_lists(pos_data, vel_data)

        # Modify frame_time if optical_flow is in type_name
        if "optical_flow" in type_name:
            midpoints = get_midpoints(frame_time)
            frame_time_pos = midpoints
            frame_time_vel = get_midpoints(midpoints)
        else:
            frame_time_pos = frame_time
            frame_time_vel = frame_time

        output_dict = {}
        for subplot_name in self.subplot_names:
            if subplot_name in ["y", "z"]:
                time_data = frame_time_pos
            else:
                time_data = frame_time_vel

            output_dict[subplot_name] = {
                "type": type_name,
                "x-axis": time_data,
                "y-axis": dictie[subplot_name],
                "colour": col,
            }
        return output_dict


class UnityData:
    """
    A class to handle and manipulate data from a Unity application.

    This class is designed to process data typically received from a Unity application,
    such as position, velocity, and frame time information. It stores this data in a
    pandas DataFrame for convenient access and manipulation.

    Attributes:
        df (pd.DataFrame): A DataFrame containing combined position, velocity, and time data.

    Methods:
        __init__(data): Initializes the UnityData instance.
        _create_df_from_json(data): Converts the Unity data from JSON to a pandas DataFrame.
        get_start_position(): Retrieves the initial position and velocity from the data.
        get_duration(): Calculates the total duration from the frame time data.
    """

    def __init__(self, data: dict) -> None:
        """
        Initializes the UnityData instance with data provided in JSON format.

        Args:
            data (dict): A dictionary containing keys 'position', 'velocity', and 'frameTime'.
                         Each key has a list of dictionaries corresponding to the frame-wise data.

        Returns:
            None
        """
        self.df = self._create_df_from_json(data)

    def _create_df_from_json(self, data: dict) -> pd.DataFrame:
        """
        Converts the Unity data from JSON format to a pandas DataFrame.

        This internal method processes the JSON data to create a DataFrame where each column
        represents a different attribute (position, velocity, time), and each row represents
        a different frame.

        Args:
            data (dict): The Unity data in JSON format.

        Returns:
            pd.DataFrame: A DataFrame with the combined data of position, velocity, and time.
        """
        df_position = pd.DataFrame(data["position"])
        df_velocity = pd.DataFrame(data["velocity"])
        df_velocity.columns = ["vx", "vy", "vz"]
        df_time = pd.DataFrame(data["frameTime"], columns=["t"])
        combined_df = pd.concat([df_position, df_velocity, df_time], axis=1)
        return combined_df

    def get_start_position(self) -> dict:
        """
        Retrieves the starting position and velocity from the data.

        This method extracts the initial y and z positions, as well as the initial y and z
        velocities from the DataFrame.

        Returns:
            dict: A dictionary containing the initial y position ('y_0'), initial y velocity ('vy_0'),
                  initial z position ('z_0'), and initial z velocity ('vz_0').
        """
        return {
            "y_0": self.df["y"][0],
            "vy_0": self.df["vy"][0],
            "z_0": self.df["z"][0],
            "vz_0": self.df["vz"][0],
        }

    def get_duration(self) -> float:
        """
        Calculates the total duration of the frame data.

        This method finds the maximum value of the time column in the DataFrame,
        which represents the total duration of the data.

        Returns:
            float: The maximum time value, representing the total duration.
        """
        return self.df["t"].max()


if __name__ == "__main__":
    data_inspect = DataInspection(
        r"M:\underwater_simulator\footage\parsed_data.json",
        model_input=DataVariant.GAUSSIAN_NOISE,
    )

    Plotting(input_dict=data_inspect.plotting_dict)
