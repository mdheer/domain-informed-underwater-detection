import json
import pandas as pd
from sklearn.metrics import mean_squared_error
from typing import Optional, Dict, List, Tuple
import shutil
import sys
import random
import yaml
import os

from src.tools.enums import DictKeys, HomePath, OpticalFlowAccuracy, WaterCurrentFilter
from src.tools.general_functions import get_midpoints

# from src.data_preprocessing.data_curation import DatasetCurator


def calculate_mse(df1: pd.DataFrame, df2: pd.DataFrame) -> Dict[str, Optional[float]]:
    """
    Calculate the Mean Squared Error (MSE) between columns of two dataframes.

    Parameters:
        df1 (pd.DataFrame): The first dataframe.
        df2 (pd.DataFrame): The second dataframe.

    Returns:
        Dict[str, Optional[float]]: A dictionary with keys as column names ('y', 'vy', 'z', 'vz') and values as MSE.
                                    If a key is not found in both dataframes, its value will be None.
    """
    mse_results = {}
    keys = ["y", "vy", "z", "vz"]

    for key in keys:
        # Ensure the key exists in both dataframes
        if key in df1.columns and key in df2.columns:
            # Align the dataframes based on their index
            aligned_df1, aligned_df2 = df1.align(df2, join="inner", axis=0)
            # Calculate MSE for the current key
            mse_results[key] = mean_squared_error(aligned_df1[key], aligned_df2[key])
        else:
            mse_results[key] = (
                None  # or some indication that the key was not found in both dataframes
            )

    return mse_results


def resample(df: pd.DataFrame, t: List[float]) -> pd.DataFrame:
    """
    Resamples a dataframe at specified time points and interpolates missing data.

    Parameters:
        df (pd.DataFrame): The dataframe to be resampled.
        t (List[float]): The time points at which the dataframe should be resampled.

    Returns:
        pd.DataFrame: The resampled and interpolated dataframe.
    """

    if "t" in df.columns:
        ts = df["t"].tolist()
        df.set_index("t", inplace=True)

    else:
        ts = df.index.tolist()

    pd.set_option("display.max_rows", None)

    ts = [item for item in ts if item not in t]
    common_index = df.index.union(t)

    # Reindex and interpolate the dataframes
    df = df.reindex(common_index).interpolate(method="linear")

    # Then interpolate backward for any remaining NaNs
    df.interpolate(method="linear", limit_direction="backward", inplace=True)
    # Drop 'x' and 'vx' columns if they exist
    if "x" in df.columns:
        df = df.drop("x", axis=1)
    if "vx" in df.columns:
        df = df.drop("vx", axis=1)

    df_filtered = df.drop(ts)

    return df_filtered


def parse_unity_dict_to_df(data_dict: Dict, frame_times: List[float]) -> pd.DataFrame:
    """
    Parses a dictionary containing Unity data into a dataframe, adding frame times.

    Parameters:
        data_dict (Dict): The dictionary containing Unity data.
        frame_times (List[float]): The frame times to be added to the dataframe.

    Returns:
        pd.DataFrame: The dataframe created from the Unity data.

    Raises:
        ValueError: If the length of frame_times does not match the length of data in data_dict.
    """

    frame_time_len = len(frame_times)

    if all(frame_time_len == len(value) for value in data_dict.values()):
        data_dict["t"] = frame_times
        df = pd.DataFrame(data_dict)
        return df

    else:
        raise ValueError(
            f"The data_dict values do not correspond with frame_time_len. Data dict is {data_dict} and frame_time_len is {frame_time_len}"
        )


def parse_dict_to_df(
    dictie: Dict, pos_times: List[float], vel_times: List[float]
) -> pd.DataFrame:
    """
    Parses a dictionary containing position and velocity data into a merged dataframe.

    Parameters:
        dictie (Dict): The dictionary containing position and velocity data.
        pos_times (List[float]): The time points for position data.
        vel_times (List[float]): The time points for velocity data.

    Returns:
        pd.DataFrame: The merged dataframe with interpolated position and velocity data.
    """

    df_pos = pd.DataFrame({"y": dictie["y"], "z": dictie["z"], "t": pos_times})
    df_vel = pd.DataFrame({"vy": dictie["vy"], "vz": dictie["vz"], "t": vel_times})
    # Set 't' as the index for both DataFrames
    df_pos.set_index("t", inplace=True)
    df_vel.set_index("t", inplace=True)

    # Merge the two dataframes on the 't' index
    df_merged = df_pos.merge(df_vel, left_index=True, right_index=True, how="outer")

    # First interpolate forward
    df_merged.interpolate(method="linear", inplace=True)

    # Then interpolate backward for any remaining NaNs
    df_merged.interpolate(method="linear", limit_direction="backward", inplace=True)
    return df_merged


def merge_lists(pos_data: List[Dict], vel_data: List[Dict]) -> Dict[str, List[float]]:
    """
    Merges lists of position and velocity data into a single dictionary.

    Parameters:
        pos_data (List[Dict]): The list of position data dictionaries.
        vel_data (List[Dict]): The list of velocity data dictionaries.

    Returns:
        Dict[str, List[float]]: A dictionary with merged position and velocity data.
    """

    return {
        "y": [pos["y"] for pos in pos_data],
        "z": [pos["z"] for pos in pos_data],
        "vy": [vel["y"] for vel in vel_data],
        "vz": [vel["z"] for vel in vel_data],
    }


def process_unity_data(data_entry: Dict) -> Dict[str, List[float]]:
    """
    Processes Unity data from a given dictionary entry to extract position and velocity lists.

    Parameters:
        data_entry (Dict): The dictionary entry containing Unity data.

    Returns:
        Dict[str, List[float]]: A dictionary with lists of 'y', 'z', 'vy', and 'vz' data.
    """

    y_lst = []
    z_lst = []
    vy_lst = []
    vz_lst = []

    unity_annotations = data_entry[DictKeys.UNITY_ANNOTATIONS.value]

    for i, j in zip(
        unity_annotations[DictKeys.UNITY_POSITIONS.value],
        unity_annotations[DictKeys.UNITY_VELOCITIES.value],
    ):
        y_lst.append(i["y"])
        z_lst.append(i["z"])
        vy_lst.append(j["y"])
        vz_lst.append(j["z"])

    return {"y": y_lst, "z": z_lst, "vy": vy_lst, "vz": vz_lst}


def process_modified_data(
    data_entry: Dict, data_type: str, pos_name: str, vel_name: str
) -> Dict[str, List[float]]:
    """
    Processes modified data from a given dictionary entry based on specified data types and names.

    Parameters:
        data_entry (Dict): The dictionary entry containing the data.
        data_type (str): The type of data to be processed (e.g., optical flow data, noisy data).
        pos_name (str): The key name for position data within the data_entry.
        vel_name (str): The key name for velocity data within the data_entry.

    Returns:
        Dict[str, List[float]]: A dictionary with lists of 'y', 'z', 'vy', and 'vz' data.
    """

    y_lst = []
    z_lst = []
    vy_lst = []
    vz_lst = []

    processed_data = data_entry[DictKeys.PROCESSED_DATA.value][data_type]
    for i, j in zip(
        processed_data[pos_name],
        processed_data[vel_name],
    ):
        y_lst.append(i["y"])
        z_lst.append(i["z"])
        vy_lst.append(j["y"])
        vz_lst.append(j["z"])

    return {"y": y_lst, "z": z_lst, "vy": vy_lst, "vz": vz_lst}


def calc_mse_diff(mse1: Dict[str, float], mse2: Dict[str, float]) -> Dict[str, float]:
    """
    Calculates the difference in MSE between two dictionaries containing MSE values.

    Parameters:
        mse1 (Dict[str, float]): The first dictionary containing MSE values.
        mse2 (Dict[str, float]): The second dictionary containing MSE values.

    Returns:
        Dict[str, float]: A dictionary containing the MSE differences for each key.
    """

    comp = {}
    for i in mse1.keys():
        mse = ((mse1[i] - mse2[i]) ** 2) / mse1[i]

        comp[i] = mse

    return comp


def filter_optical_flow(
    data_entry: Dict,
    thresh_y: float,
    thresh_vy: float,
    thresh_z: float,
    thresh_vz: float,
) -> Optional[Dict]:
    """
    Filters optical flow data based on specified threshold values.

    Parameters:
        data_entry (Dict): The dictionary entry containing the data to be filtered.
        thresh_y (float): The threshold for 'y' value filtering.
        thresh_vy (float): The threshold for 'vy' value filtering.
        thresh_z (float): The threshold for 'z' value filtering.
        thresh_vz (float): The threshold for 'vz' value filtering.

    Returns:
        Optional[Dict]: A dictionary containing the filtered data, or None if the data does not meet the criteria.
    """

    if data_entry["processed_data"]["optical_flow_data"] == {}:
        return None

    frame_time = data_entry[DictKeys.UNITY_ANNOTATIONS.value]["frameTime"]

    # Define neural network inputs
    unity_dict = process_unity_data(data_entry)

    optical_flow_dict = process_modified_data(
        data_entry,
        DictKeys.OPTICAL_FLOW_DATA.value,
        DictKeys.OPTICAL_FLOW_POSITIONS.value,
        DictKeys.OPTICAL_FLOW_VELOCITIES.value,
    )

    noisy_dict = process_modified_data(
        data_entry,
        DictKeys.NOISY_DATA.value,
        DictKeys.NOISY_DATA_POSITIONS.value,
        DictKeys.NOISY_DATA_VELOCITIES.value,
    )

    pos_times = get_midpoints(
        get_midpoints(frame_time)
    )  # to account for the fact that the projection also decreased the number
    vel_times = get_midpoints(get_midpoints(pos_times))

    unity_df = parse_unity_dict_to_df(unity_dict, frame_time)

    optical_flow_df = parse_dict_to_df(optical_flow_dict, pos_times, pos_times)
    noisy_df = parse_dict_to_df(noisy_dict, frame_time, frame_time)

    unity_df_resampled = resample(unity_df, pos_times)

    # Assuming optical_flow_df and unity_df_resampled are your dataframes
    mse_results_opt = calculate_mse(optical_flow_df, unity_df_resampled)

    mse_results_noise = calculate_mse(noisy_df, unity_df)

    diff = calc_mse_diff(mse_results_noise, mse_results_opt)

    if (
        diff["y"] < thresh_y
        and diff["vy"] < thresh_vy
        and diff["z"] < thresh_z
        and diff["vz"] < thresh_vz
    ):
        return OpticalFlowAccuracy.HIGH

    else:
        return OpticalFlowAccuracy.LOW


def filter_water_current(data_entry: Dict) -> WaterCurrentFilter:
    """
    Filters the data based on the strength and direction of the current

    Paramter:
        data_entry (Dict): dictionary containing the information of a trajectory

    Return:
        Datafilter: the type the trajectory belongs to

    """
    water_current_strength = data_entry["unity_settings"]["waterCurrentStrength"][0][
        "z"
    ]
    if water_current_strength < 0 and abs(water_current_strength) < 0.1:
        return WaterCurrentFilter.LIGHT_CURRENT

    else:
        return WaterCurrentFilter.STRONG_CURRENT


def create_dirs(relative_path: str, name: str) -> Tuple[str, str]:
    """
    Creates directories for storing data and distribution information.

    This function creates a main directory and subdirectories based on the provided relative path and name. It also handles overwriting existing directories upon user confirmation.

    Parameters:
        relative_path (str): The relative path where the directory should be created.
        name (str): The name of the main directory to be created.

    Returns:
        Tuple[str, str]: A tuple containing the paths of the distribution directory and the main directory.
    """
    dir_path = os.path.join(relative_path, name)
    distribution_path = os.path.join(dir_path, "distributions")
    # Check if the directory exists
    if os.path.exists(dir_path):
        # Prompt the user for permission to overwrite
        if input("Overwrite folder? [y/n] ") == "y":
            # Remove the existing directory
            shutil.rmtree(dir_path)
            # Recreate the directory
            os.makedirs(dir_path)
            os.makedirs(distribution_path)
        else:
            sys.exit()
    else:
        # Create the directory if it doesn't exist
        os.makedirs(dir_path)
        os.makedirs(distribution_path)

    return distribution_path, dir_path


def decrease_num_samples(
    df: pd.DataFrame, distribution: Dict, step: int
) -> pd.DataFrame:
    """
    Decreases the number of samples in the DataFrame based on the provided distribution and step size.

    This function removes a certain number of rows from the DataFrame for each type specified in the distribution, based on the given step size.

    Parameters:
        df (pd.DataFrame): The DataFrame from which the samples will be removed.
        distribution (Dict): A dictionary specifying the percentages of each data type to be removed.
        step (int): The step size indicating the number of samples to be removed.

    Returns:
        pd.DataFrame: The modified DataFrame after sample removal.
    """

    # Calculate the number of rows to remove for each type based on the percentage and step
    rows_to_remove = {
        key: int(step * (perc / 100))
        for key, perc in distribution.items()
        if "percentage_" in key
    }
    rows_to_remove = adjust_rows_to_remove_randomly(rows_to_remove, step)

    # Remove rows for each type
    for key, count in rows_to_remove.items():
        # Extract the type from the key (e.g., 'percentage_ideal_data' -> 'ideal_data')
        if "percentage_" in key:
            names = df.columns.to_list()
            parsed_key = key.replace("percentage_", "", 1)

            for col in names:
                if parsed_key in col:
                    # Filter rows where the data_type column is True
                    type_df = df[df[parsed_key]]

                    # Randomly select 'count' number of rows to remove
                    rows_to_drop = type_df.sample(
                        n=min(count, len(type_df)), random_state=1
                    ).index

                    # Drop the selected rows from the main DataFrame
                    df = df.drop(index=rows_to_drop)
                    break  # exit loop, no need to check the other keys

    return df


def adjust_rows_to_remove_randomly(rows_to_remove: Dict, step: int) -> Dict:
    """
    Randomly adjusts the number of rows to be removed from each category to match the total step size.

    Parameters:
        rows_to_remove (Dict): A dictionary containing the initial number of rows to remove for each category.
        step (int): The total number of rows to be removed.

    Returns:
        Dict: The adjusted dictionary with the updated number of rows to remove for each category.
    """

    total_rows_to_remove = sum(rows_to_remove.values())
    remaining_rows = step - total_rows_to_remove
    keys = list(rows_to_remove.keys())

    available_keys = []
    for i in keys:
        if rows_to_remove[i] != 0:
            available_keys.append(i)

    while remaining_rows > 0:
        key = random.choice(available_keys)  # Randomly select a key
        rows_to_remove[key] += 1
        remaining_rows -= 1

    return rows_to_remove


def create_yolo_yaml_file(dir_path: str, name: str, home_path: HomePath) -> None:
    """
    Creates a YAML file for YOLO model training configuration.

    This function generates a YAML file with the configuration for YOLO training, including paths and class names.

    Parameters:
        dir_path (str): The directory path where the YAML file will be saved.
        name (str): The name of the YAML file.
        home_path (HomePath): The path to the home directory

    Returns:
        None: The function writes to a file and does not return any value.
    """

    class_names = ["plastic", "fish"]

    dataset = {
        "path": home_path.value,
        "train": "yolo_data/images/train",
        "val": "yolo_data/images/val",
        "test": "yolo_data/images/test",
        "nc": len(class_names),
        "names": class_names,
    }

    file_path = os.path.join(dir_path, f"{name}.yaml")

    with open(file_path, "w") as file:
        yaml.dump(dataset, file, default_flow_style=False)
