from src.tools.enums import (
    ClassName,
    InputDataPath,
    DataVariant,
    ParameterEstimation,
    InputDataPath,
    DictKeys,
)
import json
import pandas as pd
import re
import importlib
import os
import numpy as np


def process_unity_data(data_entry: dict) -> dict:
    """
    Processes Unity data from a data entry to extract position and velocity information.

    Parameters:
        data_entry (dict): A single data entry containing Unity annotations.

    Returns:
        dict: A dictionary with processed data for y, z, vy, and vz from Unity data.
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
    data_entry: dict, data_type: str, pos_name: str, vel_name: str
) -> dict:
    """
    Processes modified data (e.g., optical flow data, noisy data) from a data entry.

    Parameters:
        data_entry (dict): A single data entry containing processed data information.
        data_type (str): The type of data to process (e.g., optical flow, noisy data).
        pos_name (str): The key name for position data in the data entry.
        vel_name (str): The key name for velocity data in the data entry.

    Returns:
        dict: A dictionary with processed data for y, z, vy, and vz from the specified data type.
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


def parse_class_name(label_name: str) -> ClassName:
    """
    Parses a label name and returns the corresponding ClassName enum member.

    Parameters:
    - label_name (str): The label name to be parsed.

    Returns:
    - ClassName: An enum member of ClassName corresponding to the parsed label name.

    Raises:
    - ValueError: If the label_name does not match any known classes.

    Examples:
    - parse_class_name("Fish") returns ClassName.FISH
    - parse_class_name("Plastic Water Bottle") returns ClassName.PLASTIC
    """

    if (
        label_name == "pb"
        or label_name == "Plastic Water Bottle"
        or label_name == "Beer_Holder_Plastic"
        or label_name == "Plastic Water Bottle "
    ):
        return ClassName.PLASTIC

    elif label_name == "Fish" or label_name == "fish":
        return ClassName.FISH

    else:
        raise ValueError(f"Label assigning failed, input type: {label_name} unknown")


def parse_input_path_dependent_on_os(data_path):

    if os.name == "nt" and data_path == InputDataPath.CLUSTER_MERGED:
        return InputDataPath.MATHIEU_MERGED

    elif os.name == "posix" and data_path == InputDataPath.MATHIEU_MERGED:
        return InputDataPath.CLUSTER_MERGED

    else:
        return data_path


def load_config(config_name):
    """
    Dynamically loads a configuration module given the configuration name.
    """
    config_module = importlib.import_module(config_name)
    return config_module


def map_data_variant_to_param_estm(data_variant: DataVariant) -> ParameterEstimation:
    """
    Maps the data variant to the ideal parameter estimation setting such that the mathematical model has an accuracy of around 92%

    Parameters:
    - data_variant (DataVariant): Type of data variant

    Returns:
    - ParameterEstimaton: Most optimal parameter estimation setting

    """
    mapping = {
        DataVariant.IDEAL: ParameterEstimation.FULL,  # mathematical model acc =
        DataVariant.GAUSSIAN_NOISE: ParameterEstimation.FULL,  # mathematical model acc =
        DataVariant.OPTICAL_FLOW: ParameterEstimation.FULL,  # mathematical model acc =
    }

    return mapping[data_variant]


def merge_lists(pos_data: list, vel_data: list) -> dict:
    """
    Merges position and velocity data lists into a dictionary with separate keys.

    Parameters:
    - pos_data (list): A list of dictionaries containing position data.
    - vel_data (list): A list of dictionaries containing velocity data.

    Returns:
    - dict: A dictionary with keys 'y', 'z', 'vy', 'vz', where each key contains a list of respective values.

    Note:
    - The function assumes that 'pos_data' and 'vel_data' are lists of dictionaries with keys 'y', 'z', and 'vy', 'vz', respectively.
    """

    return {
        "y": [pos["y"] for pos in pos_data],
        "z": [pos["z"] for pos in pos_data],
        "vy": [vel["y"] for vel in vel_data],
        "vz": [vel["z"] for vel in vel_data],
    }


def sync_timestamps(
    df1: pd.DataFrame, df2: pd.DataFrame, new_col1: dict, new_col2: dict
) -> pd.DataFrame:
    """
    Synchronizes timestamps between two dataframes and merges them.

    Parameters:
    - df1 (pd.DataFrame): The first dataframe to be merged.
    - df2 (pd.DataFrame): The second dataframe to be merged.
    - new_col1 (dict): Dictionary mapping of column names for df1.
    - new_col2 (dict): Dictionary mapping of column names for df2.

    Returns:
    - pd.DataFrame: A merged dataframe with synchronized timestamps.

    Note:
    - The function assumes that both dataframes have a 't' column for timestamps.
    - The timestamps are synchronized by creating a common index, interpolating missing values, and then merging the dataframes.
    """

    def ensure_start_at_zero_and_interpolate(df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensures that the DataFrame starts at t=0 by adding a row if necessary,
        and then interpolates to fill NaN values, including backward interpolation
        for the newly added row at t=0.

        Parameters:
        - df (pd.DataFrame): The DataFrame to process.

        Returns:
        - pd.DataFrame: The processed DataFrame with no NaN values at t=0.
        """
        # Ensure the DataFrame starts at t=0
        if df["t"].iloc[0] > 0:
            new_row = pd.DataFrame({"t": [0]}, index=[0])  # Create a new row with t=0
            df = (
                pd.concat([new_row, df], ignore_index=True)
                .sort_values(by="t")
                .reset_index(drop=True)
            )

        # Interpolate to fill NaN values, assuming linear change between points
        df.interpolate(method="linear", inplace=True, limit_direction="both")

        # If any NaNs remain (e.g., at the start), use backward fill as a last resort
        df.bfill(inplace=True)

        return df

    df1 = ensure_start_at_zero_and_interpolate(df1)
    df2 = ensure_start_at_zero_and_interpolate(df2)

    df1 = df1.copy()
    df1.rename(
        columns=new_col1,
        inplace=True,
    )

    df2 = df2.copy()
    df2.rename(
        columns=new_col2,
        inplace=True,
    )

    # Set 't' as the index in both dataframes
    df1.set_index("t", inplace=True)
    df2.set_index("t", inplace=True)

    # Create a common index
    common_index = df1.index.union(df2.index)

    # Reindex and interpolate both dataframes to the common index
    df1 = df1.reindex(common_index).interpolate(method="linear")
    df2 = df2.reindex(common_index).interpolate(method="linear")

    # Merge the dataframes on the index
    merged_df = pd.merge(df1, df2, left_index=True, right_index=True)
    pd.set_option("display.max_rows", None)

    return merged_df


def get_midpoints(times: list) -> list:
    """
    Computes the midpoints between consecutive timestamps in a given list.

    This utility function calculates the midpoint between each pair of consecutive timestamps in the input list.
    It is particularly useful for situations where an intermediate time value is needed, such as when working with
    data at intervals or for smoothing time series data.

    Parameters:
        times (list): A list of timestamps. These timestamps should be numerical values representing time.

    Returns:
        list: A list of midpoints, each being the average of two consecutive timestamps from the input list.

    Example:
        >>> get_midpoints([1, 3, 5])
        [2, 4]
    """

    return [(times[i] + times[i + 1]) / 2 for i in range(len(times) - 1)]


def read_input_path(input_type: InputDataPath) -> str:
    """
    Read the input path from a JSON configuration file.

    Parameters:
        input_type (InputDataPath): The type of input data path.

    Returns:
        str: The input data path.
    """

    with open("./config.json", "r") as file:
        config = json.load(file)

    path = config[input_type.value]

    return path


def sort_filenames(filenames):
    def extract_number(filename):
        match = re.search(r"(\d+)_train_samples", filename)
        return int(match.group(1)) if match else 0

    return sorted(filenames, key=extract_number, reverse=True)
