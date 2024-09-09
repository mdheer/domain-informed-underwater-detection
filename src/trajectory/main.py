import os
import sys
import json
from math import *
import numpy as np
import random
from typing import List, Optional, Tuple, Union, Callable

from ..trajectory.optical_flow import DenseOpticalFlow
from ..trajectory.trajectory_constructor import TrajectoryConstructor
from ..neural_network.main import DataVariant
from ..domain_knowledge.mathematical_model import ParameterEstimation


class PreprocessError(ValueError):
    pass


def sort_string_list_as_numbers(nums: List[str]) -> List[str]:
    """
    Sorts a list of string representations of numbers in ascending order.

    Parameters:
        nums (List[str]): List of string representations of numbers.

    Returns:
        List[str]: Sorted list of numbers as strings.
    """
    # Filter out non-integer strings and sort
    sorted_nums = sorted(
        [int(num) for num in nums if num.isdigit()], key=lambda x: int(x)
    )
    return [str(num) for num in sorted_nums]


def add_to_master_file(
    idx: int,
    folder_path: str,
    master_path: str,
    annotations: dict,
    settings: dict,
    processed_data: dict,
    overwrite: Union[DataVariant, None],
) -> None:
    """
    Adds new data to a master JSON file, updating existing entries if necessary.

    Parameters:
        idx (int): Identifier for the new data entry.
        folder_path (str): Path to the folder containing image files.
        master_path (str): Path to the master JSON file.
        annotations (dict): Dictionary containing annotation data.
        settings (dict): Dictionary containing settings data.
        processed_data (dict): Dictionary containing processed data.
        overwrite (DataVariant or None): Specifies if data should be overwritten for a specific DataVariant type.
    """

    # Custom sort function
    def custom_sort(filename):
        # Splitting on underscore and then on period to extract the number
        return int(filename.split("_")[1].split(".jpg")[0])

    def get_middle_bounding_box_coord(bb_coord):
        x_min = bb_coord["xMin"]
        x_max = bb_coord["xMax"]
        y_min = bb_coord["yMin"]
        y_max = bb_coord["yMax"]

        x_middle = (x_max + x_min) / 2
        y_middle = (y_max + y_min) / 2

        return {"x": x_middle, "y": y_middle}

    # Modify unity annotations to add middle bounding box
    twod_coordinates = []
    for coord in annotations["boundingBoxCoordinates"]:
        twod_coordinates.append(get_middle_bounding_box_coord(coord))

    annotations["2dCoordinates"] = twod_coordinates

    with open(master_path, "r") as f:
        master_data = json.load(f)

    data_key = f"data_{idx}"
    jpg_files = [f for f in os.listdir(folder_path) if f.endswith(".jpg")]

    # Sort using the custom sort function
    jpg_files.sort(key=custom_sort)

    # Merge existing data with new data
    if data_key in master_data:
        existing_processed_data = master_data[data_key]["processed_data"]
    else:
        existing_processed_data = {}

    # Check that optical flow data is added correctly
    merged_processed_data = {
        key: {} for key in {**processed_data, **existing_processed_data}
    }

    if type(overwrite) == DataVariant:
        overwrite_value = overwrite.type
    else:
        overwrite_value = None

    for key in merged_processed_data.keys():
        if (
            key in existing_processed_data.keys()
            and existing_processed_data[key] != {}
            and existing_processed_data[key] != None
            and key != overwrite_value
        ):
            merged_processed_data[key] = existing_processed_data[key]

        elif (
            key in processed_data.keys()
            and processed_data[key] != {}
            and processed_data[key] != None
        ):
            merged_processed_data[key] = processed_data[key]

    dict_to_write = {
        "file_paths": jpg_files,
        "unity_annotations": annotations,
        "unity_settings": settings,
        "processed_data": merged_processed_data,
    }
    master_data[data_key] = dict_to_write

    # Save updated master JSON data
    with open(master_path, "w") as f:
        json.dump(master_data, f, indent=2)


def projection(x_sim: float, y_pixel: float, z_pixel: float) -> dict:
    """
    Projects 2D pixel coordinates into 3D real-world coordinates.

    Parameters:
        x_sim (float): The x-coordinate in the simulation.
        y_pixel (float): The y-coordinate in pixel space.
        z_pixel (float): The z-coordinate in pixel space.

    Returns:
        dict: Dictionary containing the projected 'y' and 'z' real-world coordinates.
    """
    resolutionWidth = 1920
    resolutionHeight = 1080
    y_offset = 0.5  # due to the camera having a 0.5m offset in unity

    fov_vertical = 60  # degrees
    center_z = resolutionWidth / 2
    center_y = resolutionHeight / 2

    focal_length = resolutionHeight / (2 * tan(radians(fov_vertical / 2)))  # in pixels

    y_real_world = (((y_pixel - center_y) / focal_length) * x_sim - y_offset) * -1
    z_real_world = (((z_pixel - center_z) / focal_length) * x_sim) * -1

    return {"y": y_real_world, "z": z_real_world}


def apply_2d_to_3d_projections(
    annotations: dict, optical_flow: dict
) -> Tuple[List[dict], List[dict]]:
    """
    Applies 2D to 3D projections to positions and velocities in optical flow data.

    Parameters:
        annotations (dict): Dictionary containing annotations data.
        optical_flow (dict): Dictionary containing optical flow data.

    Returns:
        Tuple[List[dict], List[dict]]: Two lists containing projected locations and velocities.
    """
    projected_locations = []
    for x_position, optical_flow_pixel in zip(
        annotations["position"],
        optical_flow["position"],
    ):
        projected_locations.append(
            projection(
                x_sim=x_position["x"],
                y_pixel=optical_flow_pixel["y"],
                z_pixel=optical_flow_pixel["x"],
            )
        )

    projected_velocities = []

    for idx in range(len(annotations["frameTime"]) - 2):
        z1 = projected_locations[idx]["z"]
        z2 = projected_locations[idx + 1]["z"]
        y1 = projected_locations[idx]["y"]
        y2 = projected_locations[idx + 1]["y"]

        delta_t = annotations["frameTime"][idx + 1] - annotations["frameTime"][idx]

        vz = (z2 - z1) / delta_t
        vy = (y2 - y1) / delta_t

        projected_velocities.append({"z": vz, "y": vy})

    return projected_locations, projected_velocities


def gaussian_noise(value: float, mean: float = 0, std_dev: float = 0.4) -> float:
    """
    Adds Gaussian noise to a value.

    Parameters:
        value (float): The original value.
        mean (float, optional): Mean of the Gaussian distribution. Defaults to 0.
        std_dev (float, optional): Standard deviation of the Gaussian distribution. Defaults to 0.4.

    Returns:
        float: The value with added Gaussian noise.
    """
    return value + np.random.normal(mean, std_dev)


def jitter_noise(value: float, low: float = -0.5, high: float = 0.5) -> float:
    """
    Adds jitter noise to a value by randomly selecting a number within a specified range.

    Parameters:
        value (float): The original value.
        low (float, optional): Lower bound of the range. Defaults to -0.5.
        high (float, optional): Upper bound of the range. Defaults to 0.5.

    Returns:
        float: The value with added jitter noise.
    """
    return value + np.random.uniform(low, high)


def create_erroneous_data(gt: str, percentage: float = 0.2) -> str:
    """
    Randomly alters a given ground truth label to a different label based on a specified percentage.

    Parameters:
        gt (str): The original ground truth label.
        percentage (float, optional): Probability of changing the ground truth label. Defaults to 0.2 (20%).

    Returns:
        str: Either the original ground truth label or a randomly chosen different label.
    """

    classes = ["Plastic Water Bottle", "Fish", "Beer_Holder_Plastic"]

    if random.random() < percentage:
        alternative_classes = [cls for cls in classes if cls != gt]
        return random.choice(alternative_classes)
    else:
        return gt


def create_noisy_data(
    annotations_data: dict, noise_type: Callable[[float], float]
) -> dict:
    """
    Applies a specified noise function to position and velocity data in annotations.

    Parameters:
        annotations_data (dict): Dictionary containing 'position' and 'velocity' data.
        noise_type (Callable[[float], float]): Function to apply noise to the data.

    Returns:
        dict: Dictionary containing noisy 'position' and 'velocity' data.
    """

    noisy_data = {}
    noisy_data["position"] = []
    noisy_data["velocity"] = []

    for pos, vel in zip(annotations_data["position"], annotations_data["velocity"]):
        noisy_pos = {key: noise_type(value) for key, value in pos.items()}
        noisy_vel = {key: noise_type(value) for key, value in vel.items()}

        noisy_data["position"].append(noisy_pos)
        noisy_data["velocity"].append(noisy_vel)

    return noisy_data


def identify_missing_versions(
    data: dict, input_traj: List[DataVariant]
) -> Tuple[List[DataVariant], bool]:
    """
    Identifies the types of processed data not present in the existing dataset.

    Parameters:
        data (dict): Dictionary containing existing processed data.
        input_traj (List[DataVariant]): List of DataVariant enums to check against the data.

    Returns:
        Tuple[List[DataVariant], bool]: A tuple containing a list of missing DataVariant types and a boolean indicating the presence of erroneous labels.
    """

    data_names = list(data.keys())
    present_processed_data = data[data_names[0]]["processed_data"]
    present_processed_data_keys = present_processed_data.keys()

    present_traj = set()
    for i in present_processed_data_keys:
        if (present_processed_data[i] != {}) and (present_processed_data[i] != None):
            present_traj.add(i)

    # Extract the values from the enum members in the user input
    user_input_values = set([item.value for item in input_traj])

    # Find items in user input not in present items
    not_present = user_input_values - present_traj

    print(
        f"Detected the following types not present in data: {not_present}, adding them now."
    )

    if "erroneous_label" in present_traj:
        error_label_present = True

    else:
        error_label_present = False

    non_present_enum = [DataVariant(item) for item in not_present]

    return non_present_enum, error_label_present


def process_trajectories(
    path_master_folder: str,
    path_output_file: str,
    data_processing_input: List[DataVariant],
    error_level: float,
    extend: bool = False,
    overwrite: Optional[DataVariant] = None,
) -> None:
    """
    Processes trajectories from a master folder, generating various forms of processed data based on the input settings.

    Parameters:
        path_master_folder (str): Path to the master folder containing subfolders with data.
        path_output_file (str): Path to the output file for processed data.
        data_processing_input (List[DataVariant]): List of data processing inputs specifying the types of processing to apply.
        error_level (float): The level of error to introduce in the erroneous labels.
        extend (bool, optional): Flag to indicate if existing data should be extended. Defaults to False.
        overwrite (Optional[DataVariant], optional): Specific DataVariant type to overwrite in the existing data. Defaults to None.
    """

    error_label_present = False
    if extend:
        # check the present trajectory versions and identify missing one
        with open(path_output_file, "r") as f:
            data = json.load(f)

        data_processing_input, error_label_present = identify_missing_versions(
            data, data_processing_input
        )
    else:
        # Create empty master_file
        with open(path_output_file, "w") as f:
            json.dump({}, f)

    if overwrite != None:
        data_processing_input.append(overwrite)

    lst_folders = [
        f
        for f in os.listdir(path_master_folder)
        if os.path.isdir(os.path.join(path_master_folder, f))
    ]
    lst_folders = sort_string_list_as_numbers(lst_folders)

    # Loop through all the different labels
    idx = 1
    for folder_name in lst_folders:
        # Define the path of the current subfolder
        print(f"[Initialisation - {folder_name}]: started")
        path_current_folder = os.path.join(path_master_folder, folder_name)
        path_masks = os.path.join(path_current_folder, "masks")

        path_annotations = os.path.join(path_current_folder, "annotations.json")
        path_settings = os.path.join(path_current_folder, "settings.json")

        debug_path = os.path.join(path_current_folder, "debug")
        print(f"[Initialisation - {folder_name}]: finished")

        # Extract annotations
        with open(path_annotations, "r") as f:
            annotations_data = json.load(f)

        # Extract settings data
        with open(path_settings, "r") as f:
            settings_data = json.load(f)

        path_unpacked = os.path.join(path_master_folder, folder_name)

        processed_data = {
            item.value: {} for item in DataVariant if item != DataVariant.IDEAL
        }

        # Make sure the input trajectory is long enough
        if len(os.listdir(path_unpacked)) < 8 or len(annotations_data["frameTime"]) < 8:
            continue

        for processing_input in data_processing_input:
            if processing_input == DataVariant.OPTICAL_FLOW:
                # Optical flow
                print(f"[Optical flow - {folder_name}]: started")
                DenseOpticalFlow(
                    input_path=path_unpacked,
                    output_path=path_masks,
                )
                print(f"[Optical flow - {folder_name}]: finished")

                # Construct trajectory from optical flow output
                print(f"[Trajectory constructor - {folder_name}]: started")
                optical_flow_data = TrajectoryConstructor(
                    path_to_masks=path_masks,
                    output_file_path=path_output_file,
                    debug_path=debug_path,
                ).parse_data()
                print(f"[Trajectory constructor - {folder_name}]: finished")

                # Convert the pixel locations to real world locations
                (
                    optical_flow_data["projected_positions"],
                    optical_flow_data["projected_velocities"],
                ) = apply_2d_to_3d_projections(
                    annotations=annotations_data, optical_flow=optical_flow_data
                )

                processed_data[processing_input.value] = optical_flow_data

            elif processing_input == DataVariant.GAUSSIAN_NOISE:
                gaussian_noisy_data = create_noisy_data(
                    annotations_data, gaussian_noise
                )

                processed_data[processing_input.value] = gaussian_noisy_data

            elif processing_input == DataVariant.JITTER_NOISE:
                jitter_noisy_data = create_noisy_data(annotations_data, jitter_noise)

                processed_data[processing_input.value] = jitter_noisy_data

        if not error_label_present:
            erroneous_data = {}

            erroneous_data["objectType"] = create_erroneous_data(
                settings_data["objectType"], error_level
            )

            processed_data["erroneous_label"] = erroneous_data

        # Add to master file
        add_to_master_file(
            idx=idx,
            folder_path=path_unpacked,
            master_path=path_output_file,
            annotations=annotations_data,
            settings=settings_data,
            processed_data=processed_data,
            overwrite=overwrite,
        )

        idx += 1


if __name__ == "__main__":
    data_dir = r"M:\underwater_simulator\footage"

    process_trajectories(
        path_master_folder=data_dir,
        path_output_file=os.path.join(data_dir, "parsed_data.json"),
        parameter_estimation=ParameterEstimation.OFF,
        data_variant=DataVariant.OPTICAL_FLOW,
    )
