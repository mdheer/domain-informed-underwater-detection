import os
import json

from src.trajectory.main import (
    process_trajectories,
)

from src.tools.enums import DataVariant, InputDataPath


def read_input_path(input_type: InputDataPath):
    """
    Read the input path and extra data from the JSON file.

    Parameters:
        input_type (InputDataPath): input type from the InputDataPath class

    """
    with open("./config.json", "r") as file:
        config = json.load(file)

    path = config[input_type.value]

    return path


if __name__ == "__main__":
    # Settings
    raw_data_dir = read_input_path(
        input_type=InputDataPath.CLUSTER_MERGED
    )  # Check the possible input types

    error_level = 0.1

    data_processing_input = [
        DataVariant.OPTICAL_FLOW,
        DataVariant.GAUSSIAN_NOISE,
        DataVariant.JITTER_NOISE,
    ]

    input_to_overwrite = None

    path_master_data_file = os.path.join(raw_data_dir, "parsed_data.json")

    if not os.path.exists(path_master_data_file):
        process_trajectories(
            path_output_file=path_master_data_file,
            path_master_folder=raw_data_dir,
            data_processing_input=data_processing_input,
            error_level=error_level,
        )

    else:
        inp1 = input("Overwrite master file? [y/n] ")
        if inp1 == "y":
            process_trajectories(
                path_output_file=path_master_data_file,
                path_master_folder=raw_data_dir,
                data_processing_input=data_processing_input,
                error_level=error_level,
            )

        if inp1 == "n":
            inp2 = input("Expand master file? [y/n] ")
            if inp2 == "y":
                process_trajectories(
                    path_output_file=path_master_data_file,
                    path_master_folder=raw_data_dir,
                    data_processing_input=data_processing_input,
                    error_level=error_level,
                    extend=True,
                    overwrite=input_to_overwrite,
                )
