import os

from src.tools.general_functions import (
    read_input_path,
    load_config,
    parse_input_path_dependent_on_os,
)
from src.data_preprocessing.data_curation import DatasetGenerator, DatasetConfig


def create_dataset_config(config_dict):
    """
    Converts a dictionary from the configs structure into a DatasetConfig object.
    """
    return DatasetConfig(**config_dict)


if __name__ == "__main__":
    dataset_file_name = "small_test_set"

    config_name = f"dataset_configs.{dataset_file_name}"
    # Load the configuration module
    config = load_config(config_name)

    # Access configurations
    input_data_path = config.input_data_path
    dataset_name = config.dataset_name
    num_samples_per_fold = config.num_samples_per_fold
    k = config.k
    min_train_samples = config.min_train_samples
    step = config.step
    k_folds_sets = config.k_folds_sets

    input_data_path = parse_input_path_dependent_on_os(input_data_path)

    raw_data_dir = read_input_path(
        input_type=input_data_path
    )  # Check the possible input types

    file_path = os.path.join(raw_data_dir, "parsed_data.json")

    # Create DatasetConfig objects from the configs dictionary
    dataset_configs = {
        key: create_dataset_config(value) for key, value in config.configs.items()
    }

    generator = DatasetGenerator(
        data_path=file_path,
        dataset_name=dataset_name,
        unlabelled_data_config=dataset_configs["unlabelled_data"],
        train_config=dataset_configs["train_dataset_input"],
        validate_config=dataset_configs["validate_dataset_input"],
        test_config=dataset_configs["test_dataset_input"],
        num_samples_per_fold=num_samples_per_fold,
        k=k,
        k_folds_sets=k_folds_sets,
        min_train_samples=min_train_samples,
        step=step,
    )

    print(f"Dataset generation finished for {dataset_name} using {config_name}.")
