import pandas as pd
import json
import os
from typing import Dict, List, Union
import numpy as np
import shutil
from enum import Enum
import sys

from src.data_preprocessing.preprocessing import (
    filter_optical_flow,
    filter_water_current,
    create_dirs,
    create_yolo_yaml_file,
    decrease_num_samples,
)
from src.tools.enums import (
    HomePath,
    WaterCurrentFilter,
    DataSplit,
    OpticalFlowAccuracy,
    DataVariant,
)
from dataclasses import dataclass


@dataclass
class DatasetConfig:
    data_split: DataSplit
    number_of_datapoints: int
    data_variant: DataVariant
    filter_based_on_current: Union[WaterCurrentFilter, None]

    def to_dict(self):
        data = {}
        # Convert enum fields to their string representation
        data["number_of_datapoints"] = self.number_of_datapoints
        if isinstance(self.data_split, Enum):
            data["data_split"] = self.data_split.value
        if isinstance(self.data_variant, Enum):
            data["data_variant"] = self.data_variant.value
        if isinstance(self.filter_based_on_current, Enum):
            data["filter_based_on_current"] = self.filter_based_on_current.value
        elif self.filter_based_on_current is None:
            data["filter_based_on_current"] = None

        return data


class DatasetGenerator:

    def __init__(
        self,
        data_path=str,
        dataset_name=str,
        unlabelled_data_config=DatasetConfig,
        train_config=DatasetConfig,
        validate_config=DatasetConfig,
        test_config=DatasetConfig,
        num_samples_per_fold=int,
        k=int,
        k_folds_sets=List,
        min_train_samples=int,
        step=int,
    ) -> None:

        self.split_mapping = {
            DataSplit.UNLABELLED: unlabelled_data_config,
            DataSplit.TRAIN: train_config,
            DataSplit.VALIDATE: validate_config,
            DataSplit.TEST: test_config,
        }

        self.min_train_samples = min_train_samples
        self.step = step
        self.data_path = data_path

        relative_path = "./datasets"
        self.main_dir_path = os.path.join(relative_path, dataset_name)
        if os.path.exists(self.main_dir_path):
            inp = input(
                "Dataset already exists, do you want to the delete the existing dataset and continue? [y/n] "
            )
            if inp == "y" or inp == "Y":
                shutil.rmtree(self.main_dir_path)
                os.mkdir(self.main_dir_path)
            else:
                print("Exiting...")
                sys.exit()
        else:
            os.mkdir(self.main_dir_path)

        with open(self.data_path) as f:
            data = json.load(f)

        df = self._prepare_dataframe(data)
        self._verify_inputs(
            df=df,
            k_folds_sets=k_folds_sets,
        )

        df = self._assign_data_splits(df, k_folds_sets)
        df = self._generate_k_folds_from_df(df, num_samples_per_fold)
        distributions = self._generate_distributions_k_folds(df, k)
        self._create_training_configs(distributions)

    def _prepare_dataframe(self, data: Dict) -> pd.DataFrame:
        """
        Prepares and formats a DataFrame from the given data.

        This function parses the provided data, filtering it based on certain criteria, and then transposes it into a DataFrame format for further processing.

        Parameters:
            data (Dict): The input data to be parsed and formatted.

        Returns:
            pd.DataFrame: The processed DataFrame with the parsed and formatted data.
        """

        data_parsed = {}
        data_points = data.keys()

        for key in data_points:
            optical_flow_accuracy = filter_optical_flow(
                data_entry=data[key], thresh_y=1, thresh_vy=5, thresh_z=2, thresh_vz=15
            )
            outcome_filter = filter_water_current(data_entry=data[key])

            data_parsed[key] = {
                "optical_flow_accuracy": optical_flow_accuracy,
                "water_current_filter": outcome_filter,
            }

        df = pd.DataFrame(data_parsed).transpose()

        df["available"] = True
        df["data_split"] = None
        df["data_variant"] = None
        df["dynamic"] = None
        df["k_fold_id"] = None

        return df

    def _verify_inputs(self, df, k_folds_sets):

        # Verify that samples with current filtering are not included in the k_folds_sets
        for set_name in k_folds_sets:
            if (
                type(self.split_mapping[set_name].filter_based_on_current)
                == WaterCurrentFilter
            ):
                raise ValueError(
                    f"{set_name} is present in the k fold cross-validation, but contains a filter based on current."
                )

        # Verify that a sufficient number specific samples are present.
        total_required_strong_curr = 0
        total_required_light_curr = 0
        total = 0

        for ky in self.split_mapping.keys():
            if (
                self.split_mapping[ky].filter_based_on_current
                == WaterCurrentFilter.STRONG_CURRENT
            ):
                total_required_strong_curr += self.split_mapping[
                    ky
                ].number_of_datapoints

            elif (
                self.split_mapping[ky].filter_based_on_current
                == WaterCurrentFilter.LIGHT_CURRENT
            ):
                total_required_light_curr += self.split_mapping[ky].number_of_datapoints

            total += self.split_mapping[ky].number_of_datapoints

        num_strong_current = (
            df["water_current_filter"] == WaterCurrentFilter.STRONG_CURRENT
        ).sum()
        num_light_current = (
            df["water_current_filter"] == WaterCurrentFilter.LIGHT_CURRENT
        ).sum()

        if num_strong_current < total_required_strong_curr:
            raise ValueError(
                f"Number of present strong current values is {num_strong_current} but the required is {total_required_strong_curr}"
            )

        if num_light_current < total_required_light_curr:
            raise ValueError(
                f"Number of present light current values is {num_light_current} but the required is {total_required_light_curr}"
            )

        # Verify that the dataset can facilitate the number of samples required
        if len(df.index) < total:
            raise ValueError(
                f"The dataframe has {len(df.index)} samples and the input requires {total} samples."
            )

        for key, value in self.split_mapping.items():
            if (
                type(value.filter_based_on_current) != WaterCurrentFilter
                and key not in k_folds_sets
            ):
                raise ValueError(
                    f"{key} has no value for the water current filter, but is also not present in the k_fold cross validation. Please add!"
                )

        # print(f"df: {df}")
        # print(f"strong current count: {total_required_strong_curr}")
        # print(f"light current count: {total_required_light_curr}")
        # print(f"strong current present: {num_strong_current}")
        # print(f"light current present: {num_light_current}")

        # print(f"total numbers req: {total}")
        # print(f"total number pres: {len(df.index)}")

    def _assign_data_splits(self, df, k_folds_sets):

        # Convert lists to sets
        set1 = set(list(self.split_mapping.keys()))
        set2 = set(k_folds_sets)

        # Find elements unique to each list
        unique_to_list1 = set1 - set2
        unique_to_list2 = set2 - set1

        # Combine the unique elements
        unique_elements = unique_to_list1.union(unique_to_list2)
        self.dynamic_splits = set1 - unique_elements

        # Address filtered sets
        for static_set in unique_elements:
            dataset_config = self.split_mapping[static_set]
            desired_filter = dataset_config.filter_based_on_current
            subset = df.loc[
                (df["water_current_filter"] == desired_filter)
                & (df["available"] == True)
            ]

            num = dataset_config.number_of_datapoints

            # Select random amount of num from the subset
            random_subset = subset.sample(n=num)

            # Locate this in self.input_data and change the column type and available
            for idx in random_subset.index:
                df.at[idx, "available"] = False
                df.at[idx, "data_split"] = static_set
                df.at[idx, "data_variant"] = dataset_config.data_variant
                df.at[idx, "dynamic"] = False
                df.at[idx, "k_fold_id"] = "n.a."

        to_delete = set()
        for dynamic_set in self.dynamic_splits:
            dataset_config = self.split_mapping[dynamic_set]

            if dataset_config.number_of_datapoints == 0:
                to_delete.add(dynamic_set)
                continue

            desired_filter = dataset_config.filter_based_on_current
            subset = df.loc[(df["available"] == True)]

            num = dataset_config.number_of_datapoints

            # Select random amount of num from the subset
            random_subset = subset.sample(n=num)

            # Locate this in self.input_data and change the column type and available
            for idx in random_subset.index:
                df.at[idx, "available"] = False
                df.at[idx, "data_split"] = dynamic_set
                df.at[idx, "data_variant"] = dataset_config.data_variant
                df.at[idx, "dynamic"] = True

        self.dynamic_splits = self.dynamic_splits - to_delete
        return df

    def _generate_k_folds_from_df(self, df, samples_per_fold):
        # total_number = len(df.loc[df["dynamic"] == True].index)
        k_fold_id = 1
        for i in self.dynamic_splits:
            dynamic_set = df.loc[(df["data_split"] == i) & (df["dynamic"] == True)]
            tot_per_set = len(dynamic_set.index) / samples_per_fold
            for _ in range(int(tot_per_set)):
                subset = dynamic_set.sample(n=samples_per_fold)
                for idx in subset.index:
                    df.at[idx, "k_fold_id"] = k_fold_id

                k_fold_id += 1
                dynamic_set = dynamic_set.loc[df["k_fold_id"].isnull()]

        return df.loc[df["available"] == False]

    def _generate_distributions_k_folds(self, df, k):

        def extract_fold_ids(frame):

            datasplits_present = frame["data_split"].unique()

            properties = {"k_folds": {}, "df": {}}

            for ds in datasplits_present:
                subset = frame.loc[df["data_split"] == ds]
                ids = subset["k_fold_id"].unique()
                properties["k_folds"][ds] = ids

            properties["df"] = frame
            return properties

        def shuffle_k_folds(k_folds):
            # Extract all values except 'n.a.', and keep track of structure
            all_values = []
            structure = {}
            for key, arr in k_folds.items():
                if "n.a." in arr:
                    structure[key] = ["n.a."]  # Preserve the 'n.a.' and structure
                else:
                    all_values.extend(arr)
                    structure[key] = len(arr)  # Store length for redistribution

            # Shuffle the non-'n.a.' values
            np.random.shuffle(all_values)

            # Redistribute the values according to the original structure
            shuffled_data = {}
            index = 0
            for key, length in structure.items():
                if length == ["n.a."]:
                    shuffled_data[key] = np.array(["n.a."], dtype=object)
                else:
                    shuffled_values = all_values[index : index + length]
                    shuffled_data[key] = np.array(shuffled_values, dtype=object)
                    index += length

            return shuffled_data

        def reconfigure_df_based_on_k_folds(df, k_folds):
            modified_df = df.copy()
            for key, arr in k_folds.items():
                for id in arr:
                    condition = modified_df["k_fold_id"] == id
                    modified_df.loc[condition, ["data_split", "data_variant"]] = [
                        key,
                        self.split_mapping[key].data_variant,
                    ]
            return modified_df

        distributions = {}
        initial_distr = extract_fold_ids(df)
        distributions[f"distribution_{1}"] = initial_distr
        k_folds_shuffled = initial_distr["k_folds"]

        for i in range(2, k + 1):
            k_folds_shuffled = shuffle_k_folds(k_folds_shuffled)
            df_shuffled = reconfigure_df_based_on_k_folds(df, k_folds_shuffled)
            distributions[f"distribution_{i}"] = extract_fold_ids(df_shuffled)

        return distributions

    def _create_training_configs(self, distributions):

        def df_to_dict_with_custom_types(df):
            cp = df.copy()
            for column in cp.columns:
                # Using .loc to ensure the operation is done on the original DataFrame
                cp.loc[:, column] = cp.loc[:, column].apply(
                    lambda x: x.value if hasattr(x, "value") else x
                )
            return cp.to_dict()

        def create_output_dict(df):

            unlabelled_df = df.loc[df["data_split"] == DataSplit.UNLABELLED]
            train_df = df.loc[df["data_split"] == DataSplit.TRAIN]
            validate_df = df.loc[df["data_split"] == DataSplit.VALIDATE]
            test_df = df.loc[df["data_split"] == DataSplit.TEST]

            tot_dict = {
                "data_path": self.data_path,
                "config_train": self.split_mapping[DataSplit.TRAIN].to_dict(),
                "config_validate": self.split_mapping[DataSplit.VALIDATE].to_dict(),
                "config_test": self.split_mapping[DataSplit.TEST].to_dict(),
                "config_unlabelled": self.split_mapping[DataSplit.UNLABELLED].to_dict(),
                "train": df_to_dict_with_custom_types(train_df),
                "validate": df_to_dict_with_custom_types(validate_df),
                "test": df_to_dict_with_custom_types(test_df),
                "unlabelled": df_to_dict_with_custom_types(unlabelled_df),
            }

            return tot_dict

        def decrease_num_train_samples(df):
            # Condition for filtering
            condition = df["data_split"] == DataSplit.TRAIN

            # Filter rows that satisfy the condition and randomly sample x rows from them
            rows_to_remove = df[condition].sample(
                n=self.step, random_state=np.random.RandomState()
            )

            # Drop these rows from the original DataFrame
            cp = df.drop(rows_to_remove.index)

            # Re-evaluate the condition on the modified DataFrame
            new_condition = cp["data_split"] == DataSplit.TRAIN

            # Now calculate the length based on the re-evaluated condition
            length = len(cp[new_condition].index)

            return cp, length

        for key, value in distributions.items():
            # 1. create a folder to save the distributions
            distr_folder = os.path.join(self.main_dir_path, f"k_fold_{key}")
            os.mkdir(distr_folder)

            # 2. create distributions folder
            sub_distr_folder = os.path.join(distr_folder, "configs")
            os.mkdir(sub_distr_folder)

            # 3. loop through the various distributions
            df_decreasing_samples = value["df"].copy()
            num_train_samples = (
                df_decreasing_samples["data_split"] == DataSplit.TRAIN
            ).sum()

            while num_train_samples >= self.min_train_samples:

                tot_dict = create_output_dict(df_decreasing_samples)

                file_name = os.path.join(
                    sub_distr_folder,
                    f"{num_train_samples}_train_samples_data_distribution.json",
                )

                with open(file_name, "w") as json_file:
                    json.dump(tot_dict, json_file, indent=2)

                df_decreasing_samples, num_train_samples = decrease_num_train_samples(
                    df_decreasing_samples
                )
