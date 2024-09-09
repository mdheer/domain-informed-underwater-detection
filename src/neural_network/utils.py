import torch
import torch.nn as nn
from torch.utils.data import Dataset
import math
import numpy as np
import pandas as pd
import random
import pprint
import sys

from src.tools.enums import (
    ParameterEstimation,
    DataVariant,
    DictKeys,
    ClassName,
)
from typing import Tuple, Union

from src.tools.general_functions import (
    parse_class_name,
    get_midpoints,
    process_modified_data,
    process_unity_data,
)


class CustomDataset(Dataset):
    """
    A custom dataset class for loading and processing data from JSON files for neural network training.

    This class processes various types of data entries, including Unity simulation data and various modified data types,
    and prepares them for use in a PyTorch model.

    Attributes:
        json_data (dict): The dataset loaded from a JSON file.
        parameter_estimation (ParameterEstimation): The type of input used for the mathematical model.
        classes (dict): A dictionary mapping class names to their numerical representations.
        len (int): The fixed length for the neural network input.
        erroneous (float): The level of error introduced in the labels.

    Methods:
        __len__: Returns the number of items in the dataset.
        __getitem__: Retrieves an item from the dataset by its index.
    """

    def __init__(
        self,
        json_data: dict,
        parameter_estimation: ParameterEstimation,
        classes: dict,
        length: int,
        erroneous: float,
        alpha: float,
        unlabelled: dict = {},
    ) -> None:
        """
        Initializes the CustomDataset instance with the provided JSON data and configurations.

        Parameters:
            json_data (dict): The dataset loaded from a JSON file.
            parameter_estimation (ParameterEstimation): The type of input used for the mathematical model.
            classes (dict): A dictionary mapping class names to their numerical representations.
            length (int): The fixed length for the neural network input.
            erroneous (float): The level of error introduced in the labels.
        """

        self.json_data = {**unlabelled, **json_data}
        self.json_keys = list(self.json_data.keys())
        self.parameter_estimation = parameter_estimation
        self.classes = classes
        self.len = length
        self.erroneous = erroneous
        self.alpha = alpha

    def _validate_output_size(
        self, input_lst: dict, normalize: bool = True
    ) -> torch.Tensor:
        """
        Validates and adjusts the size of the neural network input to a fixed length.

        Parameters:
            input_lst (dict): A dictionary containing lists of data points for each dimension.
            normalize (bool, optional): Whether to normalize the data. Defaults to True.

        Returns:
            torch.Tensor: A tensor of the neural network input with a fixed length.
        """

        padding = 0
        flattened = np.array([])

        input_lst = list(input_lst.values())

        tot_len = sum(len(sublist) for sublist in input_lst)
        if tot_len > self.len:
            diff = tot_len - self.len
            decrease_factor = math.ceil(diff / len(input_lst))

            for index, sublist in enumerate(input_lst):
                input_lst[index] = sublist[:-decrease_factor]

        # Using a list comprehension to flatten the list of lists and add the separator

        for arr in input_lst:
            # normalize list
            if normalize:
                arr = arr / np.max(np.abs(arr))

            flattened = np.append(flattened, arr)

        # Pad to get a fixed length
        if self.len > len(flattened):
            flattened = np.append(
                flattened, np.array((self.len - len(flattened)) * [padding])
            )

        return torch.tensor(flattened, dtype=torch.float64)

    def _generate_neural_network_dict(
        self, data_entry: dict, data_variant: DataVariant
    ) -> dict:
        """
        Generates a dictionary for neural network input based on the specified data entry and input type.

        Parameters:
            data_entry (dict): A single data entry containing various data types.
            data_variant (DataVariant): The specified neural network input type.

        Returns:
            dict: A dictionary with processed data suitable for neural network input.
        """
        # Define neural network inputs
        if data_variant == DataVariant.IDEAL:
            return process_unity_data(data_entry)

        elif data_variant == DataVariant.OPTICAL_FLOW:
            return process_modified_data(
                data_entry,
                DataVariant.OPTICAL_FLOW.value,
                DictKeys.OPTICAL_FLOW_POSITIONS.value,
                DictKeys.OPTICAL_FLOW_VELOCITIES.value,
            )

        elif data_variant == DataVariant.GAUSSIAN_NOISE:
            return process_modified_data(
                data_entry,
                DataVariant.GAUSSIAN_NOISE.value,
                DictKeys.NOISY_DATA_POSITIONS.value,
                DictKeys.NOISY_DATA_VELOCITIES.value,
            )

        elif data_variant == DataVariant.JITTER_NOISE:
            return process_modified_data(
                data_entry,
                DataVariant.JITTER_NOISE.value,
                DictKeys.NOISY_DATA_POSITIONS.value,
                DictKeys.NOISY_DATA_VELOCITIES.value,
            )

        else:
            raise NotImplementedError(
                f"The requested setting: {data_variant} is not implemented yet."
            )

    def _generate_settings_dict(
        self, data_entry: dict, data_variant: DataVariant
    ) -> dict:
        """
        Generates a settings dictionary for the mathematical model based on the data entry and input type.

        Parameters:
            data_entry (dict): A single data entry containing Unity annotations and settings.
            data_variant (DataVariant): The specified neural network input type.

        Returns:
            dict: A dictionary containing settings for the mathematical model.
        """
        # Define mathematical model inputs
        unity_annotations = data_entry[DictKeys.UNITY_ANNOTATIONS.value]
        unity_settings = data_entry[DictKeys.UNITY_SETTINGS.value]

        tend = unity_annotations["frameTime"][-1]
        unity_settings["tend"] = tend
        unity_settings["frameTime"] = data_entry[DictKeys.UNITY_ANNOTATIONS.value][
            "frameTime"
        ]

        if data_variant == DataVariant.OPTICAL_FLOW:
            optical_flow_data_entry = data_entry[DictKeys.PROCESSED_DATA.value][
                DataVariant.OPTICAL_FLOW.value
            ]

            init_states = {
                "y_0": optical_flow_data_entry[DictKeys.OPTICAL_FLOW_POSITIONS.value][
                    0
                ]["y"],
                "z_0": optical_flow_data_entry[DictKeys.OPTICAL_FLOW_POSITIONS.value][
                    0
                ]["z"],
                "vy_0": optical_flow_data_entry[DictKeys.OPTICAL_FLOW_VELOCITIES.value][
                    0
                ]["y"],
                "vz_0": optical_flow_data_entry[DictKeys.OPTICAL_FLOW_VELOCITIES.value][
                    0
                ]["z"],
                "x_angular_0": unity_annotations[DictKeys.UNITY_ANGLES.value][0]["z"],
                "vx_angular_0": unity_annotations[
                    DictKeys.UNITY_ANGULAR_VELOCITIES.value
                ][0]["x"],
            }

        elif data_variant == DataVariant.IDEAL:
            init_states = {
                "y_0": unity_annotations[DictKeys.UNITY_POSITIONS.value][0]["y"],
                "z_0": unity_annotations[DictKeys.UNITY_POSITIONS.value][0]["z"],
                "vy_0": unity_annotations[DictKeys.UNITY_VELOCITIES.value][0]["y"],
                "vz_0": unity_annotations[DictKeys.UNITY_VELOCITIES.value][0]["z"],
                "x_angular_0": unity_annotations[DictKeys.UNITY_ANGLES.value][0]["z"],
                "vx_angular_0": unity_annotations[
                    DictKeys.UNITY_ANGULAR_VELOCITIES.value
                ][0]["x"],
            }

        elif data_variant == DataVariant.GAUSSIAN_NOISE:
            gaussian_noise_data_entry = data_entry[DictKeys.PROCESSED_DATA.value][
                DataVariant.GAUSSIAN_NOISE.value
            ]

            init_states = {
                "y_0": gaussian_noise_data_entry[DictKeys.NOISY_DATA_POSITIONS.value][
                    0
                ]["y"],
                "z_0": gaussian_noise_data_entry[DictKeys.NOISY_DATA_POSITIONS.value][
                    0
                ]["z"],
                "vy_0": gaussian_noise_data_entry[DictKeys.NOISY_DATA_VELOCITIES.value][
                    0
                ]["y"],
                "vz_0": gaussian_noise_data_entry[DictKeys.NOISY_DATA_VELOCITIES.value][
                    0
                ]["z"],
                "x_angular_0": unity_annotations[DictKeys.UNITY_ANGLES.value][0]["z"],
                "vx_angular_0": unity_annotations[
                    DictKeys.UNITY_ANGULAR_VELOCITIES.value
                ][0]["x"],
            }
        elif data_variant == DataVariant.JITTER_NOISE:
            jitter_noise_data_entry = data_entry[DictKeys.PROCESSED_DATA.value][
                DataVariant.JITTER_NOISE.value
            ]

            init_states = {
                "y_0": jitter_noise_data_entry[DictKeys.NOISY_DATA_POSITIONS.value][0][
                    "y"
                ],
                "z_0": jitter_noise_data_entry[DictKeys.NOISY_DATA_POSITIONS.value][0][
                    "z"
                ],
                "vy_0": jitter_noise_data_entry[DictKeys.NOISY_DATA_VELOCITIES.value][
                    0
                ]["y"],
                "vz_0": jitter_noise_data_entry[DictKeys.NOISY_DATA_VELOCITIES.value][
                    0
                ]["z"],
                "x_angular_0": unity_annotations[DictKeys.UNITY_ANGLES.value][0]["z"],
                "vx_angular_0": unity_annotations[
                    DictKeys.UNITY_ANGULAR_VELOCITIES.value
                ][0]["x"],
            }

        settings = {
            "config": data_entry[DictKeys.UNITY_SETTINGS.value],
            "init_states": init_states,
            "neural_network_input": data_variant,
        }

        return settings

    def _generate_label_tensor(self, label_name: str) -> torch.Tensor:
        """
        Generates a tensor for the label corresponding to the given class name.

        Parameters:
            label_name (str): The name of the class.

        Returns:
            torch.Tensor: A tensor representing the class label.
        """

        # Map labels
        label_class = parse_class_name(label_name)
        label_num = self.classes[label_class.value]

        label_tensor = torch.tensor(label_num, dtype=torch.float64)

        return label_tensor

    def _parse_dict_to_df(
        self,
        data_dict: dict,
        frame_times: list,
        data_variant: DataVariant,
    ) -> pd.DataFrame:
        """
        Parses a dictionary into a pandas DataFrame considering the frame times and neural network input type.

        Parameters:
            data_dict (dict): The data dictionary to be parsed.
            frame_times (list): A list of frame times for the data.
            data_variant (DataVariant): The type of neural network input used.

        Returns:
            pd.DataFrame: The parsed DataFrame.
        """

        frame_time_len = len(frame_times)

        if all(frame_time_len == len(value) for value in data_dict.values()):
            data_dict["t"] = frame_times
            df = pd.DataFrame(data_dict)
            return df

        elif data_variant == DataVariant.OPTICAL_FLOW:
            # To account for the fact that optical flow has less data points
            times = get_midpoints(get_midpoints(frame_times))
            data_dict["t"] = times
            df = pd.DataFrame(data_dict)
            return df

        else:
            raise ValueError(
                f"The data_dict values do not correspond with frame_time_len. Data dict is {data_dict} and frame_time_len is {frame_time_len}"
            )

    def _parse_data_variant(self, input_value: str) -> Union[DataVariant, str]:
        """
        Parses the input value to match a corresponding member of the DataVariant enum.

        This method iterates through the members of the DataVariant enum to find a match for the input value.
        If a match is found, the corresponding enum member is returned. If no match is found, the original input value
        is returned, which could indicate an error or unrecognized input.

        Parameters:
            input_value (str): A string representing the type of neural network input.

        Returns:
            Union[DataVariant, str]: The corresponding DataVariant enum member if a match is found,
                                            or the original string if no match is found.
        """
        # Loop through the enum members to find a match
        for member in DataVariant:
            if member.value == input_value:
                return member
        # Return None or raise an error if the string doesn't match any enum
        return input_value

    def __len__(self) -> int:
        """
        Returns the number of items in the dataset.

        Returns:
            int: The number of items in the dataset.
        """

        return len(self.json_data)

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, dict, torch.Tensor, torch.Tensor, dict, pd.DataFrame, str]:
        """
        Retrieves an item from the dataset by its index.

        Parameters:
            index (int): The index of the item to retrieve.

        Returns:
            Tuple[torch.Tensor, dict, torch.Tensor, torch.Tensor, dict, pd.DataFrame, str]:
            A tuple containing the neural network input tensor, the original data dictionary, the label tensor,
            the erroneous label tensor, the settings dictionary, the data DataFrame, and the data entry key.
        """

        data_entry = self.json_data[self.json_keys[index]]

        data_variant = self._parse_data_variant(data_entry["neural_network_input"])

        data_dict = self._generate_neural_network_dict(data_entry, data_variant)
        settings = self._generate_settings_dict(data_entry, data_variant)
        label_tensor = self._generate_label_tensor(
            data_entry[DictKeys.UNITY_SETTINGS.value]["objectType"]
        )

        if self.erroneous > 0:

            ground_truth = parse_class_name(
                data_entry[DictKeys.UNITY_SETTINGS.value]["objectType"]
            )

            alternatives = {
                ClassName.PLASTIC: "fish",
                ClassName.FISH: "pb",
            }

            real = {
                ClassName.PLASTIC: "pb",
                ClassName.FISH: "fish",
            }

            # Decide whether to switch based on the percentage
            if random.random() < self.erroneous:
                # Return the alternative label
                errorenous = alternatives[ground_truth]
            else:
                # Return the ground truth
                errorenous = real[ground_truth]

            erroenous_label_tensor = self._generate_label_tensor(errorenous)

        else:
            erroenous_label_tensor = self._generate_label_tensor(
                data_entry[DictKeys.UNITY_SETTINGS.value]["objectType"]
            )

        data_df = self._parse_dict_to_df(
            data_dict,
            data_entry[DictKeys.UNITY_SETTINGS.value]["frameTime"],
            data_variant,
        )

        if "unlabelled" in data_entry:
            if data_entry["unlabelled"]:
                alpha = 1
            else:
                alpha = self.alpha

        else:
            alpha = self.alpha

        return (
            self._validate_output_size(data_dict),
            data_dict,
            label_tensor,
            erroenous_label_tensor,
            settings,
            data_df,
            self.json_keys[index],
            alpha,
        )


class SimpleFNN(nn.Module):
    """
    A simple feedforward neural network model.

    This class defines a basic three-layer feedforward neural network (FNN) with ReLU activation functions.
    The model consists of an input layer, two hidden layers, and an output layer. The output size of the
    network is determined by the number of classes.

    Attributes:
        fc1 (nn.Linear): The first fully connected (linear) layer.
        relu (nn.ReLU): The ReLU activation function.
        fc2 (nn.Linear): The second fully connected (linear) layer.
        fc3 (nn.Linear): The third fully connected (linear) layer, which is the output layer.

    Parameters:
        input_size (int): The size of the input features.
        num_classes (int): The number of classes for classification.

    Example Usage:
        >>> model = SimpleFNN(input_size=1000, num_classes=2)
        >>> input_tensor = torch.randn(1, 1000)
        >>> output = model(input_tensor)
    """

    def __init__(self, input_size: int, num_classes: int, device):
        """
        Initializes the SimpleFNN model with specified input size and number of classes.

        Parameters:
            input_size (int): The number of input features for the network.
            num_classes (int): The number of classes for the output layer.
        """

        seed = 42

        # Ensures the same weights are generated every run
        if device.type == "cuda":
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        super(SimpleFNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the neural network.

        Parameters:
            x (torch.Tensor): The input tensor for the neural network.

        Returns:
            torch.Tensor: The output tensor after passing through the network.
        """
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out
