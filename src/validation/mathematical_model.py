from src.validation.utils import Comparison, Validation
from src.domain_knowledge.domain_knowledge_classifier import DomainKnowledgeClassifier
from src.train_trajectory import extract_param_bounds
from src.neural_network.utils import CustomDataset
from src.neural_network.main import custom_collate_fn, parse_input_data
from torch.utils.data import DataLoader
from src.tools.enums import InvertedSimilarity, ParameterEstimation, ClassName
import torch.nn as nn
import torch
from itertools import chain
import matplotlib.pyplot as plt

import numpy as np
import os
import json

input_path = r"M:\underwater_simulator\parsed_footage_v2023-10-24\footage\ssmall_parsed_data.json"
# input_path = r"/data/mdheer/input_data/merged_v2023-10-24_v2023-11-02/parsed_data.json"

# Extract param bounds
param_bounds = extract_param_bounds(input_path)

input_data = {
    "optical_flow": r"C:\Users\mathi\Documents\Files\University of Technology Delft\Master\MSc 2\Thesis\Code\thesis_mathieu\datasets\val_optical_flow\k_fold_distribution_1",
    "gaussian_noise": r"C:\Users\mathi\Documents\Files\University of Technology Delft\Master\MSc 2\Thesis\Code\thesis_mathieu\datasets\val_gaussian_noise\k_fold_distribution_1",
    "ideal": r"C:\Users\mathi\Documents\Files\University of Technology Delft\Master\MSc 2\Thesis\Code\thesis_mathieu\datasets\val_ideal\k_fold_distribution_1",
}

# input_data = {
#     "optical_flow": r"/home/mdheer/thesis_mathieu/datasets/val_optical_flow/k_fold_distribution_1",
#     "gaussian_noise": r"/home/mdheer/thesis_mathieu/datasets/val_gaussian_noise/k_fold_distribution_1",
#     "ideal": r"/home/mdheer/thesis_mathieu/datasets/val_ideal/k_fold_distribution_1",
# }

final_path = r"C:\Users\mathi\Documents\Files\University of Technology Delft\Master\MSc 2\Thesis\Code\thesis_mathieu\src\validation\graphs"


def verify_prediction(plastic_pred, fish_pred, label):
    if plastic_pred < fish_pred:
        if label == 1:
            return True
        elif label == 0:
            return False

    elif plastic_pred > fish_pred:
        if label == 0:
            return True
        elif label == 1:
            return False


def stringify_keys(data):
    """
    Recursively convert dictionary keys to strings if they are not already
    strings, integers, or floats. Convert PyTorch tensors to lists and ensure
    all values are also JSON-serializable.
    """
    if isinstance(data, dict):
        new_dict = {}
        for key, value in data.items():
            # Only convert the key to a string if it's not already a string, int, or float
            if not isinstance(key, (str, int, float)):
                key = str(key)
            new_dict[key] = stringify_keys(value)
        return new_dict
    elif isinstance(data, list):
        return [stringify_keys(item) for item in data]
    elif isinstance(data, torch.Tensor):
        # Convert tensors to lists
        return data.tolist()
    else:
        # Add any specific conversions for other types here if necessary
        return data


def merge_list_to_dict(list_of_dicts):
    # Initialize an empty dictionary to hold the combined results
    combined_dict = {}

    # Iterate through each dictionary in the list
    for d in list_of_dicts:
        for key, value_list in d.items():
            if key in combined_dict:
                # If the key already exists in the combined_dict, extend the existing list
                combined_dict[key] = list(chain(combined_dict[key], value_list))
            else:
                # If the key does not exist, initialize it with the current list
                combined_dict[key] = value_list
    return combined_dict


def parse_metric_to_accuracy(metrics: dict, labels: list) -> dict:
    combined_labels = []
    for tensor in labels:
        combined_labels.extend(tensor.tolist())

    final_metrics = {}

    for metric in metrics.keys():
        pred = []
        prob = []
        labs = []

        for predictions, lab in zip(metrics[metric], combined_labels):
            plastic_pred = predictions[0]
            fish_pred = predictions[1]
            probability = nn.functional.softmax(
                torch.tensor([[plastic_pred, fish_pred]]), dim=1
            )
            prob.append(probability)

            pred.append(
                verify_prediction(
                    probability[0, 0].item(), probability[0, 1].item(), lab
                )
            )

            labs.append(lab)

        # Count the number of True values
        acc = round((pred.count(True) / len(pred)) * 100, 3)
        final_metrics[metric] = {"accuracy": acc, "probabilities": prob, "label": labs}

    return final_metrics


def calc_rmse(traj_df):
    traj_columns = {
        "y": "sim_y",
        "z": "sim_z",
        "vy": "sim_vy",
        "vz": "sim_vz",
    }

    mm_columns = {
        "y": "mm_y",
        "z": "mm_z",
        "vy": "mm_vy",
        "vz": "mm_vz",
    }

    rmse = {}
    for i in mm_columns.keys():
        traj_df[f"rmse_{i}"] = np.sqrt(
            (traj_df[mm_columns[i]] - traj_df[traj_columns[i]]) ** 2
        )
        rmse[i] = traj_df[f"mse_{i}"].mean()

    rmse_y = rmse["y"]
    rmse_z = rmse["z"]
    rmse_vy = rmse["vy"]
    rmse_vz = rmse["vz"]

    return rmse_y, rmse_z, rmse_vy, rmse_vz


def create_traj_graph(dfs, data_type, param, num):
    dir_path = r".\src\validation\graphs"

    filename_graph = os.path.join(dir_path, f"{data_type}_{param}_{num}.png")

    # Creating a 2x2 plot layout
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    # fig.suptitle(parsed_data[key][DictKeys.UNITY_SETTINGS.value]["objectType"])
    axs = axs.flatten()

    titles = ["t vs. y", "t vs. z", "t vs. vy", "t vs. vz"]
    y_labels = ["mm_y", "mm_z", "mm_vy", "mm_vz"]

    for i, (title, y_label) in enumerate(zip(titles, y_labels)):
        axs[i].set_title(title)
        for name, df in dfs.items():
            axs[i].plot(df.index, df[y_label], label=name)
        axs[i].legend()
        axs[i].set_xlabel("t")
        axs[i].set_ylabel(y_label)

    plt.tight_layout()
    plt.savefig(filename_graph)
    plt.close()


# Creating a list of the enum values
param_estm_level = [member for member in ParameterEstimation.__members__.values()]
final_results = {}

for key, global_dataset_path in input_data.items():

    dataset_path = os.path.join(global_dataset_path, "configs")
    lst = os.listdir(dataset_path)
    dataset_path = os.path.join(dataset_path, lst[0])

    datapath_validation = os.path.join(global_dataset_path, "validation")

    if not os.path.isdir(datapath_validation):
        os.mkdir(datapath_validation)

    final_results[key] = {}
    for param_estm in param_estm_level:
        (train_data, _, _, _, _, _, _, _) = parse_input_data(
            master_data_file=input_path,
            info_data_file=dataset_path,
        )

        dk_classifier = DomainKnowledgeClassifier(
            mm_input=param_estm,
            param_bounds=param_bounds,
            mm_model_folder=datapath_validation,
            similarity_parameter=InvertedSimilarity.WEIGHTED_MSE,
        )

        train_set = CustomDataset(
            json_data=train_data,
            parameter_estimation=param_estm,
            classes={"plastic": 0, "fish": 1},
            length=1000,
            erroneous=0,
        )

        # Create DataLoaders for the training and validation sets
        train_loader = DataLoader(
            train_set,
            batch_size=1,
            shuffle=False,
            collate_fn=custom_collate_fn,
        )

        tot_trajectories = []

        for (
            _,
            trajectories_dict,
            labels,
            erroneous_labels,
            settings,
            mm_inputs,
            data_numbers,
            _,
        ) in train_loader:
            dk_classifier.get_teacher_output(
                batched_traj=trajectories_dict,
                batched_settings=settings,
                input_trajects=mm_inputs,
                labels=labels,
                data_numbers=data_numbers,
            )
            if labels[0] == 0:
                tot_trajectories.append(dk_classifier.df_list[0][ClassName.PLASTIC])

            elif labels[0] == 1:
                tot_trajectories.append(dk_classifier.df_list[0][ClassName.FISH])

            else:
                raise ValueError(f"label not recognized: {labels}")

        rmse_y_lst = []
        rmse_z_lst = []
        rmse_vy_lst = []
        rmse_vz_lst = []

        for traj in tot_trajectories:
            rmse_y, rmse_z, rmse_vy, rmse_vz = calc_rmse(traj)
            rmse_y_lst.append(rmse_y)
            rmse_z_lst.append(rmse_z)
            rmse_vy_lst.append(rmse_vy)
            rmse_vz_lst.append(rmse_vz)

        avg_y = sum(rmse_y_lst) / len(rmse_y_lst)
        avg_z = sum(rmse_z_lst) / len(rmse_z_lst)
        avg_vy = sum(rmse_vy_lst) / len(rmse_vy_lst)
        avg_vz = sum(rmse_vz_lst) / len(rmse_vz_lst)

        print(f"Data: {key} || {param_estm}")
        print(f"avg_y: {avg_y}, avg_z: {avg_z}, avg_vy: {avg_vy}, avg_vz: {avg_vz}")

        final_results[key][param_estm] = tot_trajectories

    new_structure = []

    num_data_points = len(tot_trajectories)

    for (i,) in range(num_data_points):
        new_dict = {}
        for param_estm in param_estm_level:
            new_dict[param_estm] = final_results[key][param_estm][i]

        new_structure.append(new_dict)

    k = 0
    for j in new_structure:
        create_traj_graph(j, key, param_estm, k)
        k += 1
