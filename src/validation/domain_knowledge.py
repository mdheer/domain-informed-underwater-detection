from src.validation.utils import Comparison, Validation
from src.domain_knowledge.domain_knowledge_classifier import DomainKnowledgeClassifier
from src.train_trajectory import extract_param_bounds
from src.neural_network.utils import CustomDataset
from src.neural_network.main import custom_collate_fn, parse_input_data
from torch.utils.data import DataLoader
from src.tools.enums import InvertedSimilarity, ParameterEstimation
import torch.nn as nn
import torch
from itertools import chain

import os
import json


# input_path = r"M:\underwater_simulator\parsed_footage_v2023-10-24\footage\ssmall_parsed_data.json"
input_path = r"/data/mdheer/input_data/merged_v2023-10-24_v2023-11-02/parsed_data.json"

# Extract param bounds
param_bounds = extract_param_bounds(input_path)

# input_data = {
#     "optical_flow": r"C:\Users\mathi\Documents\Files\University of Technology Delft\Master\MSc 2\Thesis\Code\thesis_mathieu\datasets\val_optical_flow\k_fold_distribution_1",
#     "gaussian_noise": r"C:\Users\mathi\Documents\Files\University of Technology Delft\Master\MSc 2\Thesis\Code\thesis_mathieu\datasets\val_gaussian_noise\k_fold_distribution_1",
#     "ideal": r"C:\Users\mathi\Documents\Files\University of Technology Delft\Master\MSc 2\Thesis\Code\thesis_mathieu\datasets\val_ideal\k_fold_distribution_1",
# }

input_data = {
    "optical_flow": r"/home/mdheer/thesis_mathieu/datasets/val_optical_flow/k_fold_distribution_1",
    "gaussian_noise": r"/home/mdheer/thesis_mathieu/datasets/val_gaussian_noise/k_fold_distribution_1",
    "ideal": r"/home/mdheer/thesis_mathieu/datasets/val_ideal/k_fold_distribution_1",
}

final_path = r"./src/validation/dk_val_results_cluster.json"
# final_path = r".\src\validation\dk_val_results.json"


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
            batch_size=5,
            shuffle=False,
            collate_fn=custom_collate_fn,
        )

        tot_labels = []
        tot_similarities = []
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

            tot_labels.append(labels)
            tot_similarities.append(dk_classifier.merged_similarities)

        tot_similarities = merge_list_to_dict(tot_similarities)

        final_results[key][param_estm] = {}
        final_dict = parse_metric_to_accuracy(tot_similarities, tot_labels)

        for metric, value in final_dict.items():
            print(
                f"Data: {key} || {param_estm} || Metric: {metric} - accuracy: {value['accuracy']} %"
            )
            final_results[key][param_estm][metric] = value

with open(final_path, "w") as file:
    json.dump(stringify_keys(final_results), file, indent=4)
