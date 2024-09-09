import json
import sys, os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import random
import pandas as pd
from typing import Tuple, List, Dict
import math
import time

from src.domain_knowledge.domain_knowledge_classifier import DomainKnowledgeClassifier
from src.domain_knowledge.mathematical_model import ParameterEstimation
from src.data_inspection.logger import MainLogger
from src.tools.general_functions import map_data_variant_to_param_estm

from src.neural_network.utils import (
    CustomDataset,
    SimpleFNN,
)

from src.tools.enums import (
    DataVariant,
    ParameterEstimation,
    Similarity,
    InvertedSimilarity,
    DataSplit,
)


def custom_collate_fn(
    batch: List[
        Tuple[
            torch.Tensor,
            List,
            torch.Tensor,
            torch.Tensor,
            Dict,
            pd.DataFrame,
            str,
            float,
        ]
    ]
) -> Tuple[
    torch.Tensor,
    List,
    torch.Tensor,
    torch.Tensor,
    List[Dict],
    List[pd.DataFrame],
    List[str],
    List[float],
]:
    """
    A custom collate function for PyTorch DataLoader to process batches.

    This function processes each item in the batch and returns a tuple of batched items suitable for model input.

    Parameters:
        batch (List[Tuple[torch.Tensor, List, torch.Tensor, torch.Tensor, Dict, pd.DataFrame, str, float]]): A list of tuples, each tuple containing tensors, lists, labels, erroneous labels, settings, dataframes, data names and alpha's.

    Returns:
        Tuple[torch.Tensor, List, torch.Tensor, torch.Tensor, List[Dict], List[pd.DataFrame], List[str], List[float]]: A tuple containing batched tensors, lists, labels, erroneous labels, settings, dataframes, data names and alpha's.
    """
    tensors = [item[0] for item in batch]
    lists = [item[1] for item in batch]
    labels = [item[2] for item in batch]
    erroneous_labels = [item[3] for item in batch]
    settings_list = [item[4] for item in batch]
    df_lst = [item[5] for item in batch]
    data_name = [item[6] for item in batch]
    alpha = [item[7] for item in batch]

    tensors_batch = torch.stack(tensors, dim=0)
    labels_batch = torch.stack(labels, dim=0)
    erroneous_labels_batch = torch.stack(erroneous_labels, dim=0)

    return (
        tensors_batch,
        lists,
        labels_batch,
        erroneous_labels_batch,
        settings_list,
        df_lst,
        data_name,
        alpha,
    )


def set_seed(seed: int, device: str) -> None:
    """
    Sets the seed for various random number generators for reproducibility,
    and adjusts settings based on the computing device.

    Parameters:
        seed (int): The seed value to be used for all random number generators.
        device (str): The computing device, e.g., 'cpu' or 'cuda'.
    """

    # Set seed for CPU and libraries that are device-agnostic
    np.random.seed(seed)
    random.seed(seed)

    # Check if the device is a GPU (cuda)
    if device.type == "cuda":
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # These settings are relevant for CUDA devices only
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_input_data(master_data_file: str, info_data_file: str) -> Tuple[
    Dict[str, pd.DataFrame],
    Dict[str, pd.DataFrame],
    Dict[str, pd.DataFrame],
    Dict[str, pd.DataFrame],
]:
    """
    Parses input data from JSON files to generate dictionaries of DataFrame objects.

    Parameters:
        master_data_file (str): Path to the master data JSON file.
        info_data_file (str): Path to the dataset information JSON file.

    Returns:
        Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]: A tuple containing dictionaries for train, validate, test data and unlabelled sets.
    """

    def generate_dict(data: Dict, df: pd.DataFrame, unlabelled=False) -> Dict:
        filtered_dict = {}
        for key in data.keys():
            if key in df.index:
                filtered_dict[key] = data[key]
                row = df.loc[key]
                filtered_dict[key]["neural_network_input"] = row["data_variant"]
                filtered_dict[key]["unlabelled"] = unlabelled

        return filtered_dict

    def get_single_item(lst):
        if len(lst) > 1:
            raise ValueError(f"List must contain exactly one item: {lst}")
        elif len(lst) == 1:
            return lst[0]
        else:
            return ""

    # Open the JSON file and read its contents
    with open(master_data_file, "r") as f:
        data = json.load(f)

    with open(
        info_data_file,
        "r",
    ) as f:
        dataset_info = json.load(f)

    # Process dataset data
    train_df = pd.DataFrame(dataset_info["train"])
    validate_df = pd.DataFrame(dataset_info["validate"])
    test_df = pd.DataFrame(dataset_info["test"])
    unlabelled_df = pd.DataFrame(dataset_info["unlabelled"])

    train_data_variant = get_single_item(train_df["data_variant"].unique())
    validate_data_variant = get_single_item(validate_df["data_variant"].unique())
    test_data_variant = get_single_item(test_df["data_variant"].unique())
    unlabelled_data_variant = get_single_item(unlabelled_df["data_variant"].unique())

    train_dict = generate_dict(data, train_df)
    validate_dict = generate_dict(data, validate_df)
    test_dict = generate_dict(data, test_df)
    unlabelled_dict = generate_dict(data, unlabelled_df, unlabelled=True)

    return (
        train_dict,
        validate_dict,
        test_dict,
        unlabelled_dict,
        train_data_variant,
        validate_data_variant,
        test_data_variant,
        unlabelled_data_variant,
    )


def check_values_uniformity(values: list) -> float:
    """
    Check if all values in the list are the same and return that value.
    If the values are not all the same, raise a ValueError.

    :param values: List of values to be checked.
    :return: The common value if all values are the same.
    :raises ValueError: If the values in the list are not all the same.
    """
    if not values:  # Check if the list is empty
        raise ValueError("The list is empty.")

    first_value = values[0]
    for value in values:
        if value != first_value:
            raise ValueError("Not all values in the list are the same.")

    return first_value


def reorder_values_tensor(values):
    """
    Reorders values in a tensor into a binary format based on their magnitude.

    Parameters:
        values (torch.Tensor): A tensor of shape (1, 2) containing two values.

    Returns:
        torch.Tensor: A tensor of shape (1, 2) containing 1 and 0. 1 is placed at the index of the larger value.
    """
    # Check the condition directly on the tensor elements
    if values[0, 0] > values[0, 1]:
        return torch.tensor([[1, 0]], dtype=values.dtype)
    else:
        return torch.tensor([[0, 1]], dtype=values.dtype)


class TrainNeuralNetwork:
    def __init__(
        self,
        master_data_file_path: str,
        dataset_path: str,
        dataset_config_file_path: str,
        embed_domain_knowledge: bool,
        num_epochs: int,
        error_level: float,
        plotting: bool,
        param_bounds: Dict,
        batch_size: int,
        log_path: str,
        dir_name_mm: str,
        temperature: float,
        alpha: float,
    ) -> None:
        self.master_data_file_path = master_data_file_path
        self.dataset_path = dataset_path
        self.dataset_config_file_path = dataset_config_file_path
        self.embed_domain_knowledge = embed_domain_knowledge
        self.num_epochs = num_epochs
        self.error_level = error_level
        self.plotting = plotting
        self.param_bounds = param_bounds
        self.batch_size = batch_size
        self.log_path = log_path
        self.dir_name_mm = dir_name_mm
        self.temperature = temperature
        self.alpha = alpha

        self.model = self.initialization()
        self.train()

    def check_cuda(self, device):
        """
        Checks if CUDA is available and operational on the specified device by performing extensive tests on the GPU.
        Returns True if CUDA is operational on the device, False otherwise.
        """
        if not torch.cuda.is_available():
            print(f"CUDA is not available on device {device}.")
            return False

        cuda_operational = True

        try:
            # Memory allocation test on specified CUDA device
            large_tensor = torch.rand((10000, 10000), device=device)
            print(f"Memory allocation on {device} succeeded.")

            # Simple computation test (e.g., matrix multiplication)
            computation_result = torch.matmul(large_tensor, large_tensor)
            print(f"Simple computation on {device} succeeded.")

            # Data transfer test (CPU to GPU and back)
            start_time = time.time()
            _ = large_tensor.to("cpu")
            _ = large_tensor.to(device)
            end_time = time.time()
            print(
                f"Data transfer to and from {device} succeeded, took {end_time - start_time:.2f} seconds."
            )

        except Exception as e:
            print(f"CUDA encountered an error during extensive tests on {device}: {e}")
            cuda_operational = False

        return cuda_operational

    def get_device(self):
        """
        Loops through all available CUDA devices, checks each with check_cuda(),
        and returns the CUDA device with the most available memory. Falls back to CPU if no suitable GPU is found.
        """

        max_memory = 0
        max_memory_dev_idx = 0
        selected_device = torch.device("cpu")

        for i in range(torch.cuda.device_count()):
            device = torch.device(f"cuda:{i}")
            if self.check_cuda(device):
                torch.cuda.synchronize(device=device)
                free_memory = torch.cuda.get_device_properties(
                    i
                ).total_memory - torch.cuda.memory_allocated(i)
                print(
                    f"Device {device} has {free_memory // (1024 ** 2)} MB free memory."
                )
                if free_memory > max_memory:
                    max_memory = free_memory
                    selected_device = device
                    max_memory_dev_idx = i

        if selected_device.type == "cuda":
            torch.cuda.set_device(max_memory_dev_idx)
            print(f"Selected device is {selected_device} with the most free memory.")
        else:
            print(
                "No operational CUDA device with sufficient memory was found, falling back to CPU."
            )

        return selected_device

    def initialization(self):
        self.max_input_size = 1000

        learning_rate = 0.0001

        random_seed = 41

        self.classes = {"plastic": 0, "fish": 1}

        # Check if GPU is available
        self.device = self.get_device()

        # Set the random seed for reproducibility
        set_seed(random_seed, self.device)

        ####################################################
        ################## Model settings ##################
        ####################################################

        # Create the neural network
        model = SimpleFNN(self.max_input_size, len(self.classes), self.device)

        # Loss function and optimizer
        self.criterion_ce = nn.CrossEntropyLoss()  # Original loss
        self.criterion_kl = nn.KLDivLoss(reduction="batchmean")  # Distillation loss
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        model.to(self.device)
        model.double()

        #####################################################
        ################## Dataset settings #################
        #####################################################

        (
            train_data,
            validate_data,
            test_data,
            unlabelled_data,
            data_variant_train,
            data_variant_validate,
            data_variant_test,
            data_variant_unlabelled,
        ) = parse_input_data(
            master_data_file=self.master_data_file_path,
            info_data_file=self.dataset_config_file_path,
        )

        self.param_estm_train = map_data_variant_to_param_estm(
            DataVariant(data_variant_train)
        )
        self.param_estm_validate = map_data_variant_to_param_estm(
            DataVariant(data_variant_validate)
        )
        self.param_estm_test = map_data_variant_to_param_estm(
            DataVariant(data_variant_test)
        )

        train_set = CustomDataset(
            json_data=train_data,
            parameter_estimation=self.param_estm_train,
            classes=self.classes,
            length=self.max_input_size,
            erroneous=self.error_level,
            unlabelled=unlabelled_data,
            alpha=self.alpha,
        )

        validate_set = CustomDataset(
            json_data=validate_data,
            parameter_estimation=self.param_estm_validate,
            classes=self.classes,
            length=self.max_input_size,
            erroneous=self.error_level,
            alpha=self.alpha,
        )

        test_set = CustomDataset(
            json_data=test_data,
            parameter_estimation=self.param_estm_test,
            classes=self.classes,
            length=self.max_input_size,
            erroneous=self.error_level,
            alpha=self.alpha,
        )

        # Create DataLoaders for the training and validation sets
        self.train_loader = DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=custom_collate_fn,
        )
        self.val_loader = DataLoader(
            validate_set,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=custom_collate_fn,
        )
        self.test_loader = DataLoader(
            test_set,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=custom_collate_fn,
        )

        self.main_logger = MainLogger(self.log_path)

        return model

    def train(self):
        lowest_validation_loss = 100.0
        lowest_validation_loss_epoch = 0
        loss_increase_counter = 0
        previous_loss = 0
        highest_validation_accuracy = 0

        nn_times = []
        mm_times = []

        for epoch in range(self.num_epochs):
            epoch_logger_train = self.main_logger.log_epoch(DataSplit.TRAIN)
            self.epoch_logger_validate = self.main_logger.log_epoch(DataSplit.VALIDATE)

            self.model.train()

            train_loss_tot = 0.0
            train_loss_ce = 0.0
            train_loss_kl = 0.0

            tot_num = 0
            ce_order_higher = 0
            kl_higher = 0

            for (
                trajectories_tnr,
                trajectories_dict,
                labels,
                erroneous_labels,
                settings,
                mm_inputs,
                data_numbers,
                alpha,
            ) in self.train_loader:
                parsed_alpha = check_values_uniformity(alpha)

                if parsed_alpha == 1 and self.embed_domain_knowledge == False:
                    continue

                # Batch logger object to log all the data of 1 batch
                batch_logger = epoch_logger_train.log_batch()
                ground_truth = labels

                # Change labels when value is higher than 0
                if self.error_level > 0:
                    labels = erroneous_labels

                trajectories_tnr, labels = (
                    trajectories_tnr.to(self.device),
                    labels.to(self.device),
                )

                # Domain knowledge classifier aka Teacher Model
                dk_classifier_train = DomainKnowledgeClassifier(
                    mm_input=self.param_estm_train,
                    param_bounds=self.param_bounds,
                    mm_model_folder=os.path.join(self.dataset_path, self.dir_name_mm),
                    similarity_parameter=InvertedSimilarity.WEIGHTED_MSE,
                )
                if epoch == 0 & self.embed_domain_knowledge:
                    start_time_mm = time.time()
                teacher_outputs_train = dk_classifier_train.get_teacher_output(
                    batched_traj=trajectories_dict,
                    batched_settings=settings,
                    input_trajects=mm_inputs,
                    labels=labels,
                    data_numbers=data_numbers,
                )
                teacher_outputs_train = teacher_outputs_train.to(self.device)

                teacher_outputs_soft_train = teacher_outputs_train / self.temperature
                if epoch == 0 & self.embed_domain_knowledge:
                    end_time_mm = time.time()
                    total_time_mm = end_time_mm - start_time_mm
                    print(total_time_mm)
                    mm_times.append(total_time_mm)

                batch_logger.log_batched_data(
                    "similarities", dk_classifier_train.similarities
                )
                batch_logger.log_batched_data(
                    "parameter_estimation", dk_classifier_train.param_estm
                )

                batch_logger.log_batched_data("data_point_id", data_numbers)

                self.optimizer.zero_grad()

                if epoch == 0 & self.embed_domain_knowledge:
                    start_time_nn = time.time()
                # Forward pass for student model
                student_outputs_train = self.model(trajectories_tnr)

                if epoch == 0 & self.embed_domain_knowledge:
                    end_time_nn = time.time()
                    total_time_nn = end_time_nn - start_time_nn
                    nn_times.append(total_time_nn)

                # Soften logits with temperature
                student_outputs_soft_train = student_outputs_train / self.temperature

                # Calculate loss
                labels = labels.long()

                loss_ce = self.criterion_ce(student_outputs_train, labels)

                batch_logger.log_batched_data("labels", ground_truth)
                batch_logger.log_batched_data(
                    "student_outputs", student_outputs_soft_train
                )
                batch_logger.log_batched_data(
                    "softmax_student",
                    nn.functional.softmax(student_outputs_soft_train, dim=1),
                )

                if self.embed_domain_knowledge:
                    loss_kl = self.criterion_kl(
                        nn.functional.log_softmax(student_outputs_soft_train, dim=1),
                        nn.functional.softmax(teacher_outputs_soft_train, dim=1),
                    )

                    loss = (
                        1 - parsed_alpha
                    ) * loss_ce + parsed_alpha * loss_kl * self.temperature**2

                    # print(f"loss_ce: {loss_ce}")
                    # print(f"loss_kl: {loss_kl}")
                    # print(f"ce loss: {loss_ce}, kl loss: {loss_kl*self.temperature**2}")

                    loss_kl = loss_kl * self.temperature**2

                    tot_num += 1

                    # print(f"loss_ce: {loss_ce}")
                    # print(f"loss_kl: {loss_kl}")

                    if loss_ce > loss_kl:
                        ce_order_higher += 1
                    if loss_kl > loss_ce:
                        kl_higher += 1

                else:
                    loss = loss_ce
                    loss_kl = torch.tensor(0)

                loss.backward()
                self.optimizer.step()

                train_loss_tot += loss.item()
                train_loss_ce += loss_ce.item()
                train_loss_kl += loss_kl.item()

                batch_logger.log_batched_data(
                    "softmax_teacher",
                    nn.functional.softmax(teacher_outputs_soft_train, dim=1),
                )

                batch_logger.log_batched_data(
                    "teacher_outputs", teacher_outputs_soft_train
                )

                batch_logger.log_batched_data(
                    "log_softmax_student",
                    nn.functional.log_softmax(student_outputs_soft_train, dim=1),
                )
                batch_logger.log_losses(
                    tot_loss=train_loss_tot,
                    kl_loss=train_loss_kl,
                    ce_loss=train_loss_ce,
                )
                epoch_logger_train.save_batch()

            train_loss_tot /= len(self.train_loader)
            train_loss_ce /= len(self.train_loader)
            train_loss_kl /= len(self.train_loader)

            epoch_logger_train.save_data("loss_tot", train_loss_tot)
            epoch_logger_train.save_data("loss_ce", train_loss_ce)
            epoch_logger_train.save_data("loss_kl", train_loss_kl)

            validate_loss_tot, validate_accuracy = self.validate(self.model)
            print(
                f"Epoch [{epoch+1}/{self.num_epochs}], Train loss: {train_loss_tot:.4f} , Validate loss: {validate_loss_tot:.4f}, Accuracy: {validate_accuracy:.2f}%"
            )

            print(f"tot_num is {tot_num}")
            print(f"ce loss higher {ce_order_higher}")
            print(f"kl loss higher {kl_higher}")

            if validate_loss_tot <= lowest_validation_loss:
                lowest_validation_loss = validate_loss_tot
                best_weights = self.model.state_dict()
                lowest_validation_loss_epoch = epoch + 1
            print(f"lowest_validation_loss_epoch is {lowest_validation_loss_epoch}")

            if validate_accuracy >= highest_validation_accuracy:
                highest_validation_accuracy = validate_accuracy

            self.main_logger.save_epoch_data(epoch_logger_train, epoch)
            self.main_logger.save_epoch_data(self.epoch_logger_validate, epoch)

            if validate_loss_tot > previous_loss:
                loss_increase_counter += 1

            else:
                loss_increase_counter = 0

            if loss_increase_counter >= 6:
                print("Stopped training early due to validation loss increasing")
                break

            previous_loss = validate_loss_tot

        print(
            f"Best model validation set - Epoch: {lowest_validation_loss_epoch}, Accuracy: {highest_validation_accuracy:.2f}%"
        )

        best_model_path = os.path.join(
            self.log_path, f"best_model_val_acc_{validate_accuracy}%.pt"
        )
        torch.save(best_weights, best_model_path)

        test_accuracy = self.test(
            lowest_accuracy_loss_epoch=lowest_validation_loss_epoch,
            best_model_path=best_model_path,
        )

        print(f"Best model test set - Accuracy: {test_accuracy:.2f}%")
        print(f"Average time to run neural network: {sum(nn_times)/len(nn_times)}")
        print(f"Average time to run dk knowledge: {sum(mm_times)/len(mm_times)}")

        self.main_logger.save_test_data()
        self.main_logger.save_data()
        self.main_logger.create_graphs()

    def validate(self, model):
        # Evaluation
        model.eval()
        val_correct = 0
        val_total = 0

        validate_loss_tot = 0.0
        validate_loss_ce = 0.0
        validate_loss_kl = 0.0

        with torch.no_grad():
            for (
                trajectories,
                trajectories_dict,
                labels,
                _,
                settings,
                mm_inputs,
                data_numbers,
                alpha,
            ) in self.val_loader:
                trajectories_val, labels_val = trajectories.to(self.device), labels.to(
                    self.device
                )
                self.optimizer.zero_grad()

                outputs_val = model(trajectories_val)

                batch_logger = self.epoch_logger_validate.log_batch()

                _, predicted_val = torch.max(outputs_val, 1)

                val_total += labels_val.size(0)
                val_correct += (predicted_val == labels_val).sum().item()

                # Domain knowledge classifier aka Teacher Model
                dk_classifier_validate = DomainKnowledgeClassifier(
                    mm_input=self.param_estm_validate,
                    param_bounds=self.param_bounds,
                    mm_model_folder=os.path.join(self.dataset_path, self.dir_name_mm),
                    similarity_parameter=InvertedSimilarity.WEIGHTED_MSE,
                )
                teacher_outputs_validate = dk_classifier_validate.get_teacher_output(
                    batched_traj=trajectories_dict,
                    batched_settings=settings,
                    input_trajects=mm_inputs,
                    labels=labels_val,
                    data_numbers=data_numbers,
                )
                teacher_outputs_validate = teacher_outputs_validate.to(self.device)

                teacher_outputs_soft_validate = (
                    teacher_outputs_validate / self.temperature
                )

                # Calculate loss
                labels_val = labels_val.long()
                loss_ce = self.criterion_ce(outputs_val, labels_val)
                parsed_alpha = check_values_uniformity(alpha)

                if self.embed_domain_knowledge:
                    loss_kl = self.criterion_kl(
                        nn.functional.log_softmax(outputs_val, dim=1),
                        nn.functional.softmax(teacher_outputs_soft_validate, dim=1),
                    )

                    loss = (
                        1 - parsed_alpha
                    ) * loss_ce + self.temperature**2 * parsed_alpha * loss_kl
                    # loss = loss_ce + loss_kl

                else:
                    loss = loss_ce
                    loss_kl = torch.tensor(0)

                # Logging training data
                batch_logger.log_batched_data(
                    "similarities", dk_classifier_validate.similarities
                )
                batch_logger.log_batched_data(
                    "parameter_estimation", dk_classifier_validate.param_estm
                )

                batch_logger.log_batched_data("data_point_id", data_numbers)

                batch_logger.log_batched_data(
                    "teacher_outputs", teacher_outputs_soft_validate
                )
                batch_logger.log_batched_data(
                    "softmax_teacher",
                    nn.functional.softmax(teacher_outputs_soft_validate, dim=1),
                )

                batch_logger.log_batched_data("student_outputs", outputs_val)
                batch_logger.log_batched_data(
                    "softmax_student", nn.functional.softmax(outputs_val, dim=1)
                )

                batch_logger.log_batched_data("labels", labels)

                self.epoch_logger_validate.save_batch()

                validate_loss_tot += loss.item()
                validate_loss_ce += loss_ce.item()
                validate_loss_kl += loss_kl.item()

            validate_accuracy = 100 * val_correct / val_total

            validate_loss_tot /= len(self.val_loader)
            validate_loss_ce /= len(self.val_loader)
            validate_loss_kl /= len(self.val_loader)

            self.epoch_logger_validate.save_data("loss_tot", validate_loss_tot)
            self.epoch_logger_validate.save_data("loss_ce", validate_loss_ce)
            self.epoch_logger_validate.save_data("loss_kl", validate_loss_kl)

            self.epoch_logger_validate.save_data("accuracy", validate_accuracy)

        return validate_loss_tot, validate_accuracy

    def test(self, lowest_accuracy_loss_epoch, best_model_path):
        # Perform evaluation
        test_correct = 0
        test_total = 0

        test_model = SimpleFNN(self.max_input_size, len(self.classes), self.device)
        test_model.load_state_dict(torch.load(best_model_path))
        test_model.to(self.device)
        test_model.to(torch.float64)
        test_model.eval()

        test_logger = self.main_logger.log_test(lowest_accuracy_loss_epoch)

        with torch.no_grad():
            for (
                trajectories_tnr,
                trajectories_dict,
                labels,
                _,
                settings,
                mm_inputs,
                data_numbers,
                _,
            ) in self.test_loader:
                trajectories_tnr, labels = trajectories_tnr.to(self.device), labels.to(
                    self.device
                )
                self.optimizer.zero_grad()
                outputs = test_model(trajectories_tnr)

                batch_logger = test_logger.log_batch()

                _, predicted = torch.max(outputs, 1)

                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

                # Domain knowledge classifier aka Teacher Model
                dk_classifier_test = DomainKnowledgeClassifier(
                    mm_input=self.param_estm_test,
                    param_bounds=self.param_bounds,
                    mm_model_folder=os.path.join(self.dataset_path, self.dir_name_mm),
                    similarity_parameter=InvertedSimilarity.WEIGHTED_MSE,
                )
                teacher_outputs_test = dk_classifier_test.get_teacher_output(
                    batched_traj=trajectories_dict,
                    batched_settings=settings,
                    input_trajects=mm_inputs,
                    labels=labels,
                    data_numbers=data_numbers,
                )
                teacher_outputs_test = teacher_outputs_test.to(self.device)
                teacher_outputs_soft_test = teacher_outputs_test / self.temperature

                batch_logger.log_batched_data(
                    "similarities", dk_classifier_test.similarities
                )

                batch_logger.log_batched_data("data_point_id", data_numbers)

                batch_logger.log_batched_data(
                    "parameter_estimation", dk_classifier_test.param_estm
                )
                batch_logger.log_batched_data(
                    "teacher_outputs", teacher_outputs_soft_test
                )
                batch_logger.log_batched_data("student_outputs", outputs)
                batch_logger.log_batched_data(
                    "softmax_student", nn.functional.softmax(outputs, dim=1)
                )
                batch_logger.log_batched_data(
                    "softmax_teacher",
                    nn.functional.softmax(teacher_outputs_soft_test, dim=1),
                )

                batch_logger.log_batched_data("labels", labels)

                test_logger.save_batch()

            test_accuracy = 100 * test_correct / test_total

            test_logger.save_data("test_accuracy", test_accuracy)

        return test_accuracy
