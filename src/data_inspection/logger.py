from torch.utils.tensorboard import SummaryWriter

import json
import os
import torch
import os
import pandas as pd

from src.tools.enums import DataSplit
from src.data_inspection.plotting import PlotTrainingResults


class MathematicalModelBehaviourChecker:
    """
    A class for analyzing and evaluating the behavior of two models, a mathematical model and a neural network, using provided data.

    This class focuses on comparing the performance of the mathematical model and the neural network in terms of accuracy
    and other metrics by analyzing the given dataset. It is particularly useful in scenarios where an understanding of
    each model's decision-making process and accuracy is crucial, and where comparing the performance of different types
    of models is needed.

    Attributes:
        data (dict): The dataset used for analysis, typically containing model predictions and actual labels.
        df (pd.DataFrame): A DataFrame generated from the dataset, structured for analysis.
        percentages (dict): A dictionary holding calculated performance metrics.

    Methods:
        _batch_parser: Parses a batch of data into a structured list format.
        _create_df: Creates a DataFrame from the input data for analysis.
        _correct_prediction: Determines if a model's prediction is correct based on probabilities and labels.
        _perform_analysis: Analyzes the model's predictions, calculating and printing various metrics.

    Parameters:
        data (dict): The dataset used for analysis, typically containing model predictions and actual labels.

    Example Usage:
        >>> checker = MathematicalModelBehaviourChecker(data)
        >>> checker._perform_analysis() # This will print the analysis results and store metrics in checker.percentages
    """

    def __init__(self, data: dict) -> None:
        """
        Initializes the MathematicalModelBehaviourChecker with the provided dataset.

        This constructor method sets up the class by initializing its attributes with the provided data.
        It also creates a DataFrame from the data and performs initial analysis.

        Parameters:
            data (dict): The dataset used for analysis, typically containing model predictions and actual labels.
        """
        self.data = data

        self.df = self._create_df()

        self._perform_analysis()

    def _batch_parser(self, data: dict, batch_name: str) -> list:
        """
        Parses a batch of data into a structured list format for easier analysis.

        This private method processes a batch of data from the dataset, structuring it into a list of dictionaries.
        Each dictionary contains specific information extracted from the batch, useful for analysis.

        Parameters:
            data (dict): The batch data to parse, typically containing model predictions and actual labels.
            batch_name (str): The name of the batch, used for identifying the data.

        Returns:
            list: A list of dictionaries, each representing a parsed data point from the batch.
        """
        parsed_lst = []
        cols = data.keys()

        for name in cols:
            if "data_point" in name:
                subdict = {
                    "datapoint_name": batch_name + "_" + name,
                    "student_plastic_prob": data[name]["softmax_student"][0],
                    "student_fish_prob": data[name]["softmax_student"][1],
                    "teacher_plastic_prob": data[name]["softmax_teacher"][0],
                    "teacher_fish_prob": data[name]["softmax_teacher"][1],
                    "label": data[name]["labels"],
                }

                parsed_lst.append(subdict)

        return parsed_lst

    def _create_df(self) -> pd.DataFrame:
        """
        Creates a DataFrame from the input data for comprehensive analysis.

        This private method compiles all batches of data into a single DataFrame. The DataFrame format
        facilitates easier manipulation and analysis of the data.

        Returns:
            pd.DataFrame: A DataFrame containing data from all the batches, structured for analysis.
        """

        cols = self.data.keys()
        all_batches = []

        for name in cols:
            if "batch" in name:
                batch_data = self.data[name]
                parsed_data = self._batch_parser(batch_data, name)
                all_batches.extend(parsed_data)

        df = pd.DataFrame(all_batches)
        return df

    def _correct_prediction(self, plastic: float, fish: float, label: int) -> bool:
        """
        Determines if a model's prediction is correct based on probabilities for classes.

        This private method checks if the predicted label matches the actual label, based on the comparison
        of predicted probabilities for different classes (plastic and fish in this case).

        Parameters:
            plastic (float): The model's predicted probability for the 'plastic' class.
            fish (float): The model's predicted probability for the 'fish' class.
            label (int): The actual label of the data point (0 for plastic, 1 for fish).

        Returns:
            bool: True if the model's prediction is correct; False otherwise.
        """
        if plastic >= fish:
            if label == 0:
                return True
            else:
                return False

        elif plastic < fish:
            if label == 1:
                return True
            else:
                return False

    def _perform_analysis(self):
        """
        Performs an in-depth analysis of the model's predictions and prints the outcomes.

        This method compares the predictions made by a mathematical model (mm) and a neural network (nn)
        against the actual labels, and calculates various performance metrics. It assesses the accuracy of
        each model separately, as well as instances where both models are either correct or incorrect simultaneously.
        Additionally, it identifies cases where only one model predicts correctly. The results, including accuracies
        and percentages for different scenarios, are printed and stored in the class attribute 'percentages'.

        The method leverages the DataFrame `df`, which contains the model predictions and actual labels,
        to perform this analysis. It adds two new columns to `df` to indicate whether each model's prediction
        was correct for each data point.

        Outputs:
            The method prints the following metrics:
            - Accuracy of the mathematical model and the neural network.
            - Percentage of instances where both models are correct or incorrect.
            - Percentage of instances where only one of the models is correct.

        Side Effects:
            - Modifies `self.df` by adding 'correct_mm' and 'correct_nn' columns.
            - Sets `self.percentages` with the calculated performance metrics.
        """
        # Process data
        self.df["correct_mm"] = self.df.apply(
            lambda x: self._correct_prediction(
                plastic=x["teacher_plastic_prob"],
                fish=x["teacher_fish_prob"],
                label=x["label"],
            ),
            axis=1,
        )
        self.df["correct_nn"] = self.df.apply(
            lambda x: self._correct_prediction(
                plastic=x["student_plastic_prob"],
                fish=x["student_fish_prob"],
                label=x["label"],
            ),
            axis=1,
        )

        tot_len = len(self.df)
        # Create metrics
        mm_true = self.df.loc[self.df["correct_mm"]]
        nn_true = self.df.loc[self.df["correct_nn"]]
        both_true = self.df.loc[
            (self.df["correct_mm"] == True) & (self.df["correct_nn"] == True)
        ]
        both_false = self.df.loc[
            (self.df["correct_mm"] == False) & (self.df["correct_nn"] == False)
        ]
        only_nn_true = self.df.loc[
            (self.df["correct_mm"] == False) & (self.df["correct_nn"] == True)
        ]
        only_mm_true = self.df.loc[
            (self.df["correct_mm"] == True) & (self.df["correct_nn"] == False)
        ]

        accuracy_mm = round((len(mm_true) / tot_len) * 100, 2)
        accuracy_nn = round((len(nn_true) / tot_len) * 100, 2)
        both_false_perc = round((len(both_false) / tot_len) * 100, 2)
        both_true_perc = round((len(both_true) / tot_len) * 100, 2)

        mm_true_nn_false = round((len(only_mm_true) / tot_len) * 100, 2)
        mm_false_nn_true = round((len(only_nn_true) / tot_len) * 100, 2)

        print(
            f"    -- Mathematical model accuracy: {accuracy_mm}%, neural network accuracy: {accuracy_nn}%"
        )
        print(f"    -- Both true: {both_true_perc}%, both false: {both_false_perc}%")
        print(
            f"    -- Mathematical true, neural network false: {mm_true_nn_false}%, Mathematical false, neural network true: {mm_false_nn_true}%"
        )

        self.percentages = {
            "perc_mm_correct": accuracy_mm,
            "perc_nn_correct": accuracy_nn,
            "perc_mm_only_correct": mm_true_nn_false,
            "perc_nn_only_correct": mm_false_nn_true,
        }


class BatchLogger:
    """
    A utility class for logging data from batches of datapoints in machine learning contexts.

    This class provides methods to log different types of data, such as batched data points and losses,
    which are commonly used in training and evaluating machine learning models.

    Methods:
        __init__(self) -> None:
            Initializes the BatchLogger with an empty dictionary to store the data.

        log_batched_data(self, key: str, values: list) -> None:
            Logs data in a batch format, organizing them under unique keys for each data point.
            Useful for logging outputs or intermediate values during batch processing, such as
            activations or predictions.

        log_losses(self, tot_loss: float, kl_loss: float, ce_loss: float) -> None:
            Logs different types of losses typically used in training machine learning models.
            Useful for tracking the progress of model training, especially when using multiple
            loss components like KL divergence and cross-entropy loss.

    Attributes:
        data (dict): A dictionary to store logged data. The keys represent different types of data
                     (e.g., individual data points, losses), and the values are the corresponding data.
    """

    def __init__(self) -> None:
        """Initializes the BatchLogger with an empty dictionary to store the data."""
        self.data = {}

    def log_batched_data(self, key: str, values: list) -> None:
        """
        Logs data in a batch format, organizing them under unique keys for each data point.

        This method is particularly useful for logging outputs or intermediate values during
        batch processing in machine learning, such as activations or predictions.

        Parameters:
            key (str): A string representing the name under which the batched data should be logged.
                       This could be, for example, the name of a model layer or a specific type of output.
            values (list): A list of values representing the batched data. These could be raw values,
                           tensors, or any other format depending on the context. If a value is a tensor,
                           it is converted to a list for logging.

        Note:
            If a value in the 'values' list is a PyTorch tensor, it is converted to a list using the
            `tolist()` method. This is necessary for serialization and easier logging.
        """
        for idx, output in enumerate(values):
            batch_key = f"data_point_{idx}"
            if batch_key not in self.data:
                self.data[batch_key] = {}

            if torch.is_tensor(output):
                self.data[batch_key][key] = output.tolist()
            else:
                self.data[batch_key][key] = output

    def log_losses(self, tot_loss: float, kl_loss: float, ce_loss: float) -> None:
        """
        Logs different types of losses typically used in training machine learning models.

        This method is useful for tracking the progress of model training, especially in scenarios
        where multiple loss components are used, such as a combination of KL divergence and
        cross-entropy loss.

        Parameters:
            tot_loss (float): The total loss, typically a combination of KL and CE losses. This is
                              a single scalar value representing the overall loss for a batch or an epoch.
            kl_loss (float): The Kullback-Leibler (KL) divergence loss. This is a measure of how one
                             probability distribution diverges from a second, expected probability distribution.
            ce_loss (float): The cross-entropy (CE) loss. This loss measures the performance of a
                             classification model whose output is a probability value between 0 and 1.
        """
        self.data["combined_loss"] = tot_loss
        self.data["kl_loss"] = kl_loss
        self.data["ce_loss"] = ce_loss


class EpochLogger:
    """
    A class designed for logging and managing data associated with a specific epoch during the training or evaluation of a machine learning model.

    This class facilitates the tracking and storage of data generated in each batch within an epoch, such as model outputs, metrics, or other relevant information. It is particularly useful for monitoring and analyzing the performance and behavior of a model during an individual epoch.

    Attributes:
        train_batch_number (int): Counter to keep track of the number of batches processed in the current epoch.
        data_split (DataSplit): Type of data split (e.g., 'train', 'validate') indicating the phase of the epoch.
        data (dict): A dictionary to store the logged data for each batch and additional epoch-specific information.

    Methods:
        __init__: Initializes the EpochLogger with a specific data split type.
        log_batch: Prepares to log a new batch's data.
        save_batch: Saves the data of the currently logged batch.
        save_data: Stores additional data relevant to the entire epoch.

    Parameters:
        data_split (DataSplit): An enum or similar type indicating the phase of the epoch, such as 'train' or 'validate'.
    """

    def __init__(self, data_split: DataSplit) -> None:
        """
        Initializes the EpochLogger with the provided data split.

        This constructor sets up the class by initializing its attributes, particularly setting the data split type and preparing the data storage structure.

        Parameters:
            data_split (DataSplit): The type of data split (e.g., 'train', 'validate') indicating the phase of the epoch. This helps categorize and manage epoch-specific data effectively.
        """
        self.train_batch_number = 1
        self.data_split = data_split.value
        self.data = {}

    def log_batch(self) -> BatchLogger:
        """
        Prepares for logging a new batch's data by initializing a BatchLogger.

        This method is called at the beginning of a new batch processing within an epoch. It creates a new BatchLogger instance to log data specific to the current batch.

        Returns:
            BatchLogger: An instance of BatchLogger for logging batch-specific data.
        """
        self.batch_logger = BatchLogger()
        return self.batch_logger

    def save_batch(self) -> None:
        """
        Saves the data of the currently logged batch into the epoch's data.

        After a batch is processed and its data is logged, this method should be called to store the logged data into the epoch's central data storage. It updates the 'data' attribute with the current batch's data and increments the batch counter.

        Side Effects:
            - The 'data' attribute is updated with the current batch's logged data.
            - 'train_batch_number' is incremented by 1 to reflect the processing of a new batch.
        """
        self.data[f"batch_{self.train_batch_number}"] = self.batch_logger.data
        self.train_batch_number += 1

    def save_data(self, key: str, data) -> None:
        """
        Stores additional data relevant to the entire epoch under a specific key.

        This method can be used to log any epoch-level data, such as aggregated metrics or state information, which is not specific to individual batches but to the entire epoch.

        Parameters:
            key (str): The key under which the data should be stored. This should be a descriptive string that represents the nature of the data.
            data: The data to be stored. The type of this parameter can vary depending on the nature of the data (e.g., numeric values, lists, dictionaries).

        Side Effects:
            - The 'data' attribute is updated with the new key-value pair, adding the provided epoch-level data.
        """
        self.data[key] = data


class MainLogger:
    """
    Main class for logging data during a machine learning model's training run.

    This class is responsible for managing and storing various types of logs throughout the training process,
    including epoch-specific data and test results. It supports both basic and extensive logging, allowing
    for detailed analysis and evaluation of the model's performance across different training phases.

    Attributes:
        writer (SummaryWriter): Tool for logging to TensorBoard.
        log_path (str): Path where the log files are saved.
        extensive_logging (bool): Flag to enable or disable extensive logging messages.
        logs (dict): Stores raw logs for each epoch.
        results (dict): Aggregated results, organized by DataSplit type (train, validate, test).

    Methods:
        __init__: Initializes the MainLogger with a log path and extensive logging option.
        log_epoch: Initializes an EpochLogger for a specific data split.
        log_test: Prepares logging for test data based on the best epoch number.
        save_test_data: Saves and logs test data from the test logger.
        save_epoch_data: Stores data from an EpochLogger for a specific epoch.
        save_data: Saves final log files in JSON format.
        create_graphs: Generates graphs from the logged training data.

    Parameters:
        log_path (str): Path where the log should be saved.
        extensive_logging (bool, optional): Enables extensive logging messages if set to True. Defaults to True.
    """

    def __init__(self, log_path: str, extensive_logging: bool = True) -> None:
        """
        Initializes the MainLogger with the specified log path and extensive logging preference.

        Parameters:
            log_path (str): Path where the log files and results will be saved.
            extensive_logging (bool): If True, enables more detailed logging messages. Useful for in-depth analysis.
        """
        self.writer = SummaryWriter(log_dir=log_path)

        self.log_path = log_path
        self.extensive_logging = extensive_logging
        self.logs = {}
        self.results = {
            DataSplit.TRAIN.value: [],
            DataSplit.VALIDATE.value: [],
            DataSplit.TEST.value: [],
        }

    def log_epoch(self, data_split: DataSplit) -> EpochLogger:
        """
        Initializes an EpochLogger for logging data of a specific epoch based on the data split.

        Parameters:
            data_split (DataSplit): The data split type (e.g., train, validate, test) for the epoch.

        Returns:
            EpochLogger: An instance of EpochLogger for the specified data split.
        """
        return EpochLogger(data_split)

    def log_test(self, best_epoch: int) -> EpochLogger:
        """
        Prepares the logging for test data, based on the identified best epoch.

        Parameters:
            best_epoch (int): The epoch number identified as the best during the training process.

        Returns:
            EpochLogger: An instance of EpochLogger for logging test data.
        """
        self.test_epoch_num = best_epoch
        self.test_logger = EpochLogger(DataSplit.TEST)
        return self.test_logger

    def save_test_data(self) -> None:
        """
        Saves and logs the test data accumulated in the test logger.

        If extensive logging is enabled, this method also performs a detailed analysis of the test data,
        including calculating and logging various performance metrics using MathematicalModelBehaviourChecker.
        """
        if self.extensive_logging:
            print(f"  -{DataSplit.TEST.value}:")
            behaviour_checker = MathematicalModelBehaviourChecker(self.test_logger.data)
            self.results[DataSplit.TEST.value].append(behaviour_checker.percentages)

        self.logs["test"] = self.test_logger.data

    def save_epoch_data(self, epoch_logger, epoch_num) -> None:
        """
        Stores the data collected in an EpochLogger for a given epoch.

        Parameters:
            epoch_logger (EpochLogger): The EpochLogger instance containing the data of the epoch.
            epoch_num (int): The number of the epoch whose data is being logged.

        If extensive logging is enabled, this method performs additional analysis and logging of performance metrics.
        """
        # Attributes from epoch logger
        epoch_data = epoch_logger.data
        data_split = epoch_logger.data_split

        # Add data to logs
        epoch_key = f"epoch_{epoch_num+1}"
        if epoch_key not in self.logs:
            self.logs[epoch_key] = {}

        if data_split not in self.logs[epoch_key]:
            self.logs[epoch_key][data_split] = epoch_data

        # Parse data
        if self.extensive_logging:
            print(f"  -{data_split}:")
            behaviour_checker = MathematicalModelBehaviourChecker(epoch_data)
            data_to_graph = behaviour_checker.percentages
            data_to_graph["loss_tot"] = epoch_data["loss_tot"]
            data_to_graph["loss_ce"] = epoch_data["loss_ce"]
            data_to_graph["loss_kl"] = epoch_data["loss_kl"]

            self.results[data_split].append(behaviour_checker.percentages)

    def save_data(self) -> None:
        """
        Saves the final log files in JSON format to the specified log path.

        This method writes the raw logs and structured results to separate JSON files for later analysis.
        """
        file_path_raw = os.path.join(self.log_path, "training_results_raw.json")
        file_path_structured = os.path.join(
            self.log_path, "training_results_structured.json"
        )

        self.writer.close()
        # Save updated master JSON data
        with open(file_path_raw, "w") as f:
            json.dump(self.logs, f, indent=2)

        # Save updated master JSON data
        with open(file_path_structured, "w") as f:
            json.dump(self.results, f, indent=2)

    def create_graphs(self) -> None:
        """
        Generates and saves graphs visualizing the logged training data.

        This method creates a series of graphs for both loss and accuracy metrics, providing visual insights into the training process. It utilizes the `PlotTrainingResults` class to generate graphs for combined and separate metrics across different data splits (train, validate, test). The method focuses on two key aspects: loss metrics (total loss, cross-entropy loss, KL loss) and accuracy metrics (accuracies of mathematical model and neural network, and cases where only one model is correct).

        The graphs are designed to offer a clear visual comparison between different phases of training and to highlight the performance trends of the models.

        Side Effects:
            - Generates and saves multiple graphs in the specified log path.
            - Graphs include combined and separate plots for loss and accuracy metrics across different data splits.

        Notes:
            - This method assumes that 'self.results' and 'self.log_path' are already defined.
            - The method uses predefined color schemes for different metrics for consistency and clarity in the visual representation.
        """
        # Assuming 'self.results' and 'self.log_path' are already defined
        plot = PlotTrainingResults(
            results_dict=self.results,
            log_path=self.log_path,
            test_epoch=self.test_epoch_num,
        )

        # Plotting combined and separate losses
        loss_metrics = ["loss_tot", "loss_ce", "loss_kl"]
        colours = ["coral", "steelblue", "crimson"]
        plot.plot_combined_metric(
            [DataSplit.TRAIN, DataSplit.VALIDATE], loss_metrics, colours, "Loss"
        )
        plot.plot_separate_metric(DataSplit.TRAIN, loss_metrics, colours, "Loss")
        plot.plot_separate_metric(DataSplit.VALIDATE, loss_metrics, colours, "Loss")

        # Plotting combined and separate accuracies
        accuracy_metrics = [
            "perc_mm_correct",
            "perc_nn_correct",
            "perc_mm_only_correct",
            "perc_nn_only_correct",
        ]
        colours = ["coral", "steelblue", "crimson", "olivedrab"]
        plot.plot_combined_metric(
            [DataSplit.TRAIN, DataSplit.VALIDATE, DataSplit.TEST],
            accuracy_metrics,
            colours,
            "Accuracy",
        )
        plot.plot_separate_metric(
            DataSplit.TRAIN, accuracy_metrics, colours, "Accuracy"
        )
        plot.plot_separate_metric(
            DataSplit.VALIDATE, accuracy_metrics, colours, "Accuracy"
        )
