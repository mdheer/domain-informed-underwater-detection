import os
import json
import sys
import shutil
import traceback
import matplotlib.pyplot as plt
import pandas as pd
import time

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from src.tools.enums import DictKeys, ClassName, DataVariant
from src.neural_network.main import TrainNeuralNetwork
from src.tools.general_functions import (
    read_input_path,
    sort_filenames,
    parse_class_name,
    load_config,
    parse_input_path_dependent_on_os,
    process_unity_data,
    process_modified_data,
    get_midpoints,
)

experiment_name = "param_high"
dataset_name = "small_test_set"
temperature = 1


def extract_param_bounds(file_path: str) -> dict:
    """
    Extract parameter bounds from a JSON file.

    Parameters:
        file_path (str): The path to the JSON file.

    Returns:
        dict: A dictionary containing the extracted parameter bounds. Different for each class.
    """

    with open(file_path, "r") as f:
        data = json.load(f)

    # Generate the initial dictionary
    keys = list(data.keys())
    settings = data[keys[0]][DictKeys.UNITY_SETTINGS.value]
    key_names_settings = [
        settings_key
        for settings_key in settings.keys()
        if settings_key != "waterCurrentStrength"
        and settings_key != "lightDirection"
        and settings_key != "objectType"
    ]

    init_dict_plastic = {}
    init_dict_fish = {}

    for key in key_names_settings:
        first_value_dict = data[keys[0]][DictKeys.UNITY_SETTINGS.value]
        class_name = parse_class_name(first_value_dict["objectType"])
        if class_name != ClassName.FISH:
            raise ValueError("The item at index 0 in parsed data is of wrong type!")

        if key == "swimForceVector":
            val = first_value_dict[key]["z"]
            key = "F_fz"
        else:
            val = first_value_dict[key]

        init_dict_fish[key] = {"min": val, "max": val}

    for key in key_names_settings:
        first_value_dict = data[keys[1]][DictKeys.UNITY_SETTINGS.value]
        class_name = parse_class_name(first_value_dict["objectType"])

        if class_name != ClassName.PLASTIC:
            raise ValueError("The item at index 1 in parsed data is of wrong type!")

        if key == "swimForceVector":
            val = first_value_dict[key]["z"]
            key = "F_fz"
        else:
            val = first_value_dict[key]

        init_dict_plastic[key] = {"min": val, "max": val}

    final_dict = {
        ClassName.PLASTIC: init_dict_plastic,
        ClassName.FISH: init_dict_fish,
    }

    # Calculate the bounds
    for i in keys:
        sub_dict = data[i][DictKeys.UNITY_SETTINGS.value]
        class_name = parse_class_name(sub_dict["objectType"])
        for j in final_dict[class_name].keys():
            if j == "F_fz":
                value = sub_dict["swimForceVector"]["z"]
            else:
                value = sub_dict[j]

            if value < final_dict[class_name][j]["min"]:
                final_dict[class_name][j]["min"] = value

            if value > final_dict[class_name][j]["max"]:
                final_dict[class_name][j]["max"] = value

    # Calcuate the midpoints
    for i in [ClassName.PLASTIC, ClassName.FISH]:
        for j in final_dict[i].keys():
            min = final_dict[i][j]["min"]
            max = final_dict[i][j]["max"]

            # mid = (abs(max) - abs(min)) / 2
            mid = (max + min) / 2

            final_dict[i][j]["mid"] = mid

    return final_dict


def save_training_config(
    master_data_file_path: str,
    dataset_path: str,
    dataset_config_file_path: str,
    embed_domain_knowledge: bool,
    error_level: float,
    num_epochs: int,
    plotting: bool,
    batch_size: int,
    log_path: str,
) -> None:
    """
    Save training configuration to a JSON file.

    Parameters:
        master_data_file_path (str): Path to the master data file.
        dataset_path (str): Path to the dataset.
        dataset_config_file_path (str): Path to the dataset config file.
        parameter_estimation: Value specifying the mathematical model input.
        embed_domain_knowledge: Value specifying whether to embed domain knowledge.
        error_level: Value specifying the error level.
        num_epochs (int): Number of training epochs.
        plotting: Value specifying whether to enable plotting.
        evaluation: Value specifying whether to enable evaluation.
        batch_size (int): Training batch size.
        log_path (str): Path to the log directory.
    """
    config = {
        "master_data_file_path": master_data_file_path,
        "dataset_path": dataset_path,
        "dataset_config_file_path": dataset_config_file_path,
        "embed_domain_knowledge": embed_domain_knowledge,
        "error_level": error_level,
        "num_epochs": num_epochs,
        "plotting": plotting,
        "batch_size": batch_size,
        "log_path": log_path,
    }
    path = os.path.join(log_path, "training_config.json")
    with open(path, "w") as file:
        json.dump(config, file, indent=4)


def send_slack_message(message):
    token = "enter token"
    channel = "enter channel"
    raise ValueError("Update token and channel to send slack messages")
    client = WebClient(token=token)
    try:
        response = client.chat_postMessage(channel=channel, text=message)
        print(f"Message sent successfully: {response['message']['text']}")

    except SlackApiError as e:
        print(f"Error sending message: {e.response['error']}")


def print_elapsed_time(elapsed_time):

    # Convert to a readable format
    days, rem = divmod(elapsed_time, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)

    if days > 0:
        time_str = f"{int(days)}d {int(hours)}h {int(minutes)}m {seconds:.2f}s"
    elif hours > 0:
        time_str = f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"
    elif minutes > 0:
        time_str = f"{int(minutes)}m {seconds:.2f}s"
    else:
        time_str = f"{seconds:.2f}s"

    print(f"Expermiment took: {time_str}")


class ExperimentRunner:
    def __init__(
        self,
        experiment_name: str,
        dataset_name: str,
        temperature: float,
        alpha: float,
        num_epoch: int = 15,
        batch_size: int = 5,
        error_level: int = 0.0,
    ):
        self.experiment_name = experiment_name
        self.dataset_name = dataset_name
        self.error_level = error_level

        # Settings
        self.TRAIN_MODEL = True
        self.VARYING_TRAIN_SAMPLES = True

        self.DATA_PLOTTING = False
        self.TRAINING_PLOTTING = False

        self.PERFORM_VALIDATION = False

        self.NUM_EPOCH = num_epoch
        self.BATCH_SIZE = batch_size

        self.TEMPERATURE = temperature
        self.ALPHA = alpha

        self.raw_data_dir = read_input_path(input_type=self._extract_input_path())
        self.path_master_data_file = os.path.join(self.raw_data_dir, "parsed_data.json")
        self.param_bounds = extract_param_bounds(self.path_master_data_file)

        self.root_dir_logging = "./training_logs/"
        self.root_dir_dataset = "./datasets/"

        self.dataset_path = os.path.join(self.root_dir_dataset, self.dataset_name)
        self.experiment_dir = os.path.join(self.root_dir_logging, self.experiment_name)

        if os.path.exists(self.experiment_dir):
            inp = input(
                "Experiment already exists, do you want to the delete the existing experiment and continue? [y/n] "
            )
            if inp == "y" or inp == "Y":
                shutil.rmtree(self.experiment_dir)
                os.mkdir(self.experiment_dir)
            else:
                print("Exiting...")
                sys.exit()
        else:
            os.mkdir(self.experiment_dir)

    def _extract_input_path(self):

        config_name = f"dataset_configs.{self.dataset_name}"
        # Load the configuration module
        config = load_config(config_name)

        # Access configurations
        input_data_path = config.input_data_path

        input_data_path = parse_input_path_dependent_on_os(input_data_path)

        return input_data_path

    def train_on_various_k_folds(self):

        k_folds = os.listdir(self.dataset_path)
        self.sorted_k_folds = sorted(k_folds, key=lambda x: int(x.split("_")[-1]))

        for fold in self.sorted_k_folds:
            send_slack_message(
                f"Charting new horizons! Vessel {experiment_name} is setting sail for waypoint {fold}, where unknown challenges and discoveries await. Adventure beckons!"
            )

            k_fold_path = os.path.join(self.dataset_path, fold)
            self.train_on_various_distributions(k_fold_path, fold)

    def train_on_various_distributions(self, k_fold_path, k_fold_name):
        distribution_files_path = os.path.join(k_fold_path, "configs")

        self.mm_model_dir_name = f"mathematical_model_outputs_{self.experiment_name}"
        mm_model_files_path = os.path.join(k_fold_path, self.mm_model_dir_name)

        distribution_files = sort_filenames(os.listdir(distribution_files_path))

        k_fold_dir = os.path.join(self.experiment_dir, k_fold_name)
        os.mkdir(k_fold_dir)

        if not os.path.exists(mm_model_files_path):
            os.mkdir(mm_model_files_path)

        for distr_file in distribution_files:
            distribution_file_path = os.path.join(distribution_files_path, distr_file)
            for boolie in [True, False]:
                log_folder_varying_samples = os.path.join(
                    k_fold_dir, f"exp_dk_{boolie}_{self.dataset_name}"
                )
                os.makedirs(log_folder_varying_samples, exist_ok=True)

                self.perform_training(
                    embed_dk=boolie,
                    distr_file=distr_file,
                    distribution_file_path=distribution_file_path,
                    log_folder_varying_samples=log_folder_varying_samples,
                    dataset_path=k_fold_path,
                    mm_model_dir_name=self.mm_model_dir_name,
                    k_fold_name=k_fold_name,
                )

    def perform_training(
        self,
        embed_dk,
        distr_file,
        distribution_file_path,
        log_folder_varying_samples,
        dataset_path,
        mm_model_dir_name,
        k_fold_name,
    ):

        name = os.path.basename(distr_file)[: -len(".json")]
        train_log_folder = os.path.join(log_folder_varying_samples, name)
        os.makedirs(train_log_folder, exist_ok=True)

        print(f"####### Training started #######")
        print(f"Dataset: {self.dataset_name}")
        print(f"K_fold: {k_fold_name}")
        print(f"Distribution file: {name}")
        print(f"Domain knowledge embedded: {embed_dk}")

        TrainNeuralNetwork(
            master_data_file_path=self.path_master_data_file,
            dataset_path=dataset_path,
            dataset_config_file_path=distribution_file_path,
            embed_domain_knowledge=embed_dk,
            error_level=self.error_level,
            num_epochs=self.NUM_EPOCH,
            plotting=self.TRAINING_PLOTTING,
            param_bounds=self.param_bounds,
            batch_size=self.BATCH_SIZE,
            log_path=train_log_folder,
            dir_name_mm=mm_model_dir_name,
            temperature=self.TEMPERATURE,
            alpha=self.ALPHA,
        )

        save_training_config(
            master_data_file_path=self.path_master_data_file,
            dataset_path=dataset_path,
            dataset_config_file_path=distribution_file_path,
            embed_domain_knowledge=embed_dk,
            error_level=self.error_level,
            num_epochs=self.NUM_EPOCH,
            plotting=self.TRAINING_PLOTTING,
            batch_size=self.BATCH_SIZE,
            log_path=train_log_folder,
        )

    def _plot_k_fold_averages(self):
        def calculate_k_fold_averages(row, df, name):
            k_fold_cols = [col for col in df.columns if "k_fold_distribution" in col]
            key_sums = {}
            key_counts = {}

            for col in k_fold_cols:
                for key, value in row[col].items():

                    if key in key_sums:
                        key_sums[key] += value[name]
                        key_counts[key] += 1
                    else:
                        key_sums[key] = value[name]
                        key_counts[key] = 1

            averages = {key: key_sums[key] / key_counts[key] for key in key_sums}

            return averages

        # Go through the k_fold_distribution
        k_folds = os.listdir(self.experiment_dir)
        sorted_k_folds = sorted(k_folds, key=lambda x: int(x.split("_")[-1]))
        tot_dict = {}

        for k_fold in sorted_k_folds:
            k_fold_path = os.path.join(self.experiment_dir, k_fold)
            tot_dict[k_fold] = {}

            for boolie in os.listdir(k_fold_path):
                if "True" in boolie:
                    set_type = True
                else:
                    set_type = False

                tot_dict[k_fold][set_type] = {}

                boolie_path = os.path.join(k_fold_path, boolie)
                varying_sample_runs = os.listdir(boolie_path)
                for run in varying_sample_runs:
                    n_data_point = run.split("_")[0]
                    run_path = os.path.join(boolie_path, run)
                    train_structured_path = os.path.join(
                        run_path, "training_results_structured.json"
                    )

                    with open(train_structured_path, "r") as file:
                        data = json.load(file)
                    test_percentage_nn = data["test"][0]["perc_nn_correct"]
                    test_percentage_mm = data["test"][0]["perc_mm_correct"]

                    tot_dict[k_fold][set_type][n_data_point] = {
                        "test_percentage_nn": test_percentage_nn,
                        "test_percentage_mm": test_percentage_mm,
                    }

        df = pd.DataFrame(tot_dict)

        df["k_fold_averages_nn"] = df.apply(
            lambda x: calculate_k_fold_averages(x, df, "test_percentage_nn"), axis=1
        )
        df["k_fold_averages_mm"] = df.apply(
            lambda x: calculate_k_fold_averages(x, df, "test_percentage_mm"), axis=1
        )

        # Extract the 'k_fold_averages' for 'no_dk' and 'dk_embedded'
        no_dk = df.loc[False, "k_fold_averages_nn"]
        dk_embedded = df.loc[True, "k_fold_averages_nn"]
        mm_no_dk = df.loc[False, "k_fold_averages_mm"]
        mm_dk = df.loc[False, "k_fold_averages_mm"]

        # Ensure keys are integers and sort the dictionaries by keys
        no_dk_sorted = dict(sorted(no_dk.items(), key=lambda item: int(item[0])))
        dk_embedded_sorted = dict(
            sorted(dk_embedded.items(), key=lambda item: int(item[0]))
        )
        mm_no_dk = dict(sorted(mm_no_dk.items(), key=lambda item: int(item[0])))
        mm_dk = dict(sorted(mm_dk.items(), key=lambda item: int(item[0])))

        # Extract x and y values for plotting
        no_dk_x = list(map(int, no_dk_sorted.keys()))
        no_dk_y = list(no_dk_sorted.values())

        dk_embedded_x = list(map(int, dk_embedded_sorted.keys()))
        dk_embedded_y = list(dk_embedded_sorted.values())

        mm_no_dk_x = list(map(int, mm_no_dk.keys()))
        mm_no_dk_y = list(mm_no_dk.values())

        mm_dk_x = list(map(int, mm_dk.keys()))
        mm_dk_y = list(mm_dk.values())

        # Plotting with updated colors
        plt.figure(figsize=(10, 6))

        plt.plot(
            no_dk_x, no_dk_y, label="no_dk", color="#ff7f0e", marker="o"
        )  # More beautiful orange
        plt.plot(
            dk_embedded_x,
            dk_embedded_y,
            label="dk_embedded",
            color="#1f77b4",
            marker="o",
        )  # More beautiful blue

        # Add the mathematical_model with a beautiful green
        plt.plot(
            mm_dk_x, mm_dk_y, label="mathematical_model", color="#2ca02c", marker="o"
        )  # More beautiful green

        plt.title("Accuracy vs number of datapoints for training")
        plt.ylabel("Accuracy [%]")
        plt.xlabel("# of datapoints [-]")
        plt.legend()
        plt.grid(True)
        plt.xticks(no_dk_x, rotation=55)

        plt.savefig(os.path.join(self.experiment_dir, "overview.png"))

    def _get_performance_distribution(self):

        def extract_predictions_per_datapoint(data_dict, prefix):
            rows = []  # This will store our extracted information

            for batch_name, batch_data in data_dict["test"].items():
                if "batch" in batch_name:
                    for data_point_name, data_point_info in batch_data.items():
                        # Assuming all entries under batch_data are data points to be processed
                        plastic_softmax = data_point_info["softmax_student"][0]
                        fish_softmax = data_point_info["softmax_student"][1]

                        row = {
                            f"data_point_id": data_point_info["data_point_id"],
                            f"{prefix}_softmax_student_index_zero": plastic_softmax,
                            f"{prefix}_softmax_student_index_one": fish_softmax,
                        }
                        if prefix == "dk_true":
                            row["label"] = data_point_info["labels"]

                        if (
                            data_point_info["labels"] == 0
                            and plastic_softmax > fish_softmax
                        ):
                            pred = True
                        elif (
                            data_point_info["labels"] == 0
                            and plastic_softmax < fish_softmax
                        ):
                            pred = False

                        elif (
                            data_point_info["labels"] == 1
                            and plastic_softmax < fish_softmax
                        ):
                            pred = True
                        elif (
                            data_point_info["labels"] == 1
                            and plastic_softmax > fish_softmax
                        ):
                            pred = False
                        row[f"{prefix}_prediction"] = pred
                        rows.append(row)

            # Convert the list of dictionaries into a DataFrame
            df = pd.DataFrame(rows)
            return df

        performance_distribution = []
        master_df = pd.DataFrame(columns=["data_point_id", "count"])

        for fold in self.sorted_k_folds:
            k_fold_exp_dir = os.path.join(self.experiment_dir, fold)

            dk_true_path = os.path.join(
                k_fold_exp_dir, f"exp_dk_{True}_{self.dataset_name}"
            )
            files = os.listdir(dk_true_path)
            comparision_dir_path = os.path.join(k_fold_exp_dir, "comparisons")
            os.mkdir(comparision_dir_path)

            for training_run in files:
                json_path_true = os.path.join(
                    os.path.join(dk_true_path, training_run),
                    "training_results_raw.json",
                )
                json_path_false = json_path_true.replace("True", "False")

                with open(json_path_true, "r") as file:
                    dk_true_dict = json.load(file)
                with open(json_path_false, "r") as file:
                    dk_false_dict = json.load(file)

                df_true_df = extract_predictions_per_datapoint(dk_true_dict, "dk_true")
                df_false_df = extract_predictions_per_datapoint(
                    dk_false_dict, "dk_false"
                )

                merged_df = pd.merge(df_true_df, df_false_df, on="data_point_id")

                # Adjust pandas settings to display all columns
                pd.set_option("display.max_columns", None)
                mutual_true = merged_df.loc[
                    (merged_df["dk_false_prediction"])
                    & (merged_df["dk_true_prediction"])
                ]

                dk_true_alone = merged_df.loc[
                    (merged_df["dk_true_prediction"] == True)
                    & (merged_df["dk_false_prediction"] == False)
                ]
                dk_false_alone = merged_df.loc[
                    (merged_df["dk_true_prediction"] == False)
                    & (merged_df["dk_false_prediction"] == True)
                ]

                tot_len = merged_df.shape[0]

                mutual_true_perc = mutual_true.shape[0] / tot_len
                dk_true_alone_perc = dk_true_alone.shape[0] / tot_len
                dk_false_alone_perc = dk_false_alone.shape[0] / tot_len

                data_dict = merged_df.to_dict()

                data_dict["mutual_true"] = mutual_true.shape[0]
                data_dict["dk_true_alone"] = dk_true_alone.shape[0]
                data_dict["dk_false_alone"] = dk_false_alone.shape[0]

                data_dict["mutual_true_perc"] = mutual_true_perc
                data_dict["dk_true_alone_perc"] = dk_true_alone_perc
                data_dict["dk_false_alone_perc"] = dk_false_alone_perc

                file_path = os.path.join(
                    comparision_dir_path, f"{training_run}_comparison.json"
                )

                row = {
                    "k_fold": fold,
                    "varying_training_sample": training_run,
                    "mutual_true_perc": mutual_true_perc,
                    "dk_true_alone_perc": dk_true_alone_perc,
                    "dk_false_alone_perc": dk_false_alone_perc,
                }
                performance_distribution.append(row)

                # Writing the dictionary to a file in JSON format
                with open(file_path, "w") as file:
                    json.dump(data_dict, file, indent=4)

                for data_point_id in dk_false_alone["data_point_id"].unique():
                    if not master_df[master_df["data_point_id"] == data_point_id].empty:
                        # If present, increment count by 1
                        master_df.loc[
                            master_df["data_point_id"] == data_point_id, "count"
                        ] += 1
                    else:
                        # If not present, add new row
                        new_row = pd.DataFrame(
                            {"data_point_id": [data_point_id], "count": [1]}
                        )
                        master_df = pd.concat([master_df, new_row], ignore_index=True)

        # Convert count column to numeric
        master_df["count"] = pd.to_numeric(master_df["count"])

        # Sort the DataFrame by the 'count' column in descending order
        df_sorted = master_df.sort_values(by="count", ascending=False).reset_index(
            drop=True
        )

        perf_distr_df = pd.DataFrame(performance_distribution)
        tot_len_perf_distr = perf_distr_df.shape[0]

        avg_mutual_true = perf_distr_df["mutual_true_perc"].sum() / tot_len_perf_distr
        avg_dk_true_alone = (
            perf_distr_df["dk_true_alone_perc"].sum() / tot_len_perf_distr
        )
        avg_dk_false_alone = (
            perf_distr_df["dk_false_alone_perc"].sum() / tot_len_perf_distr
        )

        overview_dict = perf_distr_df.to_dict()
        overview_dict["avg_mutual_true"] = avg_mutual_true
        overview_dict["avg_dk_true_alone"] = avg_dk_true_alone
        overview_dict["avg_dk_false_alone"] = avg_dk_false_alone

        overview_file_path = os.path.join(
            self.experiment_dir, "overview_comparison.json"
        )

        # Writing the dictionary to a file in JSON format
        with open(overview_file_path, "w") as file:
            json.dump(overview_dict, file, indent=4)

    def postprocess_results(self):

        self._plot_k_fold_averages()
        self._get_performance_distribution()

    def create_trajectory_graphs(self):
        def parse_mm_files_to_dict(mm_lst, dir_path):
            data_dict = {}
            for data_point_raw in mm_lst:
                splitted = data_point_raw.split("_")
                data_point = f"{splitted[0]}_{splitted[1]}"
                splitted[2] = splitted[2][: -len(".json")]

                if data_point not in data_dict:
                    data_dict[data_point] = {
                        splitted[2]: os.path.join(dir_path, data_point_raw)
                    }
                else:
                    data_dict[data_point][splitted[2]] = os.path.join(
                        dir_path, data_point_raw
                    )

            return data_dict

        with open(self.path_master_data_file, "r") as file:
            parsed_data = json.load(file)

        # loop through the mathematical model folders
        for fold in self.sorted_k_folds:
            k_fold_path = os.path.join(self.dataset_path, fold)
            mm_path = os.path.join(k_fold_path, self.mm_model_dir_name)
            k_fold_exp_dir = os.path.join(self.experiment_dir, fold)

            experiment_k_fold_graph_dir = os.path.join(k_fold_exp_dir, "trajectories")
            os.mkdir(experiment_k_fold_graph_dir)

            mm_files = os.listdir(mm_path)
            mm_dict = parse_mm_files_to_dict(mm_files, mm_path)

            for key, value in mm_dict.items():

                # Get mathematical model fish
                with open(value[ClassName.FISH.value], "r") as file:
                    mm_data_fish = json.load(file)

                # Get mathematical model plastic
                with open(value[ClassName.PLASTIC.value], "r") as file:
                    mm_data_plastic = json.load(file)

                mm_fish_df = pd.DataFrame(mm_data_fish["mathematical_model"])
                mm_plastic_df = pd.DataFrame(mm_data_plastic["mathematical_model"])

                # Get ideal trajectory
                t = parsed_data[key][DictKeys.UNITY_ANNOTATIONS.value]["frameTime"]
                ideal_dict = process_unity_data(parsed_data[key])
                ideal_dict["t"] = t
                ideal_df = pd.DataFrame(ideal_dict)

                data_variant = mm_data_fish["neural_network_data_input"]
                if "optical_flow" in data_variant:
                    data_variant = DictKeys.OPTICAL_FLOW_DATA.value

                if data_variant == DataVariant.GAUSSIAN_NOISE.value:
                    pos_name = DictKeys.NOISY_DATA_POSITIONS.value
                    vel_name = DictKeys.NOISY_DATA_VELOCITIES.value

                elif data_variant == DataVariant.OPTICAL_FLOW.value:
                    pos_name = DictKeys.OPTICAL_FLOW_POSITIONS.value
                    vel_name = DictKeys.OPTICAL_FLOW_VELOCITIES.value
                    t = get_midpoints(get_midpoints(t))

                if data_variant != DataVariant.IDEAL.value:
                    input_data_dict = process_modified_data(
                        data_entry=parsed_data[key],
                        data_type=data_variant,
                        pos_name=pos_name,
                        vel_name=vel_name,
                    )

                    input_data_dict["t"] = t
                    input_data_df = pd.DataFrame(input_data_dict)

                else:
                    input_data_df = ideal_df

                dfs = {
                    "mm_fish": mm_fish_df,
                    "mm_plastic": mm_plastic_df,
                    "ideal_traj": ideal_df,
                    "input_traj": input_data_df,
                }

                filename_graph = os.path.join(experiment_k_fold_graph_dir, f"{key}.png")

                # Creating a 2x2 plot layout
                fig, axs = plt.subplots(2, 2, figsize=(12, 12))
                fig.suptitle(
                    f'Object: {parsed_data[key][DictKeys.UNITY_SETTINGS.value]["objectType"]}'
                )
                axs = axs.flatten()

                titles = ["t vs. y", "t vs. z", "t vs. vy", "t vs. vz"]
                y_labels = ["y", "z", "vy", "vz"]

                for i, (title, y_label) in enumerate(zip(titles, y_labels)):
                    axs[i].set_title(title)
                    for name, df in dfs.items():
                        axs[i].plot(df["t"], df[y_label], label=name)
                    axs[i].legend()
                    axs[i].set_xlabel("t [s]")
                    if "v" in y_label:
                        axs[i].set_ylabel(f"{y_label} [m/s]")
                    else:
                        axs[i].set_ylabel(f"{y_label} [m]")

                plt.tight_layout()
                plt.savefig(filename_graph)
                plt.close()


if __name__ == "__main__":

    print("############################")
    print("#### Experiment started ####")
    print("############################")

    send_slack_message(
        f"Anchors aweigh! Launching '{experiment_name}', we voyage into the dataset '{dataset_name}' with eager hearts. May fortune favor our voyage, and may we unearth secrets as boundless as the sea."
    )
    # Start time
    start_time = time.time()

    try:
        runner = ExperimentRunner(
            experiment_name=experiment_name,
            dataset_name=dataset_name,
            num_epoch=1,
            batch_size=1,
            error_level=0,  # value between 0 and 1
            alpha=0.5,
            temperature=temperature,
        )

        runner.train_on_various_k_folds()
        runner.postprocess_results()
        runner.create_trajectory_graphs()

        send_slack_message(
            f"The expedition '{experiment_name}' with dataset '{dataset_name}' has found its way back to port, our hull filled with precious data jewels."
        )

    except Exception:
        # Capture the full traceback
        error_traceback = traceback.format_exc()
        # Send or log the error message along with the full traceback
        send_slack_message(
            "Disaster strikes! A fatal error, my undoing, brings my valiant efforts to naught. I am bested, brought to halt, a script no more. Mourn my abrupt departure."
        )
        # Log the full traceback for debugging
        print(f"Encountered an error with full traceback:\n{error_traceback}")

    # End time
    end_time = time.time()

    # Calculate elapsed time
    elapsed_time = end_time - start_time

    print_elapsed_time(elapsed_time)
