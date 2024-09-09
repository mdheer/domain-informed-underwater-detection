import os
import pandas as pd
import json
import re
import matplotlib.pyplot as plt


class JsonData:
    def __init__(self, data_dict) -> None:
        self.train_df = pd.DataFrame(data_dict["train"])
        self.validate_df = pd.DataFrame(data_dict["validate"])
        self.test_df = pd.DataFrame(data_dict["test"])

    def get_test_perc(self):
        return self.test_df["perc_nn_correct"].max()


class OverviewPlotter:
    def __init__(self, dir_path) -> None:
        self.runs_dk = []
        self.runs_no_dk = []
        self.dir_path = dir_path

    def create_graphs(self):
        dk_df = pd.DataFrame(self.runs_dk).sort_values(by="number_datapoints")
        no_dk_df = pd.DataFrame(self.runs_no_dk).sort_values(by="number_datapoints")

        plot(
            x_dk=dk_df["number_datapoints"],
            y_dk=dk_df["test_accuracy_nn"],
            x_no_dk=no_dk_df["number_datapoints"],
            y_no_dk=no_dk_df["test_accuracy_nn"],
            title="Number of data points used during training versus test accuracy",
            x_label="Number of data points [-]",
            y_label="Test accuracy [%]",
            filename=os.path.join(self.dir_path, "test_accuracy_vs_num_data_points"),
        )


def plot(x_dk, y_dk, x_no_dk, y_no_dk, title, x_label, y_label, filename):
    plt.figure(figsize=(12, 8))
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(x_dk, y_dk, label="dk")
    plt.plot(x_no_dk, y_no_dk, label="no_dk")
    plt.legend()
    plt.savefig(filename)
    plt.close()


def construct_run_graph(
    dk_data: JsonData, no_dk_data: JsonData, output_path: str
) -> None:
    dk_train = dk_data.train_df
    dk_validate = dk_data.validate_df
    dk_test = dk_data.test_df

    no_dk_train = no_dk_data.train_df
    no_dk_validate = no_dk_data.validate_df
    no_dk_test = no_dk_data.test_df

    plot(
        x_dk=dk_validate.index,
        y_dk=dk_validate["perc_nn_correct"],
        x_no_dk=no_dk_validate.index,
        y_no_dk=no_dk_validate["perc_nn_correct"],
        title="Comparison neural network correct validation dk vs no dk",
        x_label="epoch number [-]",
        y_label="neural network accuracy [%]",
        filename=os.path.join(output_path, "validation_perc_nn_correct"),
    )


def construct_graphs(
    input_dk_path: str, input_no_dk_path: str, output_path: str
) -> pd.DataFrame:
    dk_runs = os.listdir(input_dk_path)
    no_dk_runs = os.listdir(input_no_dk_path)

    dk_runs = sorted(dk_runs)
    no_dk_runs = sorted(no_dk_runs)

    plotter = OverviewPlotter(output_path)

    for dk, no_dk in zip(dk_runs, no_dk_runs):
        # Construct full paths
        full_dk_path = os.path.join(input_dk_path, dk)
        full_no_dk_path = os.path.join(input_no_dk_path, no_dk)

        data_dk_path = os.path.join(full_dk_path, "training_results_structured.json")
        data_no_dk_path = os.path.join(
            full_no_dk_path, "training_results_structured.json"
        )

        run_name_dk = int(re.search(r"(\d+)_train_samples", dk).group(1))
        run_name_no_dk = int(re.search(r"(\d+)_train_samples", no_dk).group(1))

        if run_name_dk != run_name_no_dk:
            raise ValueError(
                f"Run numbers are not the same: {run_name_dk} and {run_name_no_dk}"
            )

        run_dir = os.path.join(output_path, f"{run_name_dk}_train_samples")

        if not os.path.exists(run_dir):
            os.mkdir(run_dir)

        # Extract data from .json
        with open(data_dk_path, "r") as f:
            data_dk = json.load(f)

        with open(data_no_dk_path, "r") as f:
            data_no_dk = json.load(f)

        dk_parsed = JsonData(data_dk)
        no_dk_parsed = JsonData(data_no_dk)

        construct_run_graph(dk_parsed, no_dk_parsed, run_dir)

        plotter.runs_dk.append(
            {
                "number_datapoints": run_name_dk,
                "test_accuracy_nn": dk_parsed.get_test_perc(),
            }
        )
        plotter.runs_no_dk.append(
            {
                "number_datapoints": run_name_no_dk,
                "test_accuracy_nn": no_dk_parsed.get_test_perc(),
            }
        )

    plotter.create_graphs()


if __name__ == "__main__":
    path = "./training_logs/"
    # path = r"M:\thesis\cluster_runs\training_logs"
    no_dk_dataset = "small_test_dk_False_12.optical_flow"
    dk_dataset = "small_test_dk_True_12.optical_flow"

    # Generate output path
    output_name = dk_dataset.replace("dk_True_", "")
    output_folder = os.path.join(path, "graphs_" + output_name)
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # Generate input path
    path_dk_dataset = os.path.join(path, dk_dataset)
    path_no_dk_dataset = os.path.join(path, no_dk_dataset)

    construct_graphs(path_dk_dataset, path_no_dk_dataset, output_folder)
