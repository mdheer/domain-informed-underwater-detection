import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np


def change_label_names(lst):
    new_lst = []
    for i in lst:
        if i == "weighted_dtw":
            new_value = "dtw"

        elif i == "weighted_dfd":
            new_value = "dfd"

        elif i == "weighted_mse":
            new_value = "mse"

        elif i == "weighted_pearson_correlation":
            new_value = "pearson"

        elif i == "weighted_mad":
            new_value = "mad"

        elif i == "weighted_variance":
            new_value = "var"

        elif i == "weighted_spearman_correlation":
            new_value = "spearman"

        elif i == "weighted_hausdorff":
            new_value = "hausdorff"

        new_lst.append(new_value)
    return new_lst


def generate_different_metrics_plots(json_data, output_folder_path):
    def parse_json(data):
        rows = []
        for condition_type, conditions in data.items():
            for estimation_type, estimations in conditions.items():
                for metric, metric_values in estimations.items():
                    if metric == "weighted_variance":
                        continue
                    accuracy = metric_values.get("accuracy")
                    if accuracy is not None:
                        combined_identifier = (
                            f"{condition_type} {estimation_type.replace('.', ' ')}"
                        )
                        rows.append(
                            [condition_type, combined_identifier, accuracy, metric]
                        )
        return rows

    rows = parse_json(json_data)
    df = pd.DataFrame(
        rows, columns=["Condition Type", "Combined Identifier", "Accuracy", "Metric"]
    )

    color_palette = {
        "optical_flow": "#1f77b4",
        "gaussian_noise": "#ff7f0e",
        "ideal": "#2ca02c",
    }

    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=0.8)

    metrics = df["Metric"].unique()
    for metric in metrics:
        plt.figure(figsize=(16, 8))
        metric_df = df[df["Metric"] == metric]
        ax = sns.barplot(
            data=metric_df,
            x="Combined Identifier",
            y="Accuracy",
            hue="Condition Type",
            palette=color_palette,
            dodge=False,
        )
        plt.title(
            f"Accuracy Percentages by Condition and Parameter Estimation Setting - {metric}",
            fontsize=14,
        )
        plt.xlabel("Condition and Parameter Estimation Setting", fontsize=12)
        plt.ylabel("Accuracy (%)", fontsize=12)
        plt.xticks(rotation=45, ha="right", fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend(title="Condition Type", frameon=True, shadow=True, fontsize=10)

        # Adding the exact values on top of each bar
        for p in ax.patches:
            if p.get_width() > 0 and p.get_height() > 0:
                ax.annotate(
                    f"{p.get_height():.1f}",
                    (p.get_x() + p.get_width() / 2.0, p.get_height()),
                    ha="center",
                    va="center",
                    xytext=(0, 10),
                    textcoords="offset points",
                )

        plt.tight_layout()

        # Construct the filename and save the plot
        filename = f"1_{metric.replace(' ', '_').lower()}.png"
        filepath = os.path.join(output_folder_path, filename)
        plt.savefig(filepath)
        plt.close()  # Close the figure to free memory


def generate_comparison_plots(json_data, output_folder_path):
    # Ensure output folder exists
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    def parse_data(data):
        rows = []
        for condition_type, conditions in data.items():
            for estimation_type, estimations in conditions.items():
                for metric, metric_values in estimations.items():
                    if metric == "weighted_variance":
                        continue
                    accuracy = metric_values.get("accuracy", None)
                    if accuracy is not None:
                        rows.append(
                            {
                                "Condition Type": condition_type,
                                "Estimation Type": estimation_type,
                                "Metric": metric,
                                "Accuracy": accuracy,
                            }
                        )
        return pd.DataFrame(rows)

    df = parse_data(json_data)

    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=0.8)

    for estimation_type in df["Estimation Type"].unique():
        plt.figure(figsize=(10, 6))
        filtered_df = df[df["Estimation Type"] == estimation_type]
        ax = sns.barplot(
            data=filtered_df, x="Condition Type", y="Accuracy", hue="Metric"
        )
        ax.set_title(
            f"Accuracy by Condition and Metric - {estimation_type.replace('_', ' ').title()}",
            fontsize=15,
        )
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)  # Adjust font size of x-axis tick labels
        ax.set_xlabel("Data type", fontsize=15)
        ax.set_ylabel("Accuracy [%]", fontsize=15)
        plt.ylim(bottom=0)
        plt.ylim(top=100)
        plt.xticks(rotation=0)
        plt.legend(
            title="Metric", loc="lower right", bbox_to_anchor=(1, 0), fontsize=12
        )

        # # Adding the exact values on top of each bar
        # for p in ax.patches:
        #     if p.get_width() > 0 and p.get_height() > 0:
        #         ax.annotate(
        #             f"{p.get_height():.1f}",
        #             (p.get_x() + p.get_width() / 2.0, p.get_height()),
        #             ha="center",
        #             va="center",
        #             xytext=(0, 10),
        #             textcoords="offset points",
        #         )

        # Save the plot
        filename = f"2_accuracy_comparison_{estimation_type}.png"
        filepath = os.path.join(output_folder_path, filename)
        plt.savefig(filepath)
        plt.close()


def generate_metric_param_pair_plots(json_data, output_folder_path):
    # Ensure the output folder exists
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    def parse_data(data):
        rows = []
        for condition_type, conditions in data.items():
            for estimation_type, estimations in conditions.items():
                readable_estimation_type = (
                    estimation_type.replace("ParameterEstimation.", "")
                    .replace("_", " ")
                    .lower()
                )
                if readable_estimation_type == "off":
                    readable_estimation_type = "no param"
                for metric, metric_values in estimations.items():
                    accuracy = metric_values.get("accuracy", None)
                    if accuracy is not None:
                        param_metric_combination = (
                            f"{readable_estimation_type} {metric}"
                        )
                        rows.append(
                            {
                                "Condition Type": condition_type,
                                "Param-Metric": param_metric_combination,
                                "Accuracy": accuracy,
                                "Param": readable_estimation_type,  # For color mapping
                            }
                        )
        return pd.DataFrame(rows)

    df = parse_data(json_data)

    # Generate a unique color for each parameter estimation setting
    param_settings = df["Param"].unique()
    colors = sns.color_palette(n_colors=len(param_settings))
    color_mapping = dict(zip(param_settings, colors))

    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=0.8)

    for condition_type in df["Condition Type"].unique():
        plt.figure(figsize=(14, 7))
        condition_df = df[df["Condition Type"] == condition_type]

        # Create a barplot
        ax = sns.barplot(
            x="Param-Metric",
            y="Accuracy",
            hue="Param",
            dodge=True,
            data=condition_df,
            palette=color_mapping,
        )

        plt.title(f"Accuracy for {condition_type.title()} by Parameter and Metric")
        plt.xlabel("Parameter Estimation Setting and Metric")
        plt.ylabel("Accuracy (%)")
        plt.xticks(rotation=0)
        plt.legend(
            title="Parameter Estimation Setting",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
        )

        # Adding the exact values on top of each bar
        for p in ax.patches:
            if p.get_width() > 0 and p.get_height() > 0:
                ax.annotate(
                    f"{p.get_height():.1f}",
                    (p.get_x() + p.get_width() / 2.0, p.get_height()),
                    ha="center",
                    va="center",
                    xytext=(0, 10),
                    textcoords="offset points",
                )

        # Save the plot
        filename = f"3_{condition_type}_accuracy_by_param_and_metric.png"
        filepath = os.path.join(output_folder_path, filename)
        plt.savefig(filepath, bbox_inches="tight")
        plt.close()


def generate_normalized_probability_histograms(
    data, output_folder_path, bin_width=0.01, start=0, end=1
):
    import os

    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # Calculate bins dynamically based on start, end, and bin_width
    bins = np.arange(start, end + bin_width, bin_width)

    # Iterate through the data structure
    for condition_type, param_estimations in data.items():
        for param_estimation, metrics in param_estimations.items():
            for metric, details in metrics.items():
                if metric == "weighted_variance":
                    continue
                if "probabilities" in details:
                    probabilities = []  # Store probabilities for current metric
                    for probability_list in details["probabilities"]:
                        probabilities.extend(
                            [prob for sublist in probability_list for prob in sublist]
                        )

                    # Prepare the normalized histogram plot
                    plt.figure(figsize=(10, 6))
                    # Calculate histogram data for normalization purposes
                    counts, _ = np.histogram(probabilities, bins=bins)
                    # Normalize counts to max of 1 for displaying purposes
                    max_num = sum(counts)
                    normalized_counts = counts / max_num
                    plt.bar(
                        bins[:-1],
                        normalized_counts,
                        width=np.diff(bins),
                        edgecolor="black",
                        align="edge",
                    )
                    plt.title(
                        f"{condition_type} | {param_estimation} | {metric} (Normalized)"
                    )
                    plt.xlabel("Probability")
                    plt.ylabel("Normalized Frequency")
                    plt.xlim(start, end)
                    plt.ylim(0, 1)
                    plt.grid(axis="y", alpha=0.75)

                    # Save the plot
                    filename = f"{condition_type}_{param_estimation}_{metric}_normalized.png".replace(
                        ".", ""
                    ).replace(
                        " ", "_"
                    )
                    filepath = os.path.join(output_folder_path, filename)
                    plt.savefig(filepath)
                    plt.close()


def generate_probability_violinplots(data, output_folder_path):
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    sns.set(style="whitegrid")  # Set the seaborn style for better aesthetics

    for condition_type, param_estimations in data.items():
        for param_estimation, metrics in param_estimations.items():
            plt.figure(figsize=(12, 8))
            metric_probabilities = (
                []
            )  # To hold the flat list of probabilities for all metrics
            metric_labels = (
                []
            )  # To hold the repeated metric names for proper labeling in sns.violinplot

            # Collect probabilities and labels
            for metric, details in metrics.items():
                if metric == "weighted_variance":
                    continue
                if "probabilities" in details:
                    for probability_list in details["probabilities"]:
                        for sublist in probability_list:
                            metric_probabilities.extend(sublist)
                            metric_labels.extend([metric for _ in sublist])

            metric_labels = change_label_names(metric_labels)
            # Create a DataFrame for seaborn plotting
            data_df = pd.DataFrame(
                {"Probability": metric_probabilities, "Metric": metric_labels}
            )

            # Adjusted sns.violinplot call to use density_norm parameter
            sns.violinplot(
                x="Probability",
                y="Metric",
                data=data_df,
                inner="quartile",
                density_norm="width",
                cut=0,
            )
            plt.title(
                f"Probability Distribution {condition_type} | {param_estimation}",
                fontsize=18,
            )
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)  # Adjust font size of x-axis tick labels
            plt.xlabel(
                "Probability", fontsize=20
            )  # Increase the font size for x-axis label
            plt.ylabel("Metrics", fontsize=20)
            plt.xlim(0, 1)  # Ensure the x-axis is between 0 and 1

            # Save the plot
            filename = f"4_{condition_type}_{param_estimation}_violinplot.png".replace(
                ".", ""
            ).replace(" ", "_")
            filepath = os.path.join(output_folder_path, filename)
            plt.tight_layout()
            plt.savefig(filepath, bbox_inches="tight")
            plt.close()


def generate_combined_violinplots_all_params_horizontal(data, output_folder_path):
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    sns.set(style="whitegrid")  # Set the seaborn style

    # Find all unique metrics to iterate over
    all_metrics = set(
        metric
        for condition_data in data.values()
        for param_data in condition_data.values()
        for metric in param_data
    )

    for metric in all_metrics:
        # Initialize lists to hold plot data
        probabilities = []
        labels = []

        # Collect data for each condition type and parameter estimation setting
        for condition_type, param_data in data.items():
            for param_estimation, metrics in param_data.items():
                if metric in metrics and "probabilities" in metrics[metric]:
                    if metric == "weighted_variance":
                        continue
                    for prob_list in metrics[metric]["probabilities"]:
                        for sublist in prob_list:
                            for prob in sublist:
                                probabilities.append(prob)
                                labels.append(f"{condition_type} | {param_estimation}")

        # If there's data, plot it
        if probabilities:
            df = pd.DataFrame({"Probability": probabilities, "Label": labels})

            plt.figure(figsize=(12, 8))
            sns.violinplot(
                y="Label", x="Probability", data=df, inner="quartile", cut=0, orient="h"
            )
            plt.title(f"Combined Violin Plot for {metric}")
            plt.xlabel("Probability")
            plt.xlim(0, 1)

            # Save the plot
            filename = (
                f"5_combined_{metric}_horizontal_violinplot.png".replace(" ", "_")
                .replace("/", "_")
                .replace("|", "_")
            )
            filepath = os.path.join(output_folder_path, filename)
            plt.tight_layout()
            plt.savefig(filepath)
            plt.close()


def generate_plots_for_each_condition_and_param(data, output_folder_path):
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    sns.set(style="whitegrid")  # Set the seaborn style for better aesthetics

    # Iterate over each condition type and parameter estimation
    for condition_type, params in data.items():
        for param_estimation, metrics in params.items():
            # Collect data for all metrics under the current condition and parameter estimation
            plot_data = []
            for metric, details in metrics.items():
                if metric == "weighted_variance":
                    continue
                if "probabilities" in details:
                    for prob_list in details["probabilities"]:
                        for sublist in prob_list:
                            for prob in sublist:
                                plot_data.append(
                                    {"Metric": metric, "Probability": prob}
                                )

            if plot_data:
                # Convert the collected data into a DataFrame
                df = pd.DataFrame(plot_data)

                plt.figure(figsize=(12, 8))
                sns.violinplot(
                    y="Metric",
                    x="Probability",
                    data=df,
                    inner="quartile",
                    cut=0,
                    orient="h",
                )
                plt.title(f"Probabilities for {condition_type} | {param_estimation}")
                plt.xlabel("Probability")
                plt.ylabel("Metrics")

                # Save the plot with a filename that reflects the current condition and parameter estimation
                filename = (
                    f"6_ {condition_type}_{param_estimation}_metrics_violinplot.png".replace(
                        " ", "_"
                    )
                    .replace("|", "_")
                    .replace("/", "_")
                )
                filepath = os.path.join(output_folder_path, filename)
                plt.tight_layout()
                plt.savefig(filepath)
                plt.close()


# Example usage:
json_file_path = r"M:\thesis\validation.json"
output_folder_path = r"C:\Users\mathi\Documents\Files\University of Technology Delft\Master\MSc 2\Thesis\Final paper\Mathematical_model_comparisons"

# Read the JSON file
with open(json_file_path, "r") as file:
    json_data = json.load(file)

generate_different_metrics_plots(json_data, output_folder_path)
generate_comparison_plots(json_data, output_folder_path)
generate_metric_param_pair_plots(json_data, output_folder_path)
generate_probability_violinplots(json_data, output_folder_path)
generate_combined_violinplots_all_params_horizontal(json_data, output_folder_path)
generate_plots_for_each_condition_and_param(json_data, output_folder_path)
