from src.tools.enums import (
    ParameterEstimation,
    InputDataPath,
    InvertedSimilarity,
    DataVariant,
)
from src.domain_knowledge.domain_knowledge_classifier import DomainKnowledgeClassifier
from src.train_trajectory import read_input_path, extract_param_bounds
from src.neural_network.utils import CustomDataset
from src.neural_network.main import custom_collate_fn
import os
import json
import sys
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd


def generate_dict(pth, data_variant):
    with open(pth, "r") as file:
        data = json.load(file)

    keys_lst = list(data.keys())

    for data_num in keys_lst:
        data[data_num]["neural_network_input"] = data_variant.value
        data[data_num]["unlabelled"] = False

    return data


def get_batched_lists(ds):
    # Create DataLoaders for the training and validation sets
    train_loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        collate_fn=custom_collate_fn,
    )

    batched_traj = []
    batched_settings = []
    input_trajects = []
    labels_lst = []
    data_number = []

    for (
        _,
        trajectories_dict,
        labels,
        _,
        settings,
        mm_inputs,
        data_numbers,
        _,
    ) in train_loader:

        batched_traj.append(trajectories_dict[0])
        batched_settings.append(settings[0])
        input_trajects.append(mm_inputs[0])
        labels_lst.append(labels[0])
        data_number.append(data_numbers[0])

    return batched_traj, batched_settings, input_trajects, labels_lst, data_number


def plot_comparison(data_dict, label, file_path):
    """
    Plots simplified comparison line charts for 'sim' values (combined for both 'plastic' and 'fish')
    and 'mm' values (separately for 'plastic' and 'fish') across 4 plots, each representing a different
    variable comparison in a 2x2 layout. Dynamically sets the overall title based on the 'label' value,
    with "0" corresponding to "Label: Plastic" and "1" corresponding to "Label: Fish".

    Parameters:
    - data_dict: A dictionary with two keys ('plastic', 'fish'), each associated with a DataFrame.
                 Each DataFrame should have columns for 'sim_y', 'mm_y', 'sim_z', 'mm_z', 'sim_vz', 'mm_vz',
                 and use the DataFrame index as time (t).
    - label: An integer (0 or 1) to determine the title of the generated figure. 0 for "Plastic", 1 for "Fish".
    """
    # Determine the title based on the label value
    title_map = {0: "Label: Plastic", 1: "Label: Fish"}
    overall_title = title_map.get(
        label.item(), "Label: Unknown"
    )  # Default to "Label: Unknown" if label is not 0 or 1

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))  # Create a 2x2 grid of subplots
    fig.suptitle(
        overall_title, fontsize=16
    )  # Set the overall title for the figure based on the label

    variables = ["y", "z", "vz", "vy"]  # Define the variable parts to plot

    for i, var in enumerate(variables):
        ax = axs[i // 2, i % 2]  # Determine subplot position
        # Assuming sim data is the same for both 'plastic' and 'fish', so just plot one of them for 'sim'
        ax.plot(
            data_dict[list(data_dict.keys())[0]].index,
            data_dict[list(data_dict.keys())[0]][f"sim_{var}"],
            label=f"sim_{var}",
            linestyle="--",
        )

        # Plot 'mm' values separately for 'plastic' and 'fish'
        for key, df in data_dict.items():
            ax.plot(df.index, df[f"mm_{var}"], label=f"{key} mm_{var}")

        ax.set_title(f"t vs sim_{var} & mm_{var}")
        ax.set_xlabel("Time (t)")
        ax.set_ylabel("Values")
        ax.legend()

    plt.tight_layout(
        rect=[0, 0.03, 1, 0.95]
    )  # Adjust subplots to fit into the figure area, leaving space for the overall title
    plt.savefig(file_path)
    plt.close()


options = [
    ParameterEstimation.OFF,
    ParameterEstimation.LOW,
    ParameterEstimation.MEDIUM,
    ParameterEstimation.HIGH,
    ParameterEstimation.FULL,
]

# Settings
raw_data_dir = read_input_path(
    input_type=InputDataPath.MATHIEU_MERGED
)  # Check the possible input types

path_master_data_file = os.path.join(raw_data_dir, "parsed_data.json")
# Extract the minimum and maximum values for the parameters
param_bounds = extract_param_bounds(path_master_data_file)

main_dataset_path = r"C:\Users\mathi\Downloads\folder"
# main_dataset_path = r"/data/mdheer/parameter_estimation/"
# Names of the categories
categories = ["Total", "Correct", "Wrong", "Same"]


for op in options:

    mm_folder = os.path.join(main_dataset_path, op.value)
    graph_folder = os.path.join(main_dataset_path, f"graph_{op.value}")
    os.mkdir(mm_folder)
    os.mkdir(graph_folder)

    train_set = CustomDataset(
        json_data=generate_dict(
            pth=path_master_data_file,
            data_variant=DataVariant.OPTICAL_FLOW,
        ),
        parameter_estimation=op,
        classes={"plastic": 0, "fish": 1},
        length=1000,
        erroneous=0,
        unlabelled={},
    )

    dk_classifier = DomainKnowledgeClassifier(
        mm_input=op,
        param_bounds=param_bounds,
        mm_model_folder=mm_folder,
        similarity_parameter=InvertedSimilarity.WEIGHTED_MSE,
    )

    # Create DataLoaders for the training and validation sets
    train_loader = DataLoader(
        train_set,
        batch_size=1,
        shuffle=False,
        collate_fn=custom_collate_fn,
    )

    tot = 0
    corr = 0
    wrong = 0
    same = 0

    lst = []

    counter = 0
    for (
        _,
        trajectories_dict,
        labels,
        _,
        settings,
        mm_inputs,
        data_numbers,
        _,
    ) in train_loader:

        if counter == 50:
            break

        teacher_outputs_validate = dk_classifier.get_teacher_output(
            batched_traj=trajectories_dict,
            batched_settings=settings,
            input_trajects=mm_inputs,
            labels=labels,
            data_numbers=data_numbers,
        )

        i = teacher_outputs_validate.tolist()[0]
        j = labels.item()

        outcome = ""

        if i[0] > i[1]:
            if j == 0.0:
                corr += 1
                outcome = True
            else:
                wrong += 1
                outcome = False
        elif i[0] < i[1]:
            if j == 1.0:
                corr += 1
                outcome = True
            else:
                wrong += 1
                outcome = False
        else:
            same += 1
            outcome = False

        graph_path = os.path.join(graph_folder, data_numbers[0])
        plot_comparison(dk_classifier.df_dict, labels[0], graph_path)

        tot += 1
        dd = {
            "data_numbers": data_numbers[0],
            "teacher_output": i,
            "label": j,
            "outcome": outcome,
        }
        lst.append(dd)
        counter += 1

    # Corresponding values
    values = [tot, corr, wrong, same]
    acc = corr / tot
    print(f"{op} - Total: {tot}, Correct: {acc}, Wrong: {wrong/tot}, Same: {same/tot}")

    # Creating the bar plot
    plt.bar(categories, values)

    # Adding title and labels
    plt.title("Counts by Category")
    plt.xlabel("Category")
    plt.ylabel("Count")

    # Showing the plot
    plt.savefig(os.path.join(main_dataset_path, f"plot_{op}_{acc}.png"))
    plt.close()

    df = pd.DataFrame(lst)
    df.to_excel(os.path.join(main_dataset_path, f"{op}_{acc}.xlsx"))
