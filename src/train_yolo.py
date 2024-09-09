from ultralytics import YOLO

import os
import pandas as pd
import json
import yaml
import pandas as pd
import sys
from typing import Tuple
import re
import json
import shutil


from src.tools.general_functions import sort_filenames


def get_dataset_path(name: str) -> Tuple[str, str]:
    """
    Get the paths for a dataset and its data distribution file.

    Parameters:
        name (str): Name of the dataset.
        num_samples (int): Number of samples.

    Returns:
        Tuple[str, str]: A tuple containing the YAML file path and distribution file path.
    """
    dir_name = os.path.join(f"./datasets", name)
    yaml_path = os.path.join(dir_name, f"{name}.yaml")
    subdir_name = os.path.join(dir_name, "distributions")

    return yaml_path, subdir_name


def create_or_update_symlink(source: str, destination: str) -> None:
    """
    Create or update a symbolic link from source to destination.

    Parameters:
        source (str): Path to the source file or directory.
        destination (str): Path to the destination symlink.
    """

    # Check if the destination symlink/file already exists
    if os.path.exists(destination) or os.path.islink(destination):
        os.remove(destination)  # Remove if it exists
    os.symlink(source, destination)


def prepare_dataset(
    yaml_file_path: str,
    distribution_file_path: str,
) -> None:
    """
    Prepare a dataset by creating symbolic links based on distribution information.

    Parameters:
        yaml_file_path (str): Path to the YAML configuration file.
        distribution_file_path (str): Path to the distribution JSON file.
    """

    # Delete content files

    def clear_folders(folder_paths):
        """
        Delete the contents of the specified folders.

        Args:
        folder_paths (list of str): Paths to the folders to be cleared.
        """
        for folder_path in folder_paths:
            for item in os.listdir(folder_path):
                item_path = os.path.join(folder_path, item)
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.unlink(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            print(f"Cleared contents of {folder_path}")

    # Specify the folders to clear
    folders_to_clear = [
        "./yolo_data/images/train",
        "./yolo_data/images/val",
        "./yolo_data/images/test",
        "./yolo_data/labels/train",
        "./yolo_data/labels/val",
        "./yolo_data/labels/test",
    ]

    # Clear the folders
    clear_folders(folders_to_clear)

    # Read yaml file content
    with open(yaml_file_path, "r") as file:
        yaml_data = yaml.safe_load(file)

    with open(distribution_file_path) as file:
        distr_data = json.load(file)

    data_path = distr_data["data_path"][: -len("_parsed_data.json")]

    train_df = pd.DataFrame(distr_data["train"])
    val_df = pd.DataFrame(distr_data["validate"])
    test_df = pd.DataFrame(distr_data["test"])

    train_data = [int(i[len("data_") :]) for i in train_df.index.tolist()]
    val_data = [int(i[len("data_") :]) for i in val_df.index.tolist()]
    test_data = [int(i[len("data_") :]) for i in test_df.index.tolist()]

    main_dst_path = yaml_data["train"][: -len("/images/test")]
    dst_img_path = os.path.join(main_dst_path, "images")
    dst_label_path = os.path.join(main_dst_path, "labels")

    # Go through all the lines
    for key, specific_set in {
        "train": train_data,
        "val": val_data,
        "test": test_data,
    }.items():
        for data_point in specific_set:
            img_folder = os.path.join(data_path, str(data_point))
            label_folder = os.path.join(img_folder, "yolo_labels")
            for label in os.listdir(label_folder):
                label_path = os.path.join(label_folder, label)
                img_path = os.path.join(img_folder, label[: -len(".txt")] + ".jpg")

                dst_label = os.path.join(dst_label_path, key)
                dst_image = os.path.join(dst_img_path, key)

                create_or_update_symlink(
                    source=label_path, destination=os.path.join(dst_label, label)
                )
                create_or_update_symlink(
                    source=img_path,
                    destination=os.path.join(dst_image, label[: -len(".txt")] + ".jpg"),
                )


def get_test_dataset_path(yaml_file):
    with open(yaml_file, "r") as file:
        data = yaml.safe_load(file)
        base_path = data.get("path")
        test_path = data.get("test")
        return os.path.join(base_path, test_path) if base_path and test_path else None


def get_best_weights() -> Tuple[str, str]:
    directory = "./runs/detect"
    highest_num = -1
    latest_folder = None

    for folder in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, folder)):
            match = re.search(r"train(\d*)", folder)
            if match:
                num = int(match.group(1)) if match.group(1) else 0
                if num > highest_num:
                    highest_num = num
                    latest_folder = folder

    latest_folder_path = os.path.join(directory, latest_folder)
    weights_path = os.path.join(latest_folder_path, "weights")
    best_weights_path = os.path.join(weights_path, "best.pt")

    return best_weights_path, latest_folder_path


def read_label_file(image_path):
    """
    Read the label file corresponding to the given image file.

    Parameters:
    image_path (str): Path to the image file.

    Returns:
    list of str: Contents of the label file, with each line as an element in the list.
    """
    # Change the extension from image format (e.g., .jpg) to .txt
    label_path = os.path.splitext(image_path)[0] + ".txt"
    label_path = label_path.replace("images", "labels")

    # Check if the label file exists
    if not os.path.exists(label_path):
        print(f"No label file found for {image_path}")
        return []

    class_ids = []
    with open(label_path, "r") as file:
        for line in file:
            # Extract the class ID (first element) and convert to int
            class_id = int(line.split()[0])
            class_ids.append(class_id)

    return class_ids


# # Example usage
# image_path = "/path/to/your/image.jpg"
# labels = read_label_file(image_path)
# print(labels)


if __name__ == "__main__":
    # Create yaml file
    dataset_name = "1.ideal_train_noise_test"
    run_id = "I"
    num_epoch = 20

    yaml_path, distribution_files_path = get_dataset_path(dataset_name)
    distribution_files = sort_filenames(os.listdir(distribution_files_path))

    test_data_path = get_test_dataset_path(yaml_path)

    for distr_file in distribution_files:
        distribution_file_path = os.path.join(distribution_files_path, distr_file)
        print("Started preparing dataset")
        prepare_dataset(
            yaml_file_path=yaml_path, distribution_file_path=distribution_file_path
        )
        print("Finished preparing dataset")

        # Training
        experiment_name = f"{run_id}_{distr_file[:-len('.json')]}_{dataset_name}"
        model = YOLO()
        model.train(
            data=yaml_path, epochs=num_epoch, name=experiment_name
        )  # train the model

        # Test
        weights_path, train_folder_path = get_best_weights()
        model = YOLO(weights_path)

        # Perform inference on test data
        test_results = model(test_data_path)

        detections = []

        for result in test_results:
            file_name = result.path  # Extract the file name from the result
            for cls_idx, conf, bbox in zip(
                result.boxes.cls, result.boxes.conf, result.boxes.xyxy
            ):
                detection = {
                    "file_name": file_name,
                    "label": read_label_file(os.path.join(test_data_path, file_name)),
                    "class": int(cls_idx),
                    "confidence": float(conf),
                    "bbox": bbox.cpu()
                    .numpy()
                    .tolist(),  # Convert bbox tensor to a list
                }
                detections.append(detection)

        # Serialize the detections dictionary to a JSON string
        json_string = json.dumps(detections, indent=4)

        # Write the JSON string to a file
        with open(
            os.path.join(train_folder_path, "test_detections.json"), "w"
        ) as json_file:
            json_file.write(json_string)

        print("Detections saved to detections.json")
