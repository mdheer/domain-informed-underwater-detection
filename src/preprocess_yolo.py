import os
import random

from src.tools.enums import InputDataPath
from src.tools.general_functions import read_input_path
from src.data_preprocessing.yolo_annotation_converter import (
    process_folder,
    display_image_with_boxes,
)


if __name__ == "__main__":
    raw_folder_path = read_input_path(input_type=InputDataPath.MATHIEU_24_07)

    base_input_folder = raw_folder_path
    base_output_folder = raw_folder_path

    debug_num = 10

    # Loop through each folder numbered 1, 2, 3, etc.
    for folder_name in os.listdir(base_input_folder):
        if folder_name.isdigit():  # Check if the folder name is a number
            input_folder = os.path.join(base_input_folder, folder_name)
            output_folder = os.path.join(base_output_folder, folder_name, "yolo_labels")

            # Create the output folder if it does not exist
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            annotations_path = os.path.join(input_folder, "annotations.json")
            settings_path = os.path.join(input_folder, "settings.json")

            process_folder(input_folder, output_folder, annotations_path, settings_path)

    files = os.listdir(base_input_folder)
    random_files = random.sample(files, debug_num)

    for i in random_files:
        folder_path = os.path.join(base_input_folder, i)
        yolo_folder = os.path.join(folder_path, "yolo_labels")

        files = [f for f in os.listdir(yolo_folder) if ".txt" in f]

        yolo_label_name = random.sample(files, 1)[0]

        yolo_path = os.path.join(yolo_folder, yolo_label_name)
        img_name = yolo_label_name[: -len(".txt")] + ".jpg"
        img_path = os.path.join(folder_path, img_name)
        print(img_path)
        display_image_with_boxes(image_path=img_path, label_path=yolo_path)
