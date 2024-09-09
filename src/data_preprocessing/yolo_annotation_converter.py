import json
import os
from PIL import Image
from typing import List
import cv2
import matplotlib.pyplot as plt


class YoloAnnotationConverter:
    """
    A class to convert bounding box coordinates to YOLO format.

    This class is used for converting bounding box coordinates specified in a standard format (xMin, xMax, yMin, yMax) into the YOLO format, which is normalized to the image dimensions and specified as (class_id, x_center, y_center, width, height).

    Attributes:
        image_path (str): Path to the image file.
        img_width (int): Width of the image in pixels.
        img_height (int): Height of the image in pixels.
        settings (dict): Configuration settings including object type and other parameters.
    """

    def __init__(
        self, image_path: str, img_width: int, img_height: int, settings: dict
    ) -> None:
        """
        Initializes the YoloAnnotationConverter with image details and settings.

        Parameters:
            image_path (str): The path to the image file.
            img_width (int): The width of the image in pixels.
            img_height (int): The height of the image in pixels.
            settings (dict): A dictionary of settings, including object type and other configurations.
        """

        self.image_path = image_path
        self.img_width = img_width
        self.img_height = img_height
        self.settings = settings

    def read_json_file(self, file_path):
        with open(file_path, "r") as file:
            return json.load(file)

    def convert_to_yolo_format(self, bbox, class_id):
        """
        Converts a single bounding box to YOLO format.

        Parameters:
        - bbox: A dictionary with xMin, xMax, yMin, yMax keys.
        - class_id: The integer class ID of the object.

        Returns:
        - A list representing the YOLO formatted bounding box.
        """
        x_center = (bbox["xMin"] + bbox["xMax"]) / 2
        y_center = (bbox["yMin"] + bbox["yMax"]) / 2
        width = bbox["xMax"] - bbox["xMin"]
        height = bbox["yMax"] - bbox["yMin"]

        # Fix the coordinate system
        y_center = self.img_height - y_center

        # Normalize coordinates
        x_center /= self.img_width
        y_center /= self.img_height
        width /= self.img_width
        height /= self.img_height

        return [class_id, x_center, y_center, width, height]

    def get_class_id(self, object_type):
        """
        Determines the class ID based on the object type string.

        Parameters:
        - object_type: A string describing the object type.

        Returns:
        - An integer representing the class ID.
        """
        object_type_lower = object_type.lower()
        if "plastic" in object_type_lower:
            return 0  # Class ID for plastic
        elif "fish" in object_type_lower:
            return 1  # Class ID for fish
        else:
            return -1  # Undefined class

    def generate_yolo_annotations(self, bbox, class_id):
        """
        Wrapper function to generate YOLO annotations for one bbox.

        Parameters:
        - bbox: The bounding box dictionary.
        - class_id: The class ID for the object type.

        Returns:
        - A YOLO formatted bounding box as a list.
        """
        yolo_bbox = self.convert_to_yolo_format(bbox, class_id)
        return yolo_bbox


def get_sorted_image_files(input_folder: str) -> List[str]:
    """
    Retrieves and sorts image files from the input folder based on frame sequence.

    Parameters:
        input_folder (str): The folder containing image files.

    Returns:
        List[str]: A sorted list of image file names.
    """
    # Get list of all image files
    img_files = [
        f for f in os.listdir(input_folder) if f.endswith(".jpg") or f.endswith(".png")
    ]
    # Sort files based on the frame sequence number
    img_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
    return img_files


def process_folder(
    input_folder: str, output_folder: str, annotations_path: str, settings_path: str
) -> None:
    """
    Processes a folder of images, reads annotations and settings, and generates YOLO annotations.

    This function processes each image in the folder, reads bounding box annotations and settings, and generates corresponding YOLO annotation files.

    Parameters:
        input_folder (str): The folder containing the image files.
        output_folder (str): The folder where the YOLO annotation files will be saved.
        annotations_path (str): The path to the JSON file with bounding box data.
        settings_path (str): The path to the JSON file with settings data.
    """
    # Read settings.json
    with open(settings_path, "r") as file:
        settings = json.load(file)

    # Read annotations.json
    with open(annotations_path, "r") as file:
        annotations_data = json.load(file)
    bounding_boxes = annotations_data["boundingBoxCoordinates"]

    # Sort the image files based on the frame sequence
    img_files = get_sorted_image_files(input_folder)

    for bbox, img_file in zip(bounding_boxes, img_files):
        image_path = os.path.join(input_folder, img_file)
        # Get image dimensions
        with Image.open(image_path) as img:
            img_width, img_height = img.size

        # Create converter instance
        converter = YoloAnnotationConverter(image_path, img_width, img_height, settings)
        class_id = converter.get_class_id(settings["objectType"])

        # Generate YOLO annotation
        yolo_annotation = converter.generate_yolo_annotations(bbox, class_id)

        # Save the YOLO annotation to the output folder
        output_file_path = os.path.join(
            output_folder, os.path.splitext(img_file)[0] + ".txt"
        )
        with open(output_file_path, "w") as output_file:
            output_file.write(" ".join(map(str, yolo_annotation)) + "\n")

        print(f"Annotation for {img_file} saved to {output_file_path}")


def display_image_with_boxes(image_path, label_path):
    # Load the image
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    # Read the YOLO label file
    with open(label_path, "r") as file:
        lines = file.readlines()

    for line in lines:
        class_id, x_center, y_center, box_width, box_height = map(float, line.split())

        # Convert normalized positions to pixel values
        x_center, box_width = x_center * width, box_width * width
        y_center, box_height = y_center * height, box_height * height

        # Calculate the top left corner of the bounding box
        x_start = int(x_center - box_width / 2)
        y_start = int(y_center - box_height / 2)

        # Draw the bounding box
        cv2.rectangle(
            image,
            (x_start, y_start),
            (int(x_start + box_width), int(y_start + box_height)),
            (255, 0, 0),
            2,
        )

    # Display the image
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()
