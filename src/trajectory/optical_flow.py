import cv2
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
from typing import List, Optional


def custom_sort(input_list: List[str]) -> List[str]:
    """
    Sorts a list of filenames (assumed to be .jpg) based on the integer value embedded in each filename.

    Parameters:
        input_list (List[str]): A list of filenames to be sorted.

    Returns:
        List[str]: A sorted list of filenames.
    """
    jpg_files = [f for f in input_list if f.endswith(".jpg")]
    return sorted(jpg_files, key=lambda s: int(s.split("_")[1].split(".jpg")[0]))


class NotEnoughImagesForOpticalFlow(Exception):
    """
    Custom exception for handling cases where there are not enough images to perform optical flow.
    """

    pass


class DenseOpticalFlow:
    """
    This class computes dense optical flow between pairs of images in a specified directory and saves the
    resulting flow visualizations to an output directory.

    Attributes:
        input_path (str): Directory path where input images are stored.
        output_path (str): Directory path where output flow visualizations are to be saved.
        max_iter (int, optional): Maximum number of iterations for processing pairs of images.

    Methods:
        _optical_flow(str, str, str): Computes and saves the optical flow between two images as a heatmap.
        _is_image(str): Checks if the file is an image.
        _count_image_files(list): Counts the number of image files in a list of filenames.
        _perform_optical_flow(Optional[int]): Performs optical flow calculations on all or a specified number of image pairs in the input directory.

    """

    def __init__(
        self, input_path: str, output_path: str, max_iter: Optional[int] = None
    ) -> None:
        """
        Initializes the DenseOpticalFlow object with input and output paths and an optional maximum iteration limit.

        Parameters:
            input_path (str): Path to the directory containing the input images.
            output_path (str): Path to the directory where output flow visualizations will be saved.
            max_iter (int, optional): Maximum number of image pairs to process.
        """

        self.input_path = input_path
        self.output_path = output_path
        self.max_iter = max_iter

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        self._perform_optical_flow(max_iter=self.max_iter)

    def _optical_flow(self, image_path1: str, image_path2: str, final_path: str) -> str:
        """
        Computes and saves the optical flow between two images as a heatmap.

        Parameters:
            image_path1 (str): Path to the first image.
            image_path2 (str): Path to the second image.
            final_path (str): Path to save the output heatmap image.

        Returns:
            str: Path to the saved heatmap image.
        """

        # Load two images
        img1 = cv2.imread(image_path1)
        img2 = cv2.imread(image_path2)

        # Convert images to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

        # Convert optical flow to polar coordinates
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Create heatmap from magnitude
        heatmap = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = np.uint8(heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)

        # Display and save heatmap
        # cv2.imshow("Optical Flow", heatmap)
        filename = os.path.basename(image_path1)

        output_file_path = os.path.join(final_path, "mask" + filename)
        cv2.imwrite(output_file_path, heatmap)

        # Wait for key press and close window
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return output_file_path

    def _is_image(self, filename: str) -> bool:
        """
        Checks if the file is an image.

        Parameters:
            filename (str): The filename to check.

        Returns:
            bool: True if the file is an image, False otherwise.
        """

        try:
            with Image.open(filename) as img:
                return True
        except:
            return False

    def _count_image_files(self, filenames: list) -> int:
        """
        Counts the number of image files in a list of filenames.

        Parameters:
            filenames (list): A list of filenames.

        Returns:
            int: The count of image files.
        """

        # List of common image file extensions
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"]

        # Count filenames with image extensions
        image_count = sum(
            1
            for filename in filenames
            if any(filename.lower().endswith(ext) for ext in image_extensions)
        )

        return image_count

    def _perform_optical_flow(self, max_iter: Optional[int] = None) -> None:
        """
        Performs optical flow calculations on all or a specified number of image pairs in the input directory.

        Parameters:
            max_iter (Optional[int]): Maximum number of image pairs to process, if None, all pairs are processed.
        """

        total_lst = custom_sort(os.listdir(self.input_path))

        if (self._count_image_files(total_lst)) < 2:
            raise NotEnoughImagesForOpticalFlow(
                "The number of images present is too little to perform Optical flow"
            )

        # # Based on the value of max_iter, decide which loop to execute
        loop_range = len(total_lst) - 1 if max_iter is None else max_iter

        # loop_range = 1

        for i in tqdm(range(loop_range), desc="Performing optical flow"):
            # Get the file paths
            frame1_path = os.path.join(self.input_path, total_lst[i])
            frame2_path = os.path.join(self.input_path, total_lst[i + 1])

            # frame1_path = r"C:\Users\mathi\Downloads\optical_flow_test\images2\1.png"
            # frame2_path = r"C:\Users\mathi\Downloads\optical_flow_test\images2\2.png"

            # Check that the path leads to an image
            if self._is_image(frame1_path) and self._is_image(frame2_path):
                # Perform the optical flow operation
                self._optical_flow(frame1_path, frame2_path, self.output_path)
            else:
                print(f"{frame1_path} or {frame2_path} is not an image file")


if __name__ == "__main__":
    input_path = r"C:\Users\mathi\Downloads\optical_flow_test\images"
    output_path = r"C:\Users\mathi\Downloads\optical_flow_test\masks"

    optical_flow = DenseOpticalFlow(input_path=input_path, output_path=output_path)
