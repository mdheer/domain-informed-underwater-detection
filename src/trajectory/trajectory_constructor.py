from typing import List, Tuple, Dict
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import sys
from tqdm import tqdm

from ..trajectory.cluster_detector import CentroidFinder
from ..trajectory.optical_flow import custom_sort


class TrajectoryConstructor:
    """
    A class for constructing trajectories from masks, calculating velocities and accelerations.

    This class is used for constructing trajectories from binary masks, calculating object velocities and accelerations,
    and generating visualization outputs.

    Attributes:
        path_to_masks (str): Path to the directory containing binary masks.
        output_file_path (str): Path to the output file where the trajectory data will be saved.
        debug_path (str): Path to the directory for debugging and visualization outputs.
        width (int): Width of the images extracted from the masks.
        height (int): Height of the images extracted from the masks.
        locations (list): List of lists containing object coordinates in each frame.
        pos_lst (list): List of parsed object positions.
        vel_lst (list): List of object velocities.
        acc_lst (list): List of object accelerations.

    Methods:
        get_img_size: Get the size (width and height) of the images extracted from the masks.
        calc_centroids: Calculate centroids of objects in binary masks.
        unpack_lst: Unpack a list of tuples into separate lists.
        parse_data: Parse object positions, velocities, and accelerations into a dictionary.
        create_trajectory_graph: Create a scatter plot of trajectory coordinates.
        draw_coordinates_on_image: Draw circles at specified coordinates on an image.
        parse_locations: Parse object positions list to track a single object.
        calc_speed_and_acc: Calculate object velocities and accelerations from positions.

    """

    def __init__(
        self, path_to_masks: str, output_file_path: str, debug_path: str
    ) -> None:
        """
        Initializes the TrajectoryConstructor.

        Parameters:
            path_to_masks (str): Path to the directory containing binary masks.
            output_file_path (str): Path to the output file where the trajectory data will be saved.
            debug_path (str): Path to the directory for debugging and visualization outputs.
        """

        self.path_to_masks = path_to_masks
        self.output_file_path = output_file_path
        self.debug_path = debug_path

        if not os.path.exists(self.debug_path):
            os.makedirs(self.debug_path)

        self.width, self.height = self.get_img_size()

        # Find the centroids of the masks
        self.calc_centroids()
        self.create_trajectory_graph()

        self.pos_lst = self.parse_locations()

        self.vel_lst, self.acc_lst = self.calc_speed_and_acc(pos_lst=self.pos_lst)

    def get_img_size(self) -> Tuple[int, int]:
        """
        Get the size (width and height) of the images extracted from the masks.

        Returns:
            Tuple[int, int]: Width and height of the images.
        """

        lijstje = os.listdir(self.path_to_masks)
        img_name = lijstje[0]

        img = Image.open(os.path.join(self.path_to_masks, img_name))

        return img.size

    def calc_centroids(self) -> None:
        """
        Calculate centroids of objects in binary masks.

        This method calculates the centroids of objects in the binary masks stored in the masks directory
        and stores the centroid coordinates in the 'locations' attribute.

        """

        masks_lst = custom_sort(os.listdir(self.path_to_masks))
        self.locations = []

        for i in tqdm(masks_lst, desc="Calculting centroids"):
            mask_path = os.path.join(self.path_to_masks, i)
            centroids = CentroidFinder(
                img_path=mask_path,
                debug_path=os.path.join(self.debug_path, "centroids"),
            ).parsed_centers_lst
            self.locations.append(centroids)

    def unpack_lst(
        self, input_lst: List[Tuple[float, float]]
    ) -> Tuple[List[float], List[float]]:
        """
        Unpack a list of tuples into separate lists.

        Parameters:
            input_lst (List[Tuple[float, float]]): List of tuples to unpack.

        Returns:
            Tuple[List[float], List[float]]: Two separate lists (x_lst, y_lst).
        """
        x_lst = []
        y_lst = []

        for idx in range(len(input_lst)):
            x_lst.append(input_lst[idx][0])
            y_lst.append(input_lst[idx][1])

        return x_lst, y_lst

    def parse_data(self) -> Dict[str, List[Dict[str, float]]]:
        """
        Parse object positions, velocities, and accelerations into a dictionary.

        Returns:
            Dict[str, List[Dict[str, float]]]: Dictionary with trajectory data
            including position, velocity, acceleration, and file paths.
        """

        pos_x, pos_y = self.unpack_lst(self.pos_lst)
        vel_x, vel_y = self.unpack_lst(self.vel_lst)
        acc_x, acc_y = self.unpack_lst(self.acc_lst)

        pos_lst = []
        vel_lst = []
        acc_lst = []

        for x, y in zip(pos_x, pos_y):
            pos_lst.append({"x": x, "y": y})

        for x, y in zip(vel_x, vel_y):
            vel_lst.append({"x": x, "y": y})

        for x, y in zip(acc_x, acc_y):
            acc_lst.append({"x": x, "y": y})

        trajectory = {
            "file_paths": [
                os.path.join(self.path_to_masks, f)
                for f in custom_sort(os.listdir(self.path_to_masks))
                if os.path.isfile(os.path.join(self.path_to_masks, f))
            ],
            "position": pos_lst,
            "velocity": vel_lst,
            "acceleration": acc_lst,
        }

        return trajectory

    def create_trajectory_graph(self) -> None:
        """
        Create a scatter plot of trajectory coordinates using Matplotlib.

        This method creates a scatter plot of trajectory coordinates and saves it as an image.

        """
        # Extract the x and y coordinates from the list of lists
        x = []
        y = []
        for i in self.locations:
            for j in i:
                x.append(j[0])
                y.append(j[1])

        # Create a scatter plot of the coordinates using Matplotlib
        plt.scatter(y, x)
        plt.xlim(0, self.width)
        plt.ylim(0, self.height)
        plt.savefig(os.path.join(self.debug_path, "trajectory.jpg"))

    def draw_coordinates_on_image(
        self,
        image_path: str,
        coordinates: List[Tuple[int, int]],
        point_radius: int = 5,
        point_color: Tuple[int, int, int] = (255, 0, 0),
    ) -> None:
        """
        Draw circles at the given coordinates on the input image and save the annotated image.

        Parameters:
            image_path (str): Path to the input image.
            coordinates (List[Tuple[int, int]]): List of (x, y) coordinates to be annotated on the image.
            output_path (str, optional): Path to save the annotated image. Defaults to "output_image.jpg".
            point_radius (int, optional): Radius of the circles drawn on the image. Defaults to 5.
            point_color (Tuple[int, int, int], optional): Color of the circles drawn on the image. Defaults to (255, 0, 0).

        Returns:
            None
        """
        # Open the image
        image = Image.open(image_path)

        # Create a drawing object
        draw = ImageDraw.Draw(image)

        output_path = os.path.join(self.debug_path, "trajectory.jpg")
        # Draw points at the given coordinates
        for y, x in coordinates:
            draw.ellipse(
                (
                    x - point_radius,
                    y - point_radius,
                    x + point_radius,
                    y + point_radius,
                ),
                fill=point_color,
            )

        # Save the image with points drawn
        image.save(output_path)

    def parse_locations(self) -> List[Tuple[int, int]]:
        """
        Parse object positions list to track a single object.

        Returns:
            List[Tuple[int, int]]: List of parsed object positions.
        """

        parsed_lst = []

        def fill_empty_with_next(data):
            # Find the first non-empty element starting from the beginning
            next_valid_index = next((i for i, item in enumerate(data) if item), None)

            # If there is no non-empty element, return the original data
            if next_valid_index is None:
                return data

            # Fill the preceding elements with the next non-empty element
            for i in range(next_valid_index, -1, -1):
                data[i] = data[next_valid_index]

            # Continue the original function logic from the first non-empty element
            for i in range(next_valid_index, len(data)):
                if not data[i]:
                    data[i] = data[i - 1]

            return data

        # Required to solve small bug where list has an empty element
        self.locations = fill_empty_with_next(self.locations)
        print(self.locations)
        for i in self.locations:
            first_element = i[0]
            switch = [first_element[1], self.height - first_element[0]]
            parsed_lst.append(switch)

        return parsed_lst

    def calc_speed_and_acc(
        self, pos_lst: List[Tuple[int, int]], framerate: int = 30
    ) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """
        Calculate object velocities and accelerations from positions.

        Parameters:
            pos_lst (List[Tuple[int, int]]): List of object positions.
            framerate (int, optional): Frame rate (frames per second). Defaults to 30.

        Returns:
            Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]: Lists of object velocities and accelerations.
        """
        elps_time = 1 / framerate

        vel_lst = []
        acc_lst = []

        for idx in range(len(pos_lst) - 1):
            x1 = pos_lst[idx][0]
            y1 = pos_lst[idx][1]
            x2 = pos_lst[idx + 1][0]
            y2 = pos_lst[idx + 1][1]

            vx = (x2 - x1) / elps_time
            vy = (y2 - y1) / elps_time

            vel_lst.append([vx, vy])

        for idx in range(len(vel_lst) - 1):
            vx1 = vel_lst[idx][0]
            vy1 = vel_lst[idx][1]
            vx2 = vel_lst[idx + 1][0]
            vy2 = vel_lst[idx + 1][1]

            ax = (vx2 - vx1) / (elps_time)
            ay = (vy2 - vy1) / (elps_time)

            acc_lst.append([ax, ay])

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

        frames = range(len(vel_lst))
        ax1.plot(frames, [v[0] for v in vel_lst], label="Vx")
        ax1.plot(frames, [v[1] for v in vel_lst], label="Vy")
        ax1.set_ylabel("Velocity [pixel/s]")
        ax1.legend()

        frames = range(len(acc_lst))
        ax2.plot(frames, [a[0] for a in acc_lst], label="Ax")
        ax2.plot(frames, [a[1] for a in acc_lst], label="Ay")
        ax2.set_xlabel("Frames")
        ax2.set_ylabel("Acceleration [pixel/s^2]")
        ax2.legend()

        plt.savefig(os.path.join(self.debug_path, "speed_acc_graph.jpg"))

        return vel_lst, acc_lst


if __name__ == "__main__":
    trajectory_constructor = TrajectoryConstructor(
        path_to_masks=r"C:\Users\mathi\Documents\Files\University of Technology Delft\Master\MSc 2\Thesis\Data\Dummy_images\masks",
        output_file_path="trajss.json",
        debug_path="",
    )
