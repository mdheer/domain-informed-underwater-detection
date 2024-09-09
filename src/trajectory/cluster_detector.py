from typing import List

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import sys


class CentroidFinder:
    """
    A class to identify and locate clusters of similar colors in an image using k-means clustering.
    The class processes the image using OpenCV and NumPy libraries and visualizes the results with
    Matplotlib. It finds color-based clusters, extracts position-based clusters from these, and
    merges close centers of position-based clusters.

    Attributes:
        image (np.ndarray): The loaded image on which clustering is performed.
        img_name (str): The name of the image file.
        debug_path (str): Path for saving debug images.
        debug (bool): Indicates whether debugging is enabled.
        centers_list (list): List of centers of position-based clusters.
        parsed_centers_lst (list): List of parsed centers after merging close centers.

    Methods:
        get_image: Loads and returns an RGB image from a file.
        kmeans_clustering: Applies k-means clustering to a set of samples.
        find_colour_based_clusters: Finds clusters of similar colors in the input image.
        find_position_based_clusters: Finds position-based clusters within a binary array.
        merge_close_centers: Merges position-based clusters that are too close.
        draw_final_clusters: Draws circles at the center of each cluster and displays the image.
        debug_colour_clusters: Plots initial image and output clusters in different colors.
        debug_position_clusters: Plots input coordinates and output clusters in different colors.
        parse_centers_lst: Converts object locations from integer to float values.
    """

    def __init__(self, img_path: str, debug_path=None):
        """
        Initializes the CentroidFinder with the provided image path. It processes the image to
        find clusters and optionally saves debug images.

        Parameters:
            img_path (str): Path to the input image file.
            debug_path (str, optional): Path to save debug images. If None, debugging is disabled.
        """
        self.image = self.get_image(img_path)
        self.img_name = os.path.basename(img_path)
        self.debug_path = debug_path

        if not os.path.exists(self.debug_path):
            os.makedirs(self.debug_path)

        if self.debug_path == None:
            self.debug = False
        else:
            self.debug = True

        # Find the colour based clusters
        (
            colour_based_clusters,
            colour_based_centers,
            n_clusters,
        ) = self.find_colour_based_clusters()

        if self.debug:
            self.debug_colour_clusters(
                labels=colour_based_clusters, centers=colour_based_centers
            )

        self.centers_list = []  # Store the centers of the position-based clusters

        for i in range(
            n_clusters - 1
        ):  # k - 1 since there are k clusters, but we need k - 1 binary elements
            binary_arr = np.where(colour_based_clusters == i + 1, 1, 0).reshape(
                self.image.shape[:2]
            )  # convert array with all k clusters, to a binary array and reshaped to a 2D array. Skip the first cluster as it is background anyway
            (
                position_based_labels,
                position_based_centers,
                parsed_centers,
            ) = self.find_position_based_clusters(binary_arr)
            self.centers_list.append(parsed_centers)

            if self.debug:
                self.debug_position_clusters(
                    image=binary_arr,
                    position_cluster_labels=position_based_labels,
                    position_cluster_centers=position_based_centers,
                    n=i,
                )

        self.centers_list = self.merge_close_centers(self.centers_list)
        self.parsed_centers_lst = self.parse_centers_lst()

        if self.debug:
            self.draw_final_clusters(parsed_lst=self.parsed_centers_lst)

    def get_image(self, img_path: str) -> np.ndarray:
        """
        Loads an image from the given path and converts it to RGB format.

        Parameters:
            img_path (str): Path to the image file.

        Returns:
            np.ndarray: The loaded RGB image.
        """

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

    def kmeans_clustering(self, samples: np.ndarray, n_clusters: int) -> tuple:
        """
        Performs k-means clustering on the provided samples.

        Parameters:
            samples (np.ndarray): Array of samples for clustering.
            n_clusters (int): Number of clusters to form.

        Returns:
            tuple: A tuple containing cluster labels, cluster centers, and the number of iterations run.
        """

        parsed_samples = np.float32(samples)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

        _, labels, (centers) = cv2.kmeans(
            parsed_samples, n_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )

        return labels, centers

    def find_colour_based_clusters(self) -> tuple:
        """
        Identifies clusters based on color in the loaded image using k-means clustering.

        Returns:
            tuple: A tuple containing color-based cluster labels, cluster centers, and the number of clusters.
        """

        k = 3

        processed_img = self.image.reshape((-1, 3))
        labels, centers = self.kmeans_clustering(processed_img, k)
        labels = labels.flatten()

        centers = np.uint8(centers)

        counts = np.bincount(labels)
        img_arr = np.array(self.image)

        # Filter the cluster "background" out
        total_pixels = img_arr.shape[0] * img_arr.shape[1]
        percentages = counts / total_pixels * 100

        # Merge clusters covering more than 30%
        merged_labels = []
        for i in range(k):
            if percentages[i] > 20:
                merged_labels.append(i)

        if len(merged_labels) > 1:
            # Merge all merged_labels into a single cluster
            for i in range(len(labels)):
                if labels[i] in merged_labels:
                    labels[i] = merged_labels[0]

            # Update counts and percentages
            counts = np.bincount(labels)
            percentages = counts / total_pixels * 100

            # Update k to reflect the new number of clusters
            k = len(np.unique(labels))

        # Sort cluster indexes by their sizes
        sorted_cluster_indexes = np.argsort(counts)[::-1]

        # Create a mapping of old cluster indexes to new sequential indexes
        index_mapping = {
            old_idx: new_idx for new_idx, old_idx in enumerate(sorted_cluster_indexes)
        }

        # Reassign labels according to the new sequential indexes
        for i in range(len(labels)):
            labels[i] = index_mapping[labels[i]]

        return labels, centers, k

    def find_position_based_clusters(self, arr: np.ndarray) -> tuple:
        """
        Identifies position-based clusters within a binary array using k-means clustering.

        Parameters:
            arr (np.ndarray): A binary array representing a specific color cluster.

        Returns:
            tuple: A tuple containing position-based cluster labels, cluster centers, and the closest center.
        """

        k = 2

        ones_indices = np.argwhere(arr == 1)

        # Convert the indices to float32
        coordinates = np.float32(ones_indices)

        labels, centers = self.kmeans_clustering(coordinates, k)

        cluster_count = []
        for i in range(len(centers)):
            cluster_points = coordinates[labels.flatten() == i]
            cluster_count.append(len(cluster_points))

        closest_cluster_index = np.argmax(cluster_count)

        return labels, centers, centers[closest_cluster_index]

    def merge_close_centers(self, centers: List[np.ndarray]) -> List[np.ndarray]:
        """
        Merges centers of position-based clusters that are close to each other.

        Parameters:
            centers (List[np.ndarray]): List of cluster centers.

        Returns:
            List[np.ndarray]: Updated list of cluster centers after merging.
        """

        threshold = 0.10 * self.image.shape[1]  # 10% of the image width
        updated_centers = centers.copy()

        i = 0
        while i < len(updated_centers) - 1:
            j = i + 1
            while j < len(updated_centers):
                distance = np.linalg.norm(updated_centers[i] - updated_centers[j])
                if distance < threshold:
                    new_center = (updated_centers[i] + updated_centers[j]) / 2
                    updated_centers[i] = new_center
                    updated_centers = np.delete(updated_centers, j, axis=0)
                else:
                    j += 1
            i += 1

        return updated_centers

    def draw_final_clusters(self, parsed_lst: List[np.ndarray]) -> None:
        """
        Draws the final clusters on the image and displays it.

        Parameters:
            parsed_lst (List[np.ndarray]): List of final cluster centers.
        """
        # Draw parsed_lst clusters
        img_with_centers_parsed = self.image.copy()
        for center in parsed_lst:
            cv2.circle(
                img_with_centers_parsed,
                (int(center[1]), int(center[0])),
                10,
                (0, 255, 0),
                2,
            )

        # Plot the image
        plt.title("Parsed List")
        plt.imshow(cv2.cvtColor(img_with_centers_parsed, cv2.COLOR_BGR2RGB))

        # Save the figure
        file_final_path = os.path.join(
            self.debug_path, f"{self.img_name[:-4]}_final_clusters.jpg"
        )
        plt.savefig(file_final_path)
        plt.close()

    def debug_colour_clusters(self, labels: np.ndarray, centers: np.ndarray) -> None:
        """
        Generates debug images showing the color-based clusters.

        Parameters:
            labels (np.ndarray): Cluster labels.
            centers (np.ndarray): Cluster centers.
        """

        labels = labels.reshape(self.image.shape[:2])
        unique_labels = np.unique(labels)
        shape = labels.shape
        color = (255, 0, 0)

        # Create a new figure for the subplots
        fig, axes = plt.subplots(1, len(unique_labels) + 1, figsize=(15, 5))
        fig.suptitle("Initial Image and Output Clusters")

        # Show the initial image in the first subplot
        axes[0].imshow(self.image)
        axes[0].set_title("Initial Image")

        for idx, i in enumerate(unique_labels):
            # Create a blank image with the same shape as the labels array
            output_image = np.zeros((*shape, 3), dtype=np.uint8)

            # Set the pixel values of the cluster coordinates in the output_image to the corresponding color
            output_image[labels == i] = color

            # Show the output clusters image in a subplot
            axes[idx + 1].imshow(output_image)
            axes[idx + 1].set_title(f"Output Colour Clusters {i}")

        final_file_path = os.path.join(
            self.debug_path, f"{self.img_name[:-4]}_colour_cluster.jpg"
        )
        plt.savefig(final_file_path)
        plt.clf()
        plt.close()

    def debug_position_clusters(
        self,
        image: np.ndarray,
        position_cluster_labels: np.ndarray,
        position_cluster_centers: np.ndarray,
        n: int,
    ) -> None:
        """
        Generates debug images showing the position-based clusters.

        Parameters:
            image (np.ndarray): Binary image of a specific color cluster.
            position_cluster_labels (np.ndarray): Labels of position-based clusters.
            position_cluster_centers (np.ndarray): Centers of position-based clusters.
            n (int): Cluster number for labeling the debug image.
        """

        ones_indices = np.argwhere(image == 1)

        # Convert the indices to float32
        coordinates = np.float32(ones_indices)

        # Assign colors to different clusters
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

        unique_labels = np.unique(position_cluster_labels)
        shape = self.image.shape[:2]

        # Create a new figure for the subplots
        fig, axes = plt.subplots(1, len(unique_labels) + 1, figsize=(15, 5))
        fig.suptitle("Initial Image and Output Clusters")

        # Plot input coordinates
        input_image = np.zeros((*shape, 3), dtype=np.uint8)
        input_coordinates = coordinates.astype(int)
        input_image[input_coordinates[:, 0], input_coordinates[:, 1]] = (255, 255, 255)
        axes[0].imshow(input_image)
        axes[0].set_title("Initial image")

        for idx, i in enumerate(unique_labels):
            color = colors[i % len(colors)]

            # Create a blank image with 3 color channels
            output_image = np.zeros((*shape, 3), dtype=np.uint8)

            cluster_coordinates = coordinates[
                position_cluster_labels.flatten() == i
            ].astype(int)
            output_image[cluster_coordinates[:, 0], cluster_coordinates[:, 1]] = color

            # Show the output clusters image in a subplot
            axes[idx + 1].imshow(output_image)
            axes[idx + 1].set_title(f"Output Cluster {i}")

        file_final_path = os.path.join(
            self.debug_path, f"{self.img_name[:-4]}_position_cluster_{n}.jpg"
        )
        plt.savefig(file_final_path)
        plt.clf()
        plt.close()

    def parse_centers_lst(self) -> List[float]:
        """
        Converts cluster centers to a list of floating-point coordinates.

        Returns:
            List[float]: List of parsed cluster centers.
        """
        object_lst_per_frame = []
        for i in self.centers_list:
            object_lst_per_frame.append([float(i[0]), float(i[1])])

        return object_lst_per_frame


if __name__ == "__main__":
    r = r"C:\Users\mathi\Documents\Files\University of Technology Delft\Master\MSc 2\Thesis\Data\Dummy_images\masks\maskframe0001.jpg"
    b = r"C:\Users\mathi\Documents\Files\University of Technology Delft\Master\MSc 2\Thesis\Data\Dummy_images\masks\maskframe0245.jpg"

    centroidFinder = CentroidFinder(
        r,
        r"C:\Users\mathi\Documents\Files\University of Technology Delft\Master\MSc 2\Thesis\Data\Dummy_images\debug",
    )
