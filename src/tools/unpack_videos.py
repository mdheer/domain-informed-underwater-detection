import cv2
import os


def unpack_video_to_images(video_path, output_path):
    # Open the video file
    vidcap = cv2.VideoCapture(video_path)

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Read the frames from the video and save them as images
    count = 0
    success = True
    while success:
        success, image = vidcap.read()
        if success:
            # Save the frame as an image in the output directory
            image_path = os.path.join(output_path, f"frame{count:04d}.jpg")
            cv2.imwrite(image_path, image)
            count += 1

    # Release the video file
    vidcap.release()

    print(f"Unpacked {count} frames from {video_path} to {output_path}")


if __name__ == "__main__":
    # Example usage: unpack a video located at "path/to/video.mp4" to images in "path/to/output/directory"
    video_path = r"C:\Users\mathi\Documents\Files\University of Technology Delft\Master\MSc 2\Thesis\Data\Sim_v1\sim_results_v1.mp4"
    output_path = r"C:\Users\mathi\Documents\Files\University of Technology Delft\Master\MSc 2\Thesis\Data\Sim_v1"
    unpack_video_to_images(video_path, output_path)
