import cv2
import os

# Path to the folder containing the images
image_folder = r"C:\Users\mathi\Documents\Files\University of Technology Delft\Master\MSc 2\Thesis\Data\Dummy_images\masks"

# Path to the output video file
video_name = r"C:\Users\mathi\Documents\Files\University of Technology Delft\Master\MSc 2\Thesis\Data\Dummy_images\masks\output.avi"

# Frame rate of the output video
fps = 30

# Get a list of all the image files in the folder
images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]

# Sort the image files by name
# images.sort(key=lambda x: int(x.split(".")[0]))

# Set the size of the output video to match the size of the first image
img = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = img.shape
size = (width, height)

# Create a VideoWriter object to write the video
video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*"MJPG"), fps, size)

# Loop through the images and add them to the video
for image in images:
    img_path = os.path.join(image_folder, image)
    img = cv2.imread(img_path)
    video.write(img)

# Release the VideoWriter object and close the video file
video.release()
cv2.destroyAllWindows()
