import os
import cv2

# Set the directories for images and masks
image_dir = r"C:\Users\mathi\Documents\Files\University of Technology Delft\Master\MSc 2\Thesis\Data\Dummy_images\unpacked"
mask_dir = r"C:\Users\mathi\Documents\Files\University of Technology Delft\Master\MSc 2\Thesis\Data\Dummy_images\masks"

# Get the list of image and mask filenames
image_filenames = sorted(os.listdir(image_dir))
mask_filenames = sorted(os.listdir(mask_dir))

# Read the first image to get the dimensions and create a VideoWriter object
first_image = cv2.imread(os.path.join(image_dir, image_filenames[0]))
height, width, layers = first_image.shape
video_filename = "output_video.avi"

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*"XVID")
video_writer = cv2.VideoWriter(video_filename, fourcc, 30.0, (width * 2, height))

# Iterate through the images and masks, read them, horizontally align them, and write to the video
for image_filename, mask_filename in zip(image_filenames, mask_filenames):
    image_path = os.path.join(image_dir, image_filename)
    mask_path = os.path.join(mask_dir, mask_filename)

    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path)

    # Horizontally align the image and mask
    combined = cv2.hconcat([image, mask])

    # Write the combined image to the video
    video_writer.write(combined)

# Release the VideoWriter object
video_writer.release()
