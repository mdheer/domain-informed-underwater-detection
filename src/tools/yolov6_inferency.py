import os
import torch
import cv2
from pathlib import Path


input_path = r"C:\Users\mathi\Documents\Files\University of Technology Delft\Master\MSc 2\Thesis\Data\Dummy_images\unpacked"

image_path = os.path.join(input_path, "frame0100.jpg")
image = cv2.imread(image_path)

# Set the paths to the YOLOv6 repository and the input image
yolo_path = Path(
    r"C:\Users\mathi\Documents\Files\University of Technology Delft\Master\MSc 2\Thesis\Code\thesis_mathieu\model_repos\YOLOv6"
)

model = torch.hub.load(
    r"C:\Users\mathi\Documents\Files\University of Technology Delft\Master\MSc 2\Thesis\Code\thesis_mathieu\model_repos\YOLOv6",
    "yolov6s",
    source="local",
)

# Set the device to use for inference (either "cpu" or "cuda")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

prediction = model.predict(image_path)
# prediction = model_custom.predict(img_path)
print(prediction)
detection_results = prediction

# Draw the bounding box on the image
box = detection_results["boxes"][0]
label = detection_results["classes"][0]
score = detection_results["scores"][0]
color = (0, 255, 0)  # Green
thickness = 2
cv2.rectangle(
    image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, thickness
)
text = f"{label} {score:.2f}"
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
text_color = (0, 0, 255)  # Red
text_thickness = 2
text_size, _ = cv2.getTextSize(text, font, font_scale, text_thickness)
text_x = int(box[0])
text_y = int(box[1]) - text_size[1] - 5
cv2.putText(image, text, (text_x, text_y), font, font_scale, text_color, text_thickness)

# Display the output image with the bounding box and label
cv2.imshow("Object Detection Results", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
