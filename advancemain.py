import cv2
import random
import numpy as np
from util import get_detections  # Ensure this function is implemented correctly

# Define paths
cfg_path = r'./models/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt'
weights_path = r'./models/frozen_inference_graph.pb'
img_path = r'./images/cat.png'

# Load image
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"Image not found at {img_path}")

H, W, C = img.shape

# Load model
net = cv2.dnn.readNetFromTensorflow(weights_path, cfg_path)
if net.empty():
    raise FileNotFoundError("Failed to load model. Check paths to .pb and .pbtxt files.")

# Convert image to blob
blob = cv2.dnn.blobFromImage(img, swapRB=True, crop=False)

# Get detections (boxes and masks)
boxes, masks = get_detections(net, blob)

# Generate random colors for each class
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(80)]

# Create an empty image for mask overlay
empty_img = np.zeros((H, W, C), dtype=np.uint8)

# Detection threshold
detection_threshold = 0.5

# Process each detection
for j in range(len(masks)):
    bbox = boxes[0, 0, j]
    class_id = int(bbox[1])
    score = bbox[2]

    if score > detection_threshold:
        # Extract mask for the detected object
        mask = masks[j][class_id]
        mask = cv2.resize(mask, (W, H))  # Resize mask to original image size

        # Threshold the mask to create a binary mask
        _, mask = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)

        # Get bounding box coordinates
        x1, y1, x2, y2 = int(bbox[3] * W), int(bbox[4] * H), int(bbox[5] * W), int(bbox[6] * H)

        # Ensure coordinates are within image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)

        # Apply the mask to the empty image
        for c in range(3):
            empty_img[y1:y2, x1:x2, c] = mask[y1:y2, x1:x2] * colors[class_id][c]

# Create an overlay of the mask and the original image
overlay = cv2.addWeighted(img, 0.6, empty_img, 0.4, 0)

# Display results
cv2.imshow('Mask', empty_img)
cv2.imshow('Original Image', img)
cv2.imshow('Overlay', overlay)

cv2.waitKey(0)
cv2.destroyAllWindows()