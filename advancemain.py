import cv2
import random
import numpy as np
 
# Define paths
cfg_path = r'./models/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt'
weights_path = r'./models/frozen_inference_graph.pb'
img_path = r'./images/cat.png'
 
# Define display width
display_width = 200  # Set your desired display width

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

# Set input to the network
net.setInput(blob)

# Forward pass to get detections and masks
boxes, masks = net.forward(['detection_out_final', 'detection_masks'])

# Generate random colors for each class
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(80)]

# Create an empty image for mask visualization
empty_img = np.zeros((H, W, C), dtype=np.uint8)

# Detection threshold
detection_threshold = 0.5

# Process detections
for i in range(boxes.shape[2]):
    class_id = int(boxes[0, 0, i, 1])
    score = boxes[0, 0, i, 2]

    if score > detection_threshold:
        # Get bounding box coordinates
        x1, y1, x2, y2 = int(boxes[0, 0, i, 3] * W), int(boxes[0, 0, i, 4] * H), \
                         int(boxes[0, 0, i, 5] * W), int(boxes[0, 0, i, 6] * H)

        # Ensure coordinates are within image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)

        # Get the mask for the current detection
        mask = masks[i, class_id]
        mask = cv2.resize(mask, (x2 - x1, y2 - y1))
        _, mask = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)

        # Create a colored mask
        colored_mask = np.zeros((y2 - y1, x2 - x1, 3), dtype=np.uint8)
        for c in range(3):
            colored_mask[:, :, c] = mask * colors[class_id][c]

        # Overlay the colored mask on the empty image
        empty_img[y1:y2, x1:x2] = colored_mask

# Create an overlay of the mask and the original image
overlay = cv2.addWeighted(img, 0.6, empty_img, 0.4, 0)

# Resize images for display
def resize_with_aspect_ratio(image, width=None, height=None):
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        ratio = height / float(h)
        dim = (int(w * ratio), height)
    else:
        ratio = width / float(w)
        dim = (width, int(h * ratio))
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

# Define display dimensions
resized_img = resize_with_aspect_ratio(img, width=display_width)
resized_mask = resize_with_aspect_ratio(empty_img, width=display_width)
resized_overlay = resize_with_aspect_ratio(overlay, width=display_width)

# Display results
cv2.imshow('Original Image', resized_img)
cv2.imshow('Mask', resized_mask)
cv2.imshow('Overlay', resized_overlay)

# Save results (optional)
cv2.imwrite('output_mask.png', empty_img)
cv2.imwrite('output_overlay.png', overlay)

cv2.waitKey(0)
cv2.destroyAllWindows()