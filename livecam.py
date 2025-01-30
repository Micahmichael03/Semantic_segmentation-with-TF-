import cv2
import random
import numpy as np

# Define paths
cfg_path = r'./models/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt'
weights_path = r'./models/frozen_inference_graph.pb'

# Define display width
display_width = 200  # Set your desired display width

# Load model
net = cv2.dnn.readNetFromTensorflow(weights_path, cfg_path)
if net.empty():
    raise FileNotFoundError("Failed to load model. Check paths to .pb and .pbtxt files.")

# Generate random colors for each class
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(80)]

# Detection threshold
detection_threshold = 0.5

# Initialize video capture
cap = cv2.VideoCapture(0)  # Use 0 for default camera, or replace with video file path
if not cap.isOpened():
    raise RuntimeError("Failed to open camera.")

# Resize function for display
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

# Main loop for processing camera frames
while True:
    # Read frame from camera
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    H, W, C = frame.shape

    # Convert frame to blob
    blob = cv2.dnn.blobFromImage(frame, swapRB=True, crop=False)

    # Set input to the network
    net.setInput(blob)

    # Forward pass to get detections and masks
    boxes, masks = net.forward(['detection_out_final', 'detection_masks'])

    # Create an empty image for mask visualization
    empty_img = np.zeros((H, W, C), dtype=np.uint8)

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

    # Create an overlay of the mask and the original frame
    overlay = cv2.addWeighted(frame, 0.6, empty_img, 0.4, 0)

    # Resize frames for display
    resized_frame = resize_with_aspect_ratio(frame, width=display_width)
    resized_mask = resize_with_aspect_ratio(empty_img, width=display_width)
    resized_overlay = resize_with_aspect_ratio(overlay, width=display_width)

    # Display results
    # cv2.imshow('Original Frame', resized_frame)
    # cv2.imshow('Mask', resized_mask)
    cv2.imshow('Overlay', resized_overlay)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()