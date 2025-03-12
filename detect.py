import cv2
import torch
from ultralytics import YOLO
import os

# Define paths
TESTING_FOLDER = "/testing"
DETECT_FOLDER = "/detect1"
MODEL_FOLDER = "model"
MODEL_PATH = os.path.join(MODEL_FOLDER, "garbage_detector.pt")

# Ensure the detect folder exists
os.makedirs(DETECT_FOLDER, exist_ok=True)

# Load the trained YOLOv11m model
model = YOLO(MODEL_PATH)

# Process each testing image (1 to 4)
for i in range(1, 5):
    image_path = os.path.join(TESTING_FOLDER, f"testing_{i}.jpg")
    detect_path = os.path.join(DETECT_FOLDER, f"detect_{i}.jpg")
    
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading {image_path}")
        continue
    
    # Perform object detection
    results = model(img)
    
    # Draw bounding boxes on the image
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bbox coordinates
            confidence = box.conf[0].item()  # Get confidence score
            
            # Draw the rectangle and confidence label
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Garbage {confidence:.2f}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Save the processed image
    cv2.imwrite(detect_path, img)
    print(f"Detection completed for {image_path}, saved as {detect_path}")

print("All detections completed and saved in the detect1 folder.")