import cv2
import torch
import os
import yolov5
from ultralytics import YOLO
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import numpy as np

# Define paths
TESTING_FOLDER = "/content/drive/My Drive/Sall-e/testing"
DETECT_FOLDER = "/content/drive/My Drive/Sall-e/detect"
MODEL_FOLDER = "/content/drive/My Drive/Sall-e/model"
MODEL_PATH_SELF = os.path.join(MODEL_FOLDER, "garbage_detector.pt")
MODEL_NAME_YOLO5 = 'turhancan97/yolov5-detect-trash-classification'
MODEL_PATH_DETR = "Yorai/detr-resnet-50_finetuned_detect-waste"

# Ensure the detect folder exists
os.makedirs(DETECT_FOLDER, exist_ok=True)

# Load models
model_self = YOLO(MODEL_PATH_SELF)
model_yolo5 = yolov5.load(MODEL_NAME_YOLO5)
processor_detr = DetrImageProcessor.from_pretrained(MODEL_PATH_DETR)
model_detr = DetrForObjectDetection.from_pretrained(MODEL_PATH_DETR)

# Set YOLOv5 parameters
model_yolo5.conf = 0.25  # NMS confidence threshold
model_yolo5.iou = 0.15   # NMS IoU threshold
model_yolo5.max_det = 1000  # Maximum number of detections per image

# Process each testing image
for i in range(1, 7):
    image_path = os.path.join(TESTING_FOLDER, f"testing_{i}.jpg")
    detect_path = os.path.join(DETECT_FOLDER, f"detect_{i}.jpg")

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading {image_path}")
        continue

    # Clone image for visualization
    img_vis = img.copy()

    # Run YOLOv11m (self-trained model)
    results = model_self(img)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0].item()
            cv2.rectangle(img_vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img_vis, f"Self {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Run YOLOv5 external model
    results_yolo5 = model_yolo5(img, size=416)
    for result in results_yolo5.pred[0]:
        x1, y1, x2, y2, conf, cls = result.tolist()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_vis, f"YOLOv5 {conf:.2f}", (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Run DETR external model
    image_pil = Image.open(image_path).convert("RGB")
    inputs = processor_detr(images=image_pil, return_tensors="pt")
    with torch.no_grad():
        outputs = model_detr(**inputs)
    target_sizes = torch.tensor([image_pil.size[::-1]])
    results_detr = processor_detr.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.5)[0]
    for score, label, box in zip(results_detr["scores"], results_detr["labels"], results_detr["boxes"]):
        x1, y1, x2, y2 = map(int, box.tolist())
        confidence = score.item()
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img_vis, f"DETR {confidence:.2f}", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Save the processed image
    cv2.imwrite(detect_path, img_vis)
    print(f"Detection completed for {image_path}, saved as {detect_path}")

print("All detections completed and saved in the detect folder.")