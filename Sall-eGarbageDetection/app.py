from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, FileResponse
import os
import shutil
import cv2
import numpy as np
from PIL import Image
import uvicorn
import threading
import torch
import yolov5
from ultralytics import YOLO
from transformers import DetrImageProcessor, DetrForObjectDetection

# Initialize FastAPI app
app = FastAPI()

# Define paths
UPLOAD_FOLDER = "uploads"
OUTPUT_VIDEO = "simulation/simulation.mp4"
MODEL_FOLDER = "model"
MODEL_PATH_SELF = os.path.join(MODEL_FOLDER, "garbage_detector.pt")
MODEL_PATH_YOLO5 = os.path.join(MODEL_FOLDER, "yolov5-detect-trash-classification.pt")
MODEL_PATH_DETR = os.path.join(MODEL_FOLDER, "pytorch_model.bin")

# Ensure temp folder `uploads` exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Ensure cache directory exists and set correct permissions
CACHE_DIR = "/app/cache"
os.makedirs(CACHE_DIR, exist_ok=True)
os.chmod(CACHE_DIR, 0o777)  # Set write permissions

# Set Hugging Face cache directory to a writable location
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["HF_HOME"] = CACHE_DIR

# Load models
model_self = YOLO(MODEL_PATH_SELF)
model_yolo5 = yolov5.load(MODEL_PATH_YOLO5)
processor_detr = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", local_files_only=True, cache_dir=CACHE_DIR)
model_detr = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", local_files_only=True, cache_dir=CACHE_DIR)

# Re-trigger setup, ensure directory setup before starting up the app
import setup
setup.print_model()
setup.print_cache()

# HTML Content for UI
HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head>
    <title>Garbage Detection</title>
    <style>
        body {
            font-family: 'Roboto', sans-serif;
            background: linear-gradient(270deg, rgb(44, 13, 58), rgb(13, 58, 56));
            color: white;
            text-align: center;
            margin: 0;
            padding: 50px;
        }
        h1 {
            font-size: 40px;
            background: linear-gradient(to right, #f32170, #ff6b08, #cf23cf, #eedd44);
            -webkit-text-fill-color: transparent;
            -webkit-background-clip: text;
            font-weight: bold;
        }
        #upload-container {
            background: rgba(255, 255, 255, 0.2);
            padding: 20px;
            border-radius: 10px;
            display: inline-block;
            box-shadow: 0px 0px 10px rgba(255, 255, 255, 0.3);
        }
        #upload {
            font-size: 18px;
            padding: 10px;
            border-radius: 5px;
            border: none;
            background: #fff;
            cursor: pointer;
        }
        #loader {
            display: none;
            color: rgb(255, 94, 94);
            font-size: 18px;
            margin-top: 20px;
        }
        #outputVideo {
            display: none;
            margin-top: 20px;
            width: 70%;
            max-width: 640px;
            border-radius: 10px;
            box-shadow: 0px 0px 10px rgba(255, 255, 255, 0.3);
        }
    </style>
</head>
<body>
    <h1>Upload an Image for Garbage Detection</h1>
    <div id="upload-container">
        <input type="file" id="upload" accept="image/*">
        <p id="loader">Garbage detection model processing...</p>
    </div>
    <video id="outputVideo" controls></video>
    <script>
        document.getElementById('upload').addEventListener('change', async function(event) {
            let file = event.target.files[0];
            if (file) {
                let formData = new FormData();
                formData.append("file", file);
                document.getElementById('loader').style.display = 'block';
                let response = await fetch('/upload/', { method: 'POST', body: formData });
                document.getElementById('loader').style.display = 'none';
                document.getElementById('outputVideo').style.display = 'block';
                document.getElementById('outputVideo').src = '/video';
            }
        });
    </script>
</body>
</html>
"""

@app.get("/")
async def main():
    return HTMLResponse(content=HTML_CONTENT)

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    threading.Thread(target=process_image, args=(file_path,)).start()
    return {"message": "File uploaded successfully!"}

@app.get("/video")
async def get_video():
    return FileResponse(OUTPUT_VIDEO, media_type="video/mp4")

# Garbage detection function
def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return
    image = cv2.resize(image, (640, 640))
    
    # Run detection on all models
    detections = []
    
    # Self-trained YOLOv11m
    results_self = model_self(image)
    for result in results_self:
        for box in result.boxes:
            detections.append(box.xyxy[0].tolist())
    
    # YOLOv5 Model
    results_yolo5 = model_yolo5(image, size=416)
    for result in results_yolo5.pred[0]:
        detections.append(result[:4].tolist())
    
    # DETR Model
    image_pil = Image.open(image_path).convert("RGB")
    inputs = processor_detr(images=image_pil, return_tensors="pt")
    with torch.no_grad():
        outputs = model_detr(**inputs)
    target_sizes = torch.tensor([image_pil.size[::-1]])
    results_detr = processor_detr.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.5)[0]
    for box in results_detr["boxes"]:
        detections.append(box.tolist())
    
    # Draw bounding boxes and create video
    video_writer = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (640, 640))  # 10 FPS lower resource
    for _ in range(100):  # 5-second simulation (20fps * 5s)
        frame = image.copy()
        for box in detections:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, "Detected", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        video_writer.write(frame)
    video_writer.release()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
