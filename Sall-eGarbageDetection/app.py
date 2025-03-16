# Common endpoint: https://binkhoale1812-sall-egarbagedetection.hf.space/...
# UI Endpoint: https://binkhoale1812-sall-egarbagedetection.hf.space/ui
# Analyze API: https://binkhoale1812-sall-egarbagedetection.hf.space/analyze

# Server startup
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, FileResponse, Response, StreamingResponse
import os
import shutil
import uvicorn
import threading
import requests
import uuid
import time
# Video generation
import cv2
import numpy as np
from PIL import Image
import torch
import yolov5
import ffmpeg
from ultralytics import YOLO
from transformers import DetrImageProcessor, DetrForObjectDetection


# Initialize FastAPI app
app = FastAPI()
video_ready = {}  # Dictionary to track video status for each user


# Define paths
UPLOAD_FOLDER = "/home/user/app/cache/uploads"
OUTPUT_DIR = "/home/user/app/outputs"
OUTPUT_VIDEO_MP4 = os.path.join(OUTPUT_DIR, "simulation.mp4")  
# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Set Hugging Face cache directory to a writable location
CACHE_DIR = "/home/user/app/cache"
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["HF_HOME"] = CACHE_DIR


# Define model paths
MODEL_FOLDER = "/home/user/app/model"
MODEL_PATH_SELF = os.path.join(MODEL_FOLDER, "garbage_detector.pt")
MODEL_PATH_YOLO5 = os.path.join(MODEL_FOLDER, "yolov5-detect-trash-classification.pt")
MODEL_PATH_DETR = os.path.join(MODEL_FOLDER, "detr")


# Load models safely from the pre-downloaded directory
print("🔄 Loading models...")
try:
    # Self-trained YOLO model
    model_self = YOLO(MODEL_PATH_SELF)
    print("✅ Self-trained YOLO model loaded.")

    # YOLOv5 Model
    model_yolo5 = yolov5.load(MODEL_PATH_YOLO5)
    print("✅ YOLOv5 model loaded.")

    # DETR Model
    processor_detr = DetrImageProcessor.from_pretrained(MODEL_PATH_DETR)
    model_detr = DetrForObjectDetection.from_pretrained(MODEL_PATH_DETR)
    print("✅ DETR model loaded.")
except Exception as e:
    print(f"❌ Error loading models: {e}")

print("✅ Model loading complete. Running application.")


# Re-trigger setup, ensure directory setup before starting up the app
import setup
setup.print_model()
setup.print_cache()
# Ensure simulation.mp4 exists as a placeholder
if not os.path.exists(OUTPUT_VIDEO_MP4):
    cap = cv2.VideoWriter(OUTPUT_VIDEO_MP4, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (640, 640))
    cap.release()


# HTML Content for UI
HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head>
    <title>Sall-e Garbage Detection</title>
    <link rel="website icon" type="png" href="/icon.png" >
    <style>
        body {
            font-family: 'Roboto', sans-serif; background: linear-gradient(270deg, rgb(44, 13, 58), rgb(13, 58, 56)); color: white; text-align: center; margin: 0; padding: 50px;
        }
        h1 {
            font-size: 40px;
            background: linear-gradient(to right, #f32170, #ff6b08, #cf23cf, #eedd44);
            -webkit-text-fill-color: transparent;
            -webkit-background-clip: text;
            font-weight: bold;
        }
        #upload-container {
            background: rgba(255, 255, 255, 0.2); padding: 20px; width: 70%; border-radius: 10px; display: inline-block; box-shadow: 0px 0px 10px rgba(255, 255, 255, 0.3);
        }
        #upload {
            font-size: 18px; padding: 10px; border-radius: 5px; border: none; background: #fff; cursor: pointer;
        }
        #loader {
            margin-top: 10px; margin-left: auto; margin-right: auto; width: 60px; height: 60px; font-size: 12px; text-align: center;
        }
        p {
            margin-top: 10px; font-size: 12px; color: #3498db;
        }
        #spinner {
            border: 8px solid #f3f3f3; border-top: 8px solid rgb(117 7 7); border-radius: 50%; animation: spin 1s linear infinite; width: 40px; height: 40px; margin: auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        #outputVideo {
            margin-top: 20px; width: 70%; margin-left: auto; margin-right: auto; max-width: 640px; border-radius: 10px; box-shadow: 0px 0px 10px rgba(255, 255, 255, 0.3);
        }
        #downloadBtn {
            display: block; visibility: hidden; width: 20%; margin-top: 20px; margin-left: auto; margin-right: auto; padding: 10px 15px; font-size: 16px; background: #27ae60; color: white; border: none; border-radius: 5px; cursor: pointer; text-decoration: none;
        }
        #downloadBtn:hover {
            background: #950606;
        }
        .hidden {
            display: none;
        }
        @media (max-width: 860px) {
            h1 { font-size: 30px; }
        }
        @media (max-width: 720px) {
            h1 { font-size: 25px; }
            #upload { font-size: 15px; }
            #downloadBtn { font-size: 13px; }
        }
        @media (max-width: 580px) {
            h1 { font-size: 20px; }
            #upload { font-size: 10px; }
            #downloadBtn { font-size: 10px; }
        }
        @media (max-width: 580px) {
            h1 { font-size: 10px; }
        }
        @media (max-width: 460px) {
            #upload { font-size: 7px; }
        }
        @media (max-width: 400px) {
            h1 { font-size: 14px; }
        }
        @media (max-width: 370px) {
            h1 { font-size: 11px; }
            #upload { font-size: 5px; }
            #downloadBtn { font-size: 7px; }
        }
        @media (max-width: 330px) {
            h1 { font-size: 8px; }
            #upload { font-size: 3px; }
            #downloadBtn { font-size: 5px; }
        }
    </style>
</head>
<body>
    <h1>Upload an Image for Garbage Detection</h1>
    <div id="upload-container">
        <input type="file" id="upload" accept="image/*">
    </div>
    <div id="loader" class="loader hidden">
        <div id="spinner"></div>
        <!-- <p>Garbage detection model processing...</p> -->
    </div>
    <video id="outputVideo" class="outputVideo" controls></video>
    <a id="downloadBtn" class="downloadBtn">Download Video</a>
    <script>
        document.addEventListener("DOMContentLoaded", function() {
            document.getElementById("outputVideo").classList.add("hidden");
            document.getElementById("downloadBtn").style.visibility = "hidden";
        });
        document.getElementById('upload').addEventListener('change', async function(event) {
            event.preventDefault();
            const loader = document.getElementById("loader");
            const outputVideo = document.getElementById("outputVideo");
            const downloadBtn = document.getElementById("downloadBtn");
            let file = event.target.files[0];
            if (file) {
                let formData = new FormData();
                formData.append("file", file);
                loader.classList.remove("hidden");
                outputVideo.classList.add("hidden");
                document.getElementById("downloadBtn").style.visibility = "hidden";
                let response = await fetch('/upload/', { method: 'POST', body: formData });
                let result = await response.json();
                let user_id = result.user_id;  
                while (true) {
                    let checkResponse = await fetch(`/check_video/${user_id}`);
                    let checkResult = await checkResponse.json();
                    if (checkResult.ready) break;
                    await new Promise(resolve => setTimeout(resolve, 3000)); // Wait 3s before checking again
                }
                loader.classList.add("hidden");
                let videoUrl = `/video/${user_id}?t=${new Date().getTime()}`;
                outputVideo.src = videoUrl;
                outputVideo.load();
                outputVideo.play();
                outputVideo.setAttribute("crossOrigin", "anonymous");
                outputVideo.classList.remove("hidden");
                downloadBtn.href = videoUrl;
                document.getElementById("downloadBtn").style.visibility = "visible";
            }
        });
        document.getElementById('outputVideo').addEventListener('error', function() {
            console.log("⚠️ Video could not be played, showing download button instead.");
            document.getElementById('outputVideo').classList.add("hidden");
            document.getElementById("downloadBtn").style.visibility = "visible";
        });
    </script>
</body>
</html>
"""


@app.get("/ui")
async def main():
    return HTMLResponse(content=HTML_CONTENT)


def generate_unique_filename():
    """Generate a unique filename for each user session."""
    return str(uuid.uuid4())[:8]  # Shorter random ID


# Endpoint uploading an image
@app.post("/upload/")
async def upload_file(request: Request, file: UploadFile = File(...)):
    user_id = generate_unique_filename()
    print(f"Session id {user_id}")
    file_path = os.path.join(UPLOAD_FOLDER, f"{user_id}_{file.filename}")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    thread = threading.Thread(target=process_image, args=(file_path, user_id))
    thread.start()
    return {"message": "File uploaded successfully!", "user_id": user_id}


# Endpoint generating and accessing the video
@app.get("/video/{user_id}")
async def get_video(user_id: str):
    video_path = os.path.join(OUTPUT_DIR, f"{user_id}_simulation_h264.mp4")
    if not os.path.exists(video_path) or os.path.getsize(video_path) < 100_000:
        return Response(content="Video file not found!", status_code=404)
    def iterfile():
        with open(video_path, mode="rb") as file:
            yield from file
    return StreamingResponse(iterfile(), media_type="video/mp4", headers={
        "Content-Disposition": "inline; filename={user_id}_simulation.mp4",
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0"
    })


# Ensure video ready state, hide loader and show video + download btn
@app.get("/check_video/{user_id}")
async def check_video(user_id: str):
    """Check if the video for a specific user is ready."""
    video_path = os.path.join(OUTPUT_DIR, f"{user_id}_simulation_h264.mp4")
    min_valid_size = 100_000  # Minimum valid file size 100kB
    return {"ready": video_ready.get(user_id, False) and os.path.exists(video_path) and os.path.getsize(video_path) > min_valid_size}

# Debug endpoints
@app.get("/debug/list_files")
async def list_files():
    cache_files = os.listdir("/home/user/app/cache/")
    output_files = os.listdir("/home/user/app/outputs/")
    return {
        "cache_files": cache_files,
        "output_files": output_files
    }
@app.get("/debug/video_info/{user_id}")
async def debug_video_info(user_id: str):
    """Returns debug info about a user's video file."""
    video_path = os.path.join(OUTPUT_DIR, f"{user_id}_simulation_h264.mp4")
    if not os.path.exists(video_path):
        return {"error": f"Video file for user {user_id} not found!"}
    file_size = os.path.getsize(video_path)
    return {
        "file_path": video_path,
        "file_size": file_size,
        "playable": file_size > 100_000
    }


def convert_video_to_h264(input_video, output_video):
    """Convert video to H.264 for better compatibility."""
    try:
        ffmpeg.input(input_video).output(output_video, vcodec="libx264", format="mp4").run(overwrite_output=True)
        print(f"✅ Video converted to H.264: {output_video}")
        return output_video
    except Exception as e:
        print(f"❌ Error converting video {output_video}, revert back {input_video}: {e}")
        return input_video  # Fallback to original file


def set_file_permissions(file_path):
    '''Ensure user has read access to view file (some browser may block this)'''
    try:
        os.chmod(file_path, 0o644)  # Allow read access
        print(f"✅ File permissions set: {file_path}")
    except Exception as e:
        print(f"❌ Error setting permissions for {file_path}: {e}")


def is_video_accessible(user_id: str):
    """Checks if the video file exists and is valid without external HTTP requests."""
    video_path = os.path.join(OUTPUT_DIR, f"{user_id}_simulation_h264.mp4")
    if os.path.exists(video_path):
        file_size = os.path.getsize(video_path)
        if file_size > 100_000:  # Ensure file is large enough to be a valid video
            print(f"✅ Video file for user {user_id} is accessible: {file_size} bytes")
            return True
        else:
            print(f"⚠️ Warning: Video for user {user_id} exists but is too small ({file_size} bytes)")
    else:
        print(f"❌ Error: Video file for user {user_id} does not exist.")
    return False


# Garbage detection and video generation
def process_image(image_path, user_id):
    # Assign unique id
    global video_ready
    video_ready[user_id] = False  # Ensure video is marked as "not ready" initially
    unique_filename = f"{user_id}_simulation.mp4"
    unique_h264_filename = f"{user_id}_simulation_h264.mp4"
    video_path = os.path.join(OUTPUT_DIR, unique_filename)
    h264_path = os.path.join(OUTPUT_DIR, unique_h264_filename)

    # Process the image
    image = cv2.imread(image_path)
    if image is None:
        return
    image = cv2.resize(image, (640, 640))
    detections = []
    
    # Self-trained YOLOv11m
    print(f"🔍 Running detection with Self-trained YOLO model for {user_id} session...")
    results_self = model_self(image)
    for result in results_self:
        for box in result.boxes:
            detections.append(box.xyxy[0].tolist())
    
    # YOLOv5 Model
    print(f"🔍 Running detection with YOLOv5 model for {user_id} session...")
    results_yolo5 = model_yolo5(image, size=416)
    for result in results_yolo5.pred[0]:
        detections.append(result[:4].tolist())
    
    # DETR Model
    print(f"🔍 Running detection with DETR model for {user_id} session...")
    image_pil = Image.open(image_path).convert("RGB")
    inputs = processor_detr(images=image_pil, return_tensors="pt")
    with torch.no_grad():
        outputs = model_detr(**inputs)
    target_sizes = torch.tensor([image_pil.size[::-1]])
    results_detr = processor_detr.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.5)[0]
    for box in results_detr["boxes"]:
        detections.append(box.tolist())

    print(f"✅ Multi-modal detected {len(detections)} objects for {user_id} session.")
    
    # Save video file
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(video_path, fourcc, 10.0, (640, 640)) 
    
    # Video writer
    for _ in range(100):  # 10 second simulation by 10 FPS (10fps * 10s)
        frame = image.copy()
        for box in detections:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, "Detected", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        video_writer.write(frame)

    print(f"🎥 Video generated successfully for user {user_id}!")
    video_writer.release()
    converted_video = convert_video_to_h264(video_path, h264_path) # Convert to H.264 for better streaming compatibility
    set_file_permissions(converted_video)
    time.sleep(2)  # Short delay to ensure OS flushes the file to disk
    os.sync()      # Force flush
    video_ready[user_id] = True
    if os.path.exists(converted_video) and os.path.getsize(converted_video) > 100_000 and is_video_accessible(user_id):
        print(f"✅ Video successfully verified and saved at {converted_video} for session {user_id}")
        return h264_path
    else:
        print(f"❌ ERROR: Video file not found after processing for session {user_id}!")
        return None


# API endpoint for user to extract bbox coordination from their image
@app.post("/analyze/")
async def analyze_file(file: UploadFile = File(...)):
    user_id = generate_unique_filename()
    print(f"Analyzing image for session: {user_id}")

    # Save file
    file_path = os.path.join(UPLOAD_FOLDER, f"{user_id}_{file.filename}")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run detection
    image = cv2.imread(file_path)
    if image is None:
        return {"error": "Failed to read image"}
    image = cv2.resize(image, (640, 640))
    detections = []

    # Self-trained YOLOv11m Model
    results_self = model_self(image)
    for result in results_self:
        for box in result.boxes:
            detections.append({
                "bbox": box.xyxy[0].tolist(),
                "model": "self-trained"
            })

    # YOLOv5 Model
    results_yolo5 = model_yolo5(image, size=416)
    for result in results_yolo5.pred[0]:
        detections.append({
            "bbox": result[:4].tolist(),
            "model": "yolov5"
        })

    # DETR Model
    image_pil = Image.open(file_path).convert("RGB")
    inputs = processor_detr(images=image_pil, return_tensors="pt")
    with torch.no_grad():
        outputs = model_detr(**inputs)
    target_sizes = torch.tensor([image_pil.size[::-1]])
    results_detr = processor_detr.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.5)[0]
    for box in results_detr["boxes"]:
        detections.append({
            "bbox": box.tolist(),
            "model": "detr"
        })

    # Send analyzed response
    print(f"✅ Processed image {file.filename} for session {user_id}, detected {len(detections)} objects.")
    return {"user_id": user_id, "detections": detections}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)