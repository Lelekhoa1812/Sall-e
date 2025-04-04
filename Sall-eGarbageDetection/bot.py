from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import os
import cv2
import numpy as np
import torch
import requests
import math
import heapq
from sklearn.neighbors import NearestNeighbors
import yolov5
from ultralytics import YOLO
from transformers import DetrImageProcessor, DetrForObjectDetection

app = FastAPI()

# Model Paths
MODEL_FOLDER = "/model"
MODEL_PATH_SELF = os.path.join(MODEL_FOLDER, "garbage_detector.pt")
MODEL_PATH_YOLO5 = os.path.join(MODEL_FOLDER, "yolov5-detect-trash-classification.pt")
MODEL_PATH_DETR = os.path.join(MODEL_FOLDER, "detr")

# Load Models
print("🔄 Loading AI models locally...")
model_self = YOLO(MODEL_PATH_SELF)
model_yolo5 = yolov5.load(MODEL_PATH_YOLO5)
processor_detr = DetrImageProcessor.from_pretrained(MODEL_PATH_DETR)
model_detr = DetrForObjectDetection.from_pretrained(MODEL_PATH_DETR)
print("✅ Models loaded successfully")

# External APIs (Example Endpoints)
DRONE_API_ALTITUDE = "http://drone_api/altitude"
ROBOT_GPS_API = "http://robot_api/gps"

# Constants
CAMERA_FOV = 84  # Drone camera horizontal FOV in degrees
FRAME_DIM = (640, 640)  # Standard frame size


# ===== API Communication Functions =====
def get_drone_altitude():
    '''Obtain drone altitude from drone's API'''
    try:
        response = requests.get(DRONE_API_ALTITUDE)
        if response.status_code == 200:
            return float(response.json().get("altitude", 50.0))
    except Exception as e:
        print(f"⚠️ Failed to fetch drone altitude: {e}")
    return 50.0  # Default fallback altitude (50m)

def get_robot_gps():
    '''Obtain robot location from GPS's API'''
    try:
        response = requests.get(ROBOT_GPS_API)
        if response.status_code == 200:
            gps = response.json()
            return (gps["lat"], gps["lon"])
    except Exception as e:
        print(f"⚠️ Failed to fetch robot GPS: {e}")
    return (0.0, 0.0)  # Default fallback


# ===== AI Detection - Model Evaluation =====
def detect_garbage(frame):
    detections = []

    # Self-trained YOLOv11m
    results_self = model_self(frame)
    for result in results_self:
        for box in result.boxes:
            detections.append(box.xyxy[0].tolist())

    # YOLOv5 external model
    results_yolo5 = model_yolo5(frame, size=416)
    for result in results_yolo5.pred[0]:
        detections.append(result[:4].tolist())

    # DETR model
    inputs = processor_detr(images=frame, return_tensors="pt")
    with torch.no_grad():
        outputs = model_detr(**inputs)
    target_sizes = torch.tensor([frame.shape[:2]])
    results_detr = processor_detr.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.5)[0]
    for box in results_detr["boxes"]:
        detections.append(box.tolist())

    print(f"✅ Detected {len(detections)} garbage objects")
    return [(int(x1), int(y1)) for x1, y1, x2, y2 in detections]


# ===== Pixel to GPS / Real-World Coordinate Conversion =====
def convert_pixel_to_gps(detections, frame_dim, drone_altitude, robot_gps):
    # Ground Sampling Distance (GSD) calculation
    fov_rad = math.radians(CAMERA_FOV)
    ground_width = 2 * drone_altitude * math.tan(fov_rad / 2)  # Ground width covered by the camera
    meters_per_pixel = ground_width / frame_dim[1]  # Assuming square pixels
    # Initialise list of GPS location
    gps_list = []
    for (x, y) in detections:
        dx = (x - frame_dim[1] / 2) * meters_per_pixel
        dy = (y - frame_dim[0] / 2) * meters_per_pixel
        # For prototype: simple cartesian addition (real GPS mapping needs conversion)
        gps_x = robot_gps[0] + dx
        gps_y = robot_gps[1] + dy
        gps_list.append((gps_x, gps_y))
    return gps_list


# ===== Path Planning (KNN + A*) =====
def heuristic(a, b):
    return np.linalg.norm(np.array(a) - np.array(b)) # Euclidean distance (L2 norm) formula

def astar(start, goal, graph_points):
    '''Standard A* formula to compute shortest path'''
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {tuple(start): 0}
    f_score = {tuple(start): heuristic(start, goal)}
    # Calculate score
    while open_set:
        _, current = heapq.heappop(open_set)
        if np.allclose(current, goal, atol=1):  # Close enough
            path = [tuple(goal)]
            while tuple(current) != tuple(start):
                current = came_from[tuple(current)]
                path.append(tuple(current))
            return path[::-1]
        # A* = Score + heuristic
        for neighbor in graph_points:
            tentative_g = g_score[tuple(current)] + heuristic(current, neighbor)
            if tuple(neighbor) not in g_score or tentative_g < g_score[tuple(neighbor)]:
                came_from[tuple(neighbor)] = current
                g_score[tuple(neighbor)] = tentative_g
                f_score[tuple(neighbor)] = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[tuple(neighbor)], neighbor))
    return []

def compute_optimal_path(gps_targets, robot_gps):
    if not gps_targets:
        return []
    # Init location = start + goal
    points = np.array(gps_targets)
    start = np.array(robot_gps)
    # Use KNN to select initial closest garbage to start the A*
    neighbors = NearestNeighbors(n_neighbors=min(3, len(points)), algorithm='auto').fit(points)
    _, indices = neighbors.kneighbors([start])
    nearest = points[indices[0][0]]
    # Define the planned path 
    path = astar(start.tolist(), nearest.tolist(), points.tolist())
    print(f"🧭 Planned path with {len(path)} waypoints")
    return path


# ===== API Route: Real-Time Streaming =====
@app.post("/stream/")
async def process_stream(request: Request):
    """
    Receive real-time drone stream frame, process detection, output GPS targets and path.
    """
    # Decode the incoming video frame
    body = await request.body()
    np_arr = np.frombuffer(body, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    frame = cv2.resize(frame, FRAME_DIM)
    # Get real-time drone altitude and robot GPS location
    drone_altitude = get_drone_altitude()
    robot_gps = get_robot_gps()
    print(f"📡 Drone altitude: {drone_altitude}m | 🤖 Robot GPS: {robot_gps}")
    # Detect garbage in frame
    garbage_pixel_coords = detect_garbage(frame)
    # Convert detected pixel locations to GPS/real-world coordinates
    gps_targets = convert_pixel_to_gps(garbage_pixel_coords, frame.shape, drone_altitude, robot_gps)
    # Compute optimal path using A* and KNN
    collection_path = compute_optimal_path(gps_targets, robot_gps)
    # Send data to GPS hardware
    return JSONResponse({
        "garbage_gps_targets": gps_targets,
        "collection_path": collection_path,
        "robot_current_gps": robot_gps
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
