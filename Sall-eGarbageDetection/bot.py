from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import os
import cv2
import numpy as np
import torch
import yolov5
from ultralytics import YOLO
from transformers import DetrImageProcessor, DetrForObjectDetection
import math
from sklearn.neighbors import NearestNeighbors
import heapq

app = FastAPI()

# Model paths
MODEL_FOLDER = "/model"
MODEL_PATH_SELF = os.path.join(MODEL_FOLDER, "garbage_detector.pt")
MODEL_PATH_YOLO5 = os.path.join(MODEL_FOLDER, "yolov5-detect-trash-classification.pt")
MODEL_PATH_DETR = os.path.join(MODEL_FOLDER, "detr")

# Model loading
print("🔄 Loading AI models locally...")
model_self = YOLO(MODEL_PATH_SELF)
model_yolo5 = yolov5.load(MODEL_PATH_YOLO5)
processor_detr = DetrImageProcessor.from_pretrained(MODEL_PATH_DETR)
model_detr = DetrForObjectDetection.from_pretrained(MODEL_PATH_DETR)
print("✅ Models Loaded Successfully")

# Robot GPS properties (mock)
ROBOT_GPS = GPS_API         # Location taken from the robot GPS
DRONE_ALTITUDE = DRONE_API  # Altitude taken from the drone via API

# A* algorithm for 2D coordinate navigation
def heuristic(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def astar(start, goal, graph_points):
    """ A* from start to goal given graph points of the garbage object to find the shortest route"""
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {tuple(start): 0}
    f_score = {tuple(start): heuristic(start, goal)}
    while open_set:
        _, current = heapq.heappop(open_set)
        if np.allclose(current, goal, atol=10):
            path = [tuple(goal)]
            while tuple(current) != tuple(start):
                current = came_from[tuple(current)]
                path.append(tuple(current))
            return path[::-1]
        for neighbor in graph_points:
            tentative_g = g_score[tuple(current)] + heuristic(current, neighbor)
            if tuple(neighbor) not in g_score or tentative_g < g_score[tuple(neighbor)]:
                came_from[tuple(neighbor)] = current
                g_score[tuple(neighbor)] = tentative_g
                f_score[tuple(neighbor)] = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[tuple(neighbor)], neighbor))
    return []

# Garbage Detection API (streamed video frames)
@app.post("/stream/")
async def process_stream(request: Request):
    """
    Receive drone stream frame (640x640), process detection, output garbage GPS coordinates
    """
    body = await request.body()
    np_arr = np.frombuffer(body, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    frame = cv2.resize(frame, (640, 640))
    garbage_locations = detect_garbage(frame)
    gps_targets = convert_pixel_to_gps(garbage_locations)
    collection_path = compute_optimal_path(gps_targets)
    return JSONResponse({
        "garbage_gps_targets": gps_targets,
        "collection_path": collection_path,
        "robot_current_gps": ROBOT_GPS # Location of the robot is taken from GPS 
    })

# Model evaluation to detect garbage object
def detect_garbage(frame):
    detections = []

    # YOLOv11m self-trained
    results_self = model_self(frame)
    for result in results_self:
        for box in result.boxes:
            detections.append(box.xyxy[0].tolist())

    # YOLOv5 external
    results_yolo5 = model_yolo5(frame, size=416)
    for result in results_yolo5.pred[0]:
        detections.append(result[:4].tolist())

    # DETR external
    inputs = processor_detr(images=frame, return_tensors="pt")
    with torch.no_grad():
        outputs = model_detr(**inputs)
    target_sizes = torch.tensor([frame.shape[:2]])
    results_detr = processor_detr.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.5)[0]
    for box in results_detr["boxes"]:
        detections.append(box.tolist())

    print(f"✅ Detected {len(detections)} objects")
    return [(int(x1), int(y1)) for x1, y1, x2, y2 in detections]

# Scale up the coordination by the drone altitude
def convert_pixel_to_gps(detections):
    gps_list = []
    scale_ratio = DRONE_ALTITUDE / 30.0  # Example scale based on drone height
    for (x, y) in detections:
        real_world_x = ROBOT_GPS[0] + (x - 320) * scale_ratio * 0.1  # Convert px to meters to GPS approximation
        real_world_y = ROBOT_GPS[1] + (y - 320) * scale_ratio * 0.1
        gps_list.append((real_world_x, real_world_y))
    return gps_list

def compute_optimal_path(gps_targets):
    if not gps_targets:
        return []

    points = np.array(gps_targets)
    start = np.array(ROBOT_GPS)
    neighbors = NearestNeighbors(n_neighbors=min(3, len(points)), algorithm='auto').fit(points)
    _, indices = neighbors.kneighbors([start])

    nearest = points[indices[0][0]]
    path = astar(start.tolist(), nearest.tolist(), points.tolist())
    print(f"🧭 Planned path with {len(path)} waypoints")
    return path

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
