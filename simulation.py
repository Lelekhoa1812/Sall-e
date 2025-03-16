import os
import random
import cv2
import torch
import numpy as np
from PIL import Image
import yolov5
from ultralytics import YOLO
from transformers import DetrImageProcessor, DetrForObjectDetection

# Define paths
CROP_DIR = "crop"
MODEL_FOLDER = "model"
MODEL_PATH_SELF = os.path.join(MODEL_FOLDER, "garbage_detector.pt")
MODEL_NAME_YOLO5 = 'turhancan97/yolov5-detect-trash-classification'
MODEL_PATH_DETR = "Yorai/detr-resnet-50_finetuned_detect-waste"
VIDEO_PATH = "simulation/simulation.mp4"

# Load ocean background (resized to 640x640 as trained by all models)
ocean_img = cv2.imread("src/ocean1.jpg")
ocean_img = cv2.resize(ocean_img, (640, 640))

# Load garbage objects from cropped directory
objects = []
# Setter for occupied positions to avoid overlapping
occupied_positions = set()
classes = [d for d in os.listdir(CROP_DIR) if os.path.isdir(os.path.join(CROP_DIR, d))]
# For each/any classes as subdirectory of the crop folder
for class_name in classes:
    class_path = os.path.join(CROP_DIR, class_name)
    png_files = [f for f in os.listdir(class_path) if f.endswith(".png")]
    selected_pngs = random.sample(png_files, min(20, len(png_files)))
    # Open each garbage object images and resize to 20x20 pixels
    for png in selected_pngs:
        png_path = os.path.join(class_path, png)
        overlay = Image.open(png_path).convert("RGBA").resize((20, 20))
        # Spawn the garbage object randomly around the screen
        while True:
            x, y = random.randint(0, 600), random.randint(0, 600)
            if (x, y) not in occupied_positions:
                occupied_positions.add((x, y))
                break
        # Append to a list of positioned object with flags for collecting state
        objects.append({"image": np.array(overlay), "position": [x, y], "collected": False})

# Robot class
class Robot:
    # Define the robot's size (40x40) image, position (top-left), angle (right) and speed constant (20px/s)
    def __init__(self, image_path, speed=20):
        self.image = Image.open(image_path).convert("RGBA").resize((40, 40))
        self.image_np = np.array(self.image)
        self.position = [0, 0]
        self.angle = 0  # Initially pointing right (90 deg)
        self.speed = speed

    # Function to located the nearest garbage object 
    def find_nearest_garbage(self, objects):
        nearest = None
        min_dist = float("inf")
        for obj in objects:
            if obj["collected"]:
                continue
            # Compute the location by self position to all possible objects and find the min value
            dist = np.linalg.norm(np.array(self.position) - np.array(obj["position"]))
            if dist < min_dist:
                min_dist = dist
                nearest = obj
        return nearest

    # Function to move the robot with constant speed and rotate according to the destination coordination
    def move_towards(self, target):
        if target:
            dx, dy = np.array(target["position"]) - np.array(self.position)
            distance = np.linalg.norm([dx, dy])
            if distance > self.speed:
                dx, dy = (dx / distance) * self.speed, (dy / distance) * self.speed
            self.position[0] += int(dx)
            self.position[1] += int(dy)
            self.angle = np.degrees(np.arctan2(dy, dx))
            if distance <= self.speed:
                target["collected"] = True

# Initialize robot
robot = Robot("Sall-eGarbageDetection/sprite.png")

# Video writer (to .mp4)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(VIDEO_PATH, fourcc, 20.0, (640, 640))

# Start simulation
while True:
    frame = ocean_img.copy()
    target = robot.find_nearest_garbage(objects)
    robot.move_towards(target)
    
    # Display each detected garbage object
    for obj in objects:
        if not obj["collected"]:
            x, y = obj["position"]
            overlay = obj["image"]
            alpha_s = overlay[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s
            for c in range(3):
                frame[y:y+20, x:x+20, c] = (alpha_s * overlay[:, :, c] + alpha_l * frame[y:y+20, x:x+20, c])
            cv2.putText(frame, "Detected", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Overlay robot
    rx, ry = robot.position
    alpha_s = robot.image_np[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s
    for c in range(3):
        frame[ry:ry+40, rx:rx+40, c] = (alpha_s * robot.image_np[:, :, c] + alpha_l * frame[ry:ry+40, rx:rx+40, c])
    
    # Continously writing the robot movement until all garbage are collected
    out.write(frame)
    if all(obj["collected"] for obj in objects):
        break

# Export to destination path
out.release()
print(f"Simulation saved as {VIDEO_PATH}")