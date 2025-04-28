import os
import random
import cv2
import numpy as np
import heapq
from PIL import Image
from sklearn.neighbors import NearestNeighbors

# Define paths
CROP_DIR = "/content/drive/My Drive/Sall-e/crop"
VIDEO_PATH = "/content/drive/My Drive/Sall-e/simulation/simulation_knn_astar.mp4"
OCEAN_IMAGE = "/content/drive/My Drive/Sall-e/src/ocean1.jpg"
ROBOT_IMAGE = "/content/drive/My Drive/Sall-e/src/sprite.png"

# 1) Load ocean background
ocean_img = cv2.imread(OCEAN_IMAGE)
ocean_img = cv2.resize(ocean_img, (640, 640))

# 2) Spawn random garbage objects
objects = []
occupied_positions = set()
classes = [d for d in os.listdir(CROP_DIR) if os.path.isdir(os.path.join(CROP_DIR, d))]
# Randomly taking 20 garbages from different types
for class_name in classes:
    class_path = os.path.join(CROP_DIR, class_name)
    png_files = [f for f in os.listdir(class_path) if f.endswith(".png")]
    selected_pngs = random.sample(png_files, min(20, len(png_files)))
    for png in selected_pngs:
        png_path = os.path.join(class_path, png)
        overlay = Image.open(png_path).convert("RGBA").resize((20, 20))
        while True:
            x, y = random.randint(0, 600), random.randint(0, 600)
            if (x, y) not in occupied_positions:
                occupied_positions.add((x, y))
                break
        objects.append({
            "image": np.array(overlay),
            "position": [x, y],
            "collected": False
        })

##############################################
# 3) A* Algorithm for Pixel-Level Pathfinding
##############################################
def heuristic(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def astar(start, goal):
    """
    Because we have no actual obstacles in ocean_img,
    this is effectively a straight-line path. We'll
    do pixel stepping to emulate an A* approach.
    """
    path = []
    # If desired, implement real occupancy grid. For now, produce linear waypoints
    # from start to goal at small increments so the robot can step them.
    x1, y1 = start
    x2, y2 = goal
    dist = np.linalg.norm([x2 - x1, y2 - y1])
    steps = int(dist)  # 1 px increments
    if steps == 0:
        return [start, goal]
    for i in range(steps + 1):
        t = i / steps
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        path.append((int(x), int(y)))
    return path

##############################################
# 4) KNN TSP-Like Path Manager
##############################################
def compute_global_path(robot_pos, object_list):
    """
    Build an entire path visiting all uncollected garbage items
    using a KNN approach to pick the next closest item, then
    generate A* path for each segment.
    """
    points = [obj["position"] for obj in object_list if not obj["collected"]]
    if not points:
        return []
    # Stack of node for the full path
    full_path = []
    current = robot_pos[:]
    pending = points[:]
    # Not all collected
    while pending:
        # KNN to find nearest object from 'current'
        if len(pending) == 1:
            next_goal = pending[0]
        else:
            neighbors = NearestNeighbors(n_neighbors=1).fit(pending)
            dist, idx = neighbors.kneighbors([current])
            next_goal = pending[idx[0][0]]
        # A* from current to next_goal
        subpath = astar(tuple(current), tuple(next_goal))
        # Add subpath to full
        if subpath:
            # Remove the first, to avoid duplicates except if it is the very first path
            if full_path and subpath[0] == full_path[-1]:
                subpath = subpath[1:]
            full_path.extend(subpath)
        # Move current
        current = next_goal
        # remove next_goal from pending
        pending.remove(next_goal)
    return full_path

################################################
# 5) Robot Class (Preserves Original Movement)
################################################
class Robot:
    def __init__(self, image_path, speed=20):
        self.image = Image.open(image_path).convert("RGBA").resize((40, 40))
        self.image_np = np.array(self.image)
        self.position = [0, 0]  # top-left
        self.speed = speed  # 20 px / step
        # we'll move along the path, one step per frame
    # Define movement and speed
    def step_along_path(self, path):
        if not path:
            return
        # Move robot at 'speed' px/step
        rx, ry = self.position
        px, py = path[0]
        dx, dy = (px - rx), (py - ry)
        dist = np.linalg.norm([dx, dy])
        if dist <= self.speed:
            # We can jump directly to path[0]
            self.position = [px, py]
            path.pop(0)  # consume this waypoint
        else:
            # Move partially
            ratio = self.speed / dist
            rx += int(dx * ratio)
            ry += int(dy * ratio)
            self.position = [rx, ry]

################################################
# 6) Simulation with a Single Global Path
################################################
robot = Robot(ROBOT_IMAGE, speed=20)
# Create an entire path visiting all objects in an order
global_path = compute_global_path(robot.position, objects)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(VIDEO_PATH, fourcc, 20.0, (640, 640))

# main simulation loop
frame_count = 0
max_frames = 20000  # safety break
while frame_count < max_frames:
    frame = ocean_img.copy()

    # Step the robot along the path
    robot.step_along_path(global_path)

    # check if robot is near any uncollected object => mark collected
    for obj in objects:
        if not obj["collected"]:
            ox, oy = obj["position"]
            dist = np.linalg.norm(np.array(robot.position) - np.array([ox, oy]))
            if dist <= robot.speed:
                obj["collected"] = True

    # draw objects
    for obj in objects:
        if not obj["collected"]:
            x, y = obj["position"]
            overlay = obj["image"]
            alpha_s = overlay[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s
            for c in range(3):
                frame[y:y+20, x:x+20, c] = (alpha_s * overlay[:, :, c] +
                                           alpha_l * frame[y:y+20, x:x+20, c])
            cv2.putText(frame, "Detected", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 2)

    # draw robot
    rx, ry = robot.position
    alpha_s = robot.image_np[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s
    for c in range(3):
        frame[ry:ry+40, rx:rx+40, c] = (alpha_s * robot.image_np[:, :, c] +
                                       alpha_l * frame[ry:ry+40, rx:rx+40, c])
    out.write(frame)
    frame_count += 1

    # if all objects are collected, stop
    if all(obj["collected"] for obj in objects):
        break

out.release()
print(f"Simulation saved as {VIDEO_PATH}")