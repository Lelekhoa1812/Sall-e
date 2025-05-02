# Access: https://BinKhoaLe1812-Sall-eGarbageDetection.hf.space/ui

# ───────────────────────── app.py (Sall-e demo) ─────────────────────────
# FastAPI ▸ upload image ▸ multi-model garbage detection ▸ ADE-20K
# semantic segmentation (Water / Garbage) ▸ A* + KNN navigation ▸ H.264 video
# =======================================================================

import os, uuid, threading, shutil, time, heapq, cv2, numpy as np
from PIL import Image
import uvicorn
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, StreamingResponse, Response
from fastapi.staticfiles import StaticFiles

# ── Vision libs ─────────────────────────────────────────────────────────
import torch, yolov5, ffmpeg
from ultralytics import YOLO
from transformers import (
    DetrImageProcessor, DetrForObjectDetection,
    SegformerFeatureExtractor, SegformerForSemanticSegmentation
)
from sklearn.neighbors import NearestNeighbors

# ── Folders / files ─────────────────────────────────────────────────────
BASE        = "/home/user/app"
CACHE       = f"{BASE}/cache"
UPLOAD_DIR  = f"{CACHE}/uploads"
OUTPUT_DIR  = f"{BASE}/outputs"
MODEL_DIR   = f"{BASE}/model"
SPRITE      = f"{BASE}/sprite.png"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE     , exist_ok=True)
os.environ["TRANSFORMERS_CACHE"] = CACHE
os.environ["HF_HOME"]           = CACHE

# ── Load models once  ───────────────────────────────────────────────────
print("🔄  Loading models …")
model_self  = YOLO(f"{MODEL_DIR}/garbage_detector.pt")                 # YOLOv11(l)
model_yolo5 = yolov5.load(f"{MODEL_DIR}/yolov5-detect-trash-classification.pt")
processor_detr = DetrImageProcessor.from_pretrained(f"{MODEL_DIR}/detr")
model_detr     = DetrForObjectDetection.from_pretrained(f"{MODEL_DIR}/detr")
feat_extractor = SegformerFeatureExtractor.from_pretrained(
                    "nvidia/segformer-b4-finetuned-ade-512-512")
segformer      = SegformerForSemanticSegmentation.from_pretrained(
                    "nvidia/segformer-b4-finetuned-ade-512-512")
print("✅  Models ready\n")

# ── ADE-20K palette + custom mapping (verbatim) ─────────────────────────
# ADE20K palette
ade_palette = np.array([
    [0, 0, 0], [120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
    [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255], [230, 230, 230],
    [4, 250, 7], [224, 5, 255], [235, 255, 7], [150, 5, 61], [120, 120, 70],
    [8, 255, 51], [255, 6, 82], [143, 255, 140], [204, 255, 4], [255, 51, 7],
    [204, 70, 3], [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
    [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220], [255, 9, 92],
    [112, 9, 255], [8, 255, 214], [7, 255, 224], [255, 184, 6], [10, 255, 71],
    [255, 41, 10], [7, 255, 255], [224, 255, 8], [102, 8, 255], [255, 61, 6],
    [255, 194, 7], [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
    [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255], [140, 140, 140],
    [250, 10, 15], [20, 255, 0], [31, 255, 0], [255, 31, 0], [255, 224, 0],
    [153, 255, 0], [0, 0, 255], [255, 71, 0], [0, 235, 255], [0, 173, 255],
    [31, 0, 255], [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
    [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0], [255, 102, 0],
    [194, 255, 0], [0, 143, 255], [51, 255, 0], [0, 82, 255], [0, 255, 41],
    [0, 255, 173], [10, 0, 255], [173, 255, 0], [0, 255, 153], [255, 92, 0],
    [255, 0, 255], [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
    [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255], [255, 0, 204],
    [0, 255, 194], [0, 255, 82], [0, 10, 255], [0, 112, 255], [51, 0, 255],
    [0, 194, 255], [0, 122, 255], [0, 255, 163], [255, 153, 0], [0, 255, 10],
    [255, 112, 0], [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
    [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255], [255, 0, 31],
    [0, 184, 255], [0, 214, 255], [255, 0, 112], [92, 255, 0], [0, 224, 255],
    [112, 224, 255], [70, 184, 160], [163, 0, 255], [153, 0, 255], [71, 255, 0],
    [255, 0, 163], [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
    [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0], [10, 190, 212],
    [214, 255, 0], [0, 204, 255], [20, 0, 255], [255, 255, 0], [0, 153, 255],
    [0, 41, 255], [0, 255, 204], [41, 0, 255], [41, 255, 0], [173, 0, 255],
    [0, 245, 255], [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
    [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194], [102, 255, 0],
    [92, 0, 255]
], dtype=np.uint8)

custom_class_map = {
    "Garbage":                 [(150, 5, 61)],
    "Water":                   [(0, 102, 200), (11, 102, 255), (31, 0, 255)],
    "Grass / Vegetation":      [(10, 255, 71), (143, 255, 140)],
    "Tree / Natural Obstacle": [(4, 200, 3), (235, 12, 255), (255, 6, 82), (255, 163, 0)],
    "Sand / Soil / Ground":    [(80, 50, 50), (230, 230, 230)],
    "Buildings / Structures":  [(255, 0, 255), (184, 0, 255), (120, 120, 120), (7, 255, 224)],
    "Sky / Background":        [(180, 120, 120)],
    "Undetecable":             [(0, 0, 0)],
    "Unknown Class": []
}
TOL = 30  # RGB tolerance

# Masking zones (Garbage and Water zone to be travelable)
def build_masks(seg):
    """
    Returns three binary masks at (H,W):
    water_mask   – 1 = water
    garbage_mask – 1 = semantic “Garbage” pixels
    movable_mask – union of water & garbage (robot can travel here)
    """
    decoded = ade_palette[seg]
    water_mask   = np.zeros(seg.shape, np.uint8)
    garbage_mask = np.zeros_like(water_mask)
    # Append water pixels to water_mask
    for rgb in custom_class_map["Water"]:
        water_mask |= (np.abs(decoded - rgb).max(axis=-1) <= TOL)
    # Append gb pixels to garbage_mask
    for rgb in custom_class_map["Garbage"]:
        garbage_mask |= (np.abs(decoded - rgb).max(axis=-1) <= TOL)
    movable_mask = water_mask | garbage_mask
    return water_mask, garbage_mask, movable_mask

# ── A* and KNN over binary water grid ─────────────────────────────────
def astar(start, goal, occ):
    h   = lambda a,b: abs(a[0]-b[0])+abs(a[1]-b[1])
    N8  = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    openq=[(0,start)]; g={start:0}; came={}
    while openq:
        _,cur=heapq.heappop(openq)
        if cur==goal:
            p=[cur];                   # reconstruct
            while cur in came: cur=came[cur]; p.append(cur)
            return p[::-1]
        for dx,dy in N8:
            nx,ny=cur[0]+dx,cur[1]+dy
            if not (0<=nx<640 and 0<=ny<640): continue
            if occ[ny,nx]==0: continue
            ng=g[cur]+1
            if (nx,ny) not in g or ng<g[(nx,ny)]:
                g[(nx,ny)]=ng
                f=ng+h((nx,ny),goal)
                heapq.heappush(openq,(f,(nx,ny)))
                came[(nx,ny)]=cur
    return []

# KNN fit
def knn_path(start, targets, occ):
    todo = targets[:]; path=[]
    cur  = tuple(start)
    while todo:
        nbrs = NearestNeighbors(n_neighbors=1).fit(todo)
        _,idx = nbrs.kneighbors([cur]); nxt=tuple(todo[idx[0][0]])
        seg  = astar(cur, nxt, occ)
        if seg:
            if path and seg[0]==path[-1]: seg=seg[1:]
            path.extend(seg)
        cur  = nxt; todo.remove(list(nxt))
    return path

# ── Robot sprite/class -──────────────────────────────────────────────────
class Robot:
    def __init__(self, sprite, speed=200): # Declare the robot's physical stats and routing (position, speed, movement, path)
        self.png = np.array(Image.open(sprite).convert("RGBA").resize((40,40)))
        self.pos = [0,0]; self.speed=speed
    def step(self, path):
        if not path: return
        dx,dy = path[0][0]-self.pos[0], path[0][1]-self.pos[1]
        dist = (dx*dx+dy*dy)**0.5
        if dist<=self.speed:
            self.pos=list(path.pop(0)); return
        r=self.speed/dist; self.pos=[int(self.pos[0]+dx*r), int(self.pos[1]+dy*r)]

# ── FastAPI & HTML content (original styling) ───────────────────────────
# HTML Content for UI (streamed with FastAPI HTML renderer)
HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head>
    <title>Sall-e Garbage Detection</title>
    <link rel="website icon" type="png" href="/static/icon.png" >
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

# ── Static-web ────────────────────────────────────────────────────────── 
app = FastAPI()
app.mount("/static", StaticFiles(directory=BASE), name="static")
video_ready={}
@app.get("/ui", response_class=HTMLResponse)
def ui(): return HTML_CONTENT
def _uid(): return uuid.uuid4().hex[:8]

# ── End-points ──────────────────────────────────────────────────────────
# User upload environment img here
@app.post("/upload/")
async def upload(file:UploadFile=File(...)):
    uid=_uid(); dest=f"{UPLOAD_DIR}/{uid}_{file.filename}"
    with open(dest,"wb") as bf: shutil.copyfileobj(file.file,bf)
    threading.Thread(target=_pipeline, args=(uid,dest)).start()
    return {"user_id":uid}

# Health check, make sure the video generator is alive and debug which video id is processed (multiple video can be processed at 1 worker)
@app.get("/check_video/{uid}")
def chk(uid:str): return {"ready":video_ready.get(uid,False)}

# Where the final video being saved
@app.get("/video/{uid}")
def stream(uid:str):
    vid=f"{OUTPUT_DIR}/{uid}.mp4"
    if not os.path.exists(vid): return Response(status_code=404)
    return StreamingResponse(open(vid,"rb"), media_type="video/mp4")

# ── Core pipeline (runs in background thread) ───────────────────────────
def _pipeline(uid,img_path):
    print(f"▶️ [{uid}] processing")
    bgr=cv2.resize(cv2.imread(img_path),(640,640)); rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    pil=Image.fromarray(rgb)

    # 1- Segmentation → masking each segmented zone with pytorch
    with torch.no_grad():
        inputs = feat_extractor(pil, return_tensors="pt")
        seg_logits = segformer(**inputs).logits
    # Tensor run by CPU
    seg_tensor = seg_logits.argmax(1)[0].cpu()
    if seg_tensor.numel() == 0:
        print(f"❌ [{uid}] segmentation failed (empty tensor)")
        video_ready[uid] = True
        return
    # Resize the tensor to 640x640
    seg = cv2.resize(seg_tensor.numpy(), (640, 640), interpolation=cv2.INTER_NEAREST)
    print(f"🧪 [{uid}] segmentation input shape: {inputs['pixel_values'].shape}")
    water_mask, garbage_mask, movable_mask = build_masks(seg) # movable zone = water and garbage masks

    # 2- Garbage detection (3 models) → keep centres on water 
    detections=[]
    # Detect garbage chunks (from segmentation)
    num_cc, labels = cv2.connectedComponents(garbage_mask.astype(np.uint8))
    chunk_centres = []
    for lab in range(1, num_cc):
        ys, xs = np.where(labels == lab)
        if xs.size == 0: # safety
            continue
        chunk_centres.append([int(xs.mean()), int(ys.mean())])
    print(f"🧠 {len(chunk_centres)} garbage chunk detected")
    # Detect garbage object by within travelable zones
    for r in model_self(bgr):                      # YOLOv11 (self-trained)
        detections += [b.xyxy[0].tolist() for b in r.boxes]
    for r in model_yolo5(bgr):                     # YOLOv5
        if hasattr(r, 'pred') and len(r.pred) > 0:
            detections += [p[:4].tolist() for p in r.pred[0]]
    inp=processor_detr(images=pil,return_tensors="pt")
    with torch.no_grad(): out=model_detr(**inp)    # DETR
    post = processor_detr.post_process_object_detection(
        outputs=out,
        target_sizes=torch.tensor([pil.size[::-1]]),
        threshold=0.5
    )[0]
    detections += [b.tolist() for b in post["boxes"]]
    # centre & mask filter (the garbage lies within travelable zone are collectable)
    centres = []
    for x1, y1, x2, y2 in detections: # Define IoU heuristic
        '''
        We conduct a 30% allowance whether the center 
        of the detected garbage's bbox lies within the travelable zone
        which was segmented earlier to be the water and garbage zone
        '''
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        x1 = max(0, min(x1, 639)); y1 = max(0, min(y1, 639))
        x2 = max(0, min(x2, 639)); y2 = max(0, min(y2, 639))
        box_mask = movable_mask[y1:y2, x1:x2]              # ← use MOVABLE mask
        if box_mask.size == 0:
            continue
        if np.count_nonzero(box_mask) / box_mask.size >= 0.3:
            centres.append([int((x1 + x2) / 2), int((y1 + y2) / 2)])
    # add chunk centres and deduplicate
    centres.extend(chunk_centres)
    centres = [list(c) for c in {tuple(c) for c in centres}]
    if not centres: # No garbages within travelable zone
        print(f"🛑 [{uid}] no reachable garbage"); video_ready[uid]=True; return
    else: # Garbage within valid travelable zone
        print(f"🧠 {len(centres)} garbage objects on water selected from {len(detections)} detections")

    # 3- Global route
    robot = Robot(SPRITE)
    path  = knn_path(robot.pos, centres, movable_mask)

    # 4- Video synthesis
    out_tmp=f"{OUTPUT_DIR}/{uid}_tmp.mp4"
    vw=cv2.VideoWriter(out_tmp,cv2.VideoWriter_fourcc(*"mp4v"),10.0,(640,640))
    objs=[{"pos":p,"col":False} for p in centres]
    bg = bgr.copy()
    for _ in range(15000): # safety frames
        frame=bg.copy()
        # draw garbage
        for o in objs:
            color=(0,0,255) if not o["col"] else (0,255,0)
            x,y=o["pos"]; cv2.circle(frame,(x,y),6,color,-1)
        # robot
        robot.step(path)
        rx,ry=robot.pos; sp=robot.png
        a=sp[:,:,3]/255.; bgroi=frame[ry:ry+40,rx:rx+40]
        for c in range(3): bgroi[:,:,c]=a*sp[:,:,c]+(1-a)*bgroi[:,:,c]
        frame[ry:ry+40,rx:rx+40]=bgroi
        # collection check
        for o in objs:
            if not o["col"] and np.hypot(o["pos"][0]-rx,o["pos"][1]-ry)<=20:
                o["col"]=True
        vw.write(frame)
        if all(o["col"] for o in objs): break
        if not path: break
    vw.release()

    # 5- Convert to H.264
    final=f"{OUTPUT_DIR}/{uid}.mp4"
    ffmpeg.input(out_tmp).output(final,vcodec="libx264",pix_fmt="yuv420p").run(overwrite_output=True,quiet=True)
    os.remove(out_tmp); video_ready[uid]=True
    print(f"✅ [{uid}] video ready → {final}")

# ── Run locally (HF Space ignores since built with Docker image) ────────
if __name__=="__main__":
    uvicorn.run(app,host="0.0.0.0",port=7860)
