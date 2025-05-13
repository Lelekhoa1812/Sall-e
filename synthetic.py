# ─────────────────────────────── Setup ───────────────────────────────
import os, random, cv2, numpy as np
from PIL import Image, UnidentifiedImageError
from google.colab import files
from io import BytesIO
import IPython.display as display

# Mount Drive to access crop_dir
crop_dir = "/content/drive/My Drive/Sall-e/crop"
chunk_dir = os.path.join(crop_dir, "chunk")

# ───────────────────────── Upload Interface ──────────────────────────
print("Upload a base environment image (e.g., river or ocean):")
uploaded = files.upload()
uploaded_filename = list(uploaded.keys())[0]

# Save uploaded file locally
with open(uploaded_filename, 'wb') as f:
    f.write(uploaded[uploaded_filename])

# ────── Resize Uploaded Image to 640x640 ──────
def resize_image(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        image = image.resize((640, 640))
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    except UnidentifiedImageError:
        print(f"Error: Cannot identify image file {image_path}")
        return None

background = resize_image(uploaded_filename)
if background is None:
    raise RuntimeError("Failed to load image.")

# ────── Overlay RGBA image (PNG) onto background ──────
def overlay_image(background, overlay, x, y):
    h, w = overlay.shape[:2]
    bg_rgb = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
    alpha = overlay[:, :, 3] / 255.0
    for c in range(3):  # RGB channels
        bg_rgb[y:y+h, x:x+w, c] = (
            alpha * overlay[:, :, c] + (1 - alpha) * bg_rgb[y:y+h, x:x+w, c]
        )
    return cv2.cvtColor(bg_rgb, cv2.COLOR_RGB2BGR)

# ──────────────────────── Part 1: Object Overlays ────────────────────────
for class_name in os.listdir(crop_dir):
    class_path = os.path.join(crop_dir, class_name)
    if not os.path.isdir(class_path) or class_name == "chunk":
        continue  # skip non-dirs and chunk dir here

    pngs = [f for f in os.listdir(class_path) if f.endswith(".png")]
    selected = random.sample(pngs, min(5, len(pngs)))

    for img_name in selected:
        img_path = os.path.join(class_path, img_name)
        img = Image.open(img_path).convert("RGBA")

        # Resize to height 20px
        scale = 20 / img.size[1]
        new_size = (int(img.size[0] * scale), 20)
        img = img.resize(new_size)

        overlay = np.array(img)
        x = random.randint(0, 640 - new_size[0])
        y = random.randint(0, 640 - 30)

        background = overlay_image(background, overlay, x, y)

# ──────────────────────── Part 2: Chunk Overlays ────────────────────────
chunk_files = [f for f in os.listdir(chunk_dir) if f.endswith(".png")]
for chunk_img in chunk_files:
    chunk_path = os.path.join(chunk_dir, chunk_img)
    img = Image.open(chunk_path).convert("RGBA").resize((30, 30))
    overlay = np.array(img)

    x = random.randint(0, 610)
    y = random.randint(0, 610)
    background = overlay_image(background, overlay, x, y)

# ──────────────────────── Save + Display Result ────────────────────────
output_path = "synthetic_test_image.jpg"
cv2.imwrite(output_path, background)
print(f"✅ Generated and saved: {output_path}")

# Display inline in notebook
display.display(Image.open(output_path))