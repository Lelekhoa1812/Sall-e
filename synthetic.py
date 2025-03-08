import os
import random
import cv2
import numpy as np
from PIL import Image
import PIL

# Ensure AVIF support
try:
    import pillow_avif
except ImportError:
    print("Warning: 'pillow-avif-plugin' is not installed. Install it using 'pip install pillow-avif-plugin'.")

# Paths
crop_dir = "/crop"
testing_dir = "/testing"
ocean_images = ["ocean1.jpg", "ocean2.avif", "ocean3.jpeg", "ocean4.jpg"]

ocean_images = [os.path.join("/src", img) for img in ocean_images]
os.makedirs(testing_dir, exist_ok=True)

# Resize ocean images to 640x640 px
def resize_image(image_path):
    print(f"Processing image: {image_path}")
    
    # Process individually by ocean image types
    try:
        if image_path.lower().endswith(".avif"):
            image = Image.open(image_path).convert("RGB")
            image = image.resize((640, 640))
            image = np.array(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Could not read image {image_path}")
                return None
            image = cv2.resize(image, (640, 640))
    except PIL.UnidentifiedImageError:
        print(f"Error: Unrecognized image format {image_path}, skipping...")
        return None
    
    return image

# Get class directories
classes = [d for d in os.listdir(crop_dir) if os.path.isdir(os.path.join(crop_dir, d))]

# Function to overlay PNG objects onto an image
def overlay_image(background, overlay, x, y):
    h, w, _ = overlay.shape
    alpha_mask = overlay[:, :, 3] / 255.0
    for c in range(0, 3):
        background[y:y+h, x:x+w, c] = (alpha_mask * overlay[:, :, c] + (1 - alpha_mask) * background[y:y+h, x:x+w, c])
    return background

# Process each ocean image
for idx, ocean_img_path in enumerate(ocean_images, start=1):
    ocean_img = resize_image(ocean_img_path)
    if ocean_img is None:
        continue  # Skip if image could not be read
    
    # Select 8 random PNG images from each class
    for class_name in classes:
        class_path = os.path.join(crop_dir, class_name)
        png_files = [f for f in os.listdir(class_path) if f.endswith(".png")]
        selected_pngs = random.sample(png_files, min(8, len(png_files)))
        
        for png in selected_pngs:
            png_path = os.path.join(class_path, png)
            overlay = Image.open(png_path).convert("RGBA")
            
            # Scale down the overlay to height 10px
            original_width, original_height = overlay.size
            scale_factor = 10 / original_height
            new_width = int(original_width * scale_factor)
            overlay = overlay.resize((new_width, 10))
            
            # Convert overlay to NumPy array
            overlay_np = np.array(overlay)
            
            # Select a random position on the ocean image
            x_offset = random.randint(0, 640 - new_width)
            y_offset = random.randint(0, 640 - 10)
            
            # Overlay the image
            ocean_img = overlay_image(ocean_img, overlay_np, x_offset, y_offset)
    
    # Save the generated synthetic image with new naming convention
    output_path = os.path.join(testing_dir, f"testing_{idx}.jpg")
    cv2.imwrite(output_path, ocean_img)
    print(f"{output_path} generated.")

print("Synthetic images generated successfully!")