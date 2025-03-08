import os
import cv2
import random
import numpy as np
from glob import glob

def remove_background(image_path):
    """Remove complex backgrounds while keeping only the object with transparency."""
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive threshold to enhance object edges
    _, mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Morphological operations to remove noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Find contours and extract the largest one (assuming the object is the largest blob)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [largest_contour], -1, (255), thickness=cv2.FILLED)
    
    # Smooth edges using dilation and erosion
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.erode(mask, kernel, iterations=2)
    
    # Create an RGBA image with transparency
    b, g, r = cv2.split(img)
    rgba = cv2.merge([b, g, r, mask])
    
    return rgba

def process_images(sample_folder, output_folder, num_samples=10):
    """Process and crop objects from each class folder in 'sample' and save to 'crop'."""
    os.makedirs(output_folder, exist_ok=True)
    
    class_folders = [d for d in os.listdir(sample_folder) if os.path.isdir(os.path.join(sample_folder, d))]
    
    for class_name in class_folders:
        class_path = os.path.join(sample_folder, class_name)
        image_paths = glob(os.path.join(class_path, '*.jpg'))
        
        if not image_paths:
            continue
        
        random.shuffle(image_paths)
        selected_images = image_paths[:num_samples]
        
        class_output_folder = os.path.join(output_folder, class_name)
        os.makedirs(class_output_folder, exist_ok=True)
        
        for idx, img_path in enumerate(selected_images, start=1):
            processed_img = remove_background(img_path)
            
            if processed_img is not None:
                output_path = os.path.join(class_output_folder, f"{class_name}_{idx}.png")
                cv2.imwrite(output_path, processed_img)
                print(f"Saved: {output_path}")

# Paths
sample_folder = "/sample"
output_folder = "/crop"

# Run the processing function
process_images(sample_folder, output_folder)