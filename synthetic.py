import os
import cv2
import random
import numpy as np
from glob import glob

def resize_background(image_path, target_size=(640, 640)):
    """Resize ocean background to target size while keeping aspect ratio."""
    img = cv2.imread(image_path)
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
    return img

def load_yolo_labels(label_path):
    """Load YOLO format labels from a file."""
    labels = []
    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            cls, x_center, y_center, width, height = map(float, parts)
            labels.append((cls, x_center, y_center, width, height))
    return labels

def crop_objects(image_path, label_path):
    """Crop objects from an image based on YOLO labels."""
    img = cv2.imread(image_path)
    h, w, _ = img.shape
    objects = []
    labels = load_yolo_labels(label_path)
    
    for cls, x_c, y_c, w_rel, h_rel in labels:
        x = int((x_c - w_rel / 2) * w)
        y = int((y_c - h_rel / 2) * h)
        width = int(w_rel * w)
        height = int(h_rel * h)
        
        cropped_obj = img[y:y+height, x:x+width]
        if cropped_obj.size > 0:
            objects.append(cropped_obj)
    return objects

def generate_synthetic_images(ocean_img_path, test_img_folder, test_label_folder, output_folder, num_images=3):
    """Generate synthetic images by placing cropped objects on an ocean background."""
    os.makedirs(output_folder, exist_ok=True)
    
    test_images = glob(os.path.join(test_img_folder, '*.jpg'))
    random.shuffle(test_images)
    selected_images = test_images[:10]
    
    all_objects = []
    for img_path in selected_images:
        label_path = os.path.join(test_label_folder, os.path.basename(img_path).replace('.jpg', '.txt'))
        if os.path.exists(label_path):
            objects = crop_objects(img_path, label_path)
            all_objects.extend(objects)
    
    for i in range(num_images):
        ocean_img = resize_background(ocean_img_path)
        
        for obj in all_objects:
            obj_h, obj_w, _ = obj.shape
            x_offset = random.randint(0, 640 - obj_w)
            y_offset = random.randint(0, 640 - obj_h)
            ocean_img[y_offset:y_offset + obj_h, x_offset:x_offset + obj_w] = obj
        
        output_path = os.path.join(output_folder, f'synthetic_test_{i+1}.jpg')
        cv2.imwrite(output_path, ocean_img)
        print(f"Saved: {output_path}")

# Paths
ocean_image_path = 'src/ocean.jpg'
test_images_path = 'dataset/test/images'
test_labels_path = 'dataset/test/labels'
output_path = 'testing'

# Run the synthetic image generator
generate_synthetic_images(ocean_image_path, test_images_path, test_labels_path, output_path)
