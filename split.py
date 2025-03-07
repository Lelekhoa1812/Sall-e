import os
import shutil
import random

# Define paths
train_dir = "train"
val_dir = "valid"
test_dir = "test"

# Define split ratios
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# Ensure output directories exist
for split_dir in [train_dir, val_dir, test_dir]:
    os.makedirs(split_dir, exist_ok=True)

# Get class names (subdirectories current dir)
class_names = [d for d in os.listdir() if os.path.isdir(d) and d not in {"train", "val", "test"}]

# Process each class
for class_name in class_names:
    class_path = class_name
    images = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
    
    # Shuffle images randomly
    random.shuffle(images)

    # Compute split indices
    total_images = len(images)
    train_count = int(total_images * train_ratio)
    val_count = int(total_images * val_ratio)

    # Split images
    train_images = images[:train_count]
    val_images = images[train_count:train_count + val_count]
    test_images = images[train_count + val_count:]

    # Define destination directories for the class
    for split_name, split_images in zip(["train", "val", "test"], [train_images, val_images, test_images]):
        split_class_dir = os.path.join(split_name, class_name)
        os.makedirs(split_class_dir, exist_ok=True)
        
        # Move images
        for image in split_images:
            src = os.path.join(class_path, image)
            dst = os.path.join(split_class_dir, image)
            shutil.move(src, dst)

print("Dataset successfully split into train, val, and test sets.")
