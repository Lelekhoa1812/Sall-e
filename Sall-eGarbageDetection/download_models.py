from huggingface_hub import hf_hub_download
import os
import shutil

# Define model download directory
model_dir = "/home/user/app/model"
cache_dir = "/home/user/app/cache"

max_cache_size = 500 * 1024 * 1024  # 500MB

# Ensure the directory exists
os.makedirs(model_dir, exist_ok=True)
os.makedirs(cache_dir, exist_ok=True)

# Download DETR model (only the model weights (167MB)) and save to local model directory
print("🚀 Downloading DETR model from Hugging Face...")
hf_hub_download(repo_id="facebook/detr-resnet-50", filename="pytorch_model.bin", local_dir=f"{model_dir}/detr", cache_dir=cache_dir)

print("✅ DETR model downloaded successfully!")

if os.path.exists(cache_dir) and shutil.disk_usage(cache_dir).used > max_cache_size:
    print("🗑️ Clearing Hugging Face cache to free up space...")
    shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)