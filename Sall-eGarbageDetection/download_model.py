from huggingface_hub import snapshot_download
import os

# Define model download directory
model_dir = "/home/user/app/model"

# Ensure the directory exists
os.makedirs(model_dir, exist_ok=True)

# Download DETR model and save to local model directory
print("🚀 Downloading DETR model from Hugging Face...")
snapshot_download(repo_id="facebook/detr-resnet-50", local_dir=f"{model_dir}/detr")

print("✅ DETR model downloaded successfully!")
