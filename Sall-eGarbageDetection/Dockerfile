FROM python:3.9

WORKDIR /app

# Install system dependencies for OpenCV and FontConfig fix
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 fontconfig && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Explicitly install a compatible version of huggingface_hub
RUN pip install --no-cache-dir huggingface_hub==0.14.1 transformers yolov5 ultralytics torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

# Create model folder and download models
RUN mkdir -p /app/model /app/cache
# Grant cache permission
RUN chmod -R 777 /app/cache 

# Set environment variables for writable cache
ENV MPLCONFIGDIR="/tmp/matplotlib"
ENV YOLO_CONFIG_DIR="/tmp/Ultralytics"
ENV TRANSFORMERS_CACHE="/app/cache"
ENV HF_HOME="/app/cache"

COPY . .

# Download models from Hugging Face to the writable directory
RUN HF_HOME=/app/cache wget -O /app/model/garbage_detector.pt https://huggingface.co/BinKhoaLe1812/Garbage_Detection/resolve/main/garbage_detector.pt
RUN HF_HOME=/app/cache wget -O /app/model/detr-resnet-50_finetuned_detect-waste.bin https://huggingface.co/Yorai/detr-resnet-50_finetuned_detect-waste/resolve/main/pytorch_model.bin
RUN HF_HOME=/app/cache wget -O /app/model/yolov5-detect-trash-classification.pt https://huggingface.co/turhancan97/yolov5-detect-trash-classification/resolve/main/yolov5s.pt


# Check model and cache directory setup
# RUN python setup.py

# Expose port
EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
