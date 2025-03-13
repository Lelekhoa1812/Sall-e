FROM python:3.9

WORKDIR /app

# Install system dependencies for OpenCV and FontConfig fix
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 fontconfig && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir huggingface_hub transformers yolov5 ultralytics torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

# Create model folder and download models
RUN mkdir -p /app/model

# Download models from Hugging Face
RUN wget -O /app/model/garbage_detector.pt https://huggingface.co/BinKhoaLe1812/Garbage_Detection/resolve/main/garbage_detector.pt
RUN wget -O /app/model/detr-resnet-50_finetuned_detect-waste.bin https://huggingface.co/Yorai/detr-resnet-50_finetuned_detect-waste/resolve/main/pytorch_model.bin
RUN wget -O /app/model/yolov5-detect-trash-classification.pt https://huggingface.co/turhancan97/yolov5-detect-trash-classification/resolve/main/yolov5s.pt

# Set environment variables for write permissions
ENV MPLCONFIGDIR="/tmp/matplotlib"
ENV YOLO_CONFIG_DIR="/tmp/Ultralytics"

COPY . .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
