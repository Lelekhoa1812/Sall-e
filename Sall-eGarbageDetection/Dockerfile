FROM python:3.9

# Set working directory inside the container
WORKDIR /app

# ✅ Install system dependencies before switching users
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libglib2.0-0 fontconfig && \
    rm -rf /var/lib/apt/lists/*

# ✅ Copy entrypoint script and make it executable
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# ✅ Create a non-root user for better security
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user
WORKDIR $HOME/app

# ✅ Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# ✅ Install Hugging Face CLI and required ML dependencies
RUN pip install --no-cache-dir \
    huggingface_hub==0.14.1 transformers yolov5 ultralytics torch torchvision torchaudio \
    --extra-index-url https://download.pytorch.org/whl/cpu

# ✅ Ensure Hugging Face CLI is accessible
ENV PATH="/home/user/.local/bin:$PATH"

# ✅ Create model folder (cache is now handled via /dev/shm)
RUN mkdir -p /home/user/app/model

# ✅ Set writable environment variables
ENV MPLCONFIGDIR="/tmp/matplotlib"
ENV YOLO_CONFIG_DIR="/tmp/Ultralytics"
ENV TRANSFORMERS_CACHE="/dev/shm"
ENV HF_HOME="/dev/shm"

# ✅ Copy source code **after** user switch
COPY --chown=user . $HOME/app

# ✅ Copy the model download script with correct permission
COPY download_models.py /home/user/app/download_models.py
RUN python /home/user/app/download_models.py


# ✅ Download other necessary models
RUN wget -O $HOME/app/model/garbage_detector.pt https://huggingface.co/BinKhoaLe1812/Garbage_Detection/resolve/main/garbage_detector.pt
RUN wget -O $HOME/app/model/yolov5-detect-trash-classification.pt https://huggingface.co/turhancan97/yolov5-detect-trash-classification/resolve/main/yolov5s.pt

# ✅ Ensure correct file permissions
RUN chmod -R 777 $HOME/app

# ✅ Verify model setup before running the app
RUN python setup.py

# ✅ Expose FastAPI port
EXPOSE 7860

# ✅ Start FastAPI application using the entrypoint script
ENTRYPOINT ["/entrypoint.sh"]