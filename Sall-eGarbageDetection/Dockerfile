FROM python:3.9

# Set working directory inside the container
WORKDIR /app

# Reduce Docker image size by cleaning up unused files
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* ~/.cache/

# Install system dependencies before switching users
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libglib2.0-0 fontconfig && \
    rm -rf /var/lib/apt/lists/*

# Copy entrypoint script and make it executable
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Create a non-root user for better security
RUN useradd -m -u 1000 user
USER root
ENV HOME=/home/user

# Install FFmpeg before running the app
RUN apt-get update && apt-get install -y ffmpeg

# Install Python dependencies (including `python-multipart`)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Install Hugging Face CLI and required ML dependencies
RUN pip install --no-cache-dir \
    huggingface_hub==0.14.1 \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Ensure Hugging Face CLI is accessible
ENV PATH="/home/user/.local/bin:$PATH"

# Create model/cache folder and ensure write permissions
RUN mkdir -p /home/user/app/model /home/user/app/cache /home/user/app/cache/uploads && \
    chown -R user:user /home/user/app && \
    chmod -R 777 /home/user/app

# Switch to non-root user before copying source code
USER user
WORKDIR $HOME/app

# Copy source code **after** fixing permissions
COPY --chown=user . $HOME/app

# Copy and execute the model download script
# COPY download_models.py /home/user/app/download_models.py
# RUN python /home/user/app/download_models.py

# Manually download DETR model using `snapshot_download()`
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='facebook/detr-resnet-50', local_dir='/home/user/app/model/detr', local_dir_use_symlinks=False)"

# Download other necessary models
RUN wget -O $HOME/app/model/garbage_detector.pt https://huggingface.co/BinKhoaLe1812/Garbage_Detection/resolve/main/garbage_detector.pt
RUN wget -O $HOME/app/model/yolov5-detect-trash-classification.pt https://huggingface.co/turhancan97/yolov5-detect-trash-classification/resolve/main/yolov5s.pt

# Ensure correct file permissions
RUN chmod -R 777 $HOME/app

# Verify model setup before running the app
RUN python setup.py

# Remove previous video file each Docker restart to reduce storage
RUN rm -rf /home/user/app/outputs/*.mp4

# Expose FastAPI port
EXPOSE 7860

# Start FastAPI application using the entrypoint script
ENTRYPOINT ["/entrypoint.sh"]
