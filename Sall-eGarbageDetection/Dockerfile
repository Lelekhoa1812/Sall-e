FROM python:3.9

# Set working directory inside the container
WORKDIR /app

# Reduce Docker image size by cleaning up unused files
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* ~/.cache/

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libglib2.0-0 fontconfig \
    git wget ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Copy entrypoint script and make it executable
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Create non-root user
RUN useradd -m -u 1000 user
USER root
ENV HOME=/home/user

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Install Hugging Face CLI + required ML dependencies
RUN pip install --no-cache-dir huggingface_hub --upgrade

# Install LoveDA repository dependencies
RUN git clone https://github.com/Junjue-Wang/LoveDA.git /home/user/LoveDA
WORKDIR /home/user/LoveDA
RUN pip install --no-cache-dir -r requirements.txt || true  # (ignores errors if any)

# Return to working dir
WORKDIR $HOME/app

# Ensure Hugging Face CLI is accessible
ENV PATH="/home/user/.local/bin:$PATH"

# Create model/cache folders
RUN mkdir -p /home/user/app/model /home/user/app/cache/uploads && \
    chown -R user:user /home/user/app && chmod -R 777 /home/user/app

# Switch back to non-root user
USER user
WORKDIR $HOME/app

# Copy application source code
COPY --chown=user . $HOME/app

# Pre-download all Hugging Face models
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='facebook/detr-resnet-50', local_dir='/home/user/app/model/detr', local_dir_use_symlinks=False)"
RUN wget -O $HOME/app/model/garbage_detector.pt https://huggingface.co/BinKhoaLe1812/Garbage_Detection/resolve/main/garbage_detector.pt
RUN wget -O $HOME/app/model/yolov5-detect-trash-classification.pt https://huggingface.co/turhancan97/yolov5-detect-trash-classification/resolve/main/yolov5s.pt

# Verify model setup
RUN python setup.py

# Clean outputs folder
RUN rm -rf /home/user/app/outputs/*.mp4

# Copy static images
COPY sprite.png /home/user/app/sprite.png
COPY icon.png /home/user/app/icon.png

# Expose FastAPI port
EXPOSE 7860

# Start app
ENTRYPOINT ["/entrypoint.sh"]
