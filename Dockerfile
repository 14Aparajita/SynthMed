# Use python 3.9 as base
FROM python:3.9-slim

# Install system dependencies needed for OpenCV, PyTorch, FAISS
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set up user with uid 1000 for Hugging Face Space security
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy requirements and install
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all codebase
COPY --chown=user . .

# Create outputs/models/ directories in case they are needed
RUN mkdir -p outputs/models outputs/results outputs/logs data/processed

# Expose HF default port
EXPOSE 7860

# Start app.py (wrapped to port 7860)
CMD ["python", "app.py"]
