FROM python:3.9-slim

WORKDIR /app

# Install dependencies including required packages for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install requirements and OpenCV packages
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir opencv-python && \
    pip install --no-cache-dir opencv-contrib-python

# Copy application code
COPY . .

# Expose port
EXPOSE 5000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HEADLESS=1
ENV DOCKER_CONTAINER=1

# Run the application
CMD ["python", "run_api.py"] 