FROM python:3.9-slim

WORKDIR /app

# Install dependencies including required packages for OpenCV and PDF processing
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    # PDF processing dependencies - install all variants
    poppler-utils \
    poppler-data \
    # Required for PDF processing
    ghostscript \
    # Required for PyMuPDF
    libmupdf-dev \
    && rm -rf /var/lib/apt/lists/*

# Ensure poppler binaries are in PATH
ENV PATH="/usr/bin:/usr/local/bin:${PATH}"

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install requirements and special packages
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir opencv-python && \
    pip install --no-cache-dir opencv-contrib-python && \
    # Ensure PDF libraries are correctly installed
    pip install --no-cache-dir PyMuPDF && \
    pip install --no-cache-dir pdf2image

# Verify installation paths (helpful for debugging)
RUN echo "Checking for poppler binaries:" && \
    which pdftoppm && which pdfinfo || echo "Poppler binaries not found" && \
    echo "PyMuPDF version: " && pip show PyMuPDF | grep Version && \
    echo "pdf2image version: " && pip show pdf2image | grep Version

# Copy application code
COPY . .

# Expose port
EXPOSE 5000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HEADLESS=1
ENV DOCKER_CONTAINER=1
ENV API_HOST=http://localhost:5000

# Run the application
CMD ["python", "run_api.py"] 