FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    poppler-utils \
    poppler-data \
    ghostscript \
    libmupdf-dev \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/usr/bin:/usr/local/bin:${PATH}"

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir opencv-python && \
    pip install --no-cache-dir opencv-contrib-python && \
    pip install --no-cache-dir PyMuPDF && \
    pip install --no-cache-dir pdf2image

RUN echo "Checking for poppler binaries:" && \
    which pdftoppm && which pdfinfo || echo "Poppler binaries not found" && \
    echo "PyMuPDF version: " && pip show PyMuPDF | grep Version && \
    echo "pdf2image version: " && pip show pdf2image | grep Version

COPY . .

EXPOSE 5000

ENV PYTHONUNBUFFERED=1
ENV HEADLESS=1
ENV DOCKER_CONTAINER=1
ENV API_HOST=http://localhost:5000

CMD ["python", "run_api.py"] 