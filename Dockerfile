# Use slim base + manually install deps for OpenCV, torch, etc.
FROM python:3.9-slim

# Avoid .pyc files and force stdout logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system packages required for OpenCV, Pillow, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    build-essential wget curl git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy your app
COPY . .

# Expose port (optional)
EXPOSE 5000

# Default command
CMD ["python", "worker.py"]
