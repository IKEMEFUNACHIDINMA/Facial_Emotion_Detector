# Use Python 3.10.6 full image
FROM python:3.10.6

# Environment variables
ENV PYTHONUNBUFFERED=1 \
  PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Copy project files
COPY . /app

# Install system dependencies and pin pip
RUN apt-get update && apt-get install -y \
  build-essential \
  libglib2.0-0 \
  libsm6 \
  libxrender1 \
  libxext6 \
  && pip install "pip<24.1" setuptools wheel \
  && pip install --no-cache-dir -r requirements.txt \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

EXPOSE 10000

CMD ["python", "app.py"]
