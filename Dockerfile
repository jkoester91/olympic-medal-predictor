# Use an official lightweight Python runtime as our base layer
FROM python:3.13-slim

# Prevent Python from buffering logs or writing .pyc files inside the container
ENV PYTHONTONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Establish our working directory inside the virtual file system
WORKDIR /app

# Install standard compiler build-essential utilities cleanly
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy over requirements first to maximize Docker's layer caching efficiency
COPY requirements.txt /app/

# Upgrade pip and install our required analytical packages
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy our project source files and configuration setups into the image
COPY configs/ /app/configs/
COPY src/ /app/src/

# By default, trigger our data preprocessing script when the container executes
CMD ["python", "-u", "src/preprocess.py"]