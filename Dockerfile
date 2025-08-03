FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
# This includes build-essential for compiling C/C++ extensions,
# and gfortran for SciPy, as well as ffmpeg for video/audio processing.
RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    libgl1-mesa-glx \
    ffmpeg

# Copy requirements.txt and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# Command to run the application
CMD ["python", "app.py"]
