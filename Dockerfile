FROM python:3.10-slim

WORKDIR /app

# Install system dependencies, including ffmpeg
RUN apt-get update && apt-get install -y ffmpeg

# Copy requirements.txt and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# Command to run the application
CMD ["python", "app.py"]