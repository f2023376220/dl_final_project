# 1. Use an official lightweight Python runtime as a parent base image
FROM python:3.10-slim

# 2. Set the working directory inside the container image filesystem
WORKDIR /app

# 3. Install system-level dependencies required for image processing (Pillow)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy the application dependencies layout list into the container
COPY requirements.txt .

# 5. Install the exact library packages inside the container layer
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the necessary codebase files into the container workspace
COPY src/ /app/src/
COPY deployment/ /app/deployment/

# 7. Expose the networking port our FastAPI app communicates through
EXPOSE 8000

# 8. Define the container execution command to launch the app on startup
CMD ["uvicorn", "deployment.app:app", "--host", "0.0.0.0", "--port", "8000"]
