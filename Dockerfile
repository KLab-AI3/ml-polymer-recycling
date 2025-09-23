# Dockerfile
# filepath: ml-polymer-recycling-hf/Dockerfile

# Use official Python image for backend
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy backend files
COPY backend/ ./backend/

# Install backend dependencies
RUN pip install --no-cache-dir -r backend/requirements.txt

# Install Node.js and npm for frontend build
RUN apt-get update && \
    apt-get install -y curl && \
    curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

# Copy frontend files
COPY frontend/ ./frontend/

# Build React frontend
WORKDIR /app/frontend
RUN npm install && npm run build

# Use serve to host the frontend build
RUN npm install -g serve

# Expose Hugging Face Spaces default port
EXPOSE 7860

# Start both backend (FastAPI) and frontend (React build) using a process manager
WORKDIR /app
CMD uvicorn backend.main:app --host 0.0.0.0 --port 7860 & serve -s frontend/build -l 3000
