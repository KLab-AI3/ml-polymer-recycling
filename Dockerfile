FROM python:3.12-slim

# Install system dependencies as root
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libopenblas-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Set up Hugging Face Spaces user requirements
RUN useradd -m -u 1000 user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

# Build React frontend in a separate stage
FROM node:18-slim as frontend-builder
WORKDIR /app/frontend

# Copy only package.json and package-lock.json first for caching
COPY frontend/package*.json ./
RUN npm ci

# Now copy the rest of the frontend source
COPY frontend/ ./
RUN npm run build

# Backend stage
FROM python:3.12-slim

# Install system dependencies as root
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libopenblas-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app
USER user

# Copy only requirements.txt first for caching
COPY --chown=user requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY --chown=user backend/ ./backend/

# Copy React build from builder stage
COPY --chown=user --from=frontend-builder /app/frontend/build ./frontend/dist

# Expose Hugging Face Spaces port
EXPOSE 7860

# Healthcheck endpoint
HEALTHCHECK CMD curl --fail http://localhost:7860/api/v1/health || exit 1

# Start FastAPI (serves both API and React static files)
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
