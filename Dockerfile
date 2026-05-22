# --- STAGE 1: Frontend Builder ---
FROM node:18-slim AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# --- STAGE 2: Backend Runtime ---
FROM python:3.12-slim

# Install ONLY runtime libraries (no compilers/git)
RUN apt-get update && apt-get install -y \
    curl \
    libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

# Set up user
RUN useradd -m -u 1000 user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

# We need compilers ONLY for the pip install phase
# Use a temporary root session to install, then clean up
COPY --chown=user requirements.txt ./
RUN apt-get update && apt-get install -y build-essential \
    && pip install --no-cache-dir --user -r requirements.txt \
    && apt-get purge -y build-essential && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

USER user

# Copy backend code
COPY --chown=user backend/ ./backend/

# Copy React build from builder stage
# NOTE: Using 'build' because that is the default React output folder
COPY --chown=user --from=frontend-builder /app/frontend/build ./frontend/dist

# Expose port
EXPOSE 7860

# Start FastAPI
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
