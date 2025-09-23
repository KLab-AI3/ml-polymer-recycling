#!/usr/bin/env python3
"""
Main application entry point for single-container deployment.
Serves both React frontend and FastAPI backend on the same port.
Compatible with Hugging Face Spaces hosting requirements.
"""

import uvicorn
from backend.main import app
import os
import sys
import subprocess
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path
from fastapi import FastAPI
from pathlib import Path

api = FastAPI()
frontend_dist_path = Path(__file__).parent / "frontend" / "dist"



# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))



def ensure_frontend_built():
    """Ensure React frontend is built and available"""
    if not frontend_dist_path.exists():
        print("🔨 Frontend not found, building React application...")
        frontend_path = project_root / "frontend"

        if not frontend_path.exists():
            raise RuntimeError(
                "Frontend directory not found. Please run setup first.")

        # Build the React frontend
        build_result = subprocess.run(
            ["npm", "run", "build"],
            cwd=frontend_path,
            capture_output=True,
            text=True,
            check=True
        )

        if build_result.returncode != 0:
            print(f"❌ Frontend build failed: {build_result.stderr}")
            raise RuntimeError("Frontend build failed")

        # Move build files to dist for FastAPI static serving
        build_path = frontend_path / "build"
        if build_path.exists():
            import shutil
            if frontend_dist_path.exists():
                shutil.rmtree(frontend_dist_path)
            shutil.move(str(build_path), str(frontend_dist_path))
            print("✅ Frontend built successfully")

        frontend_dist_path = Path("frontend/dist")


frontend_dist_path = Path(__file__).parent / "frontend" / "dist"
if frontend_dist_path.exists() and frontend_dist_path.is_dir():
    app.mount("/static", StaticFiles(directory=frontend_dist_path / "static"), name="static")

    @app.get("/")
    async def serve_frontend():
        index_path = frontend_dist_path / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
        return JSONResponse(content={"error": "Frontend index.html not found"}, status_code=404)

    @app.get("/{path:path}")
    async def serve_frontend_routes(path: str):
        index_path = frontend_dist_path / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
        return JSONResponse(content={"error": "Frontend index.html not found"}, status_code=404)


def main():
    """Main application entry point"""
    print("🚀 Starting Polymer Aging ML Application...")
    print("📊 React Frontend + FastAPI Backend")
    print("🔬 Single-container deployment ready")

    # Ensure frontend is built
    try:
        ensure_frontend_built()
    except Exception as e:
        print(f"⚠️ Frontend build warning: {e}")
        print("🔄 Continuing with API-only mode...")

    # Get configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))

    print(f"🌐 Starting server on http://{host}:{port}")
    print(f"📖 API Documentation: http://{host}:{port}/api/docs")
    print(f"🎯 Frontend: http://{host}:{port}")

    # Start the server
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )


if __name__ == "__main__":
    main()
