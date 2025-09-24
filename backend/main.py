# pylint: disable=unused-argument, unused-import, redefined-outer-name,
# missing-class-docstring, wrong-import-order, missing-function-docstring
# missing-module-docstring, broad-except, too-many-locals, too-many-stat
"""
FastAPI main application.
Serves both the API endpoints and React frontend static files.
Single-container deployment for Hugging Face Spaces compatibility.
"""

import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import uuid
import statistics
import math
from pydantic import BaseModel, Field


from fastapi import FastAPI, HTTPException, File, UploadFile, BackgroundTasks, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from contextlib import asynccontextmanager

from backend.service import ml_service, MLServiceError
from backend.utils.model_manager import model_manager
from backend.utils.enhanced_ml_service import enhanced_ml_service
from .pydantic_models import (
    SpectrumData,
    AnalysisRequest,
    BatchAnalysisRequest,
    ComparisonRequest,
    PredictionResult,
    ExplanationResult,
    BatchPredictionResult,
    ComparisonResult,
    ModelInfo,
    SystemInfo,
    SystemHealth,
    ErrorResponse,
    BatchError,
)

from backend.utils.prepare_data import prepare_data as run_prepare_data
from backend.utils.train import train as run_training_job
from backend.utils.multifile import parse_spectrum_data


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    print("🚀 Starting Polymer Aging ML API...")

    # Warmup models (load them into cache)
    # Use the centralized model_manager for loading
    try:
        print("Pre-loading models via ModelManager...")
        available_models_info = model_manager.get_available_models()
        print(f"✅ Discovered {len(available_models_info)} models.")

        # Warmup with a dummy spectrum if models are available
        loaded_models_count = 0
        for model_info in available_models_info:
            if model_info.available:
                ml_service.model_manager.load_model(model_info.name) # Ensure models are loaded into ml_service's manager
                loaded_models_count += 1
        print(f"✅ {loaded_models_count} models loaded into ModelManager.")

        if loaded_models_count > 0:
            dummy_spectrum = SpectrumData(
                x_values=list(range(200, 4000, 10)),
                y_values=[0.5] * len(list(range(200, 4000, 10))),
                filename="warmup"
            )
            print("✅ Models warmed up successfully")
    except (KeyError, ValueError, RuntimeError) as e:  # Replace with specific exceptions
        print(f"⚠️ Model warmup failed: {e}")

    yield

    # Shutdown
    print("🔄 Shutting down Polymer Aging ML API...")


# --- In-memory DB for Training Jobs ---
training_jobs: Dict[str, Dict[str, Any]] = {}

# --- Pydantic Models Building Blocks for Training API ---


class PrepareDataRequest(BaseModel):
    raw_data_path: str = Field(
        ..., description="Path to the raw dataset (e.g., a single CSV or a directory).")
    output_path: str = Field(
        default="data/processed", description="Directory to save the processed train/val/test splits.")


class TrainingJobConfig(BaseModel):
    experiment_name: str = "PolymerAgingClassification"
    run_name: str = Field(
        default_factory=lambda: f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    data_dir: str = "data/processed"
    train_csv: str = "train.csv"
    val_csv: str = "validation.csv"
    model_name: str
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001
    optimizer: str = "Adam"
    loss_function: str = "CrossEntropyLoss"


class TrainingJobStatus(BaseModel):
    job_id: str
    status: str  # PENDING, RUNNING, COMPLETED, FAILED
    config: TrainingJobConfig
    progress: float = 0.0
    current_epoch: int = 0
    mlflow_run_id: Optional[str] = None
    metrics: Dict[str, list] = Field(default_factory=lambda: {
        "train_loss": [], "val_loss": []})
    error: Optional[str] = None
    created_at: str


app = FastAPI(
    title="Polymer Aging ML API",
    description="AI-driven polymer aging prediction and classification using Raman and FTIR spectroscopy",
    version="1.0.0",
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc",
    openapi_url="/api/v1/openapi.json",
    lifespan=lifespan
)

# CORS middleware for React frontend
# CORS: Allows requests from any origin.
# For production, this should be replaced with a specific list of allowed origins.
# !! CRITICAL: Review CORS settings before deploying to production.====================||
_allowed = os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:3000")
origins = [o.strip() for o in _allowed.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or ["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Error handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    # Sanitize validation errors to ensure they are JSON serializable
    cleaned_errors = []
    for error in exc.errors():
        cleaned_error = error.copy()
        if 'ctx' in cleaned_error and isinstance(cleaned_error['ctx'], dict):
            # The context can contain non-serializable objects like exceptions.
            # We'll convert all context values to strings for a safe response.
            cleaned_error['ctx'] = {k: str(v)
                                    for k, v in cleaned_error['ctx'].items()}
        cleaned_errors.append(cleaned_error)
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            error="Validation Error",
            error_code="VALIDATION_ERROR",
            details={"validation_errors": cleaned_errors},
            timestamp=datetime.now().isoformat(),
            request_id=str(uuid.uuid4())
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error=str(exc) if str(exc) else "Internal server error",
            error_code="INTERNAL_ERROR",
            details={},  # Provide an empty dictionary as the default value
            timestamp=datetime.now().isoformat(),
            request_id=str(uuid.uuid4())
        ).dict()
    )


# --- API Router for Training ---
training_router = APIRouter(prefix="/api/v1/training", tags=["Training"])


@training_router.post("/prepare-data", summary="Prepare and Split Dataset")
def prepare_data_endpoint(request: PrepareDataRequest):
    """
    Triggers the data preparation script to create train/validation/test splits.
    In a real web app, this would handle an uploaded zip file.
    """
    try:
        raw_path = Path(request.raw_data_path)
        output_path = Path(request.output_path)
        run_prepare_data(data_path=raw_path, output_path=output_path)
        return {"message": f"Data preparation complete. Splits saved to {output_path}."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@training_router.post("/start", response_model=TrainingJobStatus, status_code=202, summary="Start a New Training Job")
def start_training(config: TrainingJobConfig, background_tasks: BackgroundTasks):
    """
    Starts a new model training job in the background.
    """
    job_id = str(uuid.uuid4())
    job_status = TrainingJobStatus(
        job_id=job_id,
        status="PENDING",
        config=config,
        created_at=datetime.now().isoformat(),
    ).dict()
    training_jobs[job_id] = job_status

    # Add the long-running training task to the background
    background_tasks.add_task(
        run_training_job, config=config.dict(), jobs_db=training_jobs, job_id=job_id)

    return job_status


@training_router.get("/jobs", response_model=List[TrainingJobStatus], summary="List All Training Jobs")
def list_training_jobs():
    """Retrieves the status of all training jobs."""
    return list(training_jobs.values())


@training_router.get("/jobs/{job_id}", response_model=TrainingJobStatus, summary="Get Training Job Status")
def get_training_job_status(job_id: str):
    """Retrieves the status of a specific training job by its ID."""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Training job not found")
    return training_jobs[job_id]


# API Routes
@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/api/v1/system", response_model=SystemInfo)
async def get_system_info():
    """Get system information and available models"""
    return ml_service.get_system_info()


@app.get("/api/v1/models", response_model=List[ModelInfo])
async def get_models():
    """Get list of available models"""
    print("🔍 Fetching available models...")
    # Directly use the centralized model manager
    models = model_manager.get_available_models()
    if not models:
        print("⚠️ No models found via ModelManager. Falling back to filesystem scan (this should ideally not be needed).")
        # This fallback is now less critical as ModelManager should handle discovery
        # but keeping it for extreme resilience as per original request.
        # The ModelManager itself already checks for weight file existence.
    return models


@app.post("/api/v1/analyze", response_model=PredictionResult)
async def analyze_spectrum(request: AnalysisRequest):
    """Analyze a single spectrum"""
    try:
        result = ml_service.run_inference(
            request.spectrum,
            request.model_name,
            request.modality,
            request.include_provenance
        )
        return result
    except MLServiceError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


# ** fix-429e36db-a89a-42f9-8b64-9bdfd16b01bc
@app.post("/api/v1/explain")
async def explain_spectrum(request: AnalysisRequest):
    """Analyze a spectrum with explainability features"""
    try:
        # Ensure we pass modality and use the same include_provenance flag
        result = enhanced_ml_service.predict_with_explanation(
            request.spectrum,                  # SpectrumData
            request.model_name,                # model name
            modality=request.modality,         # pass modality (raman/ftir)
            include_feature_importance=request.include_provenance
        )
        return result
    except Exception as e:
        # Log full traceback for debugging
        import traceback, sys
        print("[explain] Error during prediction with explanation:", str(e), file=sys.stderr)
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/api/v1/explain/batch")
async def explain_batch_spectra(request: BatchAnalysisRequest):
    """Analyze multiple spectra with explainability features"""
    if len(request.spectra) > 50:  # Lower limit for explanation requests
        raise HTTPException(
            status_code=400,
            detail="Batch explainability requests limited to 50 spectra"
        )

    try:
        results = enhanced_ml_service.batch_predict_with_explanation(
            request.spectra,
            request.model_name,
            modality=request.modality, # Pass modality to the enhanced service
            include_feature_importance=request.include_provenance # Use include_provenance for feature importance
        )

        return {
            "results": results,
            "total_processed": len(results),
            "model_used": request.model_name,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
# ** fix-429e36db-a89a-42f9-8b64-9bdfd16b01bc


@app.post("/api/v1/analyze/batch", response_model=BatchPredictionResult)
async def analyze_batch(request: BatchAnalysisRequest):
    """Analyze multiple spectra in batch"""
    if len(request.spectra) > 100:
        raise HTTPException(
            status_code=400, detail="Batch size cannot exceed 100 spectra")

    start_time = datetime.now()
    results = []
    errors = []

    for spectrum in request.spectra:
        try:
            result = ml_service.run_inference(
                spectrum,
                request.model_name,
                request.modality,
                request.include_provenance
            )
            results.append(result)
        except (ValueError, KeyError, RuntimeError) as e:
            errors.append(BatchError(filename=spectrum.filename, error=str(e)))

    total_time = (datetime.now() - start_time).total_seconds()

    # Initialize summary statistics with default values
    average_confidence = 0.0
    confidence_std = 0.0
    min_confidence = 0.0
    max_confidence = 0.0
    predictions = []

    # Calculate summary statistics only on successful results
    if results:
        # Calculate summary statistics
        confidences = [r.confidence for r in results]
        predictions = [r.prediction for r in results]

        if confidences:
            average_confidence = statistics.mean(confidences)
            confidence_std = statistics.stdev(
                confidences) if len(confidences) > 1 else 0.0
            min_confidence = min(confidences)
            max_confidence = max(confidences)

    summary = {
        "total_spectra_requested": len(request.spectra),
        "total_spectra_processed": len(results),
        "total_spectra_failed": len(errors),
        "stable_count": sum(1 for p in predictions if p == 0) if results else 0,
        "weathered_count": sum(1 for p in predictions if p == 1) if results else 0,
        "average_confidence": average_confidence if results else 0.0,
        "confidence_std": confidence_std if results else 0.0,
        "min_confidence": min_confidence if results else 0.0,
        "max_confidence": max_confidence if results else 0.0
    }

    return BatchPredictionResult(
        results=results,
        errors=errors,
        summary=summary,
        total_processing_time=total_time,
        timestamp=datetime.now().isoformat()
    )


@app.post("/api/v1/compare", response_model=ComparisonResult)
async def compare_models(request: ComparisonRequest):
    """Compare multiple models on a single spectrum"""
    try:
        available_models = ml_service.get_available_models()

        if request.model_names:
            models_to_test = [
                m.name for m in available_models if m.name in request.model_names and m.available]
        else:
            models_to_test = [m.name for m in available_models if m.available]

        if not models_to_test:
            raise HTTPException(
                status_code=400, detail="No available models found")

        spectrum_id = str(uuid.uuid4())
        model_results = {}
        confidences = []
        predictions = []

        for model_name in models_to_test:
            result = ml_service.run_inference(
                request.spectrum,
                model_name,
                request.modality,
                request.include_provenance
            )
            model_results[model_name] = result
            confidences.append(result.confidence)
            predictions.append(result.prediction)

        # Calculate consensus and agreement
        if predictions:
            # Simple majority vote for consensus
            prediction_counts = {0: predictions.count(
                0), 1: predictions.count(1)}
            consensus = max(prediction_counts, key=prediction_counts.get)

            # Agreement score: percentage of models that agree with consensus
            agreement_score = prediction_counts[consensus] / len(predictions)

            # Confidence variance
            if len(confidences) > 1:
                confidence_variance = statistics.variance(confidences)
            else:
                confidence_variance = 0.0
        else:
            consensus = None
            agreement_score = 0.0
            confidence_variance = 0.0

        return ComparisonResult(
            spectrum_id=spectrum_id,
            model_results=model_results,
            consensus_prediction=consensus,
            confidence_variance=confidence_variance,
            agreement_score=agreement_score,
            timestamp=datetime.now().isoformat()
        )

    except (MLServiceError, KeyError, ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/api/v1/upload", response_model=SpectrumData)
async def upload_spectrum_file(file: UploadFile = File(...)):
    """Upload and parse a spectrum file"""
    try:
        # Read file content
        content = await file.read()

        # Parse spectrum data using existing utility
        x_data, y_data = parse_spectrum_data(
            content.decode('utf-8'), file.filename or "unknown_filename")

        return SpectrumData(
            x_values=x_data.tolist(),
            y_values=y_data.tolist(),
            filename=file.filename
        )

    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to parse spectrum file: {str(e)}") from e


# Backward compatibility routes (redirect to v1)
@app.get("/api/health")
async def health_check_legacy():
    """Legacy health check endpoint - redirects to v1"""
    return await health_check()


@app.get("/api/system")
async def get_system_info_legacy():
    """Legacy system info endpoint - redirects to v1"""
    return await get_system_info()


@app.get("/api/models")
async def get_models_legacy():
    """Legacy models endpoint - redirects to v1"""
    return await get_models()


# Static file serving for React frontend
frontend_dist_path = Path("frontend/dist")
if frontend_dist_path.exists() and frontend_dist_path.is_dir():
    # Mount static files for built React app
    app.mount(
        "/static", StaticFiles(directory="frontend/dist/static"), name="static")

    @app.get("/")
    async def serve_frontend():
        """Serve React frontend"""
        index_path = frontend_dist_path / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
        return JSONResponse(content={"error": "Frontend index.html not found"}, status_code=404)

    @app.get("/{path:path}")
    async def serve_frontend_routes(path: str):
        """Serve React frontend for all non-API routes (SPA routing)"""
        if path.startswith("api/"):
            raise HTTPException(
                status_code=404, detail="API endpoint not found")

        file_path = frontend_dist_path / path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        else:
            # For SPA routing, return index.html if it exists
            index_path = frontend_dist_path / "index.html"
            if index_path.exists():
                return FileResponse(index_path)
            raise HTTPException(status_code=404, detail="Frontend not found")
else:
    @app.get("/")
    async def root():
        """Root endpoint when frontend is not built"""
        return {
            "message": "Polymer Aging ML API",
            "status": "Frontend not built. Build React frontend and place in frontend/dist/",
            "api_docs": "/api/docs",
            "version": "1.0.0"
        }

# Include the new training router in the main application
app.include_router(training_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
