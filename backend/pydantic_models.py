# pylint: disable=unused-import
"""
Pydantic models for API request/response validation.
Maintains strict contract between React frontend and FastAPI backend.
"""

import time
from typing import List, Dict, Any, Optional, Union, Literal
from pydantic import BaseModel, Field, field_validator, model_validator
import numpy as np


class SpectrumData(BaseModel):
    """Single spectrum data for analysis"""

    x_values: List[float] = Field(..., description="Wavenumber values (cm⁻¹)")
    y_values: List[float] = Field(..., description="Intensity values")
    filename: Optional[str] = Field(None, description="Original filename")

    @field_validator("x_values", "y_values")
    @classmethod
    def validate_arrays(cls, v: List[float]) -> List[float]:
        """
        Validate that the input arrays have at least 2 values.

        Args:
            v (list): The array to validate.

        Returns:
            list: The validated array.

        Raises:
            ValueError: If the array has fewer than 2 values.
        """
        if len(v) < 2:
            raise ValueError("Arrays must have at least 2 values")
        return v

    @model_validator(mode="after")
    def validate_equal_length(self) -> "SpectrumData":
        """
        Ensure that y_values has the same length as x_values.

        Args:
            v (list): The y_values list to validate.
            values (dict): The dictionary containing other field values.

        Returns:
            list: The validated y_values list.

        Raises:
            ValueError: If y_values and x_values do not have the same length.
        """
        if len(self.x_values) != len(self.y_values):
            raise ValueError("x_values and y_values must have equal length")
        return self


class AnalysisRequest(BaseModel):
    """Request for single spectrum analysis"""

    spectrum: SpectrumData
    model_name: str = Field(..., description="Model name to use for analysis")
    modality: Literal["raman", "ftir"] = Field(
        "raman", description="Spectroscopy modality"
    )
    include_provenance: bool = Field(
        True, description="Include full provenance metadata"
    )


class BatchAnalysisRequest(BaseModel):
    """Request for batch spectrum analysis"""

    spectra: List[SpectrumData] = Field(..., min_length=1, max_length=100)
    model_name: str = Field(..., description="Model name to use for analysis")
    modality: Literal["raman", "ftir"] = Field(
        "raman", description="Spectroscopy modality"
    )
    include_provenance: bool = Field(
        True, description="Include full provenance metadata"
    )


class ComparisonRequest(BaseModel):
    """Request for multi-model comparison"""

    spectrum: SpectrumData
    model_names: Optional[List[str]] = Field(
        None, description="Models to compare (all if None)"
    )
    modality: Literal["raman", "ftir"] = Field(
        "raman", description="Spectroscopy modality"
    )
    include_provenance: bool = Field(
        True, description="Include full provenance metadata"
    )


class PreprocessingMetadata(BaseModel):
    """Preprocessing provenance metadata"""

    target_length: int = Field(..., description="Target resampling length")
    baseline_degree: int = Field(...,
                                 description="Polynomial baseline removal degree")
    smooth_window: int = Field(..., description="Smoothing window length")
    smooth_polyorder: int = Field(...,
                                  description="Smoothing polynomial order")
    normalization_method: str = Field(...,
                                      description="Normalization method applied")
    modality_validated: bool = Field(
        ..., description="Whether modality validation passed"
    )
    validation_issues: List[str] = Field(
        default_factory=list, description="Any validation issues found"
    )
    original_length: int = Field(..., description="Original spectrum length")
    wavenumber_range: List[float] = Field(
        ..., min_length=2, max_length=2, description="[min, max] wavenumber range"
    )


class QualityControlMetadata(BaseModel):
    """Quality control check results"""

    signal_to_noise_ratio: Optional[float] = Field(
        None, description="Estimated SNR")
    baseline_stability: Optional[float] = Field(
        None, description="Baseline stability metric"
    )
    spectral_resolution: Optional[float] = Field(
        None, description="Estimated spectral resolution"
    )
    cosmic_ray_detected: bool = Field(
        False, description="Cosmic ray spikes detected")
    saturation_detected: bool = Field(
        False, description="Signal saturation detected")
    issues: List[str] = Field(default_factory=list,
                              description="QC issues found")


class ModelMetadata(BaseModel):
    """Model metadata and calibration details"""

    model_name: str = Field(..., description="Model identifier")
    model_description: str = Field(..., description="Model description")
    model_version: Optional[str] = Field(None, description="Model version")
    training_date: Optional[str] = Field(
        None, description="Model training date")
    input_length: int = Field(..., description="Expected input length")
    num_classes: int = Field(..., description="Number of output classes")
    parameters_count: Optional[str] = Field(
        None, description="Number of parameters")
    performance_metrics: Dict[str, float] = Field(
        default_factory=dict, description="Training performance"
    )
    supported_modalities: List[str] = Field(
        default_factory=list, description="Supported spectroscopy modalities"
    )
    citation: Optional[str] = Field(
        None, description="Model citation/reference")
    weights_loaded: bool = Field(...,
                                 description="Whether trained weights were loaded")
    weights_path: Optional[str] = Field(
        None, description="Path to loaded weights")


class PredictionResult(BaseModel):
    """Single prediction result with full provenance"""

    prediction: int = Field(...,
                            description="Predicted class (0=Stable, 1=Weathered)")
    prediction_label: str = Field(...,
                                  description="Human-readable prediction label")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Prediction confidence score"
    )
    probabilities: List[float] = Field(..., description="Class probabilities")
    logits: List[float] = Field(..., description="Raw model logits")

    # Provenance metadata
    preprocessing: PreprocessingMetadata
    quality_control: QualityControlMetadata
    model_metadata: ModelMetadata

    # Performance tracking
    inference_time: float = Field(..., ge=0.0,
                                  description="Inference time in seconds")
    preprocessing_time: float = Field(
        ..., ge=0.0, description="Preprocessing time in seconds"
    )
    total_time: float = Field(
        ..., ge=0.0, description="Total processing time in seconds"
    )
    memory_usage_mb: float = Field(..., ge=0.0,
                                   description="Memory usage in MB")

    # Input data (for audit trail)
    original_spectrum: SpectrumData
    processed_spectrum: SpectrumData

    # Timestamps
    timestamp: str = Field(...,
                           description="Processing timestamp (ISO format)")


class BatchError(BaseModel):
    """Details of a single error within a batch request"""

    filename: Optional[str] = Field(
        None, description="Filename of the spectrum that failed"
    )
    error: str = Field(..., description="The error message")


class BatchPredictionResult(BaseModel):
    """Batch prediction results"""

    results: List[PredictionResult] = Field(
        default_factory=list, description="Individual prediction results"
    )
    errors: List[BatchError] = Field(
        default_factory=list,
        description="Errors for spectra that failed processing",
    )
    summary: Dict[str, Any] = Field(
        default_factory=dict, description="Batch summary statistics"
    )
    total_processing_time: float = Field(
        ..., ge=0.0, description="Total batch processing time"
    )
    timestamp: str = Field(..., description="Batch processing timestamp")


class ComparisonResult(BaseModel):
    """Multi-model comparison results"""

    spectrum_id: str = Field(...,
                             description="Unique identifier for the spectrum")
    model_results: Dict[str, PredictionResult] = Field(
        default_factory=dict, description="Results per model"
    )
    consensus_prediction: Optional[int] = Field(
        None, description="Consensus prediction if available"
    )
    confidence_variance: float = Field(
        ..., ge=0.0, description="Variance in confidence scores"
    )
    agreement_score: float = Field(
        ..., ge=0.0, le=1.0, description="Model agreement score"
    )
    timestamp: str = Field(..., description="Comparison timestamp")


class FeatureImportanceSummary(BaseModel):
    """Summary of feature importance scores"""
    max_importance: float
    mean_importance: float
    important_region_start: int
    important_region_end: int


class TopFeatures(BaseModel):
    """Top features identified by explainability analysis"""
    indices: List[int]
    values: List[float]


class FeatureImportance(BaseModel):
    """Feature importance results from explainability analysis"""
    method: str
    importance_scores: List[float]
    top_features: TopFeatures
    summary: FeatureImportanceSummary


class ExplanationResult(BaseModel):
    """Result from an explainability analysis"""
    prediction: int
    confidence: float
    probabilities: List[float]
    class_labels: List[str]
    model_used: str
    spectrum_filename: Optional[str] = None
    feature_importance: Optional[FeatureImportance] = None

    class Config:
        """Pydantic model configuration"""
        from_attributes = True


class ModelInfo(BaseModel):
    """Model information and capabilities"""

    name: str = Field(..., description="Model identifier")
    description: str = Field(..., description="Model description")
    input_length: int = Field(..., description="Expected input length")
    num_classes: int = Field(..., description="Number of output classes")
    supported_modalities: List[str] = Field(
        default_factory=list, description="Supported modalities"
    )
    performance: Dict[str, float] = Field(
        default_factory=dict, description="Performance metrics"
    )
    parameters: Optional[str] = Field(None, description="Parameter count")
    speed: Optional[str] = Field(None, description="Relative speed category")
    citation: Optional[str] = Field(None, description="Citation/reference")
    available: bool = Field(...,
                            description="Whether model is available for inference")


class SystemHealth(BaseModel):
    """System health metrics"""
    status: str = Field(..., description="Overall system status, e.g., 'ok'.")
    timestamp: float = Field(...,
                             description="The server timestamp of the health check.")
    models_loaded: int
    total_models: int
    memory_usage_mb: float
    torch_version: str
    cuda_available: bool


class SystemInfo(BaseModel):
    """System information and health"""

    version: str = Field(..., description="API version")
    available_models: List[ModelInfo] = Field(
        default_factory=list, description="Available models"
    )
    supported_modalities: List[str] = Field(
        default_factory=list, description="Supported spectroscopy modalities"
    )
    max_batch_size: int = Field(100, ge=1, description="Maximum batch size")
    target_spectrum_length: int = Field(
        500, ge=1, description="Target spectrum length")
    system_health: SystemHealth = Field(
        ..., description="System health metrics"
    )


class ErrorResponse(BaseModel):
    """Standardized error response"""

    error: str = Field(..., description="Error message")
    error_code: str = Field(...,
                            description="Error code for programmatic handling")
    details: Optional[Dict[str, Any]] = Field(
        None, description="Additional error details"
    )
    timestamp: str = Field(..., description="Error timestamp")
    request_id: Optional[str] = Field(
        None, description="Request ID for tracking")
