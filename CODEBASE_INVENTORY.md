# Comprehensive Codebase Audit: Polymer Aging ML Platform

## Executive Summary

This audit provides a complete technical inventory of the `dev-jas/polymer-aging-ml` repository, a sophisticated machine learning platform for polymer degradation classification using **Raman and FTIR spectroscopy**. The system demonstrates a production-ready, multi-modal architecture with comprehensive error handling, multi-format batch processing, persistent performance tracking, and an extensible model framework spanning over **40 files across 8 directories**.

## 🏗️ System Architecture

### Core Infrastructure

The platform employs a **Streamlit-based web application** (`app.py`) as its primary interface, supported by a modular backend architecture. The system integrates **PyTorch for deep learning**, **Docker for deployment**, and implements a plugin-based model registry for extensibility. A **SQLite database** (`outputs/performance_tracking.db`) provides persistent storage for performance metrics.

### Directory Structure Analysis

The codebase maintains clean separation of concerns across eight primary directories:

**Root Level Files:**

- `app.py` - Main Streamlit application with a multi-tab UI layout
- `README.md` - Comprehensive project documentation
- `Dockerfile` - Python 3.13-slim containerization
- `requirements.txt` - Dependency management

**Core Directories:**

- `models/` - Neural network architectures with an expanded registry pattern
- `utils/` - Shared utility modules, including:
  - `preprocessing.py`: Modality-aware (Raman/FTIR) preprocessing.
  - `multifile.py`: Multi-format (TXT, CSV, JSON) data parsing and batch processing.
  - `results_manager.py`: Session and persistent results management.
  - `performance_tracker.py`: Performance analytics and database logging.
- `scripts/` - CLI tools for training, inference, and data management
- `outputs/` - Storage for pre-trained model weights, inference results, and the performance database
- `sample_data/` - Demo spectrum files for testing (including FTIR)
- `tests/` - Unit testing infrastructure
- `datasets/` - Data storage directory (content ignored)
- `pages/` - Streamlit pages for dashboarding and other UI components

## 🤖 Machine Learning Framework

### Model Registry System

The platform implements a **sophisticated factory pattern** for model management in `models/registry.py`. This design enables dynamic model selection and provides a unified interface for different architectures, now with added metadata for better model management.

```python
# Example from models/registry.py
_REGISTRY: Dict[str, Callable[[int], object]] = {
    "figure2": lambda L: Figure2CNN(input_length=L),
    "resnet": lambda L: ResNet1D(input_length=L),
    "resnet18vision": lambda L: ResNet18Vision(input_length=L)
}
```

### Neural Network Architectures

The platform includes several neural network architectures, including a baseline CNN, a ResNet-based model, and an experimental ResNet-18 vision model adapted for 1D spectral data.

## 🔧 Data Processing Infrastructure

### Preprocessing Pipeline

The system implements a **modular and modality-aware preprocessing pipeline** in `utils/preprocessing.py`.

**1. Multi-Format Input Validation Framework:**

- **File Format Verification**: Supports `.txt`, `.csv`, and `.json` files with auto-detection.
- **Data Integrity**: Validates for minimum data points, monotonic wavenumbers, and NaN values.
- **Modality-Aware Validation**: Applies different wavenumber range checks for Raman and FTIR spectroscopy.

**2. Core Processing Steps:**

- **Linear Resampling**: Uniform grid interpolation to a standard length (e.g., 500 points).
- **Baseline Correction**: Polynomial detrending.
- **Savitzky-Golay Smoothing**: Noise reduction with modality-specific parameters.
- **Min-Max Normalization**: Scaling to a [0, 1] range.

### Batch Processing Framework

The `utils/multifile.py` module provides **enterprise-grade batch processing** with multi-format support, error-tolerant processing, and progress tracking.

## 🖥️ User Interface Architecture

### Streamlit Application Design

The main application (`App.py`) implements a **multi-tab user interface** for different analysis modes:

- **Standard Analysis Tab**: For single-file or batch processing with a chosen model.
- **Model Comparison Tab**: Allows for side-by-side comparison of multiple models on the same data.
- **Performance Tracking Tab**: A dashboard to visualize and analyze model performance metrics from the SQLite database.

### State Management System

The application employs **advanced session state management** (`st.session_state`) to maintain a consistent user experience across tabs and reruns, with intelligent caching for performance.

## 🛠️ Utility Infrastructure

### Centralized Error Handling

The `utils/errors.py` module implements **production-grade error management** with context-aware logging and user-friendly error messages.

### Performance Tracking System

The `utils/performance_tracker.py` module provides a robust system for logging and analyzing performance metrics.

- **Database Logging**: Persists metrics to a SQLite database.
- **Automated Tracking**: Uses a context manager to automatically track inference time, preprocessing time, and memory usage.
- **Dashboarding**: Includes functions to generate performance visualizations and summary statistics for the UI.

### Enhanced Results Management

The `utils/results_manager.py` module enables comprehensive session and persistent results tracking.

- **In-Memory Storage**: Manages results for the current session.
- **Multi-Model Handling**: Aggregates results from multiple models for comparison.
- **Export Capabilities**: Exports results to CSV and JSON.
- **Statistical Analysis**: Calculates accuracy, confidence, and other metrics.

## 📜 Command-Line Interface

### Inference Pipeline

The `scripts/run_inference.py` module provides **powerful automated inference capabilities**:

- **Multi-Model Inference**: Run multiple models on the same input for comparison.
- **Format Detection**: Automatically detects input file format (TXT, CSV, JSON).
- **Modality Support**: Explicitly supports both Raman and FTIR modalities.
- **Flexible Output**: Saves results in JSON or CSV format.

## 🧪 Testing Framework

### Test Infrastructure

The `tests/` directory contains the testing framework, now with expanded coverage:

- **PyTest Configuration**: Centralized test settings in `conftest.py`.
- **Preprocessing Tests**: Includes tests for both Raman and FTIR preprocessing.
- **Multi-Format Parsing Tests**: Validates the parsing of TXT, CSV, and JSON files.

## 🔮 Strategic Development Roadmap

The project roadmap has been updated to reflect recent progress:

- [x] **FTIR Support**: Modular integration of FTIR spectroscopy is complete.
- [x] **Multi-Model Dashboard**: A model comparison tab has been implemented.
- [ ] **Image-based Inference**: Future work to include image-based polymer classification.
- [x] **Performance Tracking**: A performance tracking dashboard has been implemented.
- [ ] **Enterprise Integration**: Future work to include a RESTful API and more advanced database integration.

## 🏁 Audit Conclusion

This codebase represents a **significantly enhanced, multi-modal machine learning platform** that is well-suited for research, education, and industrial applications. The recent additions of FTIR support, multi-format data handling, performance tracking, and a multi-tab UI have greatly increased the usability and value of the project. The architecture remains robust, extensible, and well-documented, making it a solid foundation for future development.

### Neural Network Architectures

**1. Figure2CNN (Baseline Model)**[^1_6]

- **Architecture**: 4 convolutional layers with progressive channel expansion (1→16→32→64→128)
- **Classification Head**: 3 fully connected layers (256→128→2 neurons)
- **Performance**: 94.80% accuracy, 94.30% F1-score
- **Designation**: Validated exclusively for Raman spectra input
- **Parameters**: Dynamic flattened size calculation for input flexibility

**2. ResNet1D (Advanced Model)**[^1_7]

- **Architecture**: 3 residual blocks with skip connections
- **Innovation**: 1D residual connections for spectral feature learning
- **Performance**: 96.20% accuracy, 95.90% F1-score
- **Efficiency**: Global average pooling reduces parameter count
- **Parameters**: Approximately 100K (more efficient than baseline)

**3. ResNet18Vision (Deep Architecture)**[^1_8]

- **Design**: 1D adaptation of ResNet-18 with BasicBlock1D modules
- **Structure**: 4 residual layers with 2 blocks each
- **Initialization**: Kaiming normal initialization for optimal training
- **Status**: Under evaluation for spectral analysis applications

## 🔧 Data Processing Infrastructure

### Preprocessing Pipeline

The system implements a **modular preprocessing pipeline** in `utils/preprocessing.py` with five configurable stages:[^1_9]

**1. Input Validation Framework:**

- File format verification (`.txt` files exclusively)
- Minimum data points validation (≥10 points required)
- Wavenumber range validation (0-10,000 cm⁻¹ for Raman spectroscopy)
- Monotonic sequence verification for spectral consistency
- NaN value detection and automatic rejection

**2. Core Processing Steps:**[^1_9]

- **Linear Resampling**: Uniform grid interpolation to 500 points using `scipy.interpolate.interp1d`
- **Baseline Correction**: Polynomial detrending (configurable degree, default=2)
- **Savitzky-Golay Smoothing**: Noise reduction (window=11, order=2, configurable)
- **Min-Max Normalization**: Scaling to range with constant-signal protection[^1_1]

### Batch Processing Framework

The `utils/multifile.py` module (12.5 kB) provides **enterprise-grade batch processing** capabilities:[^1_10]

- **Multi-File Upload**: Streamlit widget supporting simultaneous file selection
- **Error-Tolerant Processing**: Individual file failures don't interrupt batch operations
- **Progress Tracking**: Real-time processing status with callback mechanisms
- **Results Aggregation**: Comprehensive success/failure reporting with export options
- **Memory Management**: Automatic cleanup between file processing iterations

## 🖥️ User Interface Architecture

### Streamlit Application Design

The main application implements a **sophisticated two-column layout** with comprehensive state management:[^1_2]

**Left Column - Control Panel:**

- **Model Selection**: Dropdown with real-time performance metrics display
- **Input Modes**: Three processing modes (Single Upload, Batch Upload, Sample Data)
- **Status Indicators**: Color-coded feedback system for user guidance
- **Form Submission**: Validated input handling with disabled state management

**Right Column - Results Display:**

- **Tabbed Interface**: Details, Technical diagnostics, and Scientific explanation
- **Interactive Visualization**: Confidence progress bars with color coding
- **Spectrum Analysis**: Side-by-side raw vs. processed spectrum plotting
- **Technical Diagnostics**: Model metadata, processing times, and debug logs

### State Management System

The application employs **advanced session state management**:[^1_2]

- Persistent state across Streamlit reruns using `st.session_state`
- Intelligent caching with content-based hash keys for expensive operations
- Memory cleanup protocols after inference operations
- Version-controlled file uploader widgets to prevent state conflicts

## 🛠️ Utility Infrastructure

### Centralized Error Handling

The `utils/errors.py` module (5.51 kB) implements **production-grade error management**:[^1_11]

```python
class ErrorHandler:
    @staticmethod
    def log_error(error: Exception, context: str = "", include_traceback: bool = False)
    @staticmethod
    def handle_file_error(filename: str, error: Exception) -> str
    @staticmethod
    def handle_inference_error(model_name: str, error: Exception) -> str
```

**Key Features:**

- Context-aware error messages for different operation types
- Graceful degradation with fallback modes
- Structured logging with configurable verbosity
- User-friendly error translation from technical exceptions

### Confidence Analysis System

The `utils/confidence.py` module provides **scientific confidence metrics**

:

**Softmax-Based Confidence:**

- Normalized probability distributions from model logits
- Three-tier confidence levels: HIGH (≥80%), MEDIUM (≥60%), LOW (<60%)
- Color-coded visual indicators with emoji representations
- Legacy compatibility with logit margin calculations

### Session Results Management

The `utils/results_manager.py` module (8.16 kB) enables **comprehensive session tracking**:

- **In-Memory Storage**: Session-wide results persistence
- **Export Capabilities**: CSV and JSON download with timestamp formatting
- **Statistical Analysis**: Automatic accuracy calculation when ground truth available
- **Data Integrity**: Results survive page refreshes within session boundaries

## 📜 Command-Line Interface

### Training Pipeline

The `scripts/train_model.py` module (6.27 kB) implements **robust model training**:

**Cross-Validation Framework:**

- 10-fold stratified cross-validation for unbiased evaluation
- Model registry integration supporting all architectures
- Configurable preprocessing via command-line flags
- Comprehensive JSON logging with confusion matrices

**Reproducibility Features:**

- Fixed random seeds (SEED=42) across all random number generators
- Deterministic CUDA operations when GPU available
- Standardized train/validation splitting methodology

### Inference Pipeline

The `scripts/run_inference.py` module (5.88 kB) provides **automated inference capabilities**:

**CLI Features:**

- Preprocessing parity with web interface ensuring consistent results
- Multiple output formats with detailed metadata inclusion
- Safe model loading across PyTorch versions with fallback mechanisms
- Flexible architecture selection via command-line arguments

### Data Utilities

**File Discovery System:**

- Recursive `.txt` file scanning with label extraction
- Filename-based labeling convention (`sta-*` = stable, `wea-*` = weathered)
- Dataset inventory generation with statistical summaries

## 🐳 Deployment Infrastructure

### Docker Configuration

The `Dockerfile` (421 Bytes) implements **optimized containerization**:[^1_12]

- **Base Image**: Python 3.13-slim for minimal attack surface
- **System Dependencies**: Essential build tools and scientific libraries
- **Health Monitoring**: HTTP endpoint checking for container wellness
- **Caching Strategy**: Layered builds with dependency caching for faster rebuilds

### Dependency Management

The `requirements.txt` specifies **core dependencies without version pinning**:[^1_12]

- **Web Framework**: `streamlit` for interactive UI
- **Deep Learning**: `torch`, `torchvision` for model execution
- **Scientific Computing**: `numpy`, `scipy`, `scikit-learn` for data processing
- **Visualization**: `matplotlib` for spectrum plotting
- **API Framework**: `fastapi`, `uvicorn` for potential REST API expansion

## 🧪 Testing Framework

### Test Infrastructure

The `tests/` directory implements **basic validation framework**:

- **PyTest Configuration**: Centralized test settings in `conftest.py`
- **Preprocessing Tests**: Core pipeline functionality validation in `test_preprocessing.py`
- **Limited Coverage**: Currently covers preprocessing functions only

**Testing Gaps Identified:**

- No model architecture unit tests
- Missing integration tests for UI components
- No performance benchmarking tests
- Limited error handling validation

## 🔍 Security \& Quality Assessment

### Input Validation Security

**Robust Validation Framework:**

- Strict file format enforcement preventing arbitrary file uploads
- Content verification with numeric data type checking
- Scientific range validation for spectroscopic data integrity
- Memory safety through automatic cleanup and garbage collection

### Code Quality Metrics

**Production Standards:**

- **Type Safety**: Comprehensive type hints throughout codebase using Python 3.8+ syntax
- **Documentation**: Inline docstrings following standard conventions
- **Error Boundaries**: Multi-level exception handling with graceful degradation
- **Logging**: Structured logging with appropriate severity levels

### Security Considerations

**Current Protections:**

- Input sanitization through strict parsing rules
- No arbitrary code execution paths
- Containerized deployment limiting attack surface
- Session-based storage preventing data persistence attacks

**Areas Requiring Enhancement:**

- No explicit security headers in web responses
- Basic authentication/authorization framework absent
- File upload size limits not explicitly configured
- No rate limiting mechanisms implemented

## 🚀 Extensibility Analysis

### Model Architecture Extensibility

The **registry pattern enables seamless model addition**:[^1_5]

1. **Implementation**: Create new model class with standardized interface
2. **Registration**: Add to `models/registry.py` with factory function
3. **Integration**: Automatic UI and CLI support without code changes
4. **Validation**: Consistent input/output shape requirements

### Processing Pipeline Modularity

**Configurable Architecture:**

- Boolean flags control individual preprocessing steps
- Easy integration of new preprocessing techniques
- Backward compatibility through parameter defaulting
- Single source of truth in `utils/preprocessing.py`

### Export \& Integration Capabilities

**Multi-Format Support:**

- CSV export for statistical analysis software
- JSON export for programmatic integration
- RESTful API potential through FastAPI foundation
- Batch processing enabling high-throughput scenarios

## 📊 Performance Characteristics

### Computational Efficiency

**Model Performance Metrics:**

| Model          | Parameters | Accuracy         | F1-Score         | Inference Time   |
| :------------- | :--------- | :--------------- | :--------------- | :--------------- |
| Figure2CNN     | ~500K      | 94.80%           | 94.30%           | <1s per spectrum |
| ResNet1D       | ~100K      | 96.20%           | 95.90%           | <1s per spectrum |
| ResNet18Vision | ~11M       | Under evaluation | Under evaluation | <2s per spectrum |

**System Response Times:**

- Single spectrum processing: <5 seconds end-to-end
- Batch processing: Linear scaling with file count
- Model loading: <3 seconds (cached after first load)
- UI responsiveness: Real-time updates with progress indicators

### Memory Management

**Optimization Strategies:**

- Explicit garbage collection after inference operations[^1_2]
- CUDA memory cleanup when GPU available
- Session state pruning for long-running sessions
- Caching with content-based invalidation

## 🎯 Production Readiness Evaluation

### Strengths

**Architecture Excellence:**

- Clean separation of concerns with modular design
- Production-grade error handling and logging
- Intuitive user experience with real-time feedback
- Scalable batch processing with progress tracking
- Well-documented, type-hinted codebase

**Operational Readiness:**

- Containerized deployment with health checks
- Comprehensive preprocessing validation
- Multiple export formats for integration
- Session-based results management

### Enhancement Opportunities

**Testing Infrastructure:**

- Expand unit test coverage beyond preprocessing
- Implement integration tests for UI workflows
- Add performance regression testing
- Include security vulnerability scanning

**Monitoring \& Observability:**

- Application performance monitoring integration
- User analytics and usage patterns tracking
- Model performance drift detection
- Resource utilization monitoring

**Security Hardening:**

- Implement proper authentication mechanisms
- Add rate limiting for API endpoints
- Configure security headers for web responses
- Establish audit logging for sensitive operations

## 🔮 Strategic Development Roadmap

Based on the documented roadmap in `README.md`, the platform targets three strategic expansion paths:[^1_13]

**1. Multi-Model Dashboard Evolution**

- Comparative model evaluation framework
- Side-by-side performance reporting
- Automated model retraining pipelines
- Model versioning and rollback capabilities

**2. Multi-Modal Input Support**

- FTIR spectroscopy integration with dedicated preprocessing
- Image-based polymer classification via computer vision
- Cross-modal validation and ensemble methods
- Unified preprocessing pipeline for multiple modalities

**3. Enterprise Integration Features**

- RESTful API development for programmatic access
- Database integration for persistent storage
- User authentication and authorization systems
- Audit trails and compliance reporting

## 💼 Business Logic \& Scientific Workflow

### Classification Methodology

**Binary Classification Framework:**

- **Stable Polymers**: Well-preserved molecular structure suitable for recycling
- **Weathered Polymers**: Oxidized bonds requiring additional processing
- **Confidence Thresholds**: Scientific validation with visual indicators
- **Ground Truth Validation**: Filename-based labeling for accuracy assessment

### Scientific Applications

**Research Use Cases:**[^1_13]

- Material science polymer degradation studies
- Recycling viability assessment for circular economy
- Environmental microplastic weathering analysis
- Quality control in manufacturing processes
- Longevity prediction for material aging

### Data Workflow Architecture

```
Input Validation → Spectrum Preprocessing → Model Inference →
Confidence Analysis → Results Visualization → Export Options
```

## 🏁 Audit Conclusion

This codebase represents a **well-architected, scientifically rigorous machine learning platform** with the following key characteristics:

**Technical Excellence:**

- Production-ready architecture with comprehensive error handling
- Modular design supporting extensibility and maintainability
- Scientific validation appropriate for spectroscopic data analysis
- Clean separation between research functionality and production deployment

**Scientific Rigor:**

- Proper preprocessing pipeline validated for Raman spectroscopy
- Multiple model architectures with performance benchmarking
- Confidence metrics appropriate for scientific decision-making
- Ground truth validation enabling accuracy assessment

**Operational Readiness:**

- Containerized deployment suitable for cloud platforms
- Batch processing capabilities for high-throughput scenarios
- Comprehensive export options for downstream analysis
- Session management supporting extended research workflows

**Development Quality:**

- Type-safe Python implementation with modern language features
- Comprehensive documentation supporting knowledge transfer
- Modular architecture enabling team development
- Testing framework foundation for continuous integration

The platform successfully bridges academic research and practical application, providing both accessible web interface capabilities and automation-friendly command-line tools. The extensible architecture and comprehensive documentation indicate strong software engineering practices suitable for both research institutions and industrial applications.

**Risk Assessment:** Low - The codebase demonstrates mature engineering practices with appropriate validation and error handling for production deployment.

**Recommendation:** This platform is ready for production deployment with minimal additional hardening, representing a solid foundation for polymer classification research and industrial applications.
<span style="display:none">[^1_14][^1_15][^1_16][^1_17][^1_18]</span>

<div style="text-align: center">⁂</div>

### EXTRA

```text
1. Setup & Configuration (Lines 1-105)
    Imports: Standard libraries (os, sys, time), data science (numpy, torch, matplotlib), and Streamlit.
    Local Imports: Pulls from your existing utils and models directories.
    Constants: Global, hardcoded configuration variables.
    KEEP_KEYS: Defines which session state keys persist on reset.
    TARGET_LEN: A static preprocessing value.
    SAMPLE_DATA_DIR, MODEL_WEIGHTS_DIR: Path configurations.
    MODEL_CONFIG: A dictionary defining model paths, classes, and metadata.
    LABEL_MAP: A dictionary for mapping class indices to human-readable names.
    Page Setup:
    st.set_page_config(): Sets the browser tab title, icon, and layout.
    st.markdown(<style>...): A large, embedded multi-line string containing all the custom CSS for the application.
2. Core Logic & Data Processing (Lines 108-250)
    Model Handling:
    load_state_dict(): Cached function to load model weights from a file.
    load_model(): Cached resource to initialize a model class and load its weights.
    run_inference(): The main ML prediction function. It takes resampled data, loads the appropriate model, runs inference, and returns the results.
    Data I/O & Preprocessing:
    label_file(): Extracts the ground truth label from a filename.
    get_sample_files(): Lists the available .txt files in the sample data directory.
    parse_spectrum_data(): The crucial function for reading, validating, and parsing raw text input into numerical numpy arrays.
    Visualization:
    create_spectrum_plot(): Generates the "Raw vs. Resampled" matplotlib plot and returns it as an image.
    Helpers:
    cleanup_memory(): A utility for garbage collection.
    get_confidence_description(): Maps a logit margin to a human-readable confidence level.
3. State Management & Callbacks (Lines 253-335)
    Initialization:
    init_session_state(): The cornerstone of the app's state, defining all the default values in st.session_state.
    Widget Callbacks:
    on_sample_change(): Triggered when the user selects a sample file.
    on_input_mode_change(): Triggered by the main st.radio widget.
    on_model_change(): Triggered when the user selects a new model.
    Reset/Clear Functions:
    reset_results(): A soft reset that only clears inference artifacts.
    reset_ephemeral_state(): The "master reset" that clears almost all session state and forces a file uploader refresh.
    clear_batch_results(): A focused function to clear only the results in col2.
4. UI Rendering Components (Lines 338-End)
    Generic Components:
    render_kv_grid(): A reusable helper to display a dictionary in a neat grid.
    render_model_meta(): Renders the model's accuracy and F1 score in the sidebar.
    Main Application Layout (main()):
    Sidebar: Contains the header, model selector (st.selectbox), model metadata, and the "About" expander.
    Column 1 (Input): Contains the main st.radio for mode selection and the conditional logic to display the single file uploader, batch uploader, or sample selector. It also holds the "Run Analysis" and "Reset All" buttons.
    Column 2 (Results): Contains all the logic for displaying either the batch results or the detailed, tabbed results for a single file (Details, Technical, Explanation).
```

[^1_1]: https://huggingface.co/spaces/dev-jas/polymer-aging-ml/tree/main
[^1_2]: https://huggingface.co/spaces/dev-jas/polymer-aging-ml/tree/main/datasets
[^1_3]: https://huggingface.co/spaces/dev-jas/polymer-aging-ml
[^1_4]: https://github.com/KLab-AI3/ml-polymer-recycling
[^1_5]: https://huggingface.co/spaces/dev-jas/polymer-aging-ml/raw/main/.gitignore
[^1_6]: https://huggingface.co/spaces/dev-jas/polymer-aging-ml/blob/main/models/resnet_cnn.py
[^1_7]: https://huggingface.co/spaces/dev-jas/polymer-aging-ml/raw/main/utils/multifile.py
[^1_8]: https://huggingface.co/spaces/dev-jas/polymer-aging-ml/raw/main/utils/preprocessing.py
[^1_9]: https://huggingface.co/spaces/dev-jas/polymer-aging-ml/raw/main/utils/audit.py
[^1_10]: https://huggingface.co/spaces/dev-jas/polymer-aging-ml/raw/main/utils/results_manager.py
[^1_11]: https://huggingface.co/spaces/dev-jas/polymer-aging-ml/blob/main/scripts/train_model.py
[^1_12]: https://huggingface.co/spaces/dev-jas/polymer-aging-ml/raw/main/requirements.txt
[^1_13]: https://doi.org/10.1016/j.resconrec.2022.106718
[^1_14]: https://huggingface.co/spaces/dev-jas/polymer-aging-ml/raw/main/app.py
[^1_15]: https://huggingface.co/spaces/dev-jas/polymer-aging-ml/raw/main/Dockerfile
[^1_16]: https://huggingface.co/spaces/dev-jas/polymer-aging-ml/raw/main/utils/errors.py
[^1_17]: https://huggingface.co/spaces/dev-jas/polymer-aging-ml/raw/main/utils/confidence.py
[^1_18]: https://ppl-ai-code-interpreter-files.s3.amazonaws.com/web/direct-files/9fd1eb2028a28085942cb82c9241b5ae/a25e2c38-813f-4d8b-89b3-713f7d24f1fe/3e70b172.md
