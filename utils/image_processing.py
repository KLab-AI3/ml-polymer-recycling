"""
Image loading and transformation utilities for polymer classification.
Supports conversion of spectral images to processable data.
"""

from typing import Tuple, Optional, List, Dict
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import streamlit as st
import pandas as pd

# Use existing inference pipeline
from utils.preprocessing import preprocess_spectrum
from core_logic import run_inference


class SpectralImageProcessor:
    """Handles loading and processing of spectral images."""

    def __init__(self):
        self.support_formats = [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]
        self.default_target_size = (224, 224)

    def load_image(self, image_source) -> Optional[np.ndarray]:
        """Load image from various sources."""
        try:
            if isinstance(image_source, str):
                # File path
                img = Image.open(image_source)
            elif hasattr(image_source, "read"):
                # File-like object (Streamlit uploaded file)
                img = Image.open(image_source)
            elif isinstance(image_source, np.ndarray):
                # NumPy array
                return image_source
            else:
                raise ValueError("Unsupported image source type")

            # Convert to RGB if needed
            if img.mode != "RGB":
                img = img.convert("RGB")

            return np.array(img)

        except (FileNotFoundError, IOError, ValueError) as e:
            st.error(f"Error loading image: {e}")
            return None

    def preprocess_image(
        self,
        image: np.ndarray,
        target_size: Optional[Tuple[int, int]] = None,
        enhance_contrast: bool = True,
        apply_gaussian_blur: bool = False,
        normalize: bool = True,
    ) -> np.ndarray:
        """Preprocess image for analysis."""
        if target_size is None:
            target_size = self.default_target_size

        # Convert to PIL for processing
        img = Image.fromarray(image.astype(np.uint8))

        # Resize
        img = img.resize(target_size, Image.Resampling.LANCZOS)

        # Enhance contrast if required
        if enhance_contrast:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.2)

        # Apply Gaussian blur if requested
        if apply_gaussian_blur:
            img = img.filter(ImageFilter.GaussianBlur(radius=1))

        # Convert back to numpy
        processed = np.array(img)

        # Normalize to [0, 1] if requested
        if normalize:
            processed = processed.astype(np.float32) / 255.0

        return processed

    def extract_spectral_profile(
        self,
        image: np.ndarray,
        method: str = "average",
        roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> np.ndarray:
        """
        Extract 1D spectral profile from 2D image.

        Args:
            image: Input image array
            method: 'average', 'center_line', 'max_intensity'
            roi: Region of interest (x1, y1, x2, y2)
        """
        if roi:
            x1, y1, x2, y2 = roi
            image_roi = image[y1:y2, x1:x2]
        else:
            image_roi = image

        if len(image_roi.shape) == 3:
            # Convert to grayscale if color
            image_roi = np.mean(image_roi, axis=2)

        if method == "average":
            # Average along one axis
            profile = np.mean(image_roi, axis=0)
        elif method == "center_line":
            # Extract center line
            center_y = image_roi.shape[0] // 2
            profile = image_roi[center_y, :]
        elif method == "max_intensity":
            # Maximum intensity projection
            profile = np.max(image_roi, axis=0)
        else:
            raise ValueError(f"Unknown method: {method}")

        return profile

    def image_to_spectrum(
        self,
        image: np.ndarray,
        wavenumber_range: Tuple[float, float] = (400, 4000),
        method: str = "average",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert image to spectrum-like data."""
        # Extract 1D profile
        profile = self.extract_spectral_profile(image, method=method)

        # Create wavenumber axis
        wavenumbers = np.linspace(
            wavenumber_range[0], wavenumber_range[1], len(profile)
        )

        return wavenumbers, profile

    def detect_spectral_peaks(
        self,
        spectrum: np.ndarray,
        wavenumbers: np.ndarray,
        prominence: float = 0.1,
        height: float = 0.1,
    ) -> List[Dict[str, float]]:
        """Detect peaks in spectral data."""
        from scipy.signal import find_peaks

        peaks, properties = find_peaks(spectrum, prominence=prominence, height=height)

        peak_info = []
        for i, peak_idx in enumerate(peaks):
            peak_info.append(
                {
                    "wavenumber": wavenumbers[peak_idx],
                    "intensity": spectrum[peak_idx],
                    "prominence": properties["prominences"][i],
                    "width": (
                        properties.get("widths", [None])[i]
                        if "widths" in properties
                        else None
                    ),
                }
            )

        return peak_info

    def create_visualization(
        self,
        image: np.ndarray,
        spectrum_x: np.ndarray,
        spectrum_y: np.ndarray,
        peaks: Optional[List[Dict]] = None,
    ) -> Figure:
        """Create visualization of image and extracted spectrum."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Display image
        ax1.imshow(image, cmap="viridis" if len(image.shape) == 2 else None)
        ax1.set_title("Input Image")
        ax1.axis("off")

        # Display spectrum
        ax2.plot(
            spectrum_x, spectrum_y, "b-", linewidth=1.5, label="Extracted Spectrum"
        )

        # Mark peaks if provided
        if peaks:
            peak_wavenumbers = [p["wavenumber"] for p in peaks]
            peak_intensities = [p["intensity"] for p in peaks]
            ax2.plot(
                peak_wavenumbers,
                peak_intensities,
                "ro",
                markersize=6,
                label="Detected Peaks",
            )

        ax2.set_xlabel("Wavenumber (cm⁻¹)")
        ax2.set_ylabel("Intensity")
        ax2.set_title("Extracted Spectral Profile")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        return fig


def render_image_upload_interface():
    """Render UI for image upload and processing."""
    st.markdown("#### Image-Based Spectral Analysis")
    st.markdown(
        "Upload spectral images for analysis and conversion to spectroscopic data."
    )

    processor = SpectralImageProcessor()

    # Image upload
    uploaded_image = st.file_uploader(
        "Upload spectral image",
        type=["png", "jpg", "jpeg", "tiff", "bmp"],
        help="Upload an image containing spectral data",
    )

    if uploaded_image is not None:
        # Load and display original image
        image = processor.load_image(uploaded_image)

        if image is not None:
            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown("##### Original Image")
                st.image(image, use_column_width=True)

                # Image info
                st.write(f"**Dimensions**: {image.shape}")
                st.write(f"**Size**: {uploaded_image.size} bytes")

            with col2:
                st.markdown("##### Processing Options")

                # Processing parameters
                target_width = st.slider("Target Width", 100, 1000, 500)
                target_height = st.slider("Target Height", 100, 1000, 300)
                enhance_contrast = st.checkbox("Enhance Contrast", value=True)
                apply_blur = st.checkbox("Apply Gaussian Blur", value=False)

                # Extraction method
                extraction_method = st.selectbox(
                    "Spectrum Extraction Method",
                    ["average", "center_line", "max_intensity"],
                    help="Method for converting 2D image to 1D spectrum",
                )

                # Wavenumber range
                st.markdown("**Wavenumber Range (cm⁻¹)**")
                wn_col1, wn_col2 = st.columns(2)
                with wn_col1:
                    wn_min = st.number_input("Min", value=400.0, step=10.0)
                with wn_col2:
                    wn_max = st.number_input("Max", value=4000.0, step=10.0)

            # Process image
            if st.button("Process Image", type="primary"):
                with st.spinner("Processing image..."):
                    # Preprocess image
                    processed_image = processor.preprocess_image(
                        image,
                        target_size=(target_width, target_height),
                        enhance_contrast=enhance_contrast,
                        apply_gaussian_blur=apply_blur,
                    )

                    # Extract spectrum
                    wavenumbers, spectrum = processor.image_to_spectrum(
                        processed_image,
                        wavenumber_range=(wn_min, wn_max),
                        method=extraction_method,
                    )

                    # Detect peaks
                    peaks = processor.detect_spectral_peaks(spectrum, wavenumbers)

                    # Create visualization
                    fig = processor.create_visualization(
                        processed_image, wavenumbers, spectrum, peaks
                    )

                    # Display visualization
                    st.pyplot(fig)

                    # Display peaks information
                    if peaks:
                        st.markdown("##### Detected Peaks")
                        peak_df = pd.DataFrame(peaks)
                        peak_df["wavenumber"] = peak_df["wavenumber"].round(2)
                        peak_df["intensity"] = peak_df["intensity"].round(4)
                        st.dataframe(peak_df)

                    # Store in session state for further analysis
                    st.session_state["image_spectrum_x"] = wavenumbers
                    st.session_state["image_spectrum_y"] = spectrum
                    st.session_state["image_peaks"] = peaks

                    st.success(
                        "Image processing complete! You can now use this data for model inference."
                    )

                    # Option to run inference on extracted spectrum
                    if st.button("Run Inference on Extracted Spectrum"):

                        # Preprocess extracted spectrum
                        modality = st.session_state.get("modality_select", "raman")
                        _, y_processed = preprocess_spectrum(
                            wavenumbers, spectrum, modality=modality, target_len=500
                        )

                        # Get selected model
                        model_choice = st.session_state.get("model_select", "figure2")
                        if " " in model_choice:
                            model_choice = model_choice.split(" ", 1)[1]

                        # Run inference
                        prediction, logits_list, probs, inference_time, logits = (
                            run_inference(y_processed, model_choice)
                        )

                        if prediction is not None:
                            class_names = ["Stable", "Weathered"]
                            predicted_class = (
                                class_names[int(prediction)]
                                if prediction < len(class_names)
                                else f"Class_{prediction}"
                            )
                            confidence = max(probs) if probs and len(probs) > 0 else 0.0

                            # Display results
                            st.markdown("##### Inference Results")
                            result_col1, result_col2 = st.columns(2)

                            with result_col1:
                                st.metric("Prediction", predicted_class)
                                st.metric("Confidence", f"{confidence:.3f}")

                            with result_col2:
                                st.metric("Model Used", model_choice)
                                st.metric("Processing Time", f"{inference_time:.3f}s")

                            # Show class probabilities
                            if probs:
                                st.markdown("**Class Probabilities**")
                                for i, prob in enumerate(probs):
                                    if i < len(class_names):
                                        st.write(f"- {class_names[i]}: {prob:.4f}")


def image_to_spectrum_converter(
    image_path: str,
    wavenumber_range: Tuple[float, float] = (400, 4000),
    method: str = "average",
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert image file to spectrum data (utility function)."""
    processor = SpectralImageProcessor()

    # Load image
    image = processor.load_image(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}.")

    # Convert to spectrum
    return processor.image_to_spectrum(image, wavenumber_range, method)
