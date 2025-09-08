"""Advanced Spectroscopy Integration Module
Support dual FTIR + Raman spectroscopy with ATR-FTIR integration"""

import numpy as np
from scipy.integrate import trapezoid as trapz
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from scipy import signal
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d


@dataclass
class SpectroscopyType:
    """Define spectroscopy types and their characteristics"""

    FTIR = "FTIR"
    ATR_FTIR = "ATR-FTIR"
    RAMAN = "Raman"
    TRANSMISSION_FTIR = "Transmission-FTIR"
    REFLECTION_FTIR = "Reflection-FTIR"


@dataclass
class SpectralCharacteristics:
    """Characteristics of different spectroscopy techniques"""

    technique: str
    wavenumber_range: Tuple[float, float]  # cm-1
    typical_resolution: float  # cm-1
    sample_requirements: str
    penetration_depth: Optional[str] = None
    advantages: Optional[List[str]] = None
    limitations: Optional[List[str]] = None


# Define characteristics for each technique
SPECTRAL_CHARACTERISTICS = {
    SpectroscopyType.FTIR: SpectralCharacteristics(
        technique="FTIR",
        wavenumber_range=(400.0, 4000.0),
        typical_resolution=4.0,
        sample_requirements="Various (solid, liquid, gas)",
        penetration_depth="Variable",
        advantages=["High spectral resolution", "Wide range", "Quantitative"],
        limitations=["Water interference", "Sample preparation"],
    ),
    SpectroscopyType.ATR_FTIR: SpectralCharacteristics(
        technique="ATR-FTIR",
        wavenumber_range=(600.0, 4000.0),
        typical_resolution=4.0,
        sample_requirements="Direct solid contact",
        penetration_depth="0.5-2 μm",
        advantages=["Minimal sample prep", "Solid samples", "Quick analysis"],
        limitations=["Surface analysis only", "Pressure sensitive"],
    ),
    SpectroscopyType.RAMAN: SpectralCharacteristics(
        technique="Raman",
        wavenumber_range=(200, 3500),
        typical_resolution=1.0,
        sample_requirements="Various (solid, liquid)",
        penetration_depth="Variable",
        advantages=["Water compatible", "Non-destructive", "Molecular vibrations"],
        limitations=["Fluorescence interference", "Weak signals"],
    ),
}


class AdvancedPreprocessor:
    """Advanced preprocessing pipeline for multi-modal spectroscopy data"""

    def __init__(self):
        self.techniques_applied = []
        self.preprocessing_log = []

    def baseline_correction(
        self,
        wavenumber: np.ndarray,
        intensities: np.ndarray,
        method: str = "airpls",
        **kwargs,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Advanced baseline correction methods

        Args:
            wavenumber: Wavenumber array
            intensities: Intensity array
            method: Baseline correction method ('airpls', 'als', 'polynomial', 'rolling_ball')
            **kwargs: Method-specific parameters

        Returns:
            Corrected intensities and processing metadata
        """
        metadata = {
            "method": method,
            "original_range": (intensities.min(), intensities.max()),
        }
        corrected_intensities = intensities.copy()

        if method == "airpls":
            corrected_intensities = self._airpls_baseline(intensities, **kwargs)
        elif method == "als":
            corrected_intensities = self._als_baseline(intensities, **kwargs)
        elif method == "polynomial":
            degree = kwargs.get("degree", 3)
            coeffs = np.polyfit(wavenumber, intensities, degree)
            baseline = np.polyval(coeffs, wavenumber)
            corrected_intensities = intensities - baseline
            metadata["polynomial_degree"] = degree
        elif method == "rolling_ball":
            ball_radius = kwargs.get("radius", 50)
            corrected_intensities = self._rolling_ball_baseline(
                intensities, ball_radius
            )
            metadata["ball_radius"] = ball_radius

        self.preprocessing_log.append(f"Baseline correction: {method}")
        metadata["corrected_range"] = (
            corrected_intensities.min(),
            corrected_intensities.max(),
        )

        return corrected_intensities, metadata

    def _airpls_baseline(
        self, y: np.ndarray, lambda_: float = 1e4, itermax: int = 15
    ) -> np.ndarray:
        """
        Adaptive Iteratively Reweighted Penalized Least Squares baseline correction
        """
        m = len(y)
        D = sparse.diags([1, -2, 1], offsets=[0, -1, -2], shape=(m, m - 2))
        D = lambda_ * D.dot(D.transpose())
        w = np.ones(m)

        for i in range(itermax):
            W = sparse.spdiags(w, 0, m, m)
            Z = W + D
            z = spsolve(Z, w * y)
            d = y - z
            dn = d[d < 0]

            m_dn = np.mean(dn) if len(dn) > 0 else 0
            s_dn = np.std(dn) if len(dn) > 1 else 1

            wt = 1.0 / (1 + np.exp(2 * (d - (2 * s_dn - m_dn)) / s_dn))

            if np.linalg.norm(w - wt) / np.linalg.norm(w) < 1e-9:
                break
            w = wt

        z = spsolve(sparse.spdiags(w, 0, m, m) + D, w * y)
        return y - z

    def _als_baseline(
        self, y: np.ndarray, lambda_: float = 1e4, p: float = 0.001
    ) -> np.ndarray:
        """
        Asymmetric Least Squares baseline correction
        """
        m = len(y)
        D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(m, m - 2))
        D_t_D = D.dot(D.transpose())
        w = np.ones(m)

        for _ in range(10):
            W = sparse.spdiags(w, 0, m, m)
            Z = W + lambda_ * D_t_D
            z = spsolve(Z, w * y)
            w = p * (y > z) + (1 - p) * (y < z)

        return y - z

    def _rolling_ball_baseline(self, y: np.ndarray, radius: int) -> np.ndarray:
        """
        Rolling ball baseline correction
        """
        n = len(y)
        baseline = np.zeros_like(y)

        for i in range(n):
            start = max(0, i - radius)
            end = min(n, i + radius + 1)
            baseline[i] = np.min(y[start:end])

        return y - baseline

    def normalization(
        self,
        wavenumbers: np.ndarray,
        intensities: np.ndarray,
        method: str = "vector",
        **kwargs,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Advanced normalization methods for spectroscopy data

        Args:
            wavenumbers: Wavenumber array
            intensities: Intensity array
            method: Normalization method ('vector', 'min_max', 'standard', 'area', 'peak')
            **kwargs: Method-specific parameters

        Returns:
            Normalized intensities and processing metadata
        """
        normalized_intensities = intensities.copy()
        metadata = {"method": method, "original_std": np.std(intensities)}

        if method == "vector":
            norm = np.linalg.norm(intensities)
            normalized_intensities = intensities / norm if norm > 0 else intensities
            metadata["norm_value"] = norm
        elif method == "min_max":
            scaler = MinMaxScaler()
            normalized_intensities = scaler.fit_transform(
                intensities.reshape(-1, 1)
            ).flatten()
            metadata["min_value"] = scaler.data_min_[0]
            metadata["max_value"] = scaler.data_max_[0]
        elif method == "standard":
            scaler = StandardScaler()
            normalized_intensities = scaler.fit_transform(
                intensities.reshape(-1, 1)
            ).flatten()
            metadata["mean"] = scaler.mean_[0] if scaler.mean_ is not None else None
            metadata["std"] = scaler.scale_[0] if scaler.scale_ is not None else None
        elif method == "area":
            area = trapz(np.abs(intensities), wavenumbers)
            normalized_intensities = intensities / area if area > 0 else intensities
            metadata["area"] = area
        elif method == "peak":
            peak_idx = kwargs.get("peak_idx", np.argmax(np.abs(intensities)))
            peak_value = intensities[peak_idx]
            normalized_intensities = (
                intensities / peak_value if peak_value != 0 else intensities
            )
            metadata["peak_wavenumber"] = wavenumbers[peak_idx]
            metadata["peak_value"] = peak_value

        self.preprocessing_log.append(f"Normalization: {method}")
        metadata["normalized_std"] = np.std(normalized_intensities)

        return normalized_intensities, metadata

    def noise_reduction(
        self,
        wavenumbers: np.ndarray,
        intensities: np.ndarray,
        method: str = "savgol",
        **kwargs,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Advanced noise reduction techniques

        Args:
            wavenumbers: Wavenumber array
            intensities: Intensity array
            method: Denoising method ('savgol', 'wiener', 'median', 'gaussian')
            **kwargs: Method-specific parameters

        Returns:
            Reduced intensities and processing metadata
        """
        denoised_intensities = intensities.copy()
        metadata = {
            "method": method,
            "original_noise_level": np.std(np.diff(intensities)),
        }

        if method == "savgol":
            window_length = kwargs.get("window_length", 11)
            polyorder = kwargs.get("polyorder", 3)

            if window_length % 2 == 0:
                window_length += 1
            window_length = max(window_length, polyorder + 1)
            window_length = min(window_length, len(intensities) - 1)

            if window_length >= 3:
                denoised_intensities = signal.savgol_filter(
                    intensities, window_length, polyorder
                )
                metadata["window_length"] = window_length
                metadata["polyorder"] = polyorder
        elif method == "gaussian":
            sigma = kwargs.get("sigma", 1.0)  # Default value for sigma
            denoised_intensities = gaussian_filter1d(intensities, sigma)
            metadata["sigma"] = sigma
        elif method == "median":
            kernel_size = kwargs.get("kernel_size", 5)
            denoised_intensities = signal.medfilt(intensities, kernel_size)
            metadata["kernel_size"] = kernel_size
        elif method == "wiener":
            noise_power = kwargs.get("noise_power", None)
            denoised_intensities = signal.wiener(intensities, noise=noise_power)
            metadata["noise_power"] = noise_power

        self.preprocessing_log.append(f"Noise reduction: {method}")
        metadata["final_noise_level"] = np.std(np.diff(denoised_intensities))

        return denoised_intensities, metadata

    def technique_specific_preprocessing(
        self, wavenumbers: np.ndarray, intensities: np.ndarray, technique: str
    ) -> tuple[np.ndarray, Dict]:
        """
        Apply technique-specific preprocessing optimizations

        Args:
            wavenumbers: Wavenumber array
            intensities: Intensity array
            technique: Spectroscopy technique

        Returns:
            Processed intensities and metadata
        """
        processed_intensities = intensities.copy()
        metadata = {"technique": technique, "optimizations_applied": []}

        if technique == SpectroscopyType.ATR_FTIR:
            processed_intensities = self._atr_correction(wavenumbers, intensities)
            metadata["optimizations_applied"].append("ATR_penetration_correction")
        elif technique == SpectroscopyType.RAMAN:
            processed_intensities = self._cosmic_ray_removal(intensities)
            metadata["optimizations_applied"].append("cosmic_ray_removal")
            processed_intensities = self._fluorescence_correction(
                wavenumbers, processed_intensities
            )
            metadata["optimizations_applied"].append("fluorescence_correction")
        elif technique == SpectroscopyType.FTIR:
            processed_intensities = self._atmospheric_correction(
                wavenumbers, intensities
            )
            metadata["optimizations_applied"].append("atmospheric_correction")

        self.preprocessing_log.append(f"Technique-specific preprocessing: {technique}")
        return processed_intensities, metadata

    def _atr_correction(
        self, wavenumbers: np.ndarray, intensities: np.ndarray
    ) -> np.ndarray:
        """
        Apply ATR correction for wavelength-dependant penetration depth
        """
        correction_factor = np.sqrt(wavenumbers / np.max(wavenumbers))
        return intensities * correction_factor

    def _cosmic_ray_removal(
        self, intensities: np.ndarray, threshold: float = 3.0
    ) -> np.ndarray:
        """
        Remove cosmic ray spikes from Raman spectra
        """
        diff = np.abs(np.diff(intensities, prepend=intensities[0]))
        mean_diff = np.mean(diff)
        std_diff = np.std(diff)

        spikes = diff > (mean_diff + threshold * std_diff)
        corrected = intensities.copy()

        for i in np.where(spikes)[0]:
            if i > 0 and i < len(corrected) - 1:
                corrected[i] = (corrected[i - 1] + corrected[i + 1]) / 2

        return corrected

    def _fluorescence_correction(
        self, wavenumbers: np.ndarray, intensities: np.ndarray
    ) -> np.ndarray:
        """
        Remove fluorescence from Raman spectra
        """
        try:
            coeffs = np.polyfit(wavenumbers, intensities, deg=3)
            background = np.polyval(coeffs, wavenumbers)
            return intensities - background
        except np.linalg.LinAlgError:
            return intensities

    def _atmospheric_correction(
        self, wavenumbers: np.ndarray, intensities: np.ndarray
    ) -> np.ndarray:
        """
        Correct for atmospheric CO2 and water vapor absorption
        """
        corrected = intensities.copy()
        co2_mask = (wavenumbers >= 2350) & (wavenumbers <= 2380)
        if np.any(co2_mask):
            non_co2_idx = ~co2_mask
            if np.any(non_co2_idx):
                interp_func = interp1d(
                    wavenumbers[non_co2_idx],
                    corrected[non_co2_idx],
                    kind="linear",
                    bounds_error=False,
                    fill_value="extrapolate",
                )
                corrected[co2_mask] = interp_func(wavenumbers[co2_mask])

        return corrected


class MultiModalSpectroscopyEngine:
    """Engine for handling multi-modal spectrscopy data fusion."""

    def __init__(self):
        self.preprocessor = AdvancedPreprocessor()
        self.registered_techniques = {}
        self.fusion_strategies = [
            "concatenation",
            "weighted_average",
            "pca_fusion",
            "attention_fusion",
        ]

    def register_spectrum(
        self,
        wavenumbers: np.ndarray,
        intensities: np.ndarray,
        technique: str,
        metadata: Optional[Dict] = None,
    ) -> str:
        """
        Register a spectrum for multi-modal analysis

        Args:
            wavenumbers: Wavenumber array
            intensities: Intensity array
            technique: Spectroscopy technique type
            metadata: Additional metadata for the spectrum

        Returns:
            Spectrum ID for tracking
        """
        spectrum_id = f"{technique}_{len(self.registered_techniques)}"

        self.registered_techniques[spectrum_id] = {
            "wavenumbers": wavenumbers,
            "intensities": intensities,
            "technique": technique,
            "metadata": metadata or {},
            "characteristics": SPECTRAL_CHARACTERISTICS.get(technique),
        }

        return spectrum_id

    def preprocess_spectrum(
        self, spectrum_id: str, preprocessing_config: Optional[Dict] = None
    ) -> Dict:
        """
        Apply comprehensive preprocessing to a registered spectrum

        Args:
            spectrum_id: ID of registered spectrum
            preprocessing_config: Configuration for preprocessing steps

        Returns:
            Processing results and metadata
        """
        if spectrum_id not in self.registered_techniques:
            raise ValueError(f"Spectrum with ID {spectrum_id} not found.")

        spectrum_data = self.registered_techniques[spectrum_id]
        wavenumbers = spectrum_data["wavenumbers"]
        intensities = spectrum_data["intensities"]
        technique = spectrum_data["technique"]

        config = preprocessing_config or {}

        processed_intensities = intensities.copy()
        processing_metadata = {"steps_applied": [], "step_metadata": {}}

        if config.get("baseline_correction", True):
            method = config.get("baseline_method", "airpls")
            processed_intensities, baseline_metadata = (
                self.preprocessor.baseline_correction(
                    wavenumbers, processed_intensities, method=method
                )
            )
            processing_metadata["steps_applied"].append("baseline_correction")
            processing_metadata["step_metadata"][
                "baseline_correction"
            ] = baseline_metadata

        processed_intensities, technique_meta = (
            self.preprocessor.technique_specific_preprocessing(
                wavenumbers, processed_intensities, technique
            )
        )
        processing_metadata["steps_applied"].append("technique_specific")
        processing_metadata["step_metadata"]["technique_specific"] = technique_meta

        if config.get("noise_reduction", True):
            method = config.get("noise_method", "savgol")
            processed_intensities, noise_meta = self.preprocessor.noise_reduction(
                wavenumbers, processed_intensities, method=method
            )
            processing_metadata["steps_applied"].append("noise_reduction")
            processing_metadata["step_metadata"]["noise_reduction"] = noise_meta

        if config.get("normalization", True):
            method = config.get("norm_method", "vector")
            processed_intensities, norm_meta = self.preprocessor.normalization(
                wavenumbers, processed_intensities, method=method
            )
            processing_metadata["steps_applied"].append("normalization")
            processing_metadata["step_metadata"]["normalization"] = norm_meta

        self.registered_techniques[spectrum_id][
            "processed_intensities"
        ] = processed_intensities
        self.registered_techniques[spectrum_id][
            "processing_metadata"
        ] = processing_metadata

        return {
            "spectrum_id": spectrum_id,
            "processed_intensities": processed_intensities,
            "processing_metadata": processing_metadata,
            "quality_score": self._calculate_quality_score(
                wavenumbers, processed_intensities
            ),
        }

    def fuse_spectra(
        self,
        spectrum_ids: List[str],
        fusion_strategy: str = "concatenation",
        target_wavenumber_range: Optional[Tuple[float, float]] = None,
    ) -> Dict:
        """Fuse multiple spectra using specified strategy

        Args:
            spectrum_ids: List of spectrum IDs to fuse
            fusion_strategy: Fusion strategy ('concatenation', 'weighted_average', etc.)
            target_wavenumber_range: Common wavenumber for fusion

        Returns:
            Fused spectrum data and processing metadata
        """
        if not all(sid in self.registered_techniques for sid in spectrum_ids):
            raise ValueError("Some spectrum IDs not found")

        spectra_data = [self.registered_techniques[sid] for sid in spectrum_ids]

        if fusion_strategy == "concatenation":
            return self._concatenation_fusion(spectra_data, target_wavenumber_range)
        elif fusion_strategy == "weighted_average":
            return self._weighted_average_fusion(spectra_data, target_wavenumber_range)
        elif fusion_strategy == "pca_fusion":
            return self._pca_fusion(spectra_data, target_wavenumber_range)
        elif fusion_strategy == "attention_fusion":
            return self._attention_fusion(spectra_data, target_wavenumber_range)
        else:
            raise ValueError(
                f"Unknown or unsupported fusion strategy: {fusion_strategy}"
            )

    def _interpolate_to_common_grid(
        self,
        spectra_data: List[Dict],
        target_range: Tuple[float, float],
        num_points: int = 1000,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Interpolate all spectra to a common wavenumber grid"""
        common_wavenumbers = np.linspace(target_range[0], target_range[1], num_points)
        interpolated_intensities_list = []

        for spectrum in spectra_data:
            wavenumbers = spectrum["wavenumbers"]
            intensities = spectrum.get("processed_intensities", spectrum["intensities"])

            valid_range = (wavenumbers.min(), wavenumbers.max())
            mask = (common_wavenumbers >= valid_range[0]) & (
                common_wavenumbers <= valid_range[1]
            )

            interp_intensities = np.zeros_like(common_wavenumbers)
            if np.any(mask):
                interp_func = interp1d(
                    wavenumbers,
                    intensities,
                    kind="linear",
                    bounds_error=False,
                    fill_value=0,
                )
                interp_intensities[mask] = interp_func(common_wavenumbers[mask])

            interpolated_intensities_list.append(interp_intensities)

        return common_wavenumbers, interpolated_intensities_list

    def _concatenation_fusion(
        self, spectra_data: List[Dict], target_range: Optional[Tuple[float, float]]
    ) -> Dict:
        """Simple concatenation of spectra"""
        if target_range is None:
            min_wn = max(s["wavenumbers"].min() for s in spectra_data)
            max_wn = min(s["wavenumbers"].max() for s in spectra_data)
            target_range = (min_wn, max_wn)

        common_wn, interpolated_intensities = self._interpolate_to_common_grid(
            spectra_data, target_range
        )

        fused_intensities = np.concatenate(interpolated_intensities)
        fused_wavenumbers = np.tile(common_wn, len(spectra_data))

        return {
            "wavenumbers": fused_wavenumbers,
            "intensities": fused_intensities,
            "fusion_strategy": "concatenation",
            "source_techniques": [s["technique"] for s in spectra_data],
            "common_range": target_range,
        }

    def _weighted_average_fusion(
        self, spectra_data: List[Dict], target_range: Optional[Tuple[float, float]]
    ) -> Dict:
        """Weighted average fusion based on data quality"""
        if target_range is None:
            min_wn = max(s["wavenumbers"].min() for s in spectra_data)
            max_wn = min(s["wavenumbers"].max() for s in spectra_data)
            target_range = (min_wn, max_wn)

        common_wn, interpolated_intensities = self._interpolate_to_common_grid(
            spectra_data, target_range
        )

        weights = []
        for i, spectrum in enumerate(spectra_data):
            quality_score = self._calculate_quality_score(
                common_wn, interpolated_intensities[i]
            )
            weights.append(quality_score)

        weights = np.array(weights)
        weights_sum = np.sum(weights)
        weights = (
            weights / weights_sum
            if weights_sum > 0
            else np.full_like(weights, 1.0 / len(weights))
        )

        fused_intensities = np.zeros_like(common_wn)
        for i, intensities in enumerate(interpolated_intensities):
            fused_intensities += weights[i] * intensities

        return {
            "wavenumbers": common_wn,
            "intensities": fused_intensities,
            "fusion_strategy": "weighted_average",
            "weights": weights.tolist(),
            "source_techniques": [s["technique"] for s in spectra_data],
            "common_range": target_range,
        }

    def _pca_fusion(
        self, spectra_data: List[Dict], target_range: Optional[Tuple[float, float]]
    ) -> Dict:
        """PCA-based fusion to extract common features"""
        if target_range is None:
            min_wn = max(s["wavenumbers"].min() for s in spectra_data)
            max_wn = min(s["wavenumbers"].max() for s in spectra_data)
            target_range = (min_wn, max_wn)

        common_wn, interpolated_intensities = self._interpolate_to_common_grid(
            spectra_data, target_range
        )

        spectra_matrix = np.vstack(interpolated_intensities)

        n_components = min(len(spectra_data), 3)
        pca = PCA(n_components=n_components)
        pca.fit(spectra_matrix.T)  # Fit on features (wavenumbers)

        fused_intensities = np.dot(pca.explained_variance_ratio_, pca.components_)

        return {
            "wavenumbers": common_wn,
            "intensities": fused_intensities,
            "fusion_strategy": "pca_fusion",
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "n_components": n_components,
            "source_techniques": [s["technique"] for s in spectra_data],
            "common_range": target_range,
        }

    def _attention_fusion(
        self, spectra_data: List[Dict], target_range: Optional[Tuple[float, float]]
    ) -> Dict:
        """Attention-based fusion using a simple neural attention-like mechanism"""
        if target_range is None:
            min_wn = max(s["wavenumbers"].min() for s in spectra_data)
            max_wn = min(s["wavenumbers"].max() for s in spectra_data)
            target_range = (min_wn, max_wn)

        common_wn, interpolated_intensities = self._interpolate_to_common_grid(
            spectra_data, target_range
        )

        attention_scores = []
        for intensities in interpolated_intensities:
            variance = np.var(intensities)
            quality = self._calculate_quality_score(common_wn, intensities)
            attention_scores.append(variance * quality)

        attention_scores = np.array(attention_scores)
        exp_scores = np.exp(
            attention_scores - np.max(attention_scores)
        )  # Softmax for stability
        attention_weights = exp_scores / np.sum(exp_scores)

        fused_intensities = np.zeros_like(common_wn)
        for i, intensities in enumerate(interpolated_intensities):
            fused_intensities += attention_weights[i] * intensities

        return {
            "wavenumbers": common_wn,
            "intensities": fused_intensities,
            "fusion_strategy": "attention_fusion",
            "attention_weights": attention_weights.tolist(),
            "source_techniques": [s["technique"] for s in spectra_data],
            "common_range": target_range,
        }

    def _calculate_quality_score(
        self, wavenumbers: np.ndarray, intensities: np.ndarray
    ) -> float:
        """Calculate spectral quality score based on signal-to-noise ratio and other metrics"""
        try:
            signal_power = np.var(intensities)
            if len(intensities) < 2:
                return 0.0
            noise_power = np.var(np.diff(intensities))
            snr = signal_power / noise_power if noise_power > 0 else 1e6

            peaks, properties = find_peaks(
                intensities, prominence=0.1 * np.std(intensities)
            )
            peak_prominence = (
                np.mean(properties["prominences"]) if len(peaks) > 0 else 0
            )

            baseline_stability = 1.0 / (
                1.0 + np.std(intensities[:10]) + np.std(intensities[-10:])
            )

            quality_score = (
                np.log10(max(snr, 1)) * 0.5
                + peak_prominence * 0.3
                + baseline_stability * 0.2
            )

            return max(0, min(1, quality_score))
        except Exception:
            return 0.5

    def get_technique_recommendations(self, sample_type: str) -> List[Dict]:
        """
        Recommend optimal spectroscopy techniques for a given sample type

        Args:
            sample_type: Type of sample (e.g., 'solid_polymer', 'liquid_polymer', 'thin_film')

        Returns:
            List of recommended techniques with rationale
        """
        recommendations = []

        if sample_type in ["solid_polymer", "polymer_pellets", "polymer_film"]:
            recommendations.extend(
                [
                    {
                        "technique": SpectroscopyType.ATR_FTIR,
                        "priority": "high",
                        "rationale": "Minimal sample preparation, direct solid contact analysis",
                        "characteristics": SPECTRAL_CHARACTERISTICS[
                            SpectroscopyType.ATR_FTIR
                        ],
                    },
                    {
                        "technique": SpectroscopyType.RAMAN,
                        "priority": "medium",
                        "rationale": "Complementary vibrational information, non-destructive",
                        "characteristics": SPECTRAL_CHARACTERISTICS[
                            SpectroscopyType.RAMAN
                        ],
                    },
                ]
            )
        elif sample_type in ["liquid_polymer", "polymer_solution"]:
            recommendations.extend(
                [
                    {
                        "technique": SpectroscopyType.FTIR,
                        "priority": "high",
                        "rationale": "Versatile for liquid samples, wide spectral range",
                        "characteristics": SPECTRAL_CHARACTERISTICS[
                            SpectroscopyType.FTIR
                        ],
                    },
                    {
                        "technique": SpectroscopyType.RAMAN,
                        "priority": "high",
                        "rationale": "Water compatible, molecular vibrations",
                        "characteristics": SPECTRAL_CHARACTERISTICS[
                            SpectroscopyType.RAMAN
                        ],
                    },
                ]
            )
        elif sample_type in ["weathered_polymer", "aged_polymer"]:
            recommendations.extend(
                [
                    {
                        "technique": SpectroscopyType.ATR_FTIR,
                        "priority": "high",
                        "rationale": "Surface analysis for weathering products",
                        "characteristics": SPECTRAL_CHARACTERISTICS[
                            SpectroscopyType.ATR_FTIR
                        ],
                    },
                    {
                        "technique": SpectroscopyType.FTIR,
                        "priority": "medium",
                        "rationale": "Bulk analysis for degradation assessment",
                        "characteristics": SPECTRAL_CHARACTERISTICS[
                            SpectroscopyType.FTIR
                        ],
                    },
                ]
            )

        return recommendations


""
