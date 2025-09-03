"""
Enhanced Data Pipeline for Polymer ML Aging
Integrates with spectroscopy databases, synthetic data augmentation, and quality control
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from pathlib import Path
import requests
import json
import sqlite3
from datetime import datetime
import hashlib
import warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import pickle
import io
import base64


@dataclass
class SpectralDatabase:
    """Configuration for spectroscopy databases"""

    name: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    description: str = ""
    supported_formats: List[str] = field(default_factory=list)
    access_method: str = "api"  # "api", "download", "local"
    local_path: Optional[Path] = None


# -///////////////////////////////////////////////////
@dataclass
class PolymerSample:
    """Enhanced polymer sample information"""

    sample_id: str
    polymer_type: str
    molecular_weight: Optional[float] = None
    additives: List[str] = field(default_factory=list)
    processing_conditions: Dict[str, Any] = field(default_factory=dict)
    aging_condition: Dict[str, Any] = field(default_factory=dict)
    aging_time: Optional[float] = None  # Hours
    degradation_level: Optional[float] = None  # 0-1 Scale
    spectral_data: Dict[str, np.ndarray] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    quality_score: Optional[float] = None
    validation_status: str = "pending"  # pending, validated, rejected


# -///////////////////////////////////////////////////

# Database configurations
SPECTROSCOPY_DATABASES = {
    "FTIR_PLASTICS": SpectralDatabase(
        name="FTIR Plastics Database",
        description="Comprehensive FTIR spectra of plastic materials",
        supported_formats=["FTIR", "ATR-FTIR"],
        access_method="local",
        local_path=Path("data/databases/ftir_plastics"),
    ),
    "NIST_WEBBOOK": SpectralDatabase(
        name="NIST Chemistry WebBook",
        base_url="https://webbook.nist.gov/chemistry",
        description="NIST spectroscopic database",
        supported_formats=["FTIR", "Raman"],
        access_method="api",
    ),
    "POLYMER_DATABASE": SpectralDatabase(
        name="Polymer Spectroscopy Database",
        description="Curated polymer degradation spectra",
        supported_formats=["FTIR", "ATR-FTIR", "Raman"],
        access_method="local",
        local_path=Path("data/databases/polymer_degradation"),
    ),
}

# -///////////////////////////////////////////////////


class DatabaseConnector:
    """Connector for spectroscopy databases"""

    def __init__(self, database_config: SpectralDatabase):
        self.config = database_config
        self.connection = None
        self.cache_dir = Path("data/cache") / database_config.name.lower().replace(
            " ", "_"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def connect(self) -> bool:
        """Establish connection to database"""
        try:
            if self.config.access_method == "local":
                if self.config.local_path and self.config.local_path.exists():
                    return True
                else:
                    print(f"Local database path not found: {self.config.local_path}")
                    return False

            elif self.config.access_method == "api":
                # Test API connection
                if self.config.base_url:
                    response = requests.get(self.config.base_url, timeout=10)
                    return response.status_code == 200
                return False

            return True

        except Exception as e:
            print(f"Failed to connect to {self.config.name}: {e}")
            return False

    # -///////////////////////////////////////////////////
    def search_by_polymer_type(self, polymer_type: str, limit: int = 100) -> List[Dict]:
        """Search database for spectra by polymer type"""
        cache_key = f"search{hashlib.md5(polymer_type.encode()).hexdigest()}"
        cache_file = self.cache_dir / f"{cache_key}.json"

        # Check cache first
        if cache_file.exists():
            with open(cache_file, "r") as f:
                return json.load(f)

        results = []

        if self.config.access_method == "local":
            results = self._search_local_database(polymer_type, limit)
        elif self.config.access_method == "api":
            results = self._search_api_database(polymer_type, limit)

        # Cache results
        if results:
            with open(cache_file, "w") as f:
                json.dump(results, f)

        return results

    # -///////////////////////////////////////////////////
    def _search_local_database(self, polymer_type: str, limit: int) -> List[Dict]:
        """Search local database files"""
        results = []

        if not self.config.local_path or not self.config.local_path.exists():
            return results

        # Look for CSV files with polymer data
        for csv_file in self.config.local_path.glob("*.csv"):
            try:
                df = pd.read_csv(csv_file)

                # Search for polymer type in columns
                polymer_matches = df[
                    df.astype(str)
                    .apply(lambda x: x.str.contains(polymer_type, case=False))
                    .any(axis=1)
                ]

                for _, row in polymer_matches.head(limit).iterrows():
                    result = {
                        "source_file": str(csv_file),
                        "polymer_type": polymer_type,
                        "data": row.to_dict(),
                    }
                    results.append(result)

            except Exception as e:
                print(f"Error reading {csv_file}: {e}")
                continue

        return results

    # -///////////////////////////////////////////////////
    def _search_api_database(self, polymer_type: str, limit: int) -> List[Dict]:
        """Search API-based database"""
        results = []

        try:
            # TODO: Example API search (would need actual API endpoints)
            search_params = {"query": polymer_type, "limit": limit, "format": "json"}

            if self.config.api_key:
                search_params["api_key"] = self.config.api_key

            response = requests.get(
                f"{self.config.base_url}/search", params=search_params, timeout=30
            )

            if response.status_code == 200:
                results = response.json().get("results", [])

        except Exception as e:
            print(f"API search failed: {e}")

        return results

    # -///////////////////////////////////////////////////
    def download_spectrum(self, spectrum_id: str) -> Optional[Dict]:
        """Download specific spectrum data"""
        cache_file = self.cache_dir / f"spectrum_{spectrum_id}.json"

        # Check cache
        if cache_file.exists():
            with open(cache_file, "r") as f:
                return json.load(f)

        spectrum_data = None

        if self.config.access_method == "api":
            try:
                url = f"{self.config.base_url}/spectrum/{spectrum_id}"
                response = requests.get(url, timeout=30)

                if response.status_code == 200:
                    spectrum_data = response.json()

            except Exception as e:
                print(f"Failed to download spectrum {spectrum_id}: {e}")

        # Cache results if successful
        if spectrum_data:
            with open(cache_file, "w") as f:
                json.dump(spectrum_data, f)

        return spectrum_data


# -///////////////////////////////////////////////////
class SyntheticDataAugmentation:
    """Advanced synthetic data augmentation for spectroscopy"""

    def __init__(self):
        self.augmentation_methods = [
            "noise_addition",
            "baseline_drift",
            "intensity_scaling",
            "wavenumber_shift",
            "peak_broadening",
            "atmospheric_effects",
            "instrumental_response",
            "sample_variations",
        ]

    def augment_spectrum(
        self,
        wavenumbers: np.ndarray,
        intensities: np.ndarray,
        method: str = "random",
        num_variations: int = 5,
        intensity_factor: float = 0.1,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate augmented versions of a spectrum

        Args:
            wavenumbers: Original wavenumber array
            intensities: Original intensity array
            method: str = Augmentation method or 'random' for random selection
            num_variations: Number of variations to generate
            intensity_factor: Factor controlling augmentation intesity

        Returns:
            List of (wavenumbers, intensities) tuples
        """
        augmented_spectra = []

        for _ in range(num_variations):
            if method == "random":
                chosen_method = np.random.choice(self.augmentation_methods)
            else:
                chosen_method = method

            aug_wavenumbers, aug_intensities = self._apply_augmentation(
                wavenumbers, intensities, chosen_method, intensity_factor
            )

            augmented_spectra.append((aug_wavenumbers, aug_intensities))

        return augmented_spectra

    # -///////////////////////////////////////////////////
    def _apply_augmentation(
        self,
        wavenumbers: np.ndarray,
        intensities: np.ndarray,
        method: str,
        intensity: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply specific augmentation method"""

        aug_wavenumbers = wavenumbers.copy()
        aug_intensities = intensities.copy()

        if method == "noise_addition":
            # Add random noise
            noise_level = intensity * np.std(intensities)
            noise = np.random.normal(0, noise_level, len(intensities))
            aug_intensities += noise

        elif method == "baseline_drift":
            # Add baseline drift
            drift_amplitude = intensity * np.mean(np.abs(intensities))
            drift = drift_amplitude * np.sin(
                2 * np.pi * np.linspace(0, 2, len(intensities))
            )
            aug_intensities += drift

        elif method == "intensity_scaling":
            # Scale intensity uniformly
            scale_factor = 1.0 + intensity * (2 * np.random.random() - 1)
            aug_intensities *= scale_factor

        elif method == "wavenumber_shift":
            # Shift wavenumber axis
            shift_range = intensity * 10  # cm-1
            shift = shift_range * (2 * np.random.random() - 1)
            aug_wavenumbers += shift

        elif method == "peak_broadening":
            # Broaden peaks using convolution
            from scipy import signal

            sigma = intensity * 2  # Broadening factor
            kernel_size = int(sigma * 6) + 1
            if kernel_size % 2 == 0:
                kernel_size += 1

                if kernel_size >= 3:
                    from scipy.signal.windows import gaussian

                    kernel = gaussian(kernel_size, sigma)
                    kernel = kernel / np.sum(kernel)
                    aug_intensities = signal.convolve(
                        aug_intensities, kernel, mode="same"
                    )

        elif method == "atmospheric_effects":
            # Simulate atmospheric absorption
            co2_region = (wavenumbers >= 2320) & (wavenumbers <= 2380)
            h2o_region = (wavenumbers >= 3200) & (wavenumbers <= 3800)

            if np.any(co2_region):
                aug_intensities[co2_region] *= 1 - intensity * 0.1
            if np.any(h2o_region):
                aug_intensities[h2o_region] *= 1 - intensity * 0.05

        elif method == "instrumental_response":
            # Simulate instrumental response variations
            # Add slight frequency-dependent response
            response_curve = 1.0 + intensity * 0.1 * np.sin(
                2
                * np.pi
                * (wavenumbers - wavenumbers.min())
                / (wavenumbers.max() - wavenumbers.min())
            )
            aug_intensities *= response_curve

        elif method == "sample_variations":
            # Simulate sample-to-sample variations
            # Random peak intensity variations
            num_peaks = min(5, len(intensities) // 100)
            for _ in range(num_peaks):
                peak_center = np.random.randint(0, len(intensities))
                peak_width = np.random.randint(5, 20)
                peak_variation = intensity * (2 * np.random.random() - 1)

                start_idx = max(0, peak_center - peak_width)
                end_idx = min(len(intensities), peak_center + peak_width)

                aug_intensities[start_idx:end_idx] *= 1 + peak_variation

        return aug_wavenumbers, aug_intensities

    # -///////////////////////////////////////////////////
    def generate_synthetic_aging_series(
        self,
        base_spectrum: Tuple[np.ndarray, np.ndarray],
        num_time_points: int = 10,
        max_degradation: float = 0.8,
    ) -> List[Dict]:
        """
        Generate synthetic aging series showing progressive degradation

        Args:
            base_spectrum: (wavenumbers, intensities) for fresh sample
            num_time_points: Number of time points in series
            max_degradation: Maximum degradation level (0-1)

        Returns:
            List of aging data points
        """
        wavenumbers, intensities = base_spectrum
        aging_series = []

        # Define degradation-related spectral changes
        degradation_features = {
            "carbonyl_growth": {
                "region": (1700, 1750),  # C=0 stretch
                "intensity_change": 2.0,  # Factor increase
            },
            "oh_growth": {
                "region": (3200, 3600),  # OH stretch
                "intensity_change": 1.5,
            },
            "ch_decrease": {
                "region": (2800, 3000),  # CH stretch
                "intensity_change": 0.7,  # Factor decrease
            },
            "crystrallinity_change": {
                "region": (1000, 1200),  # Various polymer backbone changes
                "intensity_change": 0.9,
            },
        }

        for i in range(num_time_points):
            degradation_level = (i / (num_time_points - 1)) * max_degradation
            aging_time = i * 100  # hours (arbitrary scale)

            # Start with base spectrum
            aged_intensities = intensities.copy()

            # Apply degradation-related changes
            for feature, params in degradation_features.items():
                region_mask = (wavenumbers >= params["region"][0]) & (
                    wavenumbers <= params["region"][1]
                )
                if np.any(region_mask):
                    change_factor = 1.0 + degradation_level * (
                        params["intensity_change"] - 1.0
                    )
                    aged_intensities[region_mask] *= change_factor

            # Add some random variations
            aug_wavenumbers, aug_intensities = self._apply_augmentation(
                wavenumbers, aged_intensities, "noise_addition", 0.02
            )

            aging_point = {
                "aging_time": aging_time,
                "degradation_level": degradation_level,
                "wavenumbers": aug_wavenumbers,
                "intensities": aug_intensities,
                "spectral_changes": {
                    feature: degradation_level * params["intensity_change"] - 1.0
                    for feature, params in degradation_features.items()
                },
            }

            aging_series.append(aging_point)

        return aging_series


# -///////////////////////////////////////////////////
class DataQualityController:
    """Advanced data quality assessment and validation"""

    def __init__(self):
        self.quality_metrics = [
            "signal_to_noise_ratio",
            "baseline_stability",
            "peak_resolution",
            "spectral_range_coverage",
            "instrumental_artifacts",
            "data_completeness",
            "metadata_completeness",
        ]

        self.validation_rules = {
            "min_str": 10.0,
            "max_baseline_variation": 0.1,
            "min_peak_count": 3,
            "min_spectral_range": 1000.0,  # cm-1
            "max_missing_points": 0.05,  # 5% max missing data
        }

    def assess_spectrum_quality(
        self,
        wavenumbers: np.ndarray,
        intensities: np.ndarray,
        metadata: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Comprehensive quality assessment of spectral data

        Args:
            wavenumbers: Array of wavenumbers
            intensities: Array of intensities
            metadata: Optional metadata dictionary

        Returns:
            Quality assessment results
        """
        assessment = {
            "overall_score": 0.0,
            "individual_scores": {},
            "issues_found": [],
            "recommendations": [],  # Ensure this is initialized as a list
            "validation_status": "pending",
        }

        # Signal-to-noise
        snr_score, snr_value = self._assess_snr(intensities)
        assessment["individual_scores"]["snr"] = snr_score
        assessment["recommendations"] = snr_value

        if snr_value < self.validation_rules["min_snr"]:
            assessment["issues_found"].append(
                f"Low SNR: {snr_value:.1f} (min: {self.validation_rules['min_snr']})"
            )
            assessment["recommendations"].append(
                "Consider noise reduction preprocessing"
            )

        # Baseline stability
        baseline_score, baseline_variation = self._assess_baseline_stability(
            intensities
        )
        assessment["individual_scores"]["baseline"] = baseline_score
        assessment["baseline_variation"] = baseline_variation

        if baseline_variation > self.validation_rules["max_baseline_variation"]:
            assessment["issues_found"].append(
                f"Unstable baseline: {baseline_variation:.3f}"
            )
            assessment["recommendations"].append("Apply baseline correction")

        # Peak resolution and count
        peak_score, peak_count = self._assess_peak_resolution(wavenumbers, intensities)
        assessment["individual_scores"]["peaks"] = peak_score
        assessment["peak_count"] = peak_count

        if peak_count < self.validation_rules["min_peak_count"]:
            assessment["issues_found"].append(f"Few peaks detected: {peak_count}")
            assessment["recommendations"].append(
                "Check sample quality or measurement conditions"
            )

        # Spectral range coverage
        range_score, spectral_range = self._assess_spectral_range(wavenumbers)
        assessment["individual_scores"]["range"] = range_score
        assessment["spectral_range"] = spectral_range

        if spectral_range < self.validation_rules["min_spectral_range"]:
            assessment["issues_found"].append(
                f"Limited spectral range: {spectral_range:.0f} cm-1"
            )

        # Data completeness
        completeness_score, missing_fraction = self._assess_data_completeness(
            intensities
        )
        assessment["individual_scores"]["completeness"] = completeness_score
        assessment["missing_fraction"] = missing_fraction

        if missing_fraction > self.validation_rules["max_missing_points"]:
            assessment["issues_found"].append(
                f"Missing data points: {missing_fraction:.1f}%"
            )
            assessment["recommendations"].append(
                "Interpolate missing points or re-measure"
            )

        # Instrumental artifacts
        artifact_score, artifacts = self._detect_instrumental_artifacts(
            wavenumbers, intensities
        )
        assessment["individual_scores"]["artifacts"] = artifact_score
        assessment["artifacts_detected"] = artifacts

        if artifacts:
            assessment["issues_found"].extend(
                [f"Artifact detected {artifact}" for artifact in artifacts]
            )
            assessment["recommendations"].append("Apply artifact correction")

        # Metadata completeness
        metadata_score = self._assess_metadata_completeness(metadata)
        assessment["individual_scores"]["metadata"] = metadata_score

        # Calculate overall score
        scores = list(assessment["individual_scores"].values())
        assessment["overall_score"] = np.mean(scores) if scores else 0.0

        # Determine validation status
        if assessment["overall_score"] >= 0.8 and len(assessment["issues_found"]) == 0:
            assessment["validation_status"] = "validated"
        elif assessment["overall_score"] >= 0.6:
            assessment["validation_status"] = "conditional"
        else:
            assessment["validation_status"] = "rejected"

        return assessment

    # -///////////////////////////////////////////////////
    def _assess_snr(self, intensities: np.ndarray) -> Tuple[float, float]:
        """Assess signal-to-noise ratio"""
        try:
            # Estimate noise from high-frequency components
            diff_signal = np.diff(intensities)
            noise_std = np.std(diff_signal)
            signal_power = np.var(intensities)

            snr = np.sqrt(signal_power) / noise_std if noise_std > 0 else float("inf")

            # Score based on SNR values
            score = min(
                1.0, max(0.0, (np.log10(snr) - 1) / 2)
            )  # Log scale, 10-1000 range

            return score, snr
        except:
            return 0.5, 1.0

    # -///////////////////////////////////////////////////
    def _assess_baseline_stability(
        self, intensities: np.ndarray
    ) -> Tuple[float, float]:
        """Assess baseline stability"""
        try:
            # Estimate baseline from endpoints and low-frequency components
            baseline_points = np.concatenate([intensities[:10], intensities[-10]])
            baseline_variation = np.std(baseline_points) / np.mean(abs(intensities))

            score = max(0.0, 1.0 - baseline_variation * 10)  # Penalty for variation

            return score, baseline_variation

        except:
            return 0.5, 1.0

    # -///////////////////////////////////////////////////
    def _assess_peak_resolution(
        self, wavenumbers: np.ndarray, intensities: np.ndarray
    ) -> Tuple[float, int]:
        """Assess peak resolution and count"""
        try:
            from scipy.signal import find_peaks

            # Find peaks with minimum prominence
            prominence_threshold = 0.1 * np.std(intensities)
            peaks, properties = find_peaks(
                intensities, prominence=prominence_threshold, distance=5
            )

            peak_count = len(peaks)

            # Score based on peak count and prominence
            if peak_count > 0:
                avg_prominence = np.mean(properties["prominences"])
                prominence_score = min(
                    1.0, avg_prominence / (0.2 * np.std(intensities))
                )
                count_score = min(1.0, peak_count / 10)  # Normalize to ~10 peaks
                score = 0.5 * prominence_score + 0.5 * count_score
            else:
                score = 0.0

            return score, peak_count

        except:
            return 0.5, 0

    # -///////////////////////////////////////////////////
    def _assess_spectral_range(self, wavenumbers: np.ndarray) -> Tuple[float, float]:
        """Assess spectral range coverage"""
        try:
            spectral_range = wavenumbers.max() - wavenumbers.min()

            # Score based on typical FTIR range (4000 cm-1)
            score = min(1.0, spectral_range / 4000)

            return score, spectral_range

        except:
            return 0.5, 1000

    # -///////////////////////////////////////////////////
    def _assess_data_completeness(self, intensities: np.ndarray) -> Tuple[float, float]:
        """Assess data completion"""
        try:
            # Check for NaN, or zero values
            invalid_mask = (
                np.isnan(intensities) | np.isinf(intensities) | (intensities == 0)
            )
            missing_fraction = np.sum(invalid_mask) / len(intensities)

            score = max(
                0.0, 1.0 - missing_fraction * 10
            )  # Heavy penalty for missing data

            return score, missing_fraction
        except:
            return 0.5, 0.0

    # -///////////////////////////////////////////////////
    def _detect_instrumental_artifacts(
        self, wavenumbers: np.ndarray, intensities: np.ndarray
    ) -> Tuple[float, List[str]]:
        """Detect common instrumental artifacts"""
        artifacts = []

        try:
            # Check for spike artifacts (cosmic rays, electrical interference)
            diff_threshold = 5 * np.std(np.diff(intensities))
            spikes = np.where(np.abs(np.diff(intensities)) > diff_threshold)[0]

            if len(spikes) > len(intensities) * 0.01:  # More than 1% spikes
                artifacts.append("excessive_spikes")

            # Check for saturation (flat regions at max/min)
            if np.std(intensities) > 0:
                max_val = np.max(intensities)
                min_val = np.min(intensities)

                saturation_high = np.sum(intensities >= 0.99 * max_val) / len(
                    intensities
                )
                saturation_low = np.sum(intensities <= 1.01 * min_val) / len(
                    intensities
                )

                if saturation_high > 0.05:
                    artifacts.append("high_saturation")
                if saturation_low > 0.05:
                    artifacts.append("low_saturation")

            # Check for periodic noise (electrical interference)
            fft = np.fft.fft(intensities - np.mean(intensities))
            freq_domain = np.abs(fft[: len(fft) // 2])

            # Look for strong periodic components
            if len(freq_domain) > 10:
                mean_amplitude = np.mean(freq_domain)
                strong_frequencies = np.sum(freq_domain > 3 * mean_amplitude)

                if strong_frequencies > len(freq_domain) * 0.1:
                    artifacts.append("periodic_noise")

            # Score inversely related to number of artifacts
            score = max(0.0, 1.0 - len(artifacts) * 0.3)

            return score, artifacts

        except:
            return 0.5, []

    # -///////////////////////////////////////////////////
    def _assess_metadata_completeness(self, metadata: Optional[Dict]) -> float:
        """Assess completeness of metadata"""
        if metadata is None:
            return 0.0

        required_fields = [
            "sample_id",
            "measurement_date",
            "instrument_type",
            "resolution",
            "number_of_scans",
            "sample_type",
        ]

        present_fields = sum(
            1
            for field in required_fields
            if field in metadata and metadata[field] is not None
        )
        score = present_fields / len(required_fields)

        return score


# -///////////////////////////////////////////////////
class EnhancedDataPipeline:
    """Complete enhanced data pipeline integrating all components"""

    def __init__(self):
        self.database_connector = {}
        self.augmentation_engine = SyntheticDataAugmentation()
        self.quality_controller = DataQualityController()
        self.local_database_path = Path("data/enhanced_data")
        self.local_database_path.mkdir(parents=True, exist_ok=True)
        self._init_local_database()

    def _init_local_database(self):
        """Initialize local SQLite database"""
        db_path = self.local_database_path / "polymer_spectra.db"

        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            # Create main spectra table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS spectra (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sample_id TEXT UNIQUE NOT NULL,
                    polymer_type TEXT NOT NULL,
                    technique TEXT NOT NULL,
                    wavenumbers BLOB,
                    intensities BLOB,
                    metadata TEXT,
                    quality_score REAL,
                    validation_status TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    source_database TEXT
                )
            """
            )

            # Create aging data table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS aging_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sample_id TEXT,
                    aging_time REAL,
                    degradation_level REAL,
                    spectral_changes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (sample_id) REFERENCES spectra (sample_id)
                )     
            """
            )

            conn.commit()

    # -///////////////////////////////////////////////////
    def connect_to_databases(self) -> Dict[str, bool]:
        """Connect to all configured databases"""
        connection_status = {}

        for db_name, db_config in SPECTROSCOPY_DATABASES.items():
            connector = DatabaseConnector(db_config)
            self.database_connector[db_name] = connector.connect()

        return connection_status

    # -///////////////////////////////////////////////////
    def search_and_import_spectra(
        self, polymer_type: str, max_per_database: int = 50
    ) -> Dict[str, int]:
        """Search and import spectra from all connected databases"""
        import_counts = {}

        for db_name, connector in self.database_connector.items():
            try:
                search_results = connector.search_by_polymer_type(
                    polymer_type, max_per_database
                )
                imported_count = 0

                for result in search_results:
                    if self._import_spectrum_to_local(result, db_name):
                        imported_count += 1

                import_counts[db_name] = imported_count

            except Exception as e:
                print(f"Error importing from {db_name}: {e}")
                import_counts[db_name] = 0

        return import_counts

    # -///////////////////////////////////////////////////]
    def _import_spectrum_to_local(self, spectrum_data: Dict, source_db: str) -> bool:
        """Import spectrum data to local database"""
        try:
            # Extract or generate sample ID
            sample_id = spectrum_data.get(
                "sample_id", f"{source_db}_{hash(str(spectrum_data))}"
            )

            # Convert spectrum data format
            if "wavenumbers" in spectrum_data and "intensities" in spectrum_data:
                wavenumbers = np.array(spectrum_data["wavenumbers"])
                intensities = np.array(spectrum_data["intensities"])
            else:
                # Try to extract from other formats
                return False

            # Quality assessment
            metadata = spectrum_data.get("metadata", {})
            quality_assessment = self.quality_controller.assess_spectrum_quality(
                wavenumbers, intensities, metadata
            )

            # Only import if quality is acceptable
            if quality_assessment["validation_status"] == "rejected":
                return False

            # Serialize arrays
            wavenumbers_blob = pickle.dumps(wavenumbers)
            intensities_blob = pickle.dumps(intensities)
            metadata_json = json.dumps(metadata)

            # Insert into database
            db_path = self.local_database_path / "polymer_spectra.db"

            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """ 
                    INSERT OR REPLACE INTO spectra(
                        sample_id, polymer_type, technique,
                        wavenumbers, intensities, metadata,
                        quality_score, validation_status,
                        source_database)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        sample_id,
                        spectrum_data.get("polymer_type", "unknown"),
                        spectrum_data.get("technique", "FTIR"),
                        wavenumbers_blob,
                        intensities_blob,
                        metadata_json,
                        quality_assessment["overall_score"],
                        quality_assessment["validation_status"],
                        source_db,
                    ),
                )

                conn.commit()

            return True

        except Exception as e:
            print(f"Error importing spectrum: {e}")
            return False

    # -///////////////////////////////////////////////////
    def generate_synthetic_aging_dataset(
        self,
        base_polymer_type: str,
        num_samples: int = 50,
        aging_conditions: Optional[List[Dict]] = None,
    ) -> int:
        """
        Generate synthetic aging dataset for training

        Args:
            base_polymer_type: Base polymer type to use
            num_samples: Number of synthetic samples to generate
            aging_conditions: List of aging condition dictionaries

        Returns:
            Number of samples generated
        """
        if aging_conditions is None:
            aging_conditions = [
                {"temperature": 60, "humidity": 75, "uv_exposure": True},
                {"temperature": 80, "humidity": 85, "uv_exposure": True},
                {"temperature": 40, "humidity": 95, "uv_exposure": False},
                {"temperature": 100, "humidity": 50, "uv_exposure": True},
            ]

        # Get base spectra from database
        base_spectra = self.spectra_by_type(base_polymer_type, limit=10)

        if not base_spectra:
            print(f"No base spectra found for {base_polymer_type}")
            return 0

        generated_count = 0

        synthetic_id = None  # Initialize synthetic_id to avoid unbound error
        aging_series = []  # Initialize aging_series to avoid unbound error
        for base_spectrum in base_spectra:
            wavenumbers = pickle.loads(base_spectrum["wavenumbers"])
            intensities = pickle.loads(base_spectrum["intensities"])

            # Generate aging series for each condition
            for condition in aging_conditions:
                aging_series = self.augmentation_engine.generate_synthetic_aging_series(
                    (wavenumbers, intensities),
                    num_time_points=min(
                        10, num_samples // len(aging_conditions) // len(base_spectra)
                    ),
                )

            if "aging_series" in locals() and aging_series:
                for aging_point in aging_series:
                    synthetic_id = f"synthetic_{base_polymer_type}_{generated_count}"

                    # Ensure condition is properly passed into the loop
                    metadata = {
                        "synthetic": True,
                        "aging_condition": aging_conditions[
                            0
                        ],  # Use the first condition or adjust as needed
                        "aging_time": aging_point["aging_time"],
                        "degradation_level": aging_point["degradation_level"],
                    }

                    # Store synthetic spectrum
                    if self._store_synthetic_spectrum(
                        synthetic_id, base_polymer_type, aging_point, metadata
                    ):
                        generated_count += 1

        return generated_count

    def _store_synthetic_spectrum(
        self, sample_id: str, polymer_type: str, aging_point: Dict, metadata: Dict
    ) -> bool:
        """Store synthetic spectrum in local database"""
        try:
            quality_assessment = self.quality_controller.assess_spectrum_quality(
                aging_point["wavenumbers"], aging_point["intensities"], metadata
            )

            # Serialize data
            wavenumbers_blob = pickle.dumps(aging_point["wavenumbers"])
            intensities_blob = pickle.dumps(aging_point["intensities"])
            metadata_json = json.dumps(metadata)

            # Insert spectrum
            db_path = self.local_database_path / "polymer_spectra.db"

            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    INSERT INTO spectra 
                    (sample_id, polymer_type, technique, wavenumbers, intensities, 
                    metadata, quality_score, validation_status, source_database)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        sample_id,
                        polymer_type,
                        "FTIR_synthetic",
                        wavenumbers_blob,
                        intensities_blob,
                        metadata_json,
                        quality_assessment["overall_score"],
                        "validated",  # Synthetic data is pre-validated
                        "synthetic",
                    ),
                )

                # Insert aging data
                cursor.execute(
                    """
                    INSERT INTO aging_data 
                    (sample_id, aging_time, degradation_level, aging_conditions, spectral_changes)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (
                        sample_id,
                        aging_point["aging_time"],
                        aging_point["degradation_level"],
                        json.dumps(metadata["aging_conditions"]),
                        json.dumps(aging_point.get("spectral_changes", {})),
                    ),
                )

                conn.commit()

            return True

        except Exception as e:
            print(f"Error storing synthetic spectrum: {e}")
            return False

    # -///////////////////////////////////////////////////]
    def spectra_by_type(self, polymer_type: str, limit: int = 100) -> List[Dict]:
        """Retrieve spectra by polymer type from local database"""
        db_path = self.local_database_path / "polymer_spectra.db"

        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT * FROM spectra 
                WHERE polymer_type LIKE ? AND validation_status != 'rejected'
                ORDER BY quality_score DESC
                LIMIT ?
            """,
                (f"%{polymer_type}%", limit),
            )

            columns = [description[0] for description in cursor.description]
            results = [dict(zip(columns, row)) for row in cursor.fetchall()]

        return results

    # -///////////////////////////////////////////////////]
    def get_weathered_samples(self, polymer_type: Optional[str] = None) -> List[Dict]:
        """Get samples with aging/weathering data"""
        db_path = self.local_database_path / "polymer_spectra.db"

        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            query = """
                SELECT s.*, a.aging_time, a.degradation_level, a.aging_conditions
                FROM spectra s
                JOIN aging_data a ON s.sample_id = a.sample_id
                WHERE s.validation_status != 'rejected'
            """
            params = []

            if polymer_type:
                query += " AND s.polymer_type LIKE ?"
                params.append(f"%{polymer_type}%")

            query += " ORDER BY a.degradation_level"

            cursor.execute(query, params)

            columns = [description[0] for description in cursor.description]
            results = [dict(zip(columns, row)) for row in cursor.fetchall()]

        return results

    # -////////////////////////////////
    def get_database_statistics(self) -> Dict[str, Any]:
        """Get statistics about the local database"""
        db_path = self.local_database_path / "polymer_spectra.db"

        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            # Total spectra count
            cursor.execute("SELECT COUNT(*) FROM spectra")
            total_spectra = cursor.fetchone()[0]

            # By polymer type
            cursor.execute(
                """
                SELECT polymer_type, COUNT(*) as count
                FROM spectra
                GROUP BY polymer_type
                ORDER BY count DESC
            """
            )
            by_polymer_type = dict(cursor.fetchall())

            # By technique
            cursor.execute(
                """
                SELECT technique, COUNT(*) as count 
                FROM spectra 
                GROUP BY technique 
                ORDER BY count DESC
            """
            )
            by_technique = dict(cursor.fetchall())

            # By validation status
            cursor.execute(
                """
                SELECT validation_status, COUNT(*) as count 
                FROM spectra 
                GROUP BY validation_status
            """
            )
            by_validation = dict(cursor.fetchall())

            # Average quality score
            cursor.execute(
                "SELECT AVG(quality_score) FROM spectra WHERE quality_score IS NOT NULL"
            )
            avg_quality = cursor.fetchone()[0] or 0.0

            # Aging data count
            cursor.execute("SELECT COUNT(*) FROM aging_data")
            aging_samples = cursor.fetchone()[0]

            return {
                "total_spectra": total_spectra,
                "by_polymer_type": by_polymer_type,
                "by_technique": by_technique,
                "by_validation_status": by_validation,
                "average_quality_score": avg_quality,
                "aging_samples": aging_samples,
            }
