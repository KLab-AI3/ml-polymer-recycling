"""
Enhanced Data Management System for POLYMEROS
Implements contextual knowledge networks and metadata preservation
"""

import os
import json
import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import numpy as np

from utils.preprocessing import preprocess_spectrum


@dataclass
class SpectralMetadata:
    """Comprehensive metadata for spectral data"""

    filename: str
    acquisition_date: Optional[str] = None
    instrument_type: str = "Raman"
    laser_wavelength: Optional[float] = None
    integration_time: Optional[float] = None
    laser_power: Optional[float] = None
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    sample_preparation: Optional[str] = None
    operator: Optional[str] = None
    data_quality_score: Optional[float] = None
    preprocessing_history: Optional[List[str]] = None

    def __post_init__(self):
        if self.preprocessing_history is None:
            self.preprocessing_history = []

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpectralMetadata":
        return cls(**data)


@dataclass
class ProvenanceRecord:
    """Complete provenance tracking for scientific reproducibility"""

    operation: str
    timestamp: str
    parameters: Dict[str, Any]
    input_hash: str
    output_hash: str
    operator: str = "system"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProvenanceRecord":
        return cls(**data)


class ContextualSpectrum:
    """Enhanced spectral data with context and provenance"""

    def __init__(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        metadata: SpectralMetadata,
        label: Optional[int] = None,
    ):
        self.x_data = x_data
        self.y_data = y_data
        self.metadata = metadata
        self.label = label
        self.provenance: List[ProvenanceRecord] = []
        self.relationships: Dict[str, List[str]] = {
            "similar_spectra": [],
            "related_samples": [],
        }

        # Calculate initial hash
        self._update_hash()

    def _calculate_hash(self, data: np.ndarray) -> str:
        """Calculate hash of numpy array for provenance tracking"""
        return hashlib.sha256(data.tobytes()).hexdigest()[:16]

    def _update_hash(self):
        """Update data hash after modifications"""
        self.data_hash = self._calculate_hash(self.y_data)

    def add_provenance(
        self, operation: str, parameters: Dict[str, Any], operator: str = "system"
    ):
        """Add provenance record for operation"""
        input_hash = self.data_hash

        record = ProvenanceRecord(
            operation=operation,
            timestamp=datetime.now().isoformat(),
            parameters=parameters,
            input_hash=input_hash,
            output_hash="",  # Will be updated after operation
            operator=operator,
        )

        self.provenance.append(record)
        return record

    def finalize_provenance(self, record: ProvenanceRecord):
        """Finalize provenance record with output hash"""
        self._update_hash()
        record.output_hash = self.data_hash

    def apply_preprocessing(self, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Apply preprocessing with full provenance tracking"""
        record = self.add_provenance("preprocessing", kwargs)

        # Apply preprocessing
        x_processed, y_processed = preprocess_spectrum(
            self.x_data, self.y_data, **kwargs
        )

        # Update data and finalize provenance
        self.x_data = x_processed
        self.y_data = y_processed
        self.finalize_provenance(record)

        # Update metadata
        if self.metadata.preprocessing_history is None:
            self.metadata.preprocessing_history = []
        self.metadata.preprocessing_history.append(
            f"preprocessing_{datetime.now().isoformat()[:19]}"
        )

        return x_processed, y_processed

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return {
            "x_data": self.x_data.tolist(),
            "y_data": self.y_data.tolist(),
            "metadata": self.metadata.to_dict(),
            "label": self.label,
            "provenance": [p.to_dict() for p in self.provenance],
            "relationships": self.relationships,
            "data_hash": self.data_hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContextualSpectrum":
        """Deserialize from dictionary"""
        spectrum = cls(
            x_data=np.array(data["x_data"]),
            y_data=np.array(data["y_data"]),
            metadata=SpectralMetadata.from_dict(data["metadata"]),
            label=data.get("label"),
        )
        spectrum.provenance = [
            ProvenanceRecord.from_dict(p) for p in data["provenance"]
        ]
        spectrum.relationships = data["relationships"]
        spectrum.data_hash = data["data_hash"]
        return spectrum


class KnowledgeGraph:
    """Knowledge graph for managing relationships between spectra and samples"""

    def __init__(self):
        self.nodes: Dict[str, ContextualSpectrum] = {}
        self.edges: Dict[str, List[Dict[str, Any]]] = {}

    def add_spectrum(self, spectrum: ContextualSpectrum, node_id: Optional[str] = None):
        """Add spectrum to knowledge graph"""
        if node_id is None:
            node_id = spectrum.data_hash

        self.nodes[node_id] = spectrum
        self.edges[node_id] = []

        # Auto-detect relationships
        self._detect_relationships(node_id)

    def _detect_relationships(self, node_id: str):
        """Automatically detect relationships between spectra"""
        current_spectrum = self.nodes[node_id]

        for other_id, other_spectrum in self.nodes.items():
            if other_id == node_id:
                continue

            # Check for similar acquisition conditions
            if self._are_similar_conditions(current_spectrum, other_spectrum):
                self.add_relationship(node_id, other_id, "similar_conditions", 0.8)

            # Check for spectral similarity (simplified)
            similarity = self._calculate_spectral_similarity(
                current_spectrum.y_data, other_spectrum.y_data
            )
            if similarity > 0.9:
                self.add_relationship(
                    node_id, other_id, "spectral_similarity", similarity
                )

    def _are_similar_conditions(
        self, spec1: ContextualSpectrum, spec2: ContextualSpectrum
    ) -> bool:
        """Check if two spectra were acquired under similar conditions"""
        meta1, meta2 = spec1.metadata, spec2.metadata

        # Check instrument type
        if meta1.instrument_type != meta2.instrument_type:
            return False

        # Check laser wavelength (if available)
        if (
            meta1.laser_wavelength
            and meta2.laser_wavelength
            and abs(meta1.laser_wavelength - meta2.laser_wavelength) > 1.0
        ):
            return False

        return True

    def _calculate_spectral_similarity(
        self, spec1: np.ndarray, spec2: np.ndarray
    ) -> float:
        """Calculate similarity between two spectra"""
        if len(spec1) != len(spec2):
            return 0.0

        # Normalize spectra
        spec1_norm = (spec1 - np.min(spec1)) / (np.max(spec1) - np.min(spec1) + 1e-8)
        spec2_norm = (spec2 - np.min(spec2)) / (np.max(spec2) - np.min(spec2) + 1e-8)

        # Calculate correlation coefficient
        correlation = np.corrcoef(spec1_norm, spec2_norm)[0, 1]
        return max(0.0, correlation)

    def add_relationship(
        self, node1: str, node2: str, relationship_type: str, weight: float
    ):
        """Add relationship between two nodes"""
        edge = {
            "target": node2,
            "type": relationship_type,
            "weight": weight,
            "timestamp": datetime.now().isoformat(),
        }

        self.edges[node1].append(edge)

        # Add reverse edge
        reverse_edge = {
            "target": node1,
            "type": relationship_type,
            "weight": weight,
            "timestamp": datetime.now().isoformat(),
        }

        if node2 in self.edges:
            self.edges[node2].append(reverse_edge)

    def get_related_spectra(
        self, node_id: str, relationship_type: Optional[str] = None
    ) -> List[str]:
        """Get spectra related to given node"""
        if node_id not in self.edges:
            return []

        related = []
        for edge in self.edges[node_id]:
            if relationship_type is None or edge["type"] == relationship_type:
                related.append(edge["target"])

        return related

    def export_knowledge_graph(self, filepath: str):
        """Export knowledge graph to JSON file"""
        export_data = {
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "edges": self.edges,
            "metadata": {
                "created": datetime.now().isoformat(),
                "total_nodes": len(self.nodes),
                "total_edges": sum(len(edges) for edges in self.edges.values()),
            },
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2)


class EnhancedDataManager:
    """Main data management interface for POLYMEROS"""

    def __init__(self, cache_dir: str = "data_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.knowledge_graph = KnowledgeGraph()
        self.quality_thresholds = {
            "min_intensity": 10.0,
            "min_signal_to_noise": 3.0,
            "max_baseline_drift": 0.1,
        }

    def load_spectrum_with_context(
        self, filepath: str, metadata: Optional[Dict[str, Any]] = None
    ) -> ContextualSpectrum:
        """Load spectrum with automatic metadata extraction and quality assessment"""
        from scripts.plot_spectrum import load_spectrum

        # Load raw data
        x_data, y_data = load_spectrum(filepath)

        # Extract metadata
        if metadata is None:
            metadata = self._extract_metadata_from_file(filepath)

        spectral_metadata = SpectralMetadata(
            filename=os.path.basename(filepath), **metadata
        )

        # Create contextual spectrum
        spectrum = ContextualSpectrum(
            np.array(x_data), np.array(y_data), spectral_metadata
        )

        # Assess data quality
        quality_score = self._assess_data_quality(np.array(y_data))
        spectrum.metadata.data_quality_score = quality_score

        # Add to knowledge graph
        self.knowledge_graph.add_spectrum(spectrum)

        return spectrum

    def _extract_metadata_from_file(self, filepath: str) -> Dict[str, Any]:
        """Extract metadata from filename and file properties"""
        filename = os.path.basename(filepath)

        metadata = {
            "acquisition_date": datetime.fromtimestamp(
                os.path.getmtime(filepath)
            ).isoformat(),
            "instrument_type": "Raman",  # Default
        }

        # Extract information from filename patterns
        if "785nm" in filename.lower():
            metadata["laser_wavelength"] = "785.0"
        elif "532nm" in filename.lower():
            metadata["laser_wavelength"] = "532.0"

        return metadata

    def _assess_data_quality(self, y_data: np.ndarray) -> float:
        """Assess spectral data quality using multiple metrics"""
        scores = []

        # Signal intensity check
        max_intensity = np.max(y_data)
        if max_intensity >= self.quality_thresholds["min_intensity"]:
            scores.append(min(1.0, max_intensity / 1000.0))
        else:
            scores.append(0.0)

        # Signal-to-noise ratio estimation
        signal = np.mean(y_data)
        noise = np.std(y_data[y_data < np.percentile(y_data, 10)])
        snr = signal / (noise + 1e-8)

        if snr >= self.quality_thresholds["min_signal_to_noise"]:
            scores.append(min(1.0, snr / 10.0))
        else:
            scores.append(0.0)

        # Baseline stability
        baseline_variation = np.std(y_data) / (np.mean(y_data) + 1e-8)
        baseline_score = max(
            0.0,
            1.0 - baseline_variation / self.quality_thresholds["max_baseline_drift"],
        )
        scores.append(baseline_score)

        return float(np.mean(scores))

    def preprocess_with_tracking(
        self, spectrum: ContextualSpectrum, **preprocessing_params
    ) -> ContextualSpectrum:
        """Apply preprocessing with full tracking"""
        spectrum.apply_preprocessing(**preprocessing_params)
        return spectrum

    def get_preprocessing_recommendations(
        self, spectrum: ContextualSpectrum
    ) -> Dict[str, Any]:
        """Provide intelligent preprocessing recommendations based on data characteristics"""
        recommendations = {}

        y_data = spectrum.y_data

        # Baseline correction recommendation
        baseline_variation = np.std(np.diff(y_data))
        if baseline_variation > 0.05:
            recommendations["do_baseline"] = True
            recommendations["degree"] = 3 if baseline_variation > 0.1 else 2
        else:
            recommendations["do_baseline"] = False

        # Smoothing recommendation
        noise_level = np.std(y_data[y_data < np.percentile(y_data, 20)])
        if noise_level > 0.01:
            recommendations["do_smooth"] = True
            recommendations["window_length"] = 11 if noise_level > 0.05 else 7
        else:
            recommendations["do_smooth"] = False

        # Normalization is generally recommended
        recommendations["do_normalize"] = True

        return recommendations

    def save_session(self, session_name: str):
        """Save current data management session"""
        session_file = self.cache_dir / f"{session_name}_session.json"
        self.knowledge_graph.export_knowledge_graph(str(session_file))

    def load_session(self, session_name: str):
        """Load saved data management session"""
        session_file = self.cache_dir / f"{session_name}_session.json"

        if session_file.exists():
            with open(session_file, "r") as f:
                data = json.load(f)

            # Reconstruct knowledge graph
            for node_id, node_data in data["nodes"].items():
                spectrum = ContextualSpectrum.from_dict(node_data)
                self.knowledge_graph.nodes[node_id] = spectrum

            self.knowledge_graph.edges = data["edges"]
