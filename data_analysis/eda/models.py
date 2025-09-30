from pydantic import BaseModel, Field, model_validator
import faiss
import pickle
from tqdm import tqdm
import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict
from hu_faiss import compute_enhanced_features


class Contour(BaseModel):
    points: np.ndarray = Field(
        ..., description="Array of (x, y) points for the contour"
    )
    image_id: str = Field(..., description="Identifier for the source image")
    image_shape: Tuple[int, int] = Field(
        ..., description="(height, width) of the source image"
    )

    model_config = {
        "arbitrary_types_allowed": True  # Allow numpy arrays
    }

    length: Optional[float] = Field(
        default=None, description="Normalized number of contour points"
    )
    area: Optional[float] = Field(
        default=None, description="Normalized area of the contour"
    )
    max_area: float = Field(default=0.1, description="Maximum allowed normalized area")
    max_length: float = Field(
        default=0.005, description="Maximum allowed normalized length"
    )
    min_area: float = Field(default=1e-5, description="Minimum allowed normalized area")
    min_length: float = Field(
        default=1e-5, description="Minimum allowed normalized length"
    )

    @model_validator(mode="after")
    def compute_derived_fields(self):
        self.length = self.points.shape[0] / (self.image_shape[0] * self.image_shape[1])
        self.area = cv2.contourArea(self.points) / (
            self.image_shape[0] * self.image_shape[1]
        )

        if self.length < self.min_length:
            raise ValueError(f"Contour too short: {self.length}")
        # if self.length > self.max_length:
        #     raise ValueError(f"Contour too long: {self.length}")

        if self.area < self.min_area:
            raise ValueError(f"Contour area too small: {self.area}")
        # if self.area > self.max_area:
        #     raise ValueError(f"Contour area too large: {self.area}")
        return self


class ImageModel(BaseModel):
    image_id: str = Field(..., description="Identifier for the image")
    image_shape: Tuple[int, int] = Field(
        ..., description="(height, width) of the image"
    )
    contours: List[Contour] = Field(
        default_factory=list, description="List of contours for the image"
    )

    def add_contours(self, contours: List[np.ndarray]):
        # contours is a list of np.ndarray of shape (N, 2)
        for contour_points in contours:
            try:
                contour = Contour(
                    points=contour_points,
                    image_id=self.image_id,
                    image_shape=self.image_shape,
                )
                self.contours.append(contour)
            except ValueError:
                pass


class ProcrustesResult(BaseModel):
    """Result of Procrustes analysis with full transformation parameters"""

    disparity: float = Field(
        ..., description="Procrustes distance/error after alignment"
    )
    translation: Dict[str, float] = Field(
        ..., description="Translation vector (x, y) to align sketch to target"
    )
    scale: float = Field(..., description="Scale factor to apply to sketch")
    rotation_degrees: float = Field(..., description="Rotation angle in degrees")
    rotation_radians: float = Field(..., description="Rotation angle in radians")
    rotation_matrix: List[List[float]] = Field(
        ..., description="2x2 rotation matrix from SVD"
    )
    sketch_centroid: Dict[str, float] = Field(
        ..., description="Centroid of sketch contour (x, y)"
    )
    target_centroid: Dict[str, float] = Field(
        ..., description="Centroid of target contour (x, y)"
    )
    transformed_sketch_points: Optional[np.ndarray] = Field(
        default=None,
        description="Sketch points after full transformation (for visualization)",
    )

    model_config = {
        "arbitrary_types_allowed": True  # Allow numpy arrays
    }


class ContourFAISSIndex:
    def __init__(self, use_weighted_distance=True):
        self.index = None
        self.contour_metadata = []  # List of (img_path, contour_idx, contour) tuples
        self.hu_features = []
        self.use_weighted_distance = use_weighted_distance

    def build_index(self, image_models: dict[str, ImageModel]):
        """Build FAISS index from all contours in image models"""
        print("Computing enhanced features for all contours...")

        feature_vectors = []
        metadata = []

        for img_path, img_model in tqdm(image_models.items()):
            for contour_idx, contour in enumerate(img_model.contours):
                enhanced_features = compute_enhanced_features(contour.points)

                # Skip invalid features
                if not np.any(np.isnan(enhanced_features)) and not np.any(
                    np.isinf(enhanced_features)
                ):
                    feature_vectors.append(enhanced_features)
                    metadata.append((img_path, contour_idx))

        if not feature_vectors:
            raise ValueError("No valid features computed")

        # Convert to numpy array
        self.hu_features = np.array(feature_vectors, dtype=np.float32)
        self.contour_metadata = metadata

        # Optionally normalize features for better FAISS performance
        if self.use_weighted_distance:
            # Normalize each feature dimension
            mean = np.mean(self.hu_features, axis=0)
            std = np.std(self.hu_features, axis=0) + 1e-8
            self.hu_features = (self.hu_features - mean) / std
            self.feature_mean = mean
            self.feature_std = std

        # Build FAISS index (L2 distance)
        dimension = len(feature_vectors[0])  # Enhanced features dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.hu_features)

        print(
            f"Built FAISS index with {len(self.hu_features)} contours, {dimension}D features"
        )

    def search_similar_contours(self, sketch_contour: np.ndarray, k: int = 100):
        """Find k most similar contours using enhanced features"""
        if self.index is None:
            raise ValueError("Index not built yet")

        sketch_features = compute_enhanced_features(sketch_contour)
        if np.any(np.isnan(sketch_features)) or np.any(np.isinf(sketch_features)):
            print("Warning: Invalid features for sketch")
            return []

        # Apply same normalization as training data
        if self.use_weighted_distance:
            sketch_features = (sketch_features - self.feature_mean) / self.feature_std

        # Search FAISS index
        distances, indices = self.index.search(
            sketch_features.reshape(1, -1), k
        )  # Return metadata for found contours
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.contour_metadata):
                img_path, contour_idx = self.contour_metadata[idx]
                results.append((distances[0][i], img_path, contour_idx))

        return results

    def save_index(self, filepath: str):
        """Save the FAISS index and metadata"""
        faiss.write_index(self.index, f"{filepath}.faiss")
        with open(f"{filepath}_metadata.pkl", "wb") as f:
            pickle.dump(
                {
                    "contour_metadata": self.contour_metadata,
                    "hu_features": self.hu_features,
                },
                f,
            )
        print(f"Saved index to {filepath}")

    def load_index(self, filepath: str):
        """Load the FAISS index and metadata"""
        self.index = faiss.read_index(f"{filepath}.faiss")
        with open(f"{filepath}_metadata.pkl", "rb") as f:
            data = pickle.load(f)
            self.contour_metadata = data["contour_metadata"]
            self.hu_features = data["hu_features"]
        print(f"Loaded index from {filepath}")
