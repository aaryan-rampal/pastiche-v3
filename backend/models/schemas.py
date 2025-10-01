from pydantic import BaseModel, Field, model_validator, field_validator
from loguru import logger
import faiss
import pickle
from tqdm import tqdm
import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict
import sys
import os

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.image_utils import compute_hu_moments


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
        default=1e-4, description="Minimum allowed normalized length"
    )

    @model_validator(mode="after")
    def compute_derived_fields(self) -> "Contour":
        """Compute derived numeric fields for the contour.

        Returns:
            Contour: the same model instance with `length` and `area` populated.

        Raises:
            ValueError: if contour is too short or has too small area based on thresholds.
        """
        self.length = self.points.shape[0] / (self.image_shape[0] * self.image_shape[1])
        self.area = cv2.contourArea(self.points) / (
            self.image_shape[0] * self.image_shape[1]
        )

        if self.length < self.min_length:
            raise ValueError(f"Contour too short: {self.length}")

        if self.area < self.min_area:
            raise ValueError(f"Contour area too small: {self.area}")

        return self


class ImageModel(BaseModel):
    image_id: str = Field(..., description="Identifier for the image")
    image_shape: Tuple[int, int] = Field(
        ..., description="(height, width) of the image"
    )
    contours: List[Contour] = Field(
        default_factory=list, description="List of contours for the image"
    )

    def add_contours(self, contours: List[np.ndarray]) -> None:
        """Add contours (numpy arrays) to this ImageModel.

        Args:
            contours: list of numpy arrays, each shape (N,2).

        Behavior:
            Appends valid `Contour` models to `self.contours`. Invalid contours
            (that fail validation) are skipped silently.
        """
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
                # Invalid contours are ignored
                pass


class PointInput(BaseModel):
    """Input model for a list of points representing a contour."""

    points: List[List[float]] = Field(..., description="List of [x, y] coordinates")

    @field_validator("points")
    @classmethod
    def validate_points(cls, v):
        """Validate that points is a non-empty list of [x, y] coordinates."""
        if not v:
            raise ValueError("Points list cannot be empty")
        for point in v:
            if len(point) != 2:
                raise ValueError("Each point must be a list of [x, y] coordinates")
        return v

    def to_numpy(self) -> np.ndarray:
        """Convert points to numpy array."""
        return np.array(self.points, dtype=np.float32)


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
    def __init__(self, use_weighted_distance: bool = True) -> None:
        """FAISS index wrapper for contour features.

        Attributes set on build:
            - self.index: faiss index instance
            - self.contour_metadata: List[Tuple[str,int]] mapping index rows to (img_path, contour_idx)
            - self.hu_features: numpy array of feature vectors used to build the index
        """
        self.index: Optional[faiss.Index] = None
        self.contour_metadata_s3: List[Tuple[str, int]] = []  # S3 path version

        self.hu_features: Optional[np.ndarray] = None
        self.use_weighted_distance: bool = use_weighted_distance

    def build_index(self, image_models: Dict[str, ImageModel]) -> None:
        """Build FAISS index from all contours in image models.

        Side effects:
            - sets self.index (faiss Index)
            - sets self.hu_features (np.ndarray)
            - sets self.contour_metadata (list of (img_path, contour_idx))
        Returns:
            None
        """
        print("Computing enhanced features for all contours...")

        feature_vectors = []
        local_metadata = []
        s3_metadata = []

        for img_path, img_model in tqdm(image_models.items()):
            for contour_idx, contour in enumerate(img_model.contours):
                hu_moments = compute_hu_moments(contour.points)

                # Skip invalid features
                if not np.any(np.isnan(hu_moments)) and not np.any(
                    np.isinf(hu_moments)
                ):
                    feature_vectors.append(hu_moments)
                    local_metadata.append((img_path, contour_idx))

                    s3_path = "/".join(img_path.split("/")[-2:])
                    s3_metadata.append((s3_path, contour_idx))

        if not feature_vectors:
            raise ValueError("No valid features computed")

        # Convert to numpy array
        self.hu_features = np.array(feature_vectors, dtype=np.float32)
        self.contour_metadata = local_metadata
        self.contour_metadata_s3 = s3_metadata

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
            f"Built FAISS index with {self.hu_features.shape[0]} contours, {dimension}D features"
        )

    def search_similar_contours(
        self, sketch_contour: np.ndarray, k: int
    ) -> List[Tuple[float, str, int]]:
        """Find k most similar contours using enhanced features.

        Args:
            sketch_contour: numpy array of sketch points (N,2) to query
            k: number of neighbors to return

        Returns:
            List of tuples (distance, img_path, contour_idx). Distance uses
            L2 as returned by FAISS IndexFlatL2 (lower is more similar).
        """
        if self.index is None:
            raise ValueError("Index not built yet")

        sketch_moments = compute_hu_moments(sketch_contour)
        if np.any(np.isnan(sketch_moments)) or np.any(np.isinf(sketch_moments)):
            print("Warning: Invalid features for sketch")
            return []

        # Apply same normalization as training data
        # logger.info(self.use_weighted_distance)
        # if self.use_weighted_distance:
        #     sketch_moments = (sketch_moments - self.feature_mean) / self.feature_std

        # Search FAISS index
        distances, indices = self.index.search(
            sketch_moments.reshape(1, -1), k
        )  # Return metadata for found contours
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.contour_metadata_s3):
                img_path, contour_idx = self.contour_metadata_s3[idx]
                results.append((distances[0][i], img_path, contour_idx))

        return results

    def save_index(self, filepath: str) -> None:
        """Save the FAISS index and metadata to disk.

        Files produced:
            - {filepath}.faiss : binary FAISS index
            - {filepath}_metadata.pkl : pickled metadata dict with keys
              'contour_metadata' and 'hu_features'
        """
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

        with open(f"{filepath}_metadata_s3.pkl", "wb") as f:
            pickle.dump(
                {
                    "contour_metadata_s3": self.contour_metadata_s3,
                },
                f,
            )
        print(f"Saved S3 metadata to {filepath}_metadata_s3.pkl")

    def load_index(self, filepath: str) -> None:
        """Load the FAISS index and metadata from disk.

        After calling this, `self.index`, `self.contour_metadata` and
        `self.hu_features` will be populated.
        """
        self.index = faiss.read_index(f"{filepath}.faiss")

        with open(f"{filepath}_metadata_s3.pkl", "rb") as f:
            data = pickle.load(f)
            self.contour_metadata_s3 = data["contour_metadata_s3"]
        print(f"Loaded S3 metadata from {filepath}_metadata_s3.pkl")
