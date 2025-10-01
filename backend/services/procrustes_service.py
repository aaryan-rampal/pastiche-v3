"""Service for Procrustes analysis and shape alignment."""

from loguru import logger

import numpy as np
from typing import List, Tuple
import cv2
from models.schemas import ProcrustesResult
from services.contour_service import extract_contours_from_image_bytes
from core.s3_settings import get_s3_client, AWS_S3_BUCKET


def align_contours_with_transform(
    sketch_pts: np.ndarray, target_pts: np.ndarray
) -> ProcrustesResult:
    """
    Align sketch_pts to target_pts using Procrustes analysis.
    Returns full transformation parameters needed for frontend positioning.

    Args:
        sketch_pts: Sketch contour points (N, 2)
        target_pts: Target contour points (M, 2)

    Returns:
        ProcrustesResult with transformation parameters and disparity score
    """
    # Resample to same number of points
    N = min(len(sketch_pts), len(target_pts))
    sketch = sketch_pts[np.linspace(0, len(sketch_pts) - 1, N, dtype=int)].astype(float)
    target = target_pts[np.linspace(0, len(target_pts) - 1, N, dtype=int)].astype(float)

    # 1. Translation: compute centroids
    sketch_centroid = sketch.mean(axis=0)
    target_centroid = target.mean(axis=0)
    translation = target_centroid - sketch_centroid  # How much to move sketch

    # 2. Center both shapes at origin
    sketch_centered = sketch - sketch_centroid
    target_centered = target - target_centroid

    # 3. Scale: compute norms
    sketch_norm = np.sqrt((sketch_centered**2).sum())
    target_norm = np.sqrt((target_centered**2).sum())
    scale = (
        target_norm / sketch_norm if sketch_norm > 0 else 1.0
    )  # How much to scale sketch

    # 4. Normalize to unit scale
    sketch_normalized = (
        sketch_centered / sketch_norm if sketch_norm > 0 else sketch_centered
    )
    target_normalized = (
        target_centered / target_norm if target_norm > 0 else target_centered
    )

    # 5. Rotation: SVD to find optimal rotation matrix
    M = sketch_normalized.T @ target_normalized
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt  # 2x2 rotation matrix

    # Ensure proper rotation (det = 1, not reflection)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt

    # 6. Convert rotation matrix to angle
    rotation_angle = np.arctan2(R[1, 0], R[0, 0])  # radians
    rotation_degrees = np.degrees(rotation_angle)

    # 7. Apply full transform to sketch for disparity calculation
    sketch_transformed_normalized = sketch_normalized @ R.T
    disparity = np.sqrt(
        ((sketch_transformed_normalized - target_normalized) ** 2).sum()
    )

    # 8. Compute fully transformed sketch points (for visualization)
    # Apply: center -> rotate -> scale -> translate
    sketch_rotated = sketch_centered @ R.T
    sketch_scaled = sketch_rotated * scale
    transformed_sketch_points = sketch_scaled + target_centroid

    return ProcrustesResult(
        disparity=float(disparity),
        translation={"x": float(translation[0]), "y": float(translation[1])},
        scale=float(scale),
        rotation_degrees=float(rotation_degrees),
        rotation_radians=float(rotation_angle),
        rotation_matrix=R.tolist(),
        sketch_centroid={
            "x": float(sketch_centroid[0]),
            "y": float(sketch_centroid[1]),
        },
        target_centroid={
            "x": float(target_centroid[0]),
            "y": float(target_centroid[1]),
        },
        transformed_sketch_points=transformed_sketch_points,
    )


class ProcrustesService:
    """Service for computing Procrustes alignment on contour candidates."""

    def __init__(self, s3_bucket: str = None):
        """Initialize Procrustes service.

        Args:
            s3_bucket: S3 bucket name where artwork images are stored (defaults to AWS_S3_BUCKET from env)
        """
        self.s3_client = get_s3_client()
        self.s3_bucket = s3_bucket or AWS_S3_BUCKET

    def load_image_from_s3(self, s3_key: str) -> np.ndarray:
        """Load image from S3 and return as grayscale numpy array.

        Args:
            s3_key: S3 key for the image (e.g., 'baroque/image.jpg')

        Returns:
            Grayscale image as numpy array
        """
        if not s3_key.startswith("artworks/"):
            s3_key = "artworks/" + s3_key
        logger.info(f"Loading image from S3: {self.s3_bucket}/{s3_key}")
        response = self.s3_client.get_object(Bucket=self.s3_bucket, Key=s3_key)
        image_bytes = response["Body"].read()

        # Decode image
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise ValueError(f"Failed to decode image from S3: {s3_key}")

        return image

    def extract_contour_from_s3_image(
        self, s3_key: str, contour_idx: int
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Extract specific contour from an S3 image.

        Args:
            s3_key: S3 key for the image
            contour_idx: Index of contour to extract

        Returns:
            Tuple of (contour_points, image_shape)
        """
        image = self.load_image_from_s3(s3_key)
        contours = extract_contours_from_image_bytes(
            cv2.imencode(".png", image)[1].tobytes()
        )

        if contour_idx >= len(contours):
            raise ValueError(
                f"Contour index {contour_idx} out of range (found {len(contours)} contours)"
            )

        return contours[contour_idx], image.shape

    def compute_procrustes_batch(
        self,
        sketch_contour: np.ndarray,
        faiss_results: List[Tuple[float, str, int]],
        top_k: int = 10,
    ) -> List[Tuple[ProcrustesResult, str, np.ndarray, float]]:
        """Compute Procrustes alignment for FAISS candidates.

        Args:
            sketch_contour: Sketch contour points (N, 2)
            faiss_results: List of (hu_distance, img_path, contour_idx) from FAISS
            top_k: Number of top results to return after Procrustes refinement

        Returns:
            List of (ProcrustesResult, img_path, contour_points, hu_distance) sorted by disparity
        """
        procrustes_results = []
        logger.info(f"Computing Procrustes for {len(faiss_results)} candidates")

        for hu_distance, img_path, contour_idx in faiss_results:
            try:
                # Extract contour from S3 image
                contour_points, _ = self.extract_contour_from_s3_image(
                    img_path, contour_idx
                )

                # Compute Procrustes alignment
                result = align_contours_with_transform(sketch_contour, contour_points)

                procrustes_results.append(
                    (result, img_path, contour_points, hu_distance)
                )
            except Exception as e:
                print(f"Error processing {img_path}[{contour_idx}]: {e}")
                continue

        # Sort by Procrustes disparity (lower is better)
        procrustes_results.sort(key=lambda x: x[0].disparity)

        return procrustes_results[:top_k]
