import numpy as np
import cv2
from models import ProcrustesResult


def compute_hu_moments(contour_points: np.ndarray) -> np.ndarray:
    """Compute Hu moments for a given contour."""
    try:
        contour_points = contour_points.astype(np.float32)
        moments = cv2.moments(contour_points)
        if moments["m00"] == 0:
            return np.zeros(7, dtype=np.float32)

        hu_moments = cv2.HuMoments(moments).flatten()
        # Log scale transform for better numerical stability and preserve sign
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
        return hu_moments
    except Exception as e:
        print(f"Error computing Hu moments: {e}")
        return np.zeros(7, dtype=np.float32)


def compute_enhanced_features(contour_points: np.ndarray) -> np.ndarray:
    """Compute enhanced feature vector: Hu moments + additional shape descriptors"""
    try:
        hu_moments = compute_hu_moments(contour_points)
        if hu_moments is None or len(hu_moments) != 7 or np.sum(hu_moments) == 0:
            return np.zeros(7, dtype=np.float32)

        # Additional shape features
        area = cv2.contourArea(contour_points)
        perimeter = cv2.arcLength(contour_points, closed=True)

        # Compactness (circularity)
        compactness = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

        # Aspect ratio from bounding rectangle
        rect = cv2.minAreaRect(contour_points)
        width, height = rect[1]
        aspect_ratio = max(width, height) / (min(width, height) + 1e-10)

        # Extent (contour area / bounding rectangle area)
        x, y, w, h = cv2.boundingRect(contour_points)
        extent = area / (w * h) if (w * h) > 0 else 0

        # Solidity (contour area / convex hull area)
        hull = cv2.convexHull(contour_points)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0

        # Contour length (normalized)
        normalized_length = len(contour_points)

        # Combine all features
        additional_features = np.array(
            [
                compactness,
                aspect_ratio,
                extent,
                solidity,
                normalized_length,
                np.log10(area + 1),
                np.log10(perimeter + 1),
                np.log10(len(contour_points) + 1),
            ],
            dtype=np.float32,
        )

        # Concatenate Hu moments + additional features
        enhanced_features = np.concatenate([hu_moments, additional_features])
        return enhanced_features

    except Exception as e:
        # Keep behaviour consistent with previous inline implementation
        print(f"Error computing enhanced features: {e}")
        return np.zeros(15, dtype=np.float32)


def align_contours_with_transform(sketch_pts, target_pts) -> ProcrustesResult:
    """
    Align sketch_pts to target_pts using Procrustes analysis.
    Returns full transformation parameters needed for frontend positioning.
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
