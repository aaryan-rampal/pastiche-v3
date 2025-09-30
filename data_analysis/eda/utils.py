import numpy as np
import cv2


def compute_enhanced_features(contour_points: np.ndarray) -> np.ndarray:
    """Compute enhanced feature vector: Hu moments + additional shape descriptors

    This is the same implementation previously living in hu_faiss.py. It
    returns a fixed-size vector (7 Hu moments + 8 additional shape features).
    """
    try:
        contour_points = contour_points.astype(np.float32)

        # 1. Hu moments (7 features)
        moments = cv2.moments(contour_points)
        if moments["m00"] == 0:
            return np.zeros(15, dtype=np.float32)  # Increased feature size

        hu_moments = cv2.HuMoments(moments).flatten()
        # Log scale transform for better numerical stability and preserve sign
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)

        # 2. Additional shape features
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
