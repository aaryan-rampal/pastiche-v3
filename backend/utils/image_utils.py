import numpy as np
import cv2


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
