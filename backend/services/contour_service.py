"""Service for extracting contours from images."""
import cv2
import numpy as np
from typing import List


def extract_contours(
    image: np.ndarray, lo: int = 150, hi: int = 200, min_length: int = 50
) -> List[np.ndarray]:
    """Extract contours from grayscale image using Canny edge detection.

    Args:
        image: Grayscale image as numpy array
        lo: Lower threshold for Canny edge detection
        hi: Higher threshold for Canny edge detection
        min_length: Minimum contour length to keep

    Returns:
        List of contour arrays (each shape (N, 2))
    """
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    canny_img = cv2.Canny(blurred, lo, hi)
    contours, _ = cv2.findContours(canny_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours_squeezed = [c.squeeze() for c in contours if len(c) > min_length]
    return contours_squeezed


def extract_contours_from_image_bytes(
    image_bytes: bytes, lo: int = 150, hi: int = 200, min_length: int = 50
) -> List[np.ndarray]:
    """Extract contours from image bytes.

    Args:
        image_bytes: Image data as bytes (PNG, JPEG, etc.)
        lo: Lower threshold for Canny edge detection
        hi: Higher threshold for Canny edge detection
        min_length: Minimum contour length to keep

    Returns:
        List of contour arrays (each shape (N, 2))
    """
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError("Failed to decode image")

    return extract_contours(image, lo, hi, min_length)
