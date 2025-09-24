from pydantic import BaseModel, Field, model_validator
import numpy as np
import cv2
from typing import List, Tuple, Optional


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
        self.area = cv2.contourArea(self.points) / (self.image_shape[0] * self.image_shape[1])

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
            except ValueError as e:
                pass
