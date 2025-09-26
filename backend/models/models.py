from datetime import datetime
from typing import List, Optional, Tuple, Union
from pydantic import BaseModel, Field, validator


class FrontendSketch(BaseModel):
    points: List[List[int]]  # List of [x, y] points


class Artwork(BaseModel):
    id: str = Field(..., description="Unique identifier for the artwork")
    image_shape: Tuple[int, int] = Field(
        ..., description="(height, width) of the image"
    )
    url: str = Field(..., description="URL of the artwork image in S3")
    created_at: datetime = Field(..., description="Creation timestamp of the artwork")


# Request/Response Models
class SketchSubmissionRequest(BaseModel):
    contour: FrontendSketch


# lightweight Contour model for responses
class Contour(BaseModel):
    points: List[List[int]]  # List of [x, y] points
    artwork: Artwork
    image_shape: Tuple[int, int, int]


class SearchResponse(BaseModel):
    matching_contours: List[Contour]
    total: int
    query_time_ms: float
