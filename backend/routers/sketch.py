"""API router for sketch matching endpoints."""

import matplotlib.pyplot as plt
import os

from loguru import logger
from core.config import settings
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np
import io

from services.contour_service import extract_contours_from_image_bytes
from services.faiss_service import FAISSService
from services.procrustes_service import ProcrustesService
from models.schemas import PointInput

router = APIRouter(prefix="/api/sketch", tags=["sketch"])

# Initialize services (singleton pattern)
faiss_service = FAISSService()
procrustes_service = ProcrustesService()


class TransformParams(BaseModel):
    """Transformation parameters for aligning sketch to artwork."""

    scale: float = Field(..., description="Scale factor")
    rotation_degrees: float = Field(..., description="Rotation in degrees")
    rotation_radians: float = Field(..., description="Rotation in radians")
    translation: dict = Field(..., description="Translation vector {x, y}")
    sketch_centroid: dict = Field(..., description="Sketch centroid {x, y}")
    target_centroid: dict = Field(..., description="Target centroid {x, y}")


class MatchResult(BaseModel):
    """Single artwork match result."""

    artwork_path: str = Field(..., description="S3 path to artwork image")
    artwork_url: Optional[str] = Field(
        None, description="Public URL to artwork (if available)"
    )
    procrustes_score: float = Field(
        ..., description="Procrustes disparity (lower is better)"
    )
    hu_distance: float = Field(..., description="Hu moment distance from FAISS")
    transform: TransformParams = Field(
        ..., description="Transform parameters for alignment"
    )
    contour_idx: int = Field(..., description="Index of matched contour in artwork")
    matched_contour_points: List[List[float]] = Field(
        ..., description="Points of the matched contour from the artwork [[x, y], ...]"
    )


class MatchResponse(BaseModel):
    """Response containing matched artworks."""

    matches: List[MatchResult] = Field(..., description="List of matched artworks")
    sketch_contours_found: int = Field(
        ..., description="Number of contours extracted from sketch"
    )
    faiss_candidates: int = Field(
        ..., description="Number of FAISS candidates evaluated"
    )


class MatchRequestBody(BaseModel):
    """Request body for matching with points input."""

    points: PointInput = Field(
        ..., description="List of points representing the sketch contour"
    )
    top_k_faiss: int = Field(
        default=settings.faiss_top_k,
        description="Number of candidates to retrieve from FAISS",
    )
    top_k_final: int = Field(
        default=settings.procrustes_top_k,
        description="Number of final results to return after Procrustes",
    )


async def quick_helper(sketch_contour: np.ndarray) -> None:
    """Helper function to save sketch points as matplotlib image."""

    # Create output directory if it doesn't exist
    output_dir = "data_analysis/sketches"
    os.makedirs(output_dir, exist_ok=True)

    # Create timestamp for unique filename
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the sketch points
    if len(sketch_contour) > 0:
        points = np.array(sketch_contour)
        ax.plot(points[:, 0], points[:, 1], "b-", linewidth=2, markersize=4)
        ax.scatter(points[:, 0], points[:, 1], c="red", s=20, zorder=5)

        # Set equal aspect ratio and limits
        ax.set_aspect("equal")
        ax.set_xlim(points[:, 0].min() - 10, points[:, 0].max() + 10)
        ax.set_ylim(points[:, 1].min() - 10, points[:, 1].max() + 10)

        # Remove axes for cleaner look
        ax.axis("off")

        # Save the plot
        output_path = os.path.join(output_dir, f"sketch_{timestamp}.png")
        plt.savefig(output_path, dpi=150, bbox_inches="tight", transparent=True)
        plt.close()

        logger.info(f"Saved sketch visualization to {output_path}")
    else:
        logger.warning("No points to visualize")


@router.post("/match-points", response_model=MatchResponse)
async def match_sketch_points(
    request: MatchRequestBody,
) -> MatchResponse:
    """Match a sketch to artworks using points input.

    Args:
        request: Request body containing points and search parameters

    Returns:
        MatchResponse with matched artworks and transformation parameters
    """
    try:
        # Convert points to numpy array
        logger.debug("Converting points to numpy array")
        sketch_contour = request.points.to_numpy()

        if len(sketch_contour) < 3:
            raise HTTPException(
                status_code=400,
                detail="At least 3 points are required for matching.",
            )
        logger.debug(f"Sketch contour has {len(sketch_contour)} points")

        # reverse y-axis to match image coordinate system
        sketch_contour[:, 1] = -sketch_contour[:, 1]

        if settings.save_sketch_debug:
            await quick_helper(sketch_contour)
            logger.info("Saved sketch points visualization for debugging")

        # Stage 1: FAISS search for initial candidates
        faiss_results = faiss_service.search_similar_contours(
            sketch_contour, k=request.top_k_faiss
        )
        logger.debug(f"FAISS results: {faiss_results}")
        faiss_results = faiss_results[: settings.faiss_top_k]
        logger.info(f"FAISS returned {len(faiss_results)} candidates")

        if not faiss_results:
            raise HTTPException(
                status_code=404,
                detail="No matches found in FAISS index.",
            )

        # Stage 2: Procrustes refinement
        procrustes_results = procrustes_service.compute_procrustes_batch(
            sketch_contour, faiss_results, top_k=request.top_k_final
        )

        # Build all candidate matches
        matches = []
        for (
            procrustes_result,
            img_path,
            contour_points,
            hu_distance,
        ) in procrustes_results:
            # Extract contour index from faiss_results
            contour_idx = next(
                (idx for _, path, idx in faiss_results if path == img_path), 0
            )

            match = MatchResult(
                artwork_path=img_path,
                artwork_url=f"/api/sketch/image/{img_path}",  # Use proxy endpoint instead of direct S3
                procrustes_score=procrustes_result.disparity,
                hu_distance=float(hu_distance),
                transform=TransformParams(
                    scale=procrustes_result.scale,
                    rotation_degrees=procrustes_result.rotation_degrees,
                    rotation_radians=procrustes_result.rotation_radians,
                    translation=procrustes_result.translation,
                    sketch_centroid=procrustes_result.sketch_centroid,
                    target_centroid=procrustes_result.target_centroid,
                ),
                contour_idx=contour_idx,
                matched_contour_points=contour_points.tolist(),  # Convert numpy array to list
            )
            matches.append(match)

        # Use exponential distribution to select one match
        # Earlier results (better Procrustes scores) are weighted more heavily
        if len(matches) > 0:
            # Generate exponential weights (decay rate = 1.0)
            weights = np.exp(-np.arange(len(matches)))
            # Normalize to probabilities
            probabilities = weights / weights.sum()
            # Randomly select one index based on probabilities
            selected_idx = np.random.choice(len(matches), p=probabilities)
            selected_match = matches[selected_idx]
            matches = [selected_match]
            logger.info(f"Selected match {selected_idx} from {len(procrustes_results)} candidates using exponential distribution")

        logger.info(f"Returning {len(matches)} match(es)")

        return MatchResponse(
            matches=matches,
            sketch_contours_found=1,
            faiss_candidates=len(faiss_results),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.get("/image/{path:path}")
async def get_artwork_image(path: str):
    """Proxy endpoint to serve artwork images from S3.

    Args:
        path: S3 key path (e.g., 'artworks/baroque/image.jpg')

    Returns:
        StreamingResponse with the image bytes
    """
    try:
        # Ensure path starts with 'artworks/'
        if not path.startswith("artworks/"):
            path = "artworks/" + path

        logger.info(f"Fetching image from S3: {path}")

        # Get image from S3
        response = procrustes_service.s3_client.get_object(
            Bucket=procrustes_service.s3_bucket,
            Key=path
        )

        # Get content type
        content_type = response.get('ContentType', 'image/jpeg')

        # Stream the image
        return StreamingResponse(
            io.BytesIO(response['Body'].read()),
            media_type=content_type,
            headers={
                "Cache-Control": "public, max-age=86400",  # Cache for 24 hours
            }
        )

    except Exception as e:
        logger.error(f"Error fetching image {path}: {str(e)}")
        raise HTTPException(status_code=404, detail=f"Image not found: {path}")


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "faiss_loaded": faiss_service._index is not None,
        "faiss_contours": (
            len(faiss_service.get_metadata()) if faiss_service._index else 0
        ),
    }
