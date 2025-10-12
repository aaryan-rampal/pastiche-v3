"""API router for sketch matching endpoints."""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import os
import cv2

from loguru import logger
from core.config import settings
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple
import numpy as np
import io
from datetime import datetime

from services.contour_service import extract_contours_from_image_bytes
from services.faiss_service import FAISSService
from services.procrustes_service import ProcrustesService
from models.schemas import PointInput, ProcrustesResult

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


async def save_sketch_points(sketch_contour: np.ndarray) -> None:
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

        # Also save points as .npy for potential future use
        output_path_npy = os.path.join(output_dir, f"sketch_{timestamp}.npy")
        np.save(output_path_npy, points)

        logger.info(f"Saved sketch visualization to {output_path}")
    else:
        logger.warning("No points to visualize")


async def visualize_faiss_matches(
    sketch_contour: np.ndarray,
    faiss_results: List[Tuple[float, str, int]],
    timestamp: str,
    top_k: int = 5
) -> str:
    """Visualize top K FAISS matches with contours overlaid on artworks."""
    output_dir = "backend/debug_output"
    os.makedirs(output_dir, exist_ok=True)

    n_results = min(top_k, len(faiss_results))
    fig, axes = plt.subplots(1, n_results + 1, figsize=(4 * (n_results + 1), 4))
    if n_results == 0:
        axes = [axes]

    # Plot sketch
    axes[0].plot(sketch_contour[:, 0], sketch_contour[:, 1], 'b-', linewidth=2)
    axes[0].set_title("Sketch", fontsize=10, fontweight='bold')
    axes[0].axis('equal')
    axes[0].axis('off')

    # Plot top FAISS matches
    for i, (distance, img_path, contour_idx) in enumerate(faiss_results[:n_results]):
        try:
            # Load image from S3
            image = procrustes_service.load_image_from_s3(img_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            # Extract and draw contour
            contour_points, _ = procrustes_service.extract_contour_from_s3_image(img_path, contour_idx)
            contour_points_int = contour_points.astype(np.int32)

            # Draw contour on image
            cv2.polylines(image_rgb, [contour_points_int], False, (255, 0, 0), 2)

            axes[i + 1].imshow(image_rgb)
            axes[i + 1].set_title(f"#{i+1}\nDist: {distance:.4f}", fontsize=9)
            axes[i + 1].axis('off')
        except Exception as e:
            logger.error(f"Error visualizing FAISS match {i}: {e}")
            axes[i + 1].text(0.5, 0.5, f"Error loading\n{img_path}", ha='center', va='center')
            axes[i + 1].axis('off')

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"faiss_matches_{timestamp}.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved FAISS matches visualization to {output_path}")
    return output_path


async def visualize_procrustes_matches(
    sketch_contour: np.ndarray,
    procrustes_results: List[Tuple[ProcrustesResult, str, np.ndarray, float]],
    timestamp: str,
    top_k: int = 5
) -> str:
    """Visualize top K Procrustes matches with transformed sketch overlay."""
    output_dir = "backend/debug_output"
    os.makedirs(output_dir, exist_ok=True)

    n_results = min(top_k, len(procrustes_results))
    fig, axes = plt.subplots(2, n_results, figsize=(4 * n_results, 8))
    if n_results == 1:
        axes = axes.reshape(2, 1)

    for i, (procrustes_result, img_path, contour_points, hu_distance) in enumerate(procrustes_results[:n_results]):
        try:
            # Load image from S3
            image = procrustes_service.load_image_from_s3(img_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            # Top row: Image with target contour
            target_pts = contour_points.astype(np.int32)
            image_with_contour = image_rgb.copy()
            cv2.polylines(image_with_contour, [target_pts], False, (255, 0, 0), 2)

            axes[0, i].imshow(image_with_contour)
            axes[0, i].set_title(f"#{i+1} Target Contour", fontsize=9, fontweight='bold')
            axes[0, i].axis('off')

            # Bottom row: Image with overlay of transformed sketch
            image_with_overlay = image_rgb.copy()
            # Draw target contour in red
            cv2.polylines(image_with_overlay, [target_pts], False, (255, 0, 0), 2)
            # Draw transformed sketch in blue (dashed approximation with circles)
            transformed_pts = procrustes_result.transformed_sketch_points.astype(np.int32)
            for j in range(len(transformed_pts) - 1):
                cv2.line(image_with_overlay, tuple(transformed_pts[j]), tuple(transformed_pts[j+1]), (0, 0, 255), 2)

            axes[1, i].imshow(image_with_overlay)
            title = f"Disparity: {procrustes_result.disparity:.4f}\n"
            title += f"Scale: {procrustes_result.scale:.2f}, Rot: {procrustes_result.rotation_degrees:.1f}°"
            axes[1, i].set_title(title, fontsize=8)
            axes[1, i].axis('off')

        except Exception as e:
            logger.error(f"Error visualizing Procrustes match {i}: {e}")
            axes[0, i].text(0.5, 0.5, "Error", ha='center', va='center')
            axes[1, i].text(0.5, 0.5, "Error", ha='center', va='center')
            axes[0, i].axis('off')
            axes[1, i].axis('off')

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"procrustes_matches_{timestamp}.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved Procrustes matches visualization to {output_path}")
    return output_path


async def visualize_final_match(
    sketch_contour: np.ndarray,
    match: MatchResult,
    procrustes_result: ProcrustesResult,
    contour_points: np.ndarray,
    timestamp: str
) -> str:
    """Create detailed visualization of the final selected match."""
    output_dir = "backend/debug_output"
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Load image from S3
        image = procrustes_service.load_image_from_s3(match.artwork_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        fig, axes = plt.subplots(2, 2, figsize=(12, 12))

        # 1. Original sketch
        axes[0, 0].plot(sketch_contour[:, 0], sketch_contour[:, 1], 'b-', linewidth=2)
        axes[0, 0].set_title("Original Sketch", fontsize=12, fontweight='bold')
        axes[0, 0].axis('equal')
        axes[0, 0].axis('off')

        # 2. Artwork with target contour
        target_pts = contour_points.astype(np.int32)
        image_target = image_rgb.copy()
        cv2.polylines(image_target, [target_pts], False, (255, 0, 0), 3)
        axes[0, 1].imshow(image_target)
        axes[0, 1].set_title("Matched Artwork + Target Contour (RED)", fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')

        # 3. Overlay comparison
        image_overlay = image_rgb.copy()
        # Target in red
        cv2.polylines(image_overlay, [target_pts], False, (255, 0, 0), 2)
        # Transformed sketch in blue
        transformed_pts = procrustes_result.transformed_sketch_points.astype(np.int32)
        for j in range(len(transformed_pts) - 1):
            cv2.line(image_overlay, tuple(transformed_pts[j]), tuple(transformed_pts[j+1]), (0, 0, 255), 2)
        axes[1, 0].imshow(image_overlay)
        axes[1, 0].set_title("Overlay: Target (RED) + Transformed Sketch (BLUE)", fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')

        # 4. Transformation details
        axes[1, 1].axis('off')
        details_text = "TRANSFORMATION DETAILS\n" + "="*40 + "\n\n"
        details_text += f"Procrustes Disparity: {match.procrustes_score:.6f}\n"
        details_text += f"Hu Distance: {match.hu_distance:.6f}\n\n"
        details_text += f"Scale Factor: {match.transform.scale:.3f}\n"
        details_text += f"Rotation: {match.transform.rotation_degrees:.2f}° ({match.transform.rotation_radians:.4f} rad)\n\n"
        details_text += f"Translation:\n"
        details_text += f"  dx: {match.transform.translation['x']:.2f}\n"
        details_text += f"  dy: {match.transform.translation['y']:.2f}\n\n"
        details_text += f"Sketch Centroid: ({match.transform.sketch_centroid['x']:.1f}, {match.transform.sketch_centroid['y']:.1f})\n"
        details_text += f"Target Centroid: ({match.transform.target_centroid['x']:.1f}, {match.transform.target_centroid['y']:.1f})\n\n"
        details_text += f"Contours:\n"
        details_text += f"  Sketch points: {len(sketch_contour)}\n"
        details_text += f"  Target points: {len(contour_points)}\n\n"
        details_text += f"Artwork: {match.artwork_path}"

        axes[1, 1].text(0.1, 0.5, details_text, fontsize=10, family='monospace', va='center')

        plt.tight_layout()
        output_path = os.path.join(output_dir, f"final_match_{timestamp}.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved final match visualization to {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Error creating final match visualization: {e}")
        return ""


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
        # Create timestamp for this matching session
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Convert points to numpy array
        logger.info("=" * 80)
        logger.info(f"NEW SKETCH MATCHING REQUEST - Session: {timestamp}")
        logger.info("=" * 80)
        logger.info(f"Request parameters: top_k_faiss={request.top_k_faiss}, top_k_final={request.top_k_final}")

        sketch_contour = request.points.to_numpy()

        if len(sketch_contour) < 3:
            raise HTTPException(
                status_code=400,
                detail="At least 3 points are required for matching.",
            )

        logger.info(f"Sketch contour: {len(sketch_contour)} points")
        logger.info(f"Sketch X range: [{sketch_contour[:, 0].min():.2f}, {sketch_contour[:, 0].max():.2f}]")
        logger.info(f"Sketch Y range (before flip): [{sketch_contour[:, 1].min():.2f}, {sketch_contour[:, 1].max():.2f}]")
        logger.debug(f"First 5 sketch points (before flip): {sketch_contour[:5].tolist()}")

        # reverse y-axis to match image coordinate system
        sketch_contour[:, 1] = -sketch_contour[:, 1]

        logger.info(f"Sketch Y range (after flip): [{sketch_contour[:, 1].min():.2f}, {sketch_contour[:, 1].max():.2f}]")
        logger.debug(f"First 5 sketch points (after flip): {sketch_contour[:5].tolist()}")

        if settings.save_sketch_debug:
            await save_sketch_points(sketch_contour)
            logger.info("Saved sketch points visualization for debugging")

        # Stage 1: FAISS search for initial candidates
        logger.info("-" * 80)
        logger.info("STAGE 1: FAISS SEARCH")
        logger.info("-" * 80)
        faiss_results = faiss_service.search_similar_contours(
            sketch_contour, k=request.top_k_faiss
        )
        faiss_results = faiss_results[: settings.faiss_top_k]
        logger.info(f"FAISS returned {len(faiss_results)} candidates")

        # Log top 5 FAISS results
        logger.info("Top 5 FAISS matches:")
        for i, (distance, img_path, contour_idx) in enumerate(faiss_results[:5]):
            logger.info(f"  {i+1}. distance={distance:.6f}, path={img_path}, contour_idx={contour_idx}")

        if not faiss_results:
            raise HTTPException(
                status_code=404,
                detail="No matches found in FAISS index.",
            )

        # Visualize FAISS matches
        if settings.save_sketch_debug:
            await visualize_faiss_matches(sketch_contour, faiss_results, timestamp, top_k=5)

        # Stage 2: Procrustes refinement
        logger.info("-" * 80)
        logger.info("STAGE 2: PROCRUSTES REFINEMENT")
        logger.info("-" * 80)
        procrustes_results = procrustes_service.compute_procrustes_batch(
            sketch_contour, faiss_results, top_k=request.top_k_final
        )

        # Log top 5 Procrustes results
        logger.info(f"Procrustes refinement completed: {len(procrustes_results)} results")
        logger.info("Top 5 Procrustes matches:")
        for i, (procrustes_result, img_path, contour_points, hu_distance) in enumerate(procrustes_results[:5]):
            logger.info(f"  {i+1}. disparity={procrustes_result.disparity:.6f}, hu_dist={hu_distance:.6f}, path={img_path}")
            logger.info(f"      scale={procrustes_result.scale:.3f}, rotation={procrustes_result.rotation_degrees:.1f}°")
            logger.info(f"      translation=({procrustes_result.translation['x']:.1f}, {procrustes_result.translation['y']:.1f})")
            logger.info(f"      contour_points={len(contour_points)}, range: x[{contour_points[:, 0].min():.1f}, {contour_points[:, 0].max():.1f}], y[{contour_points[:, 1].min():.1f}, {contour_points[:, 1].max():.1f}]")

        # Visualize Procrustes matches
        if settings.save_sketch_debug:
            await visualize_procrustes_matches(sketch_contour, procrustes_results, timestamp, top_k=5)

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
            logger.info("-" * 80)
            logger.info("FINAL MATCH SELECTION")
            logger.info("-" * 80)
            # Generate exponential weights (decay rate = 1.0)
            weights = np.exp(-np.arange(len(matches)))
            # Normalize to probabilities
            probabilities = weights / weights.sum()
            logger.info(f"Exponential weights: {weights[:5].tolist() if len(weights) > 5 else weights.tolist()}")
            logger.info(f"Probabilities: {probabilities[:5].tolist() if len(probabilities) > 5 else probabilities.tolist()}")

            # Randomly select one index based on probabilities
            selected_idx = np.random.choice(len(matches), p=probabilities)
            selected_match = matches[selected_idx]

            logger.info(f"Selected match index: {selected_idx} from {len(procrustes_results)} candidates")
            logger.info(f"Selected match details:")
            logger.info(f"  Path: {selected_match.artwork_path}")
            logger.info(f"  Procrustes disparity: {selected_match.procrustes_score:.6f}")
            logger.info(f"  Hu distance: {selected_match.hu_distance:.6f}")
            logger.info(f"  Transform: scale={selected_match.transform.scale:.3f}, rotation={selected_match.transform.rotation_degrees:.1f}°")
            logger.info(f"  Matched contour: {len(selected_match.matched_contour_points)} points")

            matches = [selected_match]

            # Visualize final selected match
            if settings.save_sketch_debug:
                # Get the full procrustes result and contour points for the selected match
                selected_procrustes_result, _, selected_contour_points, _ = procrustes_results[selected_idx]
                await visualize_final_match(
                    sketch_contour,
                    selected_match,
                    selected_procrustes_result,
                    selected_contour_points,
                    timestamp
                )

        logger.info("=" * 80)
        logger.info(f"RETURNING {len(matches)} MATCH(ES)")
        logger.info("=" * 80)

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
            Bucket=procrustes_service.s3_bucket, Key=path
        )

        # Get content type
        content_type = response.get("ContentType", "image/jpeg")

        # Stream the image
        return StreamingResponse(
            io.BytesIO(response["Body"].read()),
            media_type=content_type,
            headers={
                "Cache-Control": "public, max-age=86400",  # Cache for 24 hours
            },
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
