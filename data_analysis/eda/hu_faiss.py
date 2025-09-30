# %%
import pandas as pd
import os
from models import ImageModel, ContourFAISSIndex, ProcrustesResult
from utils import compute_enhanced_features

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# %%
DATA_DIR: str = "/Volumes/Extreme SSD/wikiart/"
lo: int = 150
hi: int = 200

# %%
df = pd.read_csv("../../data/classes_truncated.csv")


# %%
def extract_contours(image: np.ndarray) -> list[np.ndarray]:
    """Extract contours from grayscale image"""
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    canny_img = cv2.Canny(blurred, lo, hi)
    contours, _ = cv2.findContours(canny_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours_squeezed = [c.squeeze() for c in contours if len(c) > 50]
    return contours_squeezed


# compute_enhanced_features moved to data_analysis/eda/utils.py


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


def align_contours(sketch_pts, target_pts):
    """Legacy wrapper for backward compatibility"""
    result = align_contours_with_transform(sketch_pts, target_pts)
    # Return old format: (mtx1, mtx2, disparity)
    return result.transformed_sketch_points, target_pts, result.disparity


def compute_procrustes_single(args):
    """Helper function for parallel Procrustes computation"""
    sketch_contour, hu_distance, img_path, contour_idx, contour_points = args
    try:
        _, _, procrustes_score = align_contours(sketch_contour, contour_points)
        return (procrustes_score, img_path, contour_idx, contour_points, hu_distance)
    except Exception:
        return None


def compute_procrustes_parallel(sketch_contour, faiss_results, n_workers=None):
    """Compute Procrustes scores in parallel"""
    if n_workers is None:
        n_workers = min(mp.cpu_count(), len(faiss_results))

    # Prepare arguments for parallel processing
    args_list = []
    for hu_distance, img_path, contour_idx, contour in faiss_results:
        args_list.append(
            (sketch_contour, hu_distance, img_path, contour_idx, contour.points)
        )

    # Process in parallel
    procrustes_results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all tasks
        future_to_args = {
            executor.submit(compute_procrustes_single, args): args for args in args_list
        }

        # Collect results with progress bar
        for future in tqdm(
            as_completed(future_to_args),
            total=len(args_list),
            desc="Computing Procrustes",
        ):
            result = future.result()
            if result is not None:
                procrustes_score, img_path, contour_idx, contour_points, hu_distance = (
                    result
                )
                # Reconstruct contour object for compatibility
                from models import Contour

                contour = Contour(
                    points=contour_points,
                    image_id=img_path,
                    image_shape=(100, 100),  # Dummy shape, won't be used
                )
                procrustes_results.append(
                    (procrustes_score, img_path, contour, hu_distance)
                )

    return procrustes_results


# %%

# %%
# Load and process images
print("Loading and processing images...")
image_ids: dict[str, ImageModel] = {}

for row in tqdm(df.itertuples(), total=len(df)):
    img_path = os.path.join(DATA_DIR, row.filename)
    target_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if target_img is None:
        continue

    target_shape: tuple[int, int] = (target_img.shape[0], target_img.shape[1])
    target_contours: list[np.ndarray] = extract_contours(target_img)
    img_model: ImageModel = ImageModel(image_id=img_path, image_shape=target_shape)
    img_model.add_contours(target_contours)
    if img_model.contours:  # Only add if we found contours
        image_ids[img_path] = img_model

print(f"Processed {len(image_ids)} images")

# %%
# Build FAISS index
print("Building FAISS index...")
faiss_index = ContourFAISSIndex()
faiss_index.build_index(image_ids)

# Optionally save the index
faiss_index.save_index("contour_hu_index")

# %%
# Load sketch and extract contours
sketch_path = "../sketches/edge.png"
sketch_img = cv2.imread(sketch_path, cv2.IMREAD_GRAYSCALE)
if sketch_img is None:
    raise ValueError("Failed to load sketch image")

plt.figure(figsize=(8, 8))
plt.imshow(sketch_img, cmap="gray")
plt.title("Sketch Image")
plt.axis("off")
plt.show()

# %%
# Extract sketch contours
sketch_model: ImageModel = ImageModel(
    image_id=sketch_path, image_shape=(sketch_img.shape[0], sketch_img.shape[1])
)
sketch_contours = extract_contours(sketch_img)
sketch_model.add_contours(sketch_contours)
print(f"Extracted {len(sketch_model.contours)} contours from sketch")


# %%
# Two-stage matching: FAISS + Procrustes
def find_best_matches(
    sketch_contour: np.ndarray,
    faiss_index: ContourFAISSIndex,
    top_k_faiss: int = 100,
    top_k_final: int = 5,
):
    """
    Two-stage matching:
    1. Use FAISS to find top_k_faiss candidates based on Hu moments
    2. Use Procrustes to refine and get top_k_final results
    """
    print(f"Stage 1: FAISS search for top {top_k_faiss} candidates...")
    faiss_results = faiss_index.search_similar_contours(sketch_contour, k=top_k_faiss)

    if not faiss_results:
        print("No FAISS results found")
        return []

    print(f"Stage 2: Procrustes refinement on {len(faiss_results)} candidates...")
    procrustes_results = []

    for hu_distance, img_path, contour_idx in tqdm(faiss_results):
        contour = image_ids[img_path].contours[contour_idx]
        try:
            result = align_contours_with_transform(sketch_contour, contour.points)
            procrustes_results.append((result, img_path, contour, hu_distance))
        except Exception:
            continue

    # Sort by Procrustes score (lower is better)
    procrustes_results.sort(key=lambda x: x[0].disparity)
    return procrustes_results[:top_k_final]


# %%
# Run matching for the first sketch contour
if sketch_model.contours:
    sketch_contour = sketch_model.contours[0].points

    print("Running two-stage matching...")
    best_matches = find_best_matches(
        sketch_contour, faiss_index, top_k_faiss=40_000, top_k_final=10
    )

    # Visualize results
    n_results = len(best_matches)
    if n_results > 0:
        fig, ax = plt.subplots(n_results + 1, 1, figsize=(8, 4 * (n_results + 1)))

        # Show sketch
        ax[0].imshow(sketch_img, cmap="gray")
        sketch_pts = np.array(sketch_contour, dtype=np.int32)
        ax[0].plot(sketch_pts[:, 0], sketch_pts[:, 1], color="blue", linewidth=2)
        ax[0].set_title("Sketch")
        ax[0].axis("off")

        # Show matches with transformed sketch overlay
        for i, (procrustes_result, img_path, contour, hu_distance) in enumerate(
            best_matches
        ):
            target_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if target_img is not None:
                ax[i + 1].imshow(target_img)

                # Draw original target contour in red
                target_pts = np.array(contour.points, dtype=np.int32)
                ax[i + 1].plot(
                    target_pts[:, 0],
                    target_pts[:, 1],
                    color="red",
                    linewidth=2,
                    label="Target contour",
                    alpha=0.7,
                )

                # Draw transformed sketch contour in blue (to show alignment)
                transformed_pts = np.array(
                    procrustes_result.transformed_sketch_points, dtype=np.int32
                )
                ax[i + 1].plot(
                    transformed_pts[:, 0],
                    transformed_pts[:, 1],
                    color="blue",
                    linewidth=2,
                    label="Transformed sketch",
                    alpha=0.7,
                    linestyle="--",
                )

                # Add title with all transform info
                title = (
                    f"Procrustes: {procrustes_result.disparity:.4f}, Hu: {hu_distance:.4f}\n"
                    f"Scale: {procrustes_result.scale:.2f}, "
                    f"Rotation: {procrustes_result.rotation_degrees:.1f}°"
                )
                ax[i + 1].set_title(title, fontsize=9)
                ax[i + 1].legend(loc="upper right", fontsize=8)
                ax[i + 1].axis("off")

        plt.tight_layout()
        plt.show()

        print("\nTop matches:")
        for i, (procrustes_result, img_path, contour, hu_distance) in enumerate(
            best_matches
        ):
            print(
                f"{i + 1}. Procrustes: {procrustes_result.disparity:.4f}, Hu: {hu_distance:.4f}"
            )
            print(f"   Image: {os.path.basename(img_path)}")
            print(
                f"   Transform: scale={procrustes_result.scale:.2f}, rotation={procrustes_result.rotation_degrees:.1f}°"
            )
            print(
                f"   Translation: dx={procrustes_result.translation['x']:.1f}, dy={procrustes_result.translation['y']:.1f}"
            )
    else:
        print("No matches found")
else:
    print("No contours found in sketch")

# %%
