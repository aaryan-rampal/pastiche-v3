# %%
import os
from models import ImageModel

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import faiss
from scipy.spatial import procrustes
import pickle

# %%
dir: str = "/Volumes/Extreme SSD/wikiart/wikiart_5pct"
lo: int = 150
hi: int = 200

# %%
images: list[str] = []
metadata: list[str] = []
for root, dirs, files in os.walk(dir):
    for file in files:
        if file.endswith(".jpg") and file.startswith("image"):
            images.append(os.path.join(root, file))
        elif file.endswith(".json"):
            metadata.append(os.path.join(root, file))

# %%
def extract_contours(image: np.ndarray) -> list[np.ndarray]:
    """Extract contours from grayscale image"""
    canny_img = cv2.Canny(image, lo, hi)
    contours, _ = cv2.findContours(canny_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours_squeezed = [c.squeeze() for c in contours if len(c) > 50]
    return contours_squeezed

def compute_hu_moments(contour_points: np.ndarray) -> np.ndarray:
    """Compute 7 Hu moments for a contour"""
    try:
        # Convert points to the format expected by cv2.moments
        moments = cv2.moments(contour_points.astype(np.float32))
        if moments['m00'] == 0:
            return np.zeros(7, dtype=np.float32)

        hu_moments = cv2.HuMoments(moments).flatten()
        # Take log to make them more comparable and avoid very large/small values
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
        return hu_moments.astype(np.float32)
    except Exception as e:
        print(f"Error computing Hu moments: {e}")
        return np.zeros(7, dtype=np.float32)

def align_contours(sketch_pts, target_pts):
    """Align sketch_pts to target_pts using Procrustes"""
    N = min(len(sketch_pts), len(target_pts))
    sketch_resampled = sketch_pts[np.linspace(0, len(sketch_pts) - 1, N, dtype=int)]
    target_resampled = target_pts[np.linspace(0, len(target_pts) - 1, N, dtype=int)]
    mtx1, mtx2, disparity = procrustes(target_resampled, sketch_resampled)
    return mtx1, mtx2, disparity

# %%
class ContourFAISSIndex:
    def __init__(self):
        self.index = None
        self.contour_metadata = []  # List of (img_path, contour_idx, contour) tuples
        self.hu_features = []

    def build_index(self, image_models: dict[str, ImageModel]):
        """Build FAISS index from all contours in image models"""
        print("Computing Hu moments for all contours...")

        hu_vectors = []
        metadata = []

        for img_path, img_model in tqdm(image_models.items()):
            for contour_idx, contour in enumerate(img_model.contours):
                hu_moments = compute_hu_moments(contour.points)

                # Skip invalid Hu moments
                if not np.any(np.isnan(hu_moments)) and not np.any(np.isinf(hu_moments)):
                    hu_vectors.append(hu_moments)
                    metadata.append((img_path, contour_idx, contour))

        if not hu_vectors:
            raise ValueError("No valid Hu moments computed")

        # Convert to numpy array
        self.hu_features = np.array(hu_vectors, dtype=np.float32)
        self.contour_metadata = metadata

        # Build FAISS index (L2 distance)
        dimension = 7  # Hu moments are 7-dimensional
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.hu_features)

        print(f"Built FAISS index with {len(self.hu_features)} contours")

    def search_similar_contours(self, sketch_contour: np.ndarray, k: int = 100):
        """Find k most similar contours using Hu moments"""
        if self.index is None:
            raise ValueError("Index not built yet")

        sketch_hu = compute_hu_moments(sketch_contour)
        if np.any(np.isnan(sketch_hu)) or np.any(np.isinf(sketch_hu)):
            print("Warning: Invalid Hu moments for sketch")
            return []

        # Search FAISS index
        distances, indices = self.index.search(sketch_hu.reshape(1, -1), k)

        # Return metadata for found contours
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.contour_metadata):
                img_path, contour_idx, contour = self.contour_metadata[idx]
                results.append((distances[0][i], img_path, contour_idx, contour))

        return results

    def save_index(self, filepath: str):
        """Save the FAISS index and metadata"""
        faiss.write_index(self.index, f"{filepath}.faiss")
        with open(f"{filepath}_metadata.pkl", "wb") as f:
            pickle.dump({
                'contour_metadata': self.contour_metadata,
                'hu_features': self.hu_features
            }, f)
        print(f"Saved index to {filepath}")

    def load_index(self, filepath: str):
        """Load the FAISS index and metadata"""
        self.index = faiss.read_index(f"{filepath}.faiss")
        with open(f"{filepath}_metadata.pkl", "rb") as f:
            data = pickle.load(f)
            self.contour_metadata = data['contour_metadata']
            self.hu_features = data['hu_features']
        print(f"Loaded index from {filepath}")

# %%
# Load and process images
print("Loading and processing images...")
image_ids: dict[str, ImageModel] = {}
random_images = np.random.choice(images, size=500, replace=False)

for idx, img_path in tqdm(enumerate(random_images), total=len(random_images)):
    target_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if target_img is None:
        continue

    target_shape: tuple[int, int] = (target_img.shape[0], target_img.shape[1])
    target_contours: list[np.ndarray] = extract_contours(target_img)

    if target_contours:  # Only add if we found contours
        image_ids[img_path] = ImageModel(image_id=img_path, image_shape=target_shape)
        image_ids[img_path].add_contours(target_contours)

print(f"Processed {len(image_ids)} images")

# %%
# Build FAISS index
print("Building FAISS index...")
faiss_index = ContourFAISSIndex()
faiss_index.build_index(image_ids)

# Optionally save the index
# faiss_index.save_index("contour_hu_index")

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
def find_best_matches(sketch_contour: np.ndarray, faiss_index: ContourFAISSIndex,
                     top_k_faiss: int = 100, top_k_final: int = 5):
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

    for hu_distance, img_path, contour_idx, contour in tqdm(faiss_results):
        try:
            _, _, procrustes_score = align_contours(sketch_contour, contour.points)
            procrustes_results.append((procrustes_score, img_path, contour, hu_distance))
        except Exception:
            continue

    # Sort by Procrustes score (lower is better)
    procrustes_results.sort(key=lambda x: x[0])
    return procrustes_results[:top_k_final]

# %%
# Run matching for the first sketch contour
if sketch_model.contours:
    sketch_contour = sketch_model.contours[0].points

    print("Running two-stage matching...")
    best_matches = find_best_matches(sketch_contour, faiss_index, top_k_faiss=250, top_k_final=5)

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

        # Show matches
        for i, (procrustes_score, img_path, contour, hu_distance) in enumerate(best_matches):
            target_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if target_img is not None:
                ax[i + 1].imshow(target_img)
                pts = np.array(contour.points, dtype=np.int32)
                ax[i + 1].plot(pts[:, 0], pts[:, 1], color="red", linewidth=2)
                ax[i + 1].set_title(f"Procrustes: {procrustes_score:.4f}, Hu: {hu_distance:.4f}")
                ax[i + 1].axis("off")

        plt.tight_layout()
        plt.show()

        print("\nTop matches:")
        for i, (procrustes_score, img_path, contour, hu_distance) in enumerate(best_matches):
            print(f"{i+1}. Procrustes score: {procrustes_score:.4f}, Hu distance: {hu_distance:.4f}")
            print(f"   Image: {os.path.basename(img_path)}")
    else:
        print("No matches found")
else:
    print("No contours found in sketch")

# %%
