# %%
import random
import os
from models import ImageModel, Contour

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm

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
# image is grayscale in shape (h, w)
# returns np.ndarray of shape (N, 2) for each contour
def extract_contours(image: np.ndarray) -> list[np.ndarray]:
    canny_img = cv2.Canny(image, lo, hi)

    contours: list[np.ndarray] = []
    contours, _ = cv2.findContours(canny_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # this will be a list of arrays of shape (N, 2)
    # minimum length threshold to filter noise
    contours_squeezed = [c.squeeze() for c in contours if len(c) > 50]

    return contours_squeezed


# %%
image_ids: dict[str, ImageModel] = {}
random.seed(42)
random_images = random.sample(images, k=min(500, len(images)))
print(random_images[-1])

for idx, img_path in tqdm(enumerate(random_images), total=len(random_images)):
    target_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if target_img is None:
        raise ValueError(f"Failed to load image: {img_path}")

    target_shape: tuple[int, int] = (target_img.shape[0], target_img.shape[1])  # (h, w)
    assert len(target_shape) == 2, f"Image shape is not 2D: {target_shape}"

    target_contours: list[np.ndarray] = extract_contours(target_img)

    image_ids[img_path] = ImageModel(image_id=img_path, image_shape=target_shape)
    image_ids[img_path].add_contours(target_contours)
# %%
# length of contours
relative_areas = []
relative_lengths = []
for idx, img_model in image_ids.items():
    for contour in img_model.contours:
        if contour.area is not None:
            relative_areas.append(contour.area)
        if contour.length is not None:
            relative_lengths.append(contour.length)

plt.scatter(relative_areas, relative_lengths, alpha=0.5)
plt.xlabel("Relative Contour Area")
plt.yscale("log")
plt.ylabel("Relative Contour Length")
plt.title("Distribution of Contour Areas and Lengths")
plt.show()

# %%
# calculate IQR and outliers
q1_area, q3_area = np.percentile(relative_areas, [25, 75])
iqr_area = q3_area - q1_area
lower_bound_area = max(0, q1_area - 1.5 * iqr_area)
upper_bound_area = min(1, q3_area + 1.5 * iqr_area)
print(
    "Area - IQR:",
    iqr_area,
    "Lower bound:",
    lower_bound_area,
    "Upper bound:",
    upper_bound_area,
)
q1_length, q3_length = np.percentile(relative_lengths, [25, 75])
iqr_length = q3_length - q1_length
lower_bound_length = max(0, q1_length - 1.5 * iqr_length)
upper_bound_length = min(1, q3_length + 1.5 * iqr_length)
print(
    "Length - IQR:",
    iqr_length,
    "Lower bound:",
    lower_bound_length,
    "Upper bound:",
    upper_bound_length,
)
# %%
# two histograms of contour lengths and areas
fig, ax = plt.subplots(2, 1, figsize=(12, 12))
ax[0].hist(relative_areas, bins=50)
ax[0].set_xlabel("Relative Contour Area")
ax[0].set_ylabel("Count")
ax[0].set_yscale("log")
ax[0].axvline(lower_bound_area, color="r", linestyle="dashed", linewidth=1)
ax[0].axvline(upper_bound_area, color="g", linestyle="dashed", linewidth=1)

ax[1].hist(relative_lengths, bins=50)
ax[1].set_xlabel("Relative Contour Length")
ax[1].set_ylabel("Count")
ax[1].set_yscale("log")
ax[1].axvline(lower_bound_length, color="r", linestyle="dashed", linewidth=1)
ax[1].axvline(upper_bound_length, color="g", linestyle="dashed", linewidth=1)
plt.show()

# %%
n = 0
for _, img_model in image_ids.items():
    n += len(img_model.contours)
print(n, "contours in total")

# %%
# visualize some contours
random_image: ImageModel = np.random.choice(list(image_ids.values()))
print(random_image.image_id)
img = cv2.imread(random_image.image_id, cv2.IMREAD_GRAYSCALE)

plt.figure(figsize=(8, 8))
plt.imshow(img, cmap="gray")
for contour in random_image.contours:
    pts = np.array(contour.points, dtype=np.int32)
    plt.plot(pts[:, 0], pts[:, 1], linewidth=1)
plt.title("All Contours")

# %%
# visualize histogram of contours per image
contour_counts = np.array([len(img_model.contours) for img_model in image_ids.values()])
lo, hi = np.percentile(contour_counts, [5, 95])
filtered_ends = contour_counts[(contour_counts > lo) & (contour_counts < hi)]
plt.figure(figsize=(10, 6))
plt.hist(filtered_ends, bins=30, color="skyblue", edgecolor="black")
plt.xlabel("Number of Contours per Image")
plt.ylabel("Number of Images")
plt.title("Distribution of Contours per Image")


# %%
sketch_path = "../sketches/edge.png"
sketch_img = cv2.imread(sketch_path, cv2.IMREAD_GRAYSCALE)
if sketch_img is None:
    raise ValueError("Failed to load edge image")
plt.figure(figsize=(8, 8))
plt.imshow(sketch_img, cmap="gray")
plt.title("Sketch Image")

# %%
sketch_model: ImageModel = ImageModel(
    image_id=sketch_path, image_shape=sketch_img.shape
)
sketch_contours = extract_contours(sketch_img)
sketch_model.add_contours(sketch_contours)
print(f"Extracted {len(sketch_model.contours)} contours from sketch")

# %%
# run procrustes analysis on contours
from scipy.spatial import procrustes
import heapq

def align_contours(sketch_pts, target_pts):
    """
    Align sketch_pts to target_pts using Procrustes.
    Returns aligned sketch, scale factor, rotation matrix, translation, and disparity.
    """
    # Procrustes requires same number of points → resample
    N = min(len(sketch_pts), len(target_pts))
    sketch_resampled = sketch_pts[np.linspace(0, len(sketch_pts) - 1, N, dtype=int)]
    target_resampled = target_pts[np.linspace(0, len(target_pts) - 1, N, dtype=int)]

    # scipy's procrustes returns standardized (rotated+scaled+translated) points
    mtx1, mtx2, disparity = procrustes(target_resampled, sketch_resampled)
    return mtx1, mtx2, disparity


# %%
n_minimum = 5
best_scores: list[tuple[float, str, Contour]] = []  # (score, img_path, contour)
for img_path, img_model in tqdm(image_ids.items()):
    for contour in img_model.contours:
        mtx1, mtx2, score = align_contours(sketch_pts=sketch_model.contours[0].points, target_pts=contour.points)
        heapq.heappush(best_scores, (score, img_path, contour))
# %%
best_scores = heapq.nsmallest(n_minimum, best_scores)
fig, ax = plt.subplots(n_minimum + 1, 1, figsize=(5, 5 * (n_minimum + 1)))
# show sketch in first subplot
ax[0].imshow(sketch_img, cmap="gray")
ax[0].set_title("Sketch")
ax[0].axis("off")
for i, (score, img_path, contour) in enumerate(best_scores):
    target_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if target_img is None:
        raise ValueError(f"Failed to load image: {img_path}")

    ax[i + 1].imshow(target_img)
    pts = np.array(contour.points, dtype=np.int32)
    ax[i + 1].plot(pts[:, 0], pts[:, 1], color="red", linewidth=2)
    ax[i + 1].set_title(f"Score: {score:.6f}")
    ax[i + 1].axis("off")


# %%
