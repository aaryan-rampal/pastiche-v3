# %%
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2

# %%
dir = "/Volumes/Extreme SSD/wikiart/wikiart_5pct"

# %%
images = []
metadata = []
for root, dirs, files in os.walk(dir):
    for file in files:
        if file.endswith(".jpg") and file.startswith("image"):
            images.append(os.path.join(root, file))
        elif file.endswith(".json"):
            metadata.append(os.path.join(root, file))

# %%
Image.open(images[100])


# %%
# Encode all images into embeddings
def to_edge_rgb(pil_img, low=100, high=200):
    gray = np.array(pil_img.convert("L"))
    edges = cv2.Canny(gray, low, high)
    edges_3ch = np.stack([edges] * 3, axis=-1)
    return Image.fromarray(edges_3ch)


# %%
sketch_img = Image.open("edge.png")
sketch_img


# %%
import cv2
import numpy as np
from scipy.spatial import procrustes
from PIL import ImageDraw


def extract_contours(edge_map):
    # OpenCV returns a list of numpy arrays, each contour = Nx2 points
    contours, _ = cv2.findContours(edge_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # minimum length threshold to filter noise
    return [c.squeeze() for c in contours if len(c) > 50]


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


# Get contours from sketch and target image
sketch_edges = cv2.Canny(np.array(sketch_img.convert("L")), 50, 150)
sketch_contours = extract_contours(sketch_edges)

best_score = float("inf")
best_pair = (None, None)
best_mat = (None, None)
best_img = None
random_images = np.random.choice(images, size=5, replace=False)
for target_img in tqdm(random_images):
    target_img = Image.open(target_img)
    target_edges = cv2.Canny(np.array(target_img.convert("L")), 50, 150)
    target_contours = extract_contours(target_edges)

    for s in sketch_contours:
        for t in target_contours:
            try:
                mtx1, mtx2, score = align_contours(s, t)
                if score < best_score:
                    print("New best score:", score)
                    best_score = score
                    best_pair = (s, t)
                    best_img = target_img
                    best_mat = (mtx1, mtx2)
            except Exception:
                continue

print("Best matching contour found with score:", best_score)

# Visualization: draw best matching target contour
img_vis = best_img.convert("RGB").copy()
draw = ImageDraw.Draw(img_vis)
if best_pair[1] is not None:
    # Draw as a polygon
    pts = best_pair[1]
    pts = [tuple(map(int, pt)) for pt in pts]
    draw.line(pts + [pts[0]], fill="red", width=2)
plt.figure(figsize=(8, 8))
plt.imshow(img_vis)
plt.title("Best Matching Contour (red)")
plt.axis("off")
plt.show()

# %%
import matplotlib.pyplot as plt

mtx1, mtx2 = best_mat
plt.figure(figsize=(6, 6))
plt.plot(mtx1[:, 0], mtx1[:, 1], label="Target (aligned)", color="red")
plt.plot(mtx2[:, 0], mtx2[:, 1], label="Sketch (aligned)", color="blue")
plt.legend()
plt.title("Procrustes Alignment")
plt.axis("equal")
plt.show()

# %%
