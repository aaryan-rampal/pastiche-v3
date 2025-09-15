# %%
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

# %%
dir = "/Volumes/Extreme SSD/wikiart/wikiart_5pct"
lo, hi = 150, 200

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
# Encode all images into embeddings
def to_edge_rgb(pil_img, low=lo, high=hi):
    gray = np.array(pil_img.convert("L"))
    edges = cv2.Canny(gray, low, high)
    edges_3ch = np.stack([edges] * 3, axis=-1)
    return Image.fromarray(edges_3ch)


def extract_contours(edge_map):
    # OpenCV returns a list of numpy arrays, each contour = Nx2 points
    contours, _ = cv2.findContours(edge_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # minimum length threshold to filter noise
    return [c.squeeze() for c in contours if len(c) > 50]


# %%
image_contours = {}
random_images = np.random.choice(images, size=500, replace=False)
for idx, target_img in tqdm(enumerate(random_images), total=len(random_images)):
    target_img = Image.open(target_img)
    target_edges = cv2.Canny(np.array(target_img.convert("L")), lo, hi)
    target_contours = extract_contours(target_edges)
    image_shape = np.array(target_img).shape

    image_contours[idx] = (target_contours, image_shape)
# %%
# length of contours
# lengths = [len(c) for cs in image_contours.values() for c in cs]
relative_areas = [
    cv2.contourArea(c) / (shape[0] * shape[1])
    for cs, shape in image_contours.values()
    for c in cs
]
plt.hist(relative_areas, bins=50)
plt.yscale("log")
plt.xlabel("Relative Contour Area")
plt.ylabel("Count")
plt.title("Distribution of Contour Areas")
plt.show()


# %%
# find argmax contour
max_contour = None
max_length = 0
max_image = None
for img, cs in image_contours.items():
    for c in cs:
        c = c[0]
        if len(c) > max_length:
            max_length = len(c)
            max_contour = c
            max_image = img

assert max_contour is not None
assert max_image is not None
print(f"Max contour length: {max_length} from image {max_image}")
# %%
# Visualize the longest contour
target_img = Image.open(random_images[max_image])
target_edges = cv2.Canny(np.array(target_img.convert("L")), lo, hi)
plt.imshow(target_img, cmap="gray")
plt.plot(max_contour[:, 0], max_contour[:, 1], "r-", linewidth=0.5)
plt.title("Longest Contour Overlay")

# %%
