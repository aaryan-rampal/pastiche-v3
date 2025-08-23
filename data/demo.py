# %%
import os
from PIL import Image
import torch
import clip
from PIL import Image
import faiss
from tqdm import tqdm
import os
import numpy as np

# %%
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

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

# Load CLIP (ViT-B/32 is small + fast; try ViT-L/14 for better quality)
# device = "cuda" if torch.cuda.is_available() else "cpu"
# device is mac's metal
device = "mps"
model, preprocess = clip.load("ViT-B/32", device=device)

# Example: list of image file paths
image_paths = images

# Encode all images into embeddings
embeddings = []
for path in tqdm(image_paths):
    img = preprocess(Image.open(path)).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(img)
    emb = emb / emb.norm(dim=-1, keepdim=True)  # normalize
    embeddings.append(emb.cpu().numpy())

embeddings = np.vstack(embeddings).astype("float32")


# %%
import numpy as np

d = embeddings.shape[1]  # dimension
index = faiss.IndexFlatL2(d)  # simple L2 index
index.add(embeddings)  # add all embeddings
print("Stored", index.ntotal, "embeddings in vector store")


# %%
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def make_random_contour(size=256, n_points=10):
    # Generate random points and smooth them
    xs = np.linspace(0, size, n_points)
    ys = np.random.randint(20, size - 20, n_points)

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(size / 64, size / 64), dpi=64)
    ax.plot(xs, ys, color="black", linewidth=3)
    ax.axis("off")
    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    fig.canvas.draw()

    # Convert figure to PIL image
    data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))

    plt.close(fig)
    img = Image.fromarray(data)
    return img.convert("RGB")


# Make a random contour
sketch_img = make_random_contour()
sketch_img


# %%
try:
    sketch_input = preprocess(sketch_img).unsqueeze(0).to(device)
    print(f"Sketch input shape: {sketch_input.shape}")

    with torch.no_grad():
        sketch_emb = model.encode_image(sketch_input)
    sketch_emb = sketch_emb / sketch_emb.norm(dim=-1, keepdim=True)
    print(f"Sketch embedding shape: {sketch_emb.shape}")

    # Convert to numpy for FAISS search
    sketch_emb_np = sketch_emb.cpu().numpy().astype("float32")
    print(f"Sketch embedding numpy shape: {sketch_emb_np.shape}")

    # Search FAISS
    D, I = index.search(sketch_emb_np, k=5)
    print(f"Search results - Distances: {D}")
    print(f"Search results - Indices: {I}")

    # Get matching image paths (only valid indices)
    valid_indices = I[0][I[0] < len(image_paths)]
    matches = [Image.open(image_paths[i]) for i in valid_indices[:5]]
    match_paths = [image_paths[i] for i in valid_indices[:5]]

    print("Match paths:")
    for path in match_paths:
        print(f"  {path}")

except Exception as e:
    print(f"Error in search: {e}")
    import traceback

    traceback.print_exc()

# %%
# %%
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
import clip
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage


# %%
class CLIPAttentionExtractor:
    def __init__(self, model):
        self.model = model
        self.attention_weights = []
        self.hooks = []

    def register_hooks(self):
        """Register hooks to capture attention weights"""

        def attention_hook(module, input, output):
            # For CLIP ViT, attention weights are in the multi-head attention
            if hasattr(module, "num_heads"):
                # Store attention weights
                self.attention_weights.append(output)

        # Register hooks on attention layers
        for name, module in self.model.visual.transformer.named_modules():
            if "attn" in name and hasattr(module, "num_heads"):
                hook = module.register_forward_hook(attention_hook)
                self.hooks.append(hook)

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def get_attention_map(self, image_input, layer_idx=-1):
        """Extract attention map from specified layer"""
        self.attention_weights = []
        self.register_hooks()

        with torch.no_grad():
            _ = self.model.encode_image(image_input)

        self.remove_hooks()

        if self.attention_weights:
            # Use the specified layer's attention
            attn = self.attention_weights[layer_idx]
            return attn
        return None


# %%
def get_patch_similarities(model, sketch_img, target_img, patch_size=32, stride=16):
    """
    Slide a window across the target image and compute similarity with sketch
    """
    device = next(model.parameters()).device

    # Preprocess sketch
    sketch_input = preprocess(sketch_img).unsqueeze(0).to(device)
    with torch.no_grad():
        sketch_emb = model.encode_image(sketch_input)
        sketch_emb = sketch_emb / sketch_emb.norm(dim=-1, keepdim=True)

    # Convert target image to numpy for patch extraction
    target_np = np.array(target_img)
    h, w = target_np.shape[:2]

    similarities = []
    positions = []
    patches = []

    # Slide window across image
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            # Extract patch
            patch = target_np[y : y + patch_size, x : x + patch_size]
            patch_pil = Image.fromarray(patch)

            # Resize patch to 224x224 for CLIP
            patch_resized = patch_pil.resize((224, 224))
            patch_input = preprocess(patch_resized).unsqueeze(0).to(device)

            # Get patch embedding
            with torch.no_grad():
                patch_emb = model.encode_image(patch_input)
                patch_emb = patch_emb / patch_emb.norm(dim=-1, keepdim=True)

            # Compute similarity
            similarity = torch.cosine_similarity(sketch_emb, patch_emb).item()

            similarities.append(similarity)
            positions.append((x, y))
            patches.append(patch_pil)

    return similarities, positions, patches


# %%
def create_heatmap_overlay(
    target_img, similarities, positions, patch_size=32, stride=16
):
    """Create a heatmap overlay showing similarity scores"""
    target_np = np.array(target_img)
    h, w = target_np.shape[:2]

    # Create heatmap
    heatmap = np.zeros((h, w))

    for sim, (x, y) in zip(similarities, positions):
        # Fill the patch area with similarity score
        heatmap[y : y + patch_size, x : x + patch_size] = np.maximum(
            heatmap[y : y + patch_size, x : x + patch_size], sim
        )

    # Normalize heatmap
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    # Create colored heatmap
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(target_img)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(target_img)
    plt.imshow(heatmap, alpha=0.6, cmap="hot")
    plt.title("Similarity Heatmap")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    return heatmap


# %%
def highlight_best_matches(target_img, similarities, positions, patch_size=32, top_k=3):
    """Highlight the top-k matching patches with bounding boxes"""
    # Get top-k matches
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    # Create image with bounding boxes
    img_with_boxes = target_img.copy()
    draw = ImageDraw.Draw(img_with_boxes)

    colors = ["red", "yellow", "green", "blue", "purple"]

    for i, idx in enumerate(top_indices):
        x, y = positions[idx]
        similarity = similarities[idx]
        color = colors[i % len(colors)]

        # Draw bounding box
        draw.rectangle([x, y, x + patch_size, y + patch_size], outline=color, width=3)

        # Add similarity score
        draw.text((x, y - 15), f"{similarity:.3f}", fill=color)

    return img_with_boxes, top_indices


# %%
def gradcam_clip(model, image_input, target_layer=None):
    """
    Simple GradCAM-like visualization for CLIP
    """
    device = image_input.device
    image_input.requires_grad_(True)

    # Forward pass
    image_features = model.encode_image(image_input)

    # Use the mean of features as the target (could also use specific dimensions)
    target = image_features.mean()

    # Backward pass
    model.zero_grad()
    target.backward(retain_graph=True)

    # Get gradients
    gradients = image_input.grad

    # Create activation map
    # Take mean across color channels and convert to positive values
    activation_map = torch.mean(gradients.abs(), dim=1, keepdim=True)
    activation_map = F.interpolate(
        activation_map, size=(224, 224), mode="bilinear", align_corners=False
    )

    # Normalize
    activation_map = activation_map - activation_map.min()
    activation_map = activation_map / (activation_map.max() + 1e-8)

    return activation_map.squeeze().cpu().numpy()


# %%
# Example usage with your existing code
def analyze_matches_with_localization(sketch_img, match_images, model, preprocess):
    """Complete analysis pipeline"""
    device = next(model.parameters()).device

    fig, axes = plt.subplots(3, len(match_images) + 1, figsize=(20, 15))

    # Show sketch in first column
    for row in range(3):
        axes[row, 0].imshow(sketch_img)
        axes[row, 0].set_title("Sketch Query" if row == 0 else "")
        axes[row, 0].axis("off")

    for i, match_img in enumerate(match_images):
        col = i + 1

        # Row 1: Original matched image
        axes[0, col].imshow(match_img)
        axes[0, col].set_title(f"Match {i + 1}")
        axes[0, col].axis("off")

        # Row 2: Patch-based similarity heatmap
        print(f"Analyzing match {i + 1} with patch-based similarity...")
        similarities, positions, patches = get_patch_similarities(
            model, sketch_img, match_img, patch_size=64, stride=32
        )

        heatmap = np.zeros(np.array(match_img).shape[:2])
        for sim, (x, y) in zip(similarities, positions):
            patch_size = 64
            heatmap[y : y + patch_size, x : x + patch_size] = np.maximum(
                heatmap[y : y + patch_size, x : x + patch_size], sim
            )

        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        axes[1, col].imshow(match_img)
        axes[1, col].imshow(heatmap, alpha=0.6, cmap="hot")
        axes[1, col].set_title(f"Similarity Heatmap")
        axes[1, col].axis("off")

        # Row 3: Highlighted best matching regions
        img_with_boxes, top_indices = highlight_best_matches(
            match_img, similarities, positions, patch_size=64, top_k=3
        )
        axes[2, col].imshow(img_with_boxes)
        axes[2, col].set_title("Top Matching Regions")
        axes[2, col].axis("off")

    plt.tight_layout()
    plt.show()

    return similarities, positions


# %%
# Alternative: Simple gradient-based attention
def simple_attention_visualization(model, sketch_img, target_img, preprocess):
    """Simple attention visualization using gradients"""
    device = next(model.parameters()).device

    # Preprocess images
    sketch_input = preprocess(sketch_img).unsqueeze(0).to(device)
    target_input = preprocess(target_img).unsqueeze(0).to(device)
    target_input.requires_grad_(True)

    # Get embeddings
    with torch.no_grad():
        sketch_emb = model.encode_image(sketch_input)
        sketch_emb = sketch_emb / sketch_emb.norm(dim=-1, keepdim=True)

    # Forward pass for target (with gradients)
    target_emb = model.encode_image(target_input)
    target_emb = target_emb / target_emb.norm(dim=-1, keepdim=True)

    # Compute similarity and backpropagate
    similarity = torch.cosine_similarity(sketch_emb, target_emb)
    model.zero_grad()
    similarity.backward()

    # Get attention map from gradients
    gradients = target_input.grad
    attention_map = torch.mean(gradients.abs(), dim=1, keepdim=True)
    attention_map = F.interpolate(
        attention_map, size=target_img.size[::-1], mode="bilinear"
    )
    attention_map = attention_map.squeeze().cpu().numpy()

    # Normalize
    attention_map = (attention_map - attention_map.min()) / (
        attention_map.max() - attention_map.min() + 1e-8
    )

    # Visualize
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(sketch_img)
    plt.title("Sketch Query")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(target_img)
    plt.title("Target Image")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(target_img)
    plt.imshow(attention_map, alpha=0.6, cmap="hot")
    plt.title(f"Attention Map (sim: {similarity.item():.3f})")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    return attention_map


# %%
# Replace your final show_images cell with this enhanced version:

if "matches" in locals() and matches:
    print("Analyzing matches with localization...")

    # Method 1: Patch-based analysis (most reliable)
    target_img = matches[0]  # Analyze first match
    similarities, positions, patches = get_patch_similarities(
        model, sketch_img, target_img, patch_size=64, stride=32
    )

    # Create heatmap
    heatmap = create_heatmap_overlay(target_img, similarities, positions)

    # Highlight best matches
    img_with_boxes, top_indices = highlight_best_matches(
        target_img, similarities, positions, patch_size=64, top_k=5
    )

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(sketch_img)
    plt.title("Query Sketch")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(target_img)
    plt.title("Best Match")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(img_with_boxes)
    plt.title("Localized Regions")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # Method 2: Gradient-based attention
    print("Computing gradient-based attention...")
    attention_map = simple_attention_visualization(
        model, sketch_img, target_img, preprocess
    )

    # Method 3: Analyze all matches
    print("Full analysis of all matches...")
    analyze_matches_with_localization(sketch_img, matches[1:], model, preprocess)

else:
    print("No matches found to analyze")


# %%
