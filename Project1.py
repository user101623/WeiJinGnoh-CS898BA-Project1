import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats
import random
import re
from matplotlib.gridspec import GridSpec

# Part 2, Task 1: Find and Print Basic Image Statistics
img = cv2.imread('HW1_IMG_CS898BA.png')

if img is None:
    print("Error: Image not found.")
    exit()

# Split into individual color channels
r_channel = img[:, :, 2]
g_channel = img[:, :, 1]
b_channel = img[:, :, 0]

# Blue channel
print(f"  Min: {np.min(b_channel)}")
print(f"  Max: {np.max(b_channel)}")
print(f"  Average: {np.mean(b_channel):.2f}")
print(f"  Median: {np.median(b_channel)}")
print(f"  Mode: {int(stats.mode(b_channel, axis=None, keepdims=True).mode.item())}")
print(f"  Skew: {stats.skew(b_channel, axis=None)}")
print(f"  Range: {int(np.ptp(b_channel))}")
print(f"  Standard Deviation: {np.std(b_channel)}")
print(f"  Variance: {np.var(b_channel)}")

# Green Channel
print("\n[Green Channel]")
print(f"  Min: {np.min(g_channel)}")
print(f"  Max: {np.max(g_channel)}")
print(f"  Average: {np.mean(g_channel):.4f}")
print(f"  Median: {np.median(g_channel)}")
print(f"  Mode: {int(stats.mode(g_channel, axis=None, keepdims=True).mode.item())}")
print(f"  Skew: {stats.skew(g_channel, axis=None)}")
print(f"  Range: {int(np.ptp(g_channel))}")
print(f"  Standard Deviation: {np.std(g_channel)}")
print(f"  Variance: {np.var(g_channel)}")

# Red Channel
print("\n[Red Channel]")
print(f"  Min: {np.min(r_channel)}")
print(f"  Max: {np.max(r_channel)}")
print(f"  Average: {np.mean(r_channel):.4f}")
print(f"  Median: {np.median(r_channel)}")
print(f"  Mode: {int(stats.mode(r_channel, axis=None, keepdims=True).mode.item())}")
print(f"  Skew: {stats.skew(r_channel, axis=None):.4f}")
print(f"  Range: {int(np.ptp(r_channel))}")
print(f"  Standard Deviation: {np.std(r_channel):.4f}")
print(f"  Variance: {np.var(r_channel):.4f}")

# Part 2, Task 2: Converting to different color spaces

# Creating an output directory to save converted images
os.makedirs("output_images", exist_ok=True)

# Original BGR image
bgr_img = img.copy()

# Grayscale image
grayscale_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

# Binary Threshold using Otsu's thresholding
_, bt_img = cv2.threshold(grayscale_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# HSV image
hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)

# CIELAB image
lab_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)

# HLS image
hls_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HLS)


# Part 2, Task 3: Normalize lighting using Histogram Equalization
# Normalize lighting using Histogram Equalization
h, s, v = cv2.split(hsv_img)
v = cv2.equalizeHist(v)
hsv_normalized = cv2.merge([h, s, v])

# Part 2, Task 4: Convert back to RGB color space
converted_bgr_img = cv2.cvtColor(hsv_normalized, cv2.COLOR_HSV2BGR)

# Save all images
cv2.imwrite("output_images/base_original.png", bgr_img)
cv2.imwrite("output_images/base_grayscale.png", grayscale_img)
cv2.imwrite("output_images/base_binary.png", bt_img)
cv2.imwrite("output_images/base_hsv.png", hsv_img)
cv2.imwrite("output_images/base_cielab.png", lab_img)
cv2.imwrite("output_images/base_hls.png", hls_img)
cv2.imwrite("output_images/base_rgb_normalized.png", converted_bgr_img)
print("Saved all 7 images successfully")

# Part 2, Task 6: Applying Affine Transformations
h_dim, w_dim = img.shape[:2]

# Define the 7 base images into a dictionary for looping
base_images = {
    "original": bgr_img,
    "grayscale": grayscale_img,
    "binary": bt_img,
    "hsv": hsv_img,
    "cielab": lab_img,
    "hls": hls_img,
    "rgb_normalized": converted_bgr_img
}

affine_transformation_configs = [
    ("rotate", 45,     "translate", (20, -10)),  # original
    ("rotate", 90,     "scale", 1.5),            # grayscale
    ("shear", 0.2,     "translate", (-15, 30)),  # binary
    ("rotate", -30,    "scale", 0.8),            # hsv
    ("shear", -0.15,   "translate", (40, 40)),   # cielab
    ("rotate", 120,    "scale", 1.2),            # hls
    ("shear", 0.3,     "rotate", 180)            # rgb_normalized
]

# Creating a pool to store all 21 tracking tuples in the format: (filename, image)
compilation_image_pool = []

for idx, (base_name, base_img) in enumerate(base_images.items()):
    # 1. Append and save the base image variant first (1 of 21)
    cv2.imwrite(f"output_images/base_{base_name}.png", base_img)
    compilation_image_pool.append((f"base_{base_name}", base_img))

    # 2. Extract the 2 unique transformations assigned to this image
    t1_type, t1_param, t2_type, t2_param = affine_transformation_configs[idx]

    # Apply Transformation 1
    if t1_type == "rotate":
        M = cv2.getRotationMatrix2D((w_dim // 2, h_dim // 2), t1_param, 1.0)
        t1_img = cv2.warpAffine(base_img, M, (w_dim, h_dim))
    elif t1_type == "shear":
        M = np.float32([[1, t1_param, 0], [0, 1, 0]])
        t1_img = cv2.warpAffine(base_img, M, (w_dim, h_dim))
        
    cv2.imwrite(f"output_images/base_{base_name}_t1.png", t1_img)
    compilation_image_pool.append((f"base_{base_name}_t1", t1_img))
    
    # --- Apply Transformation 2 ---
    if t2_type == "translate":
        tx, ty = t2_param
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        t2_img = cv2.warpAffine(base_img, M, (w_dim, h_dim))
    elif t2_type == "scale":
        scaled = cv2.resize(base_img, None, fx=t2_param, fy=t2_param, interpolation=cv2.INTER_LINEAR)
        # Handle canvas sizing so matrix bounds don't crash or distort shape tracking
        if t2_param > 1.0:
            t2_img = scaled[0:h_dim, 0:w_dim]
        else:
            t2_img = np.zeros_like(base_img)
            h_s, w_s = scaled.shape[:2]
            t2_img[0:h_s, 0:w_s] = scaled
            
    cv2.imwrite(f"output_images/base_{base_name}_t2.png", t2_img)
    compilation_image_pool.append((f"base_{base_name}_t2", t2_img))

print(f"Total images in compilation image pool = {len(compilation_image_pool)} / 21")

# Part 2, Task 8: Gaussian Blur

# Define the 7 required levels of sigma
sigmas = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]

# Final image pool that contains all 168 images
final_image_pool = []
total_images_generated = 0

for label_name, current_frame in compilation_image_pool:
    # 1. Save the baseline unblurred version (Sigma = 0.0)
    name_0 = f"{label_name}_sigma_0.0"
    cv2.imwrite(f"output_images/{name_0}.png", current_frame)
    final_image_pool.append((name_0, current_frame))
    total_images_generated += 1
    
    # 2. Sweep through the 7 specific sigma variations
    for s in sigmas:
        blurred_frame = cv2.GaussianBlur(current_frame, (0, 0), sigmaX=s, sigmaY=s)
        
        name_s = f"{label_name}_sigma_{s}"
        cv2.imwrite(f"output_images/{name_s}.png", blurred_frame)
        final_image_pool.append((name_s, blurred_frame))
        total_images_generated += 1
print(f"Successfully exported {total_images_generated} images.")

# Part 3, Task 1-3: Creating Subsets

# Setting seed
random.seed(42)

# Copy and shuffle the final image pool to create a randomized version for subset selection
shuffled_pool = final_image_pool.copy()
random.shuffle(shuffled_pool)

# Partition into 4 equal subsets of 42 images each
subset_size = 42
subsets = [shuffled_pool[i:i + subset_size] for i in range(0, len(shuffled_pool), subset_size)]

# Select Subset 2 for our active edge detection evaluation pipeline
active_subset = subsets[1] 
print (f"Chosen subset: Subset 2 with {len(active_subset)} images.")

# Part 3, Task 4: Edge Detection
os.makedirs("output_edges", exist_ok=True)
os.makedirs("output_plots", exist_ok=True)

BG_COLOR = '#363636'
TEXT_COLOR = 'white'
generated_plot_paths = []

COLOR_SPACE_LABEL = {
    '': 'BGR', 'grayscale': 'Grayscale', 'binary': 'Binary',
    'HSV': 'HSV', 'CIELAB': 'CIELAB', 'HLS': 'HLS', 'equalized': 'Equalized'
}

# Prewitt directional kernels
px = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
py = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)

for img_name, img_matrix in active_subset:
    # 1. Edge Detection Logic
    gray = cv2.cvtColor(img_matrix, cv2.COLOR_BGR2GRAY) if len(img_matrix.shape) == 3 else img_matrix.copy()
    filters = {
        "sobel": np.uint8(np.clip(cv2.magnitude(cv2.Sobel(gray, cv2.CV_64F, 1, 0), cv2.Sobel(gray, cv2.CV_64F, 0, 1)), 0, 255)),
        "laplacian": np.uint8(np.clip(np.absolute(cv2.Laplacian(gray, cv2.CV_64F)), 0, 255)),
        "canny": cv2.Canny(gray, 50, 150),
        "prewitt": np.uint8(np.clip(cv2.magnitude(cv2.filter2D(gray, -1, px).astype(float), cv2.filter2D(gray, -1, py).astype(float)), 0, 255))
    }
    for name, edge_img in filters.items():
        cv2.imwrite(f"output_edges/{img_name}_edge_{name}.png", edge_img)

    # 2. Metadata Parsing
    t_idx = 0 if "_t1" in img_name else 1 if "_t2" in img_name else 0
    cfg = affine_transformation_configs[t_idx % len(affine_transformation_configs)]
    t_details = f"{cfg[0].capitalize()}({cfg[1]}), {cfg[2].capitalize()}({cfg[3]})"
    sigma_val = re.search(r'sigma_([\d.]+)', img_name).group(1) if 'sigma_' in img_name else "0.0"
    cs_key = img_name.split('_')[1]
    header_str = (f"Original\n→ Color Space: {COLOR_SPACE_LABEL.get(cs_key, cs_key.capitalize())}\n"
                  f"→ Transformation: {t_details}\n→ Gaussian Blur (σ): {sigma_val}")

    # 3. Canvas Initialization (3x3 GridSpec)
    fig = plt.figure(figsize=(10, 12), facecolor=BG_COLOR)
    fig.text(0.5, 0.97, header_str, ha='center', va='top', color=TEXT_COLOR, fontsize=12, family='monospace', linespacing=1.8)
    gs = GridSpec(3, 3, figure=fig, top=0.76, bottom=0.04, left=0.04, right=0.96, hspace=0.25, wspace=0.1)
    
    panels = {
        (0, 1): ('Sobel Edge', filters['sobel']),
        (1, 0): ('Laplacian Edge', filters['laplacian']),
        (1, 1): ('Input Image', img_matrix),
        (1, 2): ('Canny Edge', filters['canny']),
        (2, 1): ('Prewitt Edge', filters['prewitt']),
    }
    
    # 4. Render
    for (r, c), (title, im) in panels.items():
        ax = fig.add_subplot(gs[r, c])
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB) if im is not None and im.ndim == 3 else im, cmap='gray')
        ax.set_title(title, color=TEXT_COLOR, fontsize=11, pad=8)
        ax.set_facecolor(BG_COLOR)
        ax.axis('off')

    plt.savefig(f"output_plots/{img_name}_comparison.png", facecolor=BG_COLOR, dpi=100)
    plt.close(fig)

# Part 3, Task 8: Comparison Plots
for img_name, img_matrix in active_subset:
    # 1. Precise Extraction
    # Map t1/t2 to index 0/1 from your config list
    t_idx = 0 if "_t1" in img_name else 1 if "_t2" in img_name else 0
    cfg = affine_transformation_configs[t_idx % len(affine_transformation_configs)]
    t_details = f"{cfg[0].capitalize()}({cfg[1]}), {cfg[2].capitalize()}({cfg[3]})"
    
    sigma_val = re.search(r'sigma_([\d.]+)', img_name).group(1) if 'sigma_' in img_name else "0.0"
    cs_key = img_name.split('_')[1] # Extracts 'binary', 'hsv', etc.

    # 2. Build Header String
    header_str = (
        f"Original\n"
        f"→ Color Space: {COLOR_SPACE_LABEL.get(cs_key, cs_key.capitalize())}\n"
        f"→ Transformation: {t_details}\n"
        f"→ Gaussian Blur (σ): {sigma_val}"
    )

    # 3. Load Images
    sobel = cv2.imread(f"output_edges/{img_name}_edge_sobel.png")
    laplacian = cv2.imread(f"output_edges/{img_name}_edge_laplacian.png")
    canny = cv2.imread(f"output_edges/{img_name}_edge_canny.png")
    prewitt = cv2.imread(f"output_edges/{img_name}_edge_prewitt.png")

    # 4. Create 3x3 Grid
    fig = plt.figure(figsize=(10, 12), facecolor=BG_COLOR)
    fig.text(0.5, 0.97, header_str, ha='center', va='top',
             color=TEXT_COLOR, fontsize=12, family='monospace', linespacing=1.8)

    gs = GridSpec(3, 3, figure=fig, top=0.76, bottom=0.04, left=0.04, right=0.96, hspace=0.25, wspace=0.1)
    
    panels = {
        (0, 1): ('Sobel Edge', sobel),
        (1, 0): ('Laplacian Edge', laplacian),
        (1, 1): ('Input Image', img_matrix),
        (1, 2): ('Canny Edge', canny),
        (2, 1): ('Prewitt Edge', prewitt),
    }
    
    for (r, c), (title, im) in panels.items():
        ax = fig.add_subplot(gs[r, c])
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB) if im is not None and im.ndim == 3 else im, cmap='gray')
        ax.set_title(title, color=TEXT_COLOR, fontsize=11, pad=8)
        ax.set_facecolor(BG_COLOR)
        ax.axis('off')

    out_path = os.path.join('output_plots', f"{img_name}_comparison.png")
    fig.savefig(out_path, facecolor=BG_COLOR, dpi=100)
    plt.close(fig)
    generated_plot_paths.append(out_path)

# 5. README Injection
README_PATH = 'README.md'
if os.path.exists(README_PATH):
    # Filter paths
    comparison_samples = [p for p in generated_plot_paths if "_comparison.png" in p]
    readme_samples = random.sample(comparison_samples, min(6, len(comparison_samples)))
    
    with open(README_PATH, 'r+') as f:
        content = f.read()
        marker = "# Output Examples"
        if marker in content:
            head, tail = content.split(marker, 1)
            new_content = head + marker + "\n\n" + \
                          "\n".join([f"![{os.path.basename(p)}]({p})" for p in readme_samples]) + \
                          (tail.split('\n\n', 1)[-1] if len(tail.split('\n\n', 1)) > 1 else "")
            f.seek(0)
            f.write(new_content)
            f.truncate()
print("\nSUCCESS")