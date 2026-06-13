# Homework 1: Image Analysis and Computer Vision
**Course:** CS898BA

## Setup and Execution
1. Clone the repository.
2. Install all required Python and OpenCV libraries using the command:
   `pip install -r requirements.txt`
3. Run `python Project1.py`

---
## Code Explanations

### Part 2
**Task 1**  
Explanation: Extracts individual color channel matrices and applies global statistical operators (np.min, np.mean, stats.mode, stats.skew) with axis=None to evaluate intensity distributions without altering array dimensions.

**Task 2**  
Explanation: Converts the input BGR image matrix into grayscale, HSV, CIELAB, and HLS representations using cv2.cvtColor to isolate specific visual features across different structural channel dimensions.

**Task 3**  
Explanation: Binarizes the grayscale image using Otsu's Thresholding via cv2.threshold, which automatically calculates the mathematically optimal intensity cutoff by maximizing inter-class variance between foreground and background pixels.

**Task 5**  
Explanation: Normalizes uneven scene lighting by isolating the brightness component (V channel) in the HSV color space, stretching its dynamic contrast range globally via cv2.equalizeHist before converting back to BGR.

**Task 6 & 7**  
Explanation: Applies 14 unique affine transformations across the 7 base images using custom 2x3 geometric transformation matrices (cv2.warpAffine) to introduce controlled spatial variations—such as midpoint rotations, directional translations, and horizontal shearing.

**Task 8**  
Explanation: Executes a multi-scale Gaussian Blur Sweep across 7 different standard deviation intensities (s = 0.5 to 3.5) using cv2.GaussianBlur with automatic kernel window sizing (ksize=(0,0)) to simulate progressive out-of-focus camera noise.

**Result discussion for Part 2, Task 8:**  
At low sigma values (0.5 to 1.0), minor background fuzz and small textures are cleaned up while keeping the main edges of objects sharp and clear so they are easy to find. On the other hand, at high sigma (1.5 to 3.5) values, the image is heavily blurred to wipe out fine details, leaving behind only large outlines which helps focus on the biggest shapes while ignoring background clutter.

### Part 3
**Task 1 - 3**  
Explanation: Initializes a deterministic random seed (42) to shuffle the global image pool, ensuring reproducible results. The pool is then partitioned into four equal subsets of 42 images, with 'Subset 2' selected to serve as the active testing set for edge detection analysis.

**Task 4**  
Explanation: Performs multi-operator edge extraction on the active image subset. Implements Sobel, Laplacian, Canny, and Prewitt filters to convert grayscale intensities into high-contrast edge maps, capturing structural boundaries via spatial gradient calculation and thresholding.

**Task 6**  
Explanation: Iterates through the generated filter dictionary to write isolated edge-detected image arrays to the 'output_edges/' directory, utilizing standardized filenames for precise tracking and subsequent plotting retrieval.

**Task 8**  
Explanation: Constructs a 3x3 diamond-layout visualization for each image, embedding processing metadata (Color Space, Transformation, Gaussian Sigma) into a monospace header. Automates the generation of comparison figures and programmatically injects a random sample of 6 plots into 'README.md' for final reporting.

### Part 3, Task 5 Discussion  
1. **Sobel**  
   * Pros: Fast and easy to use; provides a good balance between noise and edge detail.
   * Cons: Edges can appear thick, leading to less precise boundaries.

2. **Laplacian**
   * Pros: Very fast; excellent at finding rapid intensity changes.
   * Cons: Highly sensitive to noise, which often creates "false" edges. It also produces double-lined edges.

3. **Canny**
   * Pros: Generally the best; creates thin, clean, and accurate edges while filtering out noise.
   * Cons: More complex and computationally heavy than the others.

4. **Prewitt**
   * Pros: Very simple and fast for detecting horizontal or vertical lines.
   * Cons: Less accurate than Sobel and struggles with noise.

**Conclusion:** For this project, Sobel is the most effective all-around performer, providing the most consistent and usable edge maps. Prewitt serves as a strong secondary option, particularly for bright or high-contrast images where it offers a cleaner output by avoiding the excessive edge sensitivity seen in Sobel.

---

## Output Examples

![Comparison 1](output_plots/base_hls_sigma_0.5_comparison.png)
![Comparison 2](output_plots/base_hls_t1_sigma_0.0_comparison.png)
![Comparison 3](output_plots/base_hls_t2_sigma_0.5_comparison.png)
![Comparison 4](output_plots/base_hls_t1_sigma_2.0_comparison.png)
![Comparison 5](output_plots/base_cielab_t2_sigma_0.0_comparison.png)
![Comparison 6](output_plots/base_grayscale_t2_sigma_0.5_comparison.png)