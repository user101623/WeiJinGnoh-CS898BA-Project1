# Homework 1: Image Analysis and Computer Vision
** Course:**: CS898BA

## Setup and Execution
1. Clone the repository
2. Install all required Python and OpenCV libraries using the command:
```
pip install -r requirements.txt
```
3. Run `python3 Project1.py

# Code Explanations
Part 2, Task 1
Explanation: Extracts individual color channel matrices and applies global statistical operators (np.min, np.mean, stats.mode, stats.skew) with axis=None to evaluate intensity distributions without altering array dimensions.

Part 2, Task 2
Explanation: Converts the input BGR image matrix into grayscale, HSV, CIELAB, and HLS representations using cv2.cvtColor to isolate specific visual features across different structural channel dimensions.

Part 2, Task 3
Explanation: Binarizes the grayscale image using Otsu's Thresholding via cv2.threshold, which automatically calculates the mathematically optimal intensity cutoff by maximizing inter-class variance between foreground and background pixels.

Part 2, Task 5
Explanation: Normalizes uneven scene lighting by isolating the brightness component (V channel) in the HSV color space, stretching its dynamic contrast range globally via cv2.equalizeHist before converting back to BGR.

Part 2, Task 6 & 7
Explanation: Applies 14 unique affine transformations across the 7 base images using custom 2x3 geometric transformation matrices (cv2.warpAffine) to introduce controlled spatial variations—such as midpoint rotations, directional translations, and horizontal shearing.

Part 2, Task 8 & 9
Explanation: Executes a multi-scale Gaussian Blur Sweep across 7 different standard deviation intensities (s = 0.5 to 3.5) using cv2.GaussianBlur with automatic kernel window sizing (ksize=(0,0)) to simulate progressive out-of-focus camera noise.

Result discussion for Part 2, Task 9:
At low sigma values (0.5 to 1.0), minor background fuzz and small textures are cleanted up while keeping the main edges of objects sharp and clear so they are easy to find. On the other hand, at high sigma (1.5 to 3.5) values, the image is heavily blurred to wipe out fine details, leaving behind only large outlines which helps focus on the biggest shapes while ignoring background clutter.