# AI Usage Log

| Date and Time | Prompt | Tool | Response Synopsis | Change |
| :--- | :--- | :--- | :--- | :--- |
| 06/13/2026 11:55 AM | "guide on which function to use to find image statistics for each channel" | Gemini | Provided direct NumPy and SciPy function mappings (`np.min`, `stats.mode`, `stats.skew`) using `axis=None` parameters to run global 2D array analysis. | Implemented it for Part 2, Task 1  |
| 06/13/2026 12:08 PM | "how to perform binary thresholding using otsu thresholding in python for image" | Gemini | Explained parameters and return signatures for `cv2.threshold` when combined with the `cv2.THRESH_OTSU` flag. | Implemented it for Part 2, Task 2 |
| 06/13/2026 12:09 PM | "how to normalize lighting using histogram equalization for image in python?" | Gemini | Detailed the process of isolating illumination from color by splitting HSV channels, equalizing the V channel, and re-merging. | Implemented it for Part 2, Task 3 . |
| 06/13/2026 12:22 PM | "how to use affine transformation on images like scaling, shearing, rotation in python" | Gemini | Provided technical documentation and NumPy/OpenCV matrix code examples for geometric image shifts (`cv2.getRotationMatrix2D`, `cv2.warpAffine`, `cv2.resize`). | Implemented it for Part 2, Task 6 |
| 06/13/2026 12:35 PM | "how to apply gaussian blur to image given the levels of sigma in python" | Gemini | Explained dynamic `cv2.GaussianBlur` kernel window size derivation using the `(0, 0)` tuple with explicit `sigmaX` and `sigmaY` assignments. | Implemented it for Task 2, Part 8 |