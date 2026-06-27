import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import re

# Part 2: Image Preprocessing and Normalization
img = cv2.imread('HW1_IMG_CS898BA.png')

if img is None:
    print("Error: Image not found.")
    exit()

# Split into individual color channels
r_channel = img[:, :, 2]
g_channel = img[:, :, 1]
b_channel = img[:, :, 0]

# Apply Histogram Equalization
r_eq = cv2.equalizeHist(r_channel)
g_eq = cv2.equalizeHist(g_channel)
b_eq = cv2.equalizeHist(b_channel)

# Merge back the channels
normalized_img = cv2.merge([b_eq, g_eq, r_eq])

# Save the normalized color image
cv2.imwrite('HW2_IMG_CS898BA_Normalized.png', normalized_img)

# Part 3: Threshold-Based Segmentation
# 1. Convert the image to grayscale
gray_img = cv2.cvtColor(normalized_img, cv2.COLOR_BGR2GRAY)

# 2. Apply Otsu’s thresholding
otsu_val, otsu_mask = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 3. Apply Adaptive thresholding
adaptive_mask = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# Extract foreground object for Otsu thresholding  
otsu_foreground = cv2.bitwise_and(normalized_img, normalized_img, mask=otsu_mask)

# Extract foreground object for Adaptive thresholding 
adaptive_foreground = cv2.bitwise_and(normalized_img, normalized_img, mask=adaptive_mask)

# Save the resulting masks and extractions
cv2.imwrite('HW2_CS898BA_Otsu_Mask.png', otsu_mask)
cv2.imwrite('HW2_CS898BA_Otsu_Foreground.png', otsu_foreground)
cv2.imwrite('HW2_CS898BA_Adaptive_Mask.png', adaptive_mask)
cv2.imwrite('HW2_CS898BA_Adaptive_Foreground.png', adaptive_foreground)