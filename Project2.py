import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import re

# Part 2, Task 1: Find and Print Basic Image Statistics
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