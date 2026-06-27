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

# Set random seed for reproducibility
cv2.setRNGSeed(42)

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

# Part 4: Classical and Optimization-Based Segmentation
# 1. Prepare HSV image and enhance contrast
hsv_img = cv2.cvtColor(normalized_img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv_img)

# Boost brightness (V) slightly to help K-Means separate dark shadows from the figure
v_enhanced = cv2.convertScaleAbs(v, alpha=1.1, beta=15)
hsv_enhanced = cv2.merge([h, s, v_enhanced])

# 2. Flatten for clustering
pixel_data = np.float32(hsv_enhanced.reshape((-1, 3)))
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

# 3. Test clusters and save for inspection
k_range = [3, 4, 5]
for k in k_range:
    _, labels, centers = cv2.kmeans(pixel_data, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    
    # Sort by brightness (V channel)
    brightness_order = np.argsort(centers[:, 2])
    label_map = np.take(brightness_order, labels.reshape(hsv_img.shape[:2]))
    
    # Save a mask for each cluster to identify the most appropriate cluster
    for cluster_id in range(k):
        mask = np.where(label_map == cluster_id, 255, 0).astype(np.uint8)
        cv2.imwrite(f'Cluster_Analysis_K{k}_ID{cluster_id}.png', mask)

# 4. Final selection 
OPTIMAL_K = 5
FIGURE_CLUSTER_ID = 4

_, labels, centers = cv2.kmeans(pixel_data, OPTIMAL_K, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
brightness_order = np.argsort(centers[:, 2])
label_map = np.take(brightness_order, labels.reshape(hsv_img.shape[:2]))

k_means_mask = np.where(label_map == FIGURE_CLUSTER_ID, 255, 0).astype(np.uint8)
kmeans_result = cv2.bitwise_and(normalized_img, normalized_img, mask=k_means_mask)
cv2.imwrite('HW2_CS898BA_KMeans_Final_Mask.png', k_means_mask)
cv2.imwrite('HW2_CS898BA_KMeans_Foreground.png', kmeans_result)