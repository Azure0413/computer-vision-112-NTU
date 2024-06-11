# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 01:15:49 2024

@author: Eric
"""

import numpy as np

# Load the data from the file
keypoints_gt = np.load('./testdata/1_gt.npy')

# Print the loaded data
print(keypoints_gt)

import numpy as np
import cv2
import argparse

# Parse command-line argument for the input image path
parser = argparse.ArgumentParser(description='Convert PNG image to NumPy array and display it')
parser.add_argument('--image_path', default='./testdata/1.png', help='Path to the PNG image')
args = parser.parse_args()

# Read the PNG image
image = cv2.imread(args.image_path, cv2.IMREAD_GRAYSCALE)

# Convert the image to a NumPy array
image_array = np.array(image)

# Display the NumPy array
print("NumPy array representation of the image:")
print(image_array)