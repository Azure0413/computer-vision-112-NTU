# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 21:22:33 2024

@author: Eric
"""

import json

# Read the first annotations.json file
with open('../hw2_data/p2_data/train/annotations.json', 'r') as file:
    data1 = json.load(file)

# Read the second annotations.json file
with open('./train_new/annotations.json', 'r') as file:
    data2 = json.load(file)

# Merge filenames and labels
merged_filenames = data1['filenames'] + data2['filenames']
merged_labels = data1['labels'] + data2['labels']

# Create a new dictionary for the merged data
merged_data = {
    'filenames': merged_filenames,
    'labels': merged_labels
}

# Write the merged data to a new annotations.json file
with open('annotations.json', 'w') as file:
    json.dump(merged_data, file)