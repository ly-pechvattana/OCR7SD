"""
Filename: img_rename.py
Author: LY Pechvattana
Date: 10 April 2026
Version: 1.0
Description: This code auto rename the image files in the specified directory with a 
sequential number format based on the folder name. 

Input: Image files in the specified directory (e.g., './img/raw')
Output: Renamed image files in the same directory with the format 'foldername_001.jpg
"""

import os

# Define the image directory
img_dir = './img/raw' ## --> Change this to image directory

# Get the folder name (last part of the path)
folder_name = os.path.basename(img_dir)

# Get all image files and sort them
image_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])

# Rename each file with folder_name + sequential number
for index, filename in enumerate(image_files, start=1):
    file_extension = os.path.splitext(filename)[1]
    new_filename = f'{folder_name}_{index:03d}{file_extension}'
    old_path = os.path.join(img_dir, filename)
    new_path = os.path.join(img_dir, new_filename)
    os.rename(old_path, new_path)
    print(f'Renamed: {filename} → {new_filename}')
