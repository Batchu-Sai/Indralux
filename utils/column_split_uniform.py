import os
import cv2
import numpy as np
from PIL import Image

def split_into_n_columns(img_path, output_dir, n_columns):
    """
    Uniformly splits an RGB image into n vertical columns.

    Parameters:
        img_path (str): Full image path
        output_dir (str): Where to save cropped columns
        n_columns (int): Number of equal slices to split into

    Returns:
        List of file paths to saved column images
    """
    os.makedirs(output_dir, exist_ok=True)
    img = cv2.imread(img_path)
    if img is None or len(img.shape) != 3:
        raise ValueError(f"Invalid or non-RGB image: {img_path}")

    height, width, _ = img.shape
    step = width // n_columns
    base = os.path.splitext(os.path.basename(img_path))[0]
    paths = []

    for i in range(n_columns):
        x0 = i * step
        x1 = width if i == n_columns - 1 else (i + 1) * step
        col_img = img[:, x0:x1]
        path = os.path.join(output_dir, f"{base}_col{i+1}.png")
        Image.fromarray(cv2.cvtColor(col_img, cv2.COLOR_BGR2RGB)).save(path)
        paths.append(path)

    return paths

