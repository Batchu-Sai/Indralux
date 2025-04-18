import os
import cv2
import numpy as np
from PIL import Image

def manual_override_split(img_path, output_dir, rois):
    """
    Manually split an image into regions based on bounding boxes.

    Parameters:
        img_path (str): path to full image
        output_dir (str): directory to save cropped panels
        rois (list of tuples): [(x0, x1), (x1, x2), ...] column bounds

    Returns:
        List of file paths to the saved cropped images
    """
    os.makedirs(output_dir, exist_ok=True)
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Cannot read image at {img_path}")
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    base = os.path.splitext(os.path.basename(img_path))[0]
    paths = []

    for i, (x0, x1) in enumerate(rois):
        if x1 - x0 <= 0 or x0 < 0 or x1 > img.shape[1]:
            print(f"[WARNING] Skipping invalid ROI: {(x0, x1)}")
            continue
        col = img[:, x0:x1]
        save_path = os.path.join(output_dir, f"{base}_manual_col{i+1}.png")
        Image.fromarray(col).save(save_path)
        paths.append(save_path)

    return paths
