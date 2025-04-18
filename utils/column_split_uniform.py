import numpy as np
from PIL import Image
import os
import cv2

def split_into_n_columns(img_path, output_dir, n_cols):
    os.makedirs(output_dir, exist_ok=True)
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"[split_into_n_columns] Could not load image: {img_path}")

    height, width = img.shape[:2]
    col_width = width // n_cols
    base = os.path.splitext(os.path.basename(img_path))[0]
    saved = []

    for i in range(n_cols):
        x0 = i * col_width
        x1 = (i + 1) * col_width if i < n_cols - 1 else width
        col = img[:, x0:x1]

        if col.shape[1] == 0:
            raise ValueError(f"[split_into_n_columns] Column {i+1} has zero width!")

        save_path = os.path.join(output_dir, f"{base}_col{i+1}.png")
        success = cv2.imwrite(save_path, col)

        if not success:
            raise IOError(f"[split_into_n_columns] Failed to save panel image: {save_path}")
        saved.append(save_path)

    return saved

