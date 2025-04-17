import cv2
import os
import numpy as np
from PIL import Image

def split_into_n_columns(img_path, output_dir, n_cols):
    os.makedirs(output_dir, exist_ok=True)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE if len(img_path.split('.')) > 1 and img_path.split('.')[-1] == "png" else cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Image could not be read.")
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    h, w = img.shape[:2]
    step = w // n_cols
    base = os.path.splitext(os.path.basename(img_path))[0]
    paths = []

    for i in range(n_cols):
        x0 = i * step
        x1 = (i + 1) * step if i < n_cols - 1 else w
        col_img = img[:, x0:x1]
        out_path = os.path.join(output_dir, f"{base}_col{i+1}.png")
        cv2.imwrite(out_path, col_img)
        paths.append(out_path)

    return paths

