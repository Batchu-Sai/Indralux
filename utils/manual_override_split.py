import cv2
import os
import numpy as np
from PIL import Image

def manual_override_split(img_path, output_dir, manual_bounds):
    os.makedirs(output_dir, exist_ok=True)
    img = cv2.imread(img_path)
    base = os.path.splitext(os.path.basename(img_path))[0]
    saved = []

    for i, (x0, x1) in enumerate(manual_bounds):
        col = img[:, x0:x1]
        out_path = os.path.join(output_dir, f"{base}_manual_col{i+1}.png")
        Image.fromarray(cv2.cvtColor(col, cv2.COLOR_BGR2RGB)).save(out_path)
        saved.append(out_path)

    return saved
