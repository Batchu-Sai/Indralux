import cv2
import os
from PIL import Image

def split_columns_fixed(img_path, output_dir, n_columns):
    os.makedirs(output_dir, exist_ok=True)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    col_width = w // n_columns
    base = os.path.splitext(os.path.basename(img_path))[0]

    for i in range(n_columns):
        x0 = i * col_width
        x1 = (i + 1) * col_width if i < n_columns - 1 else w
        col_img = Image.fromarray(img[:, x0:x1])
        col_img.save(os.path.join(output_dir, f"{base}_col{i+1}_manual{n_columns}.png"))
