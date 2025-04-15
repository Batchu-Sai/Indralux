import cv2, os
import numpy as np
from PIL import Image
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

def split_columns_improved(img_path, output_dir, sigma=8, prominence=0.015):
    os.makedirs(output_dir, exist_ok=True)

    # Read image safely
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Image {img_path} could not be loaded.")

    # Remove horizontal artifacts like white bars
    cleaned = img.copy()
    row_sums = np.mean(cleaned, axis=1)
    high_rows = np.where(row_sums > 245)[0]  # over-bright rows
    cleaned[high_rows] = np.median(cleaned)

    # Column projection
    proj = np.sum(cleaned, axis=0).astype(np.float32)
    smoothed = gaussian_filter1d(proj, sigma=sigma)
    norm_proj = (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min())
    inverted = 1 - norm_proj

    # Detect valleys between columns
    peaks, _ = find_peaks(inverted, distance=img.shape[1] // 20, prominence=prominence)
    boundaries = [0] + list(peaks) + [img.shape[1]]

    base = os.path.splitext(os.path.basename(img_path))[0]
    count = 0

    for i in range(len(boundaries) - 1):
        x0, x1 = boundaries[i], boundaries[i + 1]
        if x1 - x0 < 20:
            continue
        col_img = cleaned[:, x0:x1]
        if col_img.shape[1] < 10:
            continue
        Image.fromarray(col_img).save(
            os.path.join(output_dir, f"{base}_col{i + 1}_improved.png")
        )
        count += 1

    if count == 0:
        raise RuntimeError(f"No columns detected in image {img_path}")

def split_into_n_columns(img_path, output_dir, n_cols):
    os.makedirs(output_dir, exist_ok=True)
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not load image: {img_path}")

    height, width = img.shape[:2]
    col_width = width // n_cols
    base = os.path.splitext(os.path.basename(img_path))[0]
    saved = []

    for i in range(n_cols):
        x0 = i * col_width
        x1 = (i + 1) * col_width if i < n_cols - 1 else width
        col = img[:, x0:x1]
        save_path = os.path.join(output_dir, f"{base}_col{i+1}.png")
        Image.fromarray(cv2.cvtColor(col, cv2.COLOR_BGR2RGB)).save(save_path)
        saved.append(save_path)

    return saved
