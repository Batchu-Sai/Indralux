import cv2, os
import numpy as np
from PIL import Image
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

def split_columns_improved(img_path, output_dir, sigma=8, prominence=0.015):
    os.makedirs(output_dir, exist_ok=True)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    proj = np.sum(img, axis=0).astype(np.float32)
    smoothed = gaussian_filter1d(proj, sigma=sigma)
    norm_proj = (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min())
    inverted = 1 - norm_proj
    peaks, _ = find_peaks(inverted, distance=img.shape[1] // 20, prominence=prominence)
    boundaries = [0] + list(peaks) + [img.shape[1]]
    base = os.path.splitext(os.path.basename(img_path))[0]

    for i in range(len(boundaries) - 1):
        x0, x1 = boundaries[i], boundaries[i+1]
        if x1 - x0 < 20: continue
        col_img = Image.fromarray(img[:, x0:x1])
        col_img.save(os.path.join(output_dir, f"{base}_col{i+1}_improved.png"))
