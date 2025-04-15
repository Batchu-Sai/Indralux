import cv2, os
import numpy as np
from PIL import Image
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

def split_columns_improved(img_path, output_dir, sigma=8, prominence=0.015, min_width=20):
    os.makedirs(output_dir, exist_ok=True)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"❌ Could not read image: {img_path}")
        return

    proj = np.sum(img, axis=0).astype(np.float32)
    smoothed = gaussian_filter1d(proj, sigma=sigma)
    norm_proj = (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min() + 1e-6)
    inverted = 1 - norm_proj
    peaks, _ = find_peaks(inverted, distance=img.shape[1] // 20, prominence=prominence)

    if len(peaks) == 0:
        print(f"⚠️ No peaks found for {img_path} — skipping.")
        return

    boundaries = [0] + list(peaks) + [img.shape[1]]
    base = os.path.splitext(os.path.basename(img_path))[0]
    n_saved = 0

    for i in range(len(boundaries) - 1):
        x0, x1 = boundaries[i], boundaries[i+1]
        if x1 - x0 < min_width:
            continue
        col_img = Image.fromarray(img[:, x0:x1])
        save_path = os.path.join(output_dir, f"{base}_col{i+1}_improved.png")
        col_img.save(save_path)
        n_saved += 1

    print(f"✅ {n_saved} columns saved from {base}")
