# core/overlay.py

import cv2
import numpy as np
from skimage.measure import regionprops
import matplotlib.cm as cm

def draw_colored_overlay_with_cv2(image_rgb, label_mask, df):
    overlay = image_rgb.copy()
    overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)  # OpenCV uses BGR

    props = regionprops(label_mask)

    for region in props:
        if region.label not in df['Cell_ID'].values:
            continue

        mask = (label_mask == region.label).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            color = tuple(np.random.randint(0, 255, 3).tolist())
            cv2.drawContours(overlay, contours, -1, color, 1)

    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    return overlay

