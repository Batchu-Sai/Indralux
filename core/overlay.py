import numpy as np
import cv2
from skimage import measure
import random

def draw_colored_overlay_with_cv2(image, labels, df):
    overlay = image.copy()
    rng = np.random.default_rng(seed=42)

    for region in measure.regionprops(labels):
        if region.label not in df["Cell_ID"].values:
            continue
        mask = labels == region.label
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        color = tuple(int(c) for c in rng.integers(50, 255, size=3))
        cv2.drawContours(overlay, contours, -1, color, 1)

        y, x = map(int, region.centroid)
        label = str(region.label)
        cv2.putText(overlay, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

    return overlay
