# core/overlay.py

import cv2
import numpy as np
from skimage.measure import regionprops
import matplotlib.cm as cm

def draw_colored_overlay_with_cv2(img, labels, df):
    overlay = img.copy()
    cmap = cm.get_cmap("viridis", np.max(labels) + 1)

    for region in regionprops(labels):
        if region.label in df['Cell_ID'].values:
            y, x = map(int, region.centroid)
            color = tuple((np.array(cmap(region.label)[:3]) * 255).astype(int))
            cv2.drawContours(overlay, [region.convex_image.astype(np.uint8)], -1, color, 1)
            cv2.putText(overlay, str(region.label), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

    return overlay
