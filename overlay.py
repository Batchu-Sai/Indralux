
import cv2
import numpy as np
from skimage import measure

def draw_colored_overlay_with_cv2(img_rgb, labels, df, label_field='Cell_ID'):
    overlay = img_rgb.copy()
    for _, row in df.iterrows():
        cell_id = int(row[label_field])
        mask = labels == cell_id
        contours = measure.find_contours(mask, 0.5)
        for contour in contours:
            contour = np.round(contour).astype(np.int32)
            for y, x in contour:
                cv2.circle(overlay, (x, y), 1, (0, 255, 0), -1)
        cy, cx = np.mean(np.argwhere(mask), axis=0).astype(int)
        cv2.putText(overlay, str(cell_id), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    return overlay
