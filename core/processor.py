import os
import cv2
import numpy as np
from skimage import filters, feature, segmentation, measure, morphology
from scipy import ndimage as ndi
import pandas as pd

def process_with_breaks(img_path, n_columns=1, column_labels=None):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    if img.ndim == 2:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = img[:, :, :3]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    channels = img_rgb.shape[2]

    f_actin = img_rgb[:, :, 0] if channels >= 1 else np.zeros_like(img_rgb[:, :, 0])
    ve_cadherin = img_rgb[:, :, 1] if channels >= 2 else np.zeros_like(img_rgb[:, :, 0])
    dapi = img_rgb[:, :, 2] if channels >= 3 else np.zeros_like(img_rgb[:, :, 0])

    actin_thresh = filters.threshold_otsu(f_actin)
    borders = f_actin > actin_thresh

    dapi_blur = filters.gaussian(dapi, sigma=1)
    dapi_thresh = filters.threshold_otsu(dapi_blur)
    nuclei = dapi_blur > dapi_thresh

    distance = ndi.distance_transform_edt(nuclei)
    local_maxi = feature.peak_local_max(distance, labels=nuclei, footprint=np.ones((3, 3)), exclude_border=False)
    markers = np.zeros_like(nuclei, dtype=int)
    for i, (y, x) in enumerate(local_maxi, start=1):
        markers[y, x] = i

    elevation_map = filters.sobel(f_actin)
    segmentation_labels = segmentation.watershed(elevation_map, markers, mask=borders)

    results = []
    for region in measure.regionprops(segmentation_labels, intensity_image=ve_cadherin):
        if region.area < 100:
            continue
        full_mask = segmentation_labels == region.label
        interior = morphology.binary_erosion(full_mask, morphology.disk(3))
        periphery = full_mask ^ interior

        ve_per = np.mean(ve_cadherin[periphery]) if np.any(periphery) else 0
        ve_cyto = np.mean(ve_cadherin[interior]) if np.any(interior) else 1
        f_per = np.mean(f_actin[periphery]) if np.any(periphery) else 0
        f_cyto = np.mean(f_actin[interior]) if np.any(interior) else 1
        dapi_mean = np.mean(dapi[region.coords[:, 0], region.coords[:, 1]]) if region.coords.size else 0
        ve_ratio = ve_per / (ve_cyto + 1e-6)
        f_ratio = f_per / (f_cyto + 1e-6)
        column_id = int(region.centroid[1] // (img.shape[1] / n_columns))
        column_label = column_labels[column_id] if column_labels and column_id < len(column_labels) else str(column_id)

        skel = morphology.skeletonize(periphery)
        n_breaks = measure.label(skel).max()

        results.append({
            "Cell_ID": region.label,
            "Centroid_X": region.centroid[1],
            "Column_ID": column_id,
            "Column_Label": column_label,
            "Periphery_Intensity_VE": ve_per,
            "Cytoplasm_Intensity_VE": ve_cyto,
            "VE_Ratio": ve_ratio,
            "Periphery_Intensity_F": f_per,
            "Cytoplasm_Intensity_F": f_cyto,
            "F_Ratio": f_ratio,
            "DAPI_Intensity": dapi_mean,
            "Periphery_Breaks": n_breaks
        })

    df = pd.DataFrame(results)
    return df, segmentation_labels, img_rgb
