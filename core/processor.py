import os
import cv2
import numpy as np
from skimage import filters, feature, segmentation, measure, morphology
from scipy import ndimage as ndi
import pandas as pd

def process_with_breaks(img_path, n_columns=1, column_labels=None, marker_channels=None):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    if len(img.shape) == 2 or img.shape[2] == 1:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        channels = {"ve": 0, "f": 0, "dapi": 0}
    else:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        channels = marker_channels or {"ve": 1, "f": 0, "dapi": 2}

    f_actin = img_rgb[:, :, channels["f"]]
    ve_cadherin = img_rgb[:, :, channels["ve"]]
    dapi = img_rgb[:, :, channels["dapi"]]

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
        if region.area < 100: continue
        mask = segmentation_labels == region.label
        interior = morphology.binary_erosion(mask, morphology.disk(3))
        periphery = mask ^ interior
        skel = morphology.skeletonize(periphery)
        n_breaks = measure.label(skel).max()
        column_id = int(region.centroid[1] // (img.shape[1] / n_columns))
        column_label = column_labels[column_id] if column_labels and column_id < len(column_labels) else str(column_id)

        results.append({
            "Cell_ID": region.label,
            "Centroid_X": region.centroid[1],
            "Column_ID": column_id,
            "Column_Label": column_label,
            "Periphery_Intensity_VE": np.mean(ve_cadherin[periphery]),
            "Cytoplasm_Intensity_VE": np.mean(ve_cadherin[interior]),
            "VE_Ratio": np.mean(ve_cadherin[periphery]) / (np.mean(ve_cadherin[interior]) + 1e-6),
            "Periphery_Intensity_F": np.mean(f_actin[periphery]),
            "Cytoplasm_Intensity_F": np.mean(f_actin[interior]),
            "F_Ratio": np.mean(f_actin[periphery]) / (np.mean(f_actin[interior]) + 1e-6),
            "DAPI_Intensity": np.mean(dapi[region.coords[:, 0], region.coords[:, 1]]),
            "Periphery_Breaks": n_breaks
        })

    df = pd.DataFrame(results)
    return df, segmentation_labels, img_rgb
