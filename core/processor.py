import os
import cv2
import numpy as np
import pandas as pd
from skimage import filters, feature, segmentation, measure, morphology
from scipy import ndimage as ndi

def process_with_breaks(img_path, n_columns=1, column_labels=None, channel_map=None):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Could not read image.")

    if len(img.shape) == 2:
        img_rgb = np.stack([img] * 3, axis=-1)
    elif img.shape[2] == 4:
        img_rgb = img[:, :, :3]
    else:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if channel_map is None:
        channel_map = {'F-actin': 0, 'VE-cadherin': 1, 'DAPI': 2}

    f_actin = img_rgb[:, :, channel_map.get('F-actin', 0)]
    ve_cadherin = img_rgb[:, :, channel_map.get('VE-cadherin', 1)]
    dapi = img_rgb[:, :, channel_map.get('DAPI', 2)]

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

        ve_per = np.mean(ve_cadherin[periphery])
        ve_cyto = np.mean(ve_cadherin[interior])
        f_per = np.mean(f_actin[periphery])
        f_cyto = np.mean(f_actin[interior])
        dapi_mean = np.mean(dapi[region.coords[:, 0], region.coords[:, 1]])
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
