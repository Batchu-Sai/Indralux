import os
import cv2
import numpy as np
import pandas as pd
from skimage import filters, feature, segmentation, measure, morphology
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from skimage.measure import label

# —————————————————————————————
# Fragmentation via Skeletonization
# —————————————————————————————
def compute_junction_fragmentation(periphery):
    skel = skeletonize(periphery)
    return label(skel).max()

# —————————————————————————————
# Per-cell metric extraction
# —————————————————————————————
def extract_cell_metrics(cell_id, mask, ve_cadherin, f_actin, dapi, nuclei_mask,
                         column_id, column_label, centroid_x, global_ve=None, global_f=None):
    interior = morphology.binary_erosion(mask, morphology.disk(3))
    periphery = mask ^ interior
    nucleus_overlap = nuclei_mask & mask

    ve_per = np.mean(ve_cadherin[periphery])
    ve_cyto = np.mean(ve_cadherin[interior])
    f_per = np.mean(f_actin[periphery])
    f_cyto = np.mean(f_actin[interior])

    dapi_mean = np.mean(dapi[nucleus_overlap]) if np.any(nucleus_overlap) else 0
    nucleus_area = np.sum(nucleus_overlap)

    ve_ratio = ve_per / (ve_cyto + 1e-6)
    f_ratio = f_per / (f_cyto + 1e-6)

    ve_global_ratio = ve_per / (global_ve + 1e-6) if global_ve is not None else None
    f_global_ratio = f_per / (global_f + 1e-6) if global_f is not None else None

    breaks = compute_junction_fragmentation(periphery)

    return {
        "Cell_ID": cell_id,
        "Area": np.sum(mask),
        "Column_ID": column_id,
        "Time_Label": column_label,
        "Centroid_X": centroid_x,
        "VE_Periphery_Intensity": ve_per,
        "VE_Cytoplasm_Intensity": ve_cyto,
        "VE_Intensity_Ratio": ve_ratio,
        "F_Periphery_Intensity": f_per,
        "F_Cytoplasm_Intensity": f_cyto,
        "F_Intensity_Ratio": f_ratio,
        "DAPI_Intensity": dapi_mean,
        "Nucleus_Area": nucleus_area,
        "N_Periphery_Breaks": breaks,
        "VE_Global_Ratio": ve_global_ratio,
        "F_Global_Ratio": f_global_ratio
    }

# —————————————————————————————
# Wrapper for full image processing
# —————————————————————————————
def process_with_breaks(img_path, n_columns=1, column_labels=None, channel_map=None):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    is_gray = len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1)
    if is_gray:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if len(img.shape) == 2 else np.repeat(img, 3, axis=2)
    else:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Default fallback
    if channel_map is None:
        channel_map = {"F-Actin": 0, "VE-Cadherin": 1, "DAPI": 2}

    def safe_channel(marker_name):
        ch = channel_map.get(marker_name, None)
        return img_rgb[:, :, ch] if ch is not None and ch < img_rgb.shape[2] else np.zeros_like(img_rgb[:, :, 0])

    f_actin = safe_channel("F-Actin")
    ve_cadherin = safe_channel("VE-Cadherin")
    dapi = safe_channel("DAPI")

    # Segmentation
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

    global_ve = np.mean(ve_cadherin)
    global_f = np.mean(f_actin)

    results = []
    for region in measure.regionprops(segmentation_labels):
        if region.area < 100:
            continue
        mask = segmentation_labels == region.label
        centroid_x = region.centroid[1]
        column_id = int(centroid_x // (img.shape[1] / n_columns))
        column_label = column_labels[column_id] if column_labels and column_id < len(column_labels) else str(column_id)

        metrics = extract_cell_metrics(
            cell_id=region.label,
            mask=mask,
            ve_cadherin=ve_cadherin,
            f_actin=f_actin,
            dapi=dapi,
            nuclei_mask=nuclei,
            column_id=column_id,
            column_label=column_label,
            centroid_x=centroid_x,
            global_ve=global_ve,
            global_f=global_f
        )
        results.append(metrics)

    df = pd.DataFrame(results)
    return df, segmentation_labels, img_rgb

