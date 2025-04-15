

# Optimized processor and feature extractor

def process_with_breaks(img_path, n_columns=1, column_labels=None):
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    f_actin = img_rgb[:, :, 0]
    ve_cadherin = img_rgb[:, :, 1]
    dapi = img_rgb[:, :, 2]

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

        ve_intensity = np.mean(ve_cadherin[periphery])
        f_intensity = np.mean(f_actin[periphery])
        cytoplasm_ve = np.mean(ve_cadherin[interior])
        cytoplasm_f = np.mean(f_actin[interior])
        dapi_intensity = np.mean(dapi[region.coords[:, 0], region.coords[:, 1]])

        ve_ratio = ve_intensity / (cytoplasm_ve + 1e-6)
        f_ratio = f_intensity / (cytoplasm_f + 1e-6)

        cell_column = int(region.centroid[1] // (img.shape[1] / n_columns))
        column_label = column_labels[cell_column] if column_labels and cell_column < len(column_labels) else str(cell_column + 1)

        region_mask = segmentation_labels == region.label
        skeleton = morphology.skeletonize(region_mask)
        n_breaks = measure.label(skeleton).max()

        results.append({
            "Cell_ID": region.label,
            "Area": region.area,
            "Centroid_X": region.centroid[1],
            "Column_ID": cell_column,
            "Column_Label": column_label,
            "Periphery_Intensity_VE": ve_intensity,
            "Cytoplasm_Intensity_VE": cytoplasm_ve,
            "VE_Ratio": ve_ratio,
            "Periphery_Intensity_F": f_intensity,
            "Cytoplasm_Intensity_F": cytoplasm_f,
            "F_Ratio": f_ratio,
            "DAPI_Intensity": dapi_intensity,
            "Periphery_Breaks": n_breaks
        })

    df = pd.DataFrame(results)
    return df, segmentation_labels, img_rgb


def compute_junction_fragmentation(periphery):
    skel = skeletonize(periphery)
    return label(skel).max()  # Number of segments in skeleton

def extract_cell_metrics(cell_id, mask, ve_cadherin, f_actin, dapi, nuclei_mask, column_id, column_label, centroid_x, global_ve=None, global_f=None):
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
