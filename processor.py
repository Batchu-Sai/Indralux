# Optimized processor and feature extractor
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
