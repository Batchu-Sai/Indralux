import pandas as pd
import numpy as np
import cv2
from skimage.measure import regionprops
from skimage.morphology import convex_hull_image

def add_morphological_metrics(df, labels):
    regions = regionprops(labels)
    morph_data = []

    for r in regions:
        if r.label in df['Cell_ID'].values:
            solidity = r.solidity
            circularity = (4 * np.pi * r.area) / (r.perimeter ** 2) if r.perimeter > 0 else 0
            aspect_ratio = r.major_axis_length / (r.minor_axis_length + 1e-6)
            morph_data.append({
                "Cell_ID": r.label,
                "Solidity": solidity,
                "Circularity": circularity,
                "Aspect_Ratio": aspect_ratio
            })

    morph_df = pd.DataFrame(morph_data)
    return morph_df

def add_extended_metrics(df, labels):
    # Normalize and compute disruption index
    ve_safe = df['VE_Ratio'].replace([np.inf, -np.inf], np.nan).fillna(0)
    f_safe = df['F_Ratio'].replace([np.inf, -np.inf], np.nan).fillna(0)
    dapi_norm = df['DAPI_Intensity'] / (df['DAPI_Intensity'].max() + 1e-6)
    breaks_norm = df['Periphery_Breaks'] / (df['Periphery_Breaks'].max() + 1e-6)

    disruption_index = 1 / (ve_safe + 1e-6) + (1 - f_safe) + dapi_norm + breaks_norm
    df['Disruption_Index'] = disruption_index

    return df

def add_ve_snr(df, labels, ve_channel, pad=10):
    snr_list = []

    for region in df.itertuples():
        mask = labels == region.Cell_ID
        dilated = cv2.dilate(mask.astype(np.uint8), np.ones((pad, pad), dtype=np.uint8), iterations=1)
        background = (dilated > 0) & (~mask)

        bg_vals = ve_channel[background]
        if len(bg_vals) == 0 or np.std(bg_vals) == 0:
            snr = None
        else:
            periphery = mask ^ cv2.erode(mask.astype(np.uint8), None)
            signal = ve_channel[periphery]
            snr = (np.mean(signal) - np.mean(bg_vals)) / (np.std(bg_vals) + 1e-6)

        snr_list.append(snr)

    df["VE_SNR"] = snr_list
    return df
