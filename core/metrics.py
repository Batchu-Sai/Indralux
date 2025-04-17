import pandas as pd
import numpy as np
from skimage.measure import regionprops
from skimage.morphology import convex_hull_image
import cv2

def add_morphological_metrics(df, labels):
    regions = regionprops(labels)
    morph_data = []
    for r in regions:
        if r.label in df['Cell_ID'].values:
            morph_data.append({
                "Cell_ID": r.label,
                "Solidity": r.solidity,
                "Circularity": (4 * np.pi * r.area) / (r.perimeter ** 2 + 1e-6),
                "Aspect_Ratio": r.major_axis_length / (r.minor_axis_length + 1e-6)
            })
    return pd.DataFrame(morph_data)

def add_extended_metrics(df, labels):
    df['Disruption_Index'] = (
        1 / (df['VE_Ratio'] + 1e-6) +
        (1 - df['F_Ratio']) +
        df['DAPI_Intensity'] / (df['DAPI_Intensity'].max() + 1e-6) +
        df['Periphery_Breaks'] / (df['Periphery_Breaks'].max() + 1e-6)
    )
    return df

def add_ve_snr(df, labels, ve_channel, pad=10):
    snr_list = []
    for region in df.itertuples():
        mask = labels == region.Cell_ID
        dilated = cv2.dilate(mask.astype(np.uint8), np.ones((pad, pad), dtype=np.uint8), iterations=1)
        background = (dilated > 0) & (~mask)
        bg_vals = ve_channel[background]
        if len(bg_vals) == 0 or np.std(bg_vals) == 0:
            snr = np.nan
        else:
            periphery = mask ^ cv2.erode(mask.astype(np.uint8), None)
            signal = ve_channel[periphery]
            snr = (np.mean(signal) - np.mean(bg_vals)) / (np.std(bg_vals) + 1e-6)
        snr_list.append(snr)
    df["VE_SNR"] = snr_list
    return df
