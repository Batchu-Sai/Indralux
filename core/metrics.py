
import pandas as pd
import numpy as np
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
    return pd.DataFrame(morph_data)

def add_extended_metrics(df, labels):
    # Normalize and compute disruption index
    df['Disruption_Index'] = (
        1 / (df['VE_Ratio'] + 1e-6) +
        (1 - df['F_Ratio']) +
        df['DAPI_Intensity'] / (df['DAPI_Intensity'].max() + 1e-6) +
        df['Periphery_Breaks'] / (df['Periphery_Breaks'].max() + 1e-6)
    )
    return df

def add_ve_snr(df, labels, ve_channel):
    from skimage.measure import regionprops

    signal_means = []
    bg_mean = np.mean(ve_channel[ve_channel < np.percentile(ve_channel, 10)])
    bg_std = np.std(ve_channel[ve_channel < np.percentile(ve_channel, 10)])

    props = regionprops(labels, intensity_image=ve_channel)
    for r in props:
        if r.label in df['Cell_ID'].values:
            signal_mean = r.mean_intensity
            snr = (signal_mean - bg_mean) / (bg_std + 1e-6)
            signal_means.append({
                "Cell_ID": r.label,
                "VE_SNR": snr
            })
    snr_df = pd.DataFrame(signal_means)
    return df.merge(snr_df, on="Cell_ID", how="left")
