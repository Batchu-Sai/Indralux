
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

def plot_spatial_disruption_map(df, save_path):
    if 'Column_Label' not in df.columns or 'VE_Ratio' not in df.columns:
        raise ValueError("Required columns missing: 'Column_Label', 'VE_Ratio'")

    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, x='Column_Label', y='VE_Ratio', palette='viridis')
    plt.title("VE-Cadherin Ratio Across Columns")
    plt.xlabel("Treatment / Timepoint")
    plt.ylabel("VE_Ratio")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_disruption_index(df, save_path):
    if 'Column_Label' not in df.columns or 'Disruption_Index' not in df.columns:
        raise ValueError("Required columns missing: 'Column_Label', 'Disruption_Index'")

    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, x='Column_Label', y='Disruption_Index', palette='magma')
    plt.title("Disruption Index Across Columns")
    plt.xlabel("Treatment / Timepoint")
    plt.ylabel("Disruption Index")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
