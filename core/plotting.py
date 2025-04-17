import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy.stats import kruskal

def plot_metric_trends_manual(df, metrics, save_path):
    if "Column_Label" not in df.columns:
        raise ValueError("Column_Label is missing from the DataFrame.")

    plt.figure(figsize=(10, 5 * len(metrics)), dpi=300)

    for idx, metric in enumerate(metrics):
        plt.subplot(len(metrics), 1, idx + 1)
        sns.boxplot(data=df, x="Column_Label", y=metric, linewidth=1.2, fliersize=3)
        sns.stripplot(data=df, x="Column_Label", y=metric, color='black', alpha=0.4, jitter=True, size=2)
        plt.title(f"{metric} Across Conditions")
        plt.xlabel("")
        plt.ylabel(metric)

    sns.despine()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_spatial_disruption_map(df, save_path, test="kruskal", posthoc="dunn", show_sig=False):
    df = df[df['VE_Ratio'].notnull() & df['Column_Label'].notnull()]

    plt.figure(figsize=(8, 5), dpi=300)
    ax = sns.boxplot(data=df, x='Column_Label', y='VE_Ratio', linewidth=1.2, fliersize=3)
    sns.stripplot(data=df, x='Column_Label', y='VE_Ratio', color='black', alpha=0.4, jitter=True, size=2)

    if show_sig and posthoc == "dunn":
        try:
            pvals = sp.posthoc_dunn(df, val_col="VE_Ratio", group_col="Column_Label", p_adjust="bonferroni")
            for i, col1 in enumerate(pvals.columns):
                for j, col2 in enumerate(pvals.columns):
                    if i < j and pvals.loc[col1, col2] < 0.05:
                        y = df['VE_Ratio'].max() + 0.02 * (j - i)
                        ax.plot([i, j], [y, y], lw=1.2, color='black')
                        ax.text((i + j) / 2, y + 0.01, "*", ha='center', va='bottom', fontsize=10)
        except Exception as e:
            print(f"Posthoc Dunn test failed: {e}")

    plt.title("VE-Cadherin Ratio Across Columns", fontsize=11)
    plt.xlabel("Treatment Condition", fontsize=10)
    plt.ylabel("VE Ratio (Periphery / Cytoplasm)", fontsize=10)
    sns.despine()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
