import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import scikit_posthocs as sp

def plot_spatial_disruption_map(df, save_path, test="kruskal", posthoc="dunn", show_sig=False):
    df = df[df['VE_Ratio'].notnull() & df['Column_Label'].notnull()]
    plt.figure(figsize=(8, 5), dpi=300)
    ax = sns.boxplot(data=df, x='Column_Label', y='VE_Ratio', linewidth=1.2, fliersize=3)
    sns.stripplot(data=df, x='Column_Label', y='VE_Ratio', color='black', alpha=0.4, jitter=True, size=2)

    if show_sig and posthoc == "dunn":
        pvals = sp.posthoc_dunn(df, val_col="VE_Ratio", group_col="Column_Label", p_adjust="bonferroni")
        for i, col1 in enumerate(pvals.columns):
            for j, col2 in enumerate(pvals.columns):
                if i < j and pvals.loc[col1, col2] < 0.05:
                    y = df['VE_Ratio'].max()
                    ax.plot([i, j], [y + 0.05, y + 0.05], lw=1.5, color='black')
                    ax.text((i + j) / 2, y + 0.06, "*", ha='center', va='bottom')

    plt.title("VE-Cadherin Ratio Across Columns", fontsize=11)
    plt.xlabel("Treatment Condition", fontsize=10)
    plt.ylabel("VE Ratio (Periphery / Cytoplasm)", fontsize=10)
    sns.despine()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_metric_trends_manual(df, metrics, save_path):
    order = {'Control': 0, '5': 5, '10': 10, '15': 15, '20': 20, '30': 30, '40': 40}
    df['Timepoint_Sort'] = df['Column_Label'].map(order)
    df = df.sort_values("Timepoint_Sort")
    plt.figure(figsize=(10, 6), dpi=300)

    for metric in metrics:
        means, sems = [], []
        for tp in df['Column_Label'].unique():
            vals = df[df['Column_Label'] == tp][metric].dropna()
            means.append(vals.mean())
            sems.append(stats.sem(vals) if len(vals) > 1 else 0)
        plt.errorbar(df['Column_Label'].unique(), means, yerr=sems, label=metric, marker='o', capsize=4)

    plt.title("Metric Trends Across Treatment Conditions", fontsize=12)
    plt.xlabel("Treatment Condition", fontsize=10)
    plt.ylabel("Metric (mean ± SEM)", fontsize=10)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
