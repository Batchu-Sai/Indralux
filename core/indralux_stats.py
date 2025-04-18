import pandas as pd
import numpy as np
from scipy.stats import kruskal
import scikit_posthocs as sp
import os

def run_statistical_tests(df, metric=None, group_col="Column_Label", output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    ignore_cols = ['Cell_ID', 'Column_ID', 'Centroid_X', 'Timepoint_Sort']
    numeric_cols = [metric] if metric else [col for col in df.columns if col not in ignore_cols and df[col].dtype in [np.float64, np.float32, np.int64]]
    stats_summary = []

    for metric in numeric_cols:
        try:
            groups = [g[metric].dropna().values for _, g in df.groupby(group_col) if len(g[metric].dropna()) > 1]
            if len(groups) < 2:
                continue
            h_stat, p_val = kruskal(*groups)
            significant = "Yes" if p_val < 0.05 else "No"
            stats_summary.append({
                "Metric": metric,
                "H_statistic": h_stat,
                "p_value": p_val,
                "Significant": significant
            })
            if significant == "Yes":
                dunn = sp.posthoc_dunn(df, val_col=metric, group_col=group_col, p_adjust="bonferroni")
                dunn.to_csv(os.path.join(output_dir, f"Dunn_posthoc_{metric}.csv"))
        except Exception as e:
            stats_summary.append({
                "Metric": metric,
                "H_statistic": None,
                "p_value": None,
                "Significant": "Error",
                "Error": str(e)
            })

    result_df = pd.DataFrame(stats_summary)
    result_df.to_csv(os.path.join(output_dir, "Kruskal_summary.csv"), index=False)
    return result_df
