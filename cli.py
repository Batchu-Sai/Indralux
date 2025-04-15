import argparse
from processor import process_with_breaks
from metrics import add_extended_metrics, add_morphological_metrics, add_ve_snr
from overlay import draw_colored_overlay_with_labels
from plotting import plot_spatial_disruption_map
import cv2
import pandas as pd
import os

def main():
    parser = argparse.ArgumentParser(description="Indralux CLI")
    parser.add_argument("--input", required=True, help="Path to image")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--n_columns", type=int, default=4)
    parser.add_argument("--column_labels", nargs='+', default=None)
    parser.add_argument("--min_nucleus_area", type=int, default=100)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    df, seg_labels, rgb_img = process_with_breaks(args.input, args.n_columns, args.column_labels)
    df = df[df["Nucleus_Area"] >= args.min_nucleus_area]

    morph_df = add_morphological_metrics(df, seg_labels)
    df = pd.merge(df, morph_df, on="Cell_ID")

    extended_df = add_extended_metrics(df, seg_labels)
    df = pd.merge(df, extended_df, on="Cell_ID")

    ve_img = cv2.cvtColor(cv2.imread(args.input), cv2.COLOR_BGR2RGB)[:, :, 1]
    df = add_ve_snr(df, seg_labels, ve_img)

    df.to_csv(os.path.join(args.output, "cell_metrics.csv"), index=False)

    overlay = draw_colored_overlay_with_labels(rgb_img, seg_labels, df)
    overlay_path = os.path.join(args.output, "overlay_labeled.png")
    cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    plot_path = os.path.join(args.output, "disruption_map.png")
    plot_spatial_disruption_map(df, plot_path)

if __name__ == "__main__":
    main()
