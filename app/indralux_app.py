import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import tempfile, os, cv2, sys

# Enable parent directory access
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Core logic
from core.processor import process_with_breaks
from core.metrics import add_morphological_metrics, add_extended_metrics, add_ve_snr
from core.overlay import draw_colored_overlay_with_cv2
from core.plotting import plot_spatial_disruption_map, plot_metric_trends_manual
from core.indralux_stats import run_statistical_tests

# Utilities
from utils.pptx_extract import extract_clean_images_from_pptx
from utils.column_split_improved import split_columns_improved

# ——— PAGE HEADER ———
st.set_page_config(page_title="Indralux", page_icon="assets/favicon_32.png", layout="wide")
st.image("assets/indralux_final_logo.png", width=300)
st.markdown("<h2 style='text-align: center;'>Quantifying endothelial disruption — pixel by pixel</h2>", unsafe_allow_html=True)
st.markdown("---")

from utils.pptx_extract import extract_clean_images_from_pptx
from utils.column_split_uniform import split_into_n_columns


from utils.pptx_extract import extract_clean_images_from_pptx
from utils.column_split_uniform import split_into_n_columns

# Track per-slide results
batch_results = {}

if st.checkbox("📂 Upload .pptx for Batch Analysis"):
    pptx_file = st.file_uploader("Upload your .pptx file", type=["pptx"])

    if pptx_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pptx") as tmp:
            tmp.write(pptx_file.read())
            pptx_path = tmp.name

        extract_dir = os.path.join(tempfile.gettempdir(), "pptx_clean_images")
        os.makedirs(extract_dir, exist_ok=True)
        clean_imgs = extract_clean_images_from_pptx(pptx_path, extract_dir)

        if not clean_imgs:
            st.error("No clean image objects found.")
        else:
            st.success(f"Found {len(clean_imgs)} image(s).")
            selected = st.selectbox("Select slide image to analyze:", clean_imgs)

            if selected:
                img_path = os.path.join(extract_dir, selected)
                st.image(img_path, caption=f"Preview: {selected}", use_container_width=True)

                n_cols = st.number_input(f"How many columns in {selected}?", min_value=1, max_value=12, value=4, key="ncol_" + selected)
                labels_str = st.text_input("Column labels (comma-separated)", value="Control,5,10,15", key="lab_" + selected)
                col_labels = [lbl.strip() for lbl in labels_str.split(",")]
                if len(col_labels) != n_cols:
                    st.warning("Label count doesn't match number of columns.")

                if st.button("Run analysis on this slide", key="run_" + selected):
                    split_dir = os.path.join(tempfile.gettempdir(), "split_columns")
                    os.makedirs(split_dir, exist_ok=True)
                    col_paths = split_into_n_columns(img_path, split_dir, n_cols)

                    per_col_data = []
                    for idx, col_path in enumerate(col_paths):
                        try:
                            col_label = col_labels[idx] if idx < len(col_labels) else f"Col{idx+1}"
                            df, labels, img_rgb = process_with_breaks(col_path, n_columns=1, column_labels=[col_label])
                            df = pd.merge(df, add_morphological_metrics(df, labels), on="Cell_ID", how="left")
                            df = pd.merge(df, add_extended_metrics(df, labels), on="Cell_ID", how="left")
                            df = add_ve_snr(df, labels, img_rgb[:, :, 1])
                            df["Slide_Image"] = selected
                            df["Panel_Label"] = col_label
                            per_col_data.append(df)
                        except Exception as e:
                            st.error(f"Failed to process {col_path}: {e}")

                    if per_col_data:
                        slide_df = pd.concat(per_col_data, ignore_index=True)
                        batch_results[selected] = slide_df

                        st.success(f"Metrics complete for {selected}")
                        st.dataframe(slide_df.head())

                        # Optional trend plot
                        st.markdown("#### Metric Trends (per slide)")
                        metric_cols = [col for col in slide_df.columns if slide_df[col].dtype in ['float64', 'int64'] and col not in ['Cell_ID']]
                        safe_defaults = [m for m in ["DAPI_Intensity", "VE_Ratio", "Disruption_Index"] if m in metric_cols]
                        chosen_metrics = st.multiselect("Select metrics to plot:", metric_cols, default=safe_defaults, key="plot_" + selected)

                        if chosen_metrics:
                            fig_path = os.path.join(tempfile.gettempdir(), f"trend_{selected}.png")
                            plot_metric_trends_manual(slide_df, chosen_metrics, fig_path)
                            st.image(fig_path, caption="Trend Plot", use_container_width=True)

                        # Optional stats
                        st.markdown("#### Statistical Analysis (per slide)")
                        safe_stat_defaults = [m for m in ["VE_Ratio", "Disruption_Index"] if m in metric_cols]
                        stat_cols = st.multiselect("Select metrics to test:", metric_cols, default=safe_stat_defaults, key="stats_" + selected)


                        if stat_cols:
                            stats_df = run_statistical_tests(slide_df[["Column_Label"] + stat_cols])
                            st.dataframe(stats_df)
                            stats_path = os.path.join(tempfile.gettempdir(), f"stats_{selected}.csv")
                            stats_df.to_csv(stats_path, index=False)
                            st.download_button("Download Stats CSV", open(stats_path, "rb"), f"{selected}_stats.csv")

                        # Export metrics
                        metrics_out = os.path.join(tempfile.gettempdir(), f"{selected}_metrics.csv")
                        slide_df.to_csv(metrics_out, index=False)
                        st.download_button("Download Slide Metrics CSV", open(metrics_out, "rb"), f"{selected}_metrics.csv")

# Export all collected results
if batch_results:
    full_df = pd.concat(batch_results.values(), ignore_index=True)
    st.markdown("## Download All Batch Metrics")
    full_csv = os.path.join(tempfile.gettempdir(), "indralux_full_batch.csv")
    full_df.to_csv(full_csv, index=False)
    st.download_button("Download Full CSV", open(full_csv, "rb"), "indralux_batch_all_metrics.csv")



# ——— SINGLE IMAGE ANALYSIS ———
st.markdown("## 📸 Upload Single Microscopy Image")
uploaded_file = st.file_uploader("Upload a fluorescent microscopy image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    column_labels = st.text_input("Enter column labels (comma-separated):", "Control,5,15,30")
    column_labels = [label.strip() for label in column_labels.split(",")]

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        img_path = tmp.name

    st.image(img_path, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Processing image..."):
        try:
            df, labels, img_rgb = process_with_breaks(img_path, n_columns=len(column_labels), column_labels=column_labels)

            morph_df = add_morphological_metrics(df, labels).drop(columns=["Column_Label"], errors="ignore")
            ext_df = add_extended_metrics(df, labels).drop(columns=["Column_Label"], errors="ignore")

            df = pd.merge(df, morph_df, on="Cell_ID", how="left")
            df = pd.merge(df, ext_df, on="Cell_ID", how="left")
            df = add_ve_snr(df, labels, img_rgb[:, :, 1])

            st.success("Segmentation and metrics complete.")
        except Exception as e:
            st.error(f"Failed to process image: {e}")
            st.stop()

    st.dataframe(df.head())

    if st.checkbox("Show overlay with labels"):
        overlay = draw_colored_overlay_with_cv2(img_rgb, labels, df)
        overlay_path = os.path.join(tempfile.gettempdir(), "overlay.png")
        cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        st.image(overlay_path, caption="Overlay", use_container_width=True)

    if st.checkbox("Show metric trends"):
        if not df.empty:
            metrics = [col for col in df.columns if df[col].dtype in ['float64', 'int64'] and col not in ['Column_ID', 'Cell_ID']]
            defaults = [m for m in ["DAPI_Intensity", "VE_Ratio", "Disruption_Index"] if m in metrics]
            selected_metrics = st.multiselect("Select metrics to plot:", options=metrics, default=defaults)

            if "Column_Label" not in df.columns:
                st.error("Column_Label missing from DataFrame.")
            elif not selected_metrics:
                st.warning("Please select metrics.")
            else:
                fig_path = os.path.join(tempfile.gettempdir(), "trend_plot.png")
                plot_metric_trends_manual(df, selected_metrics, fig_path)
                st.image(fig_path, caption="Metric Trends", use_container_width=True)

    if st.checkbox("Run statistical tests"):
        numeric_cols = [col for col in df.columns if df[col].dtype in ['float64', 'int64'] and col not in ['Column_ID', 'Cell_ID']]
        stat_metrics = st.multiselect("Select metrics for analysis:", options=numeric_cols, default=[m for m in ["VE_Ratio", "Disruption_Index"] if m in numeric_cols])
        if stat_metrics and "Column_Label" in df.columns:
            result_df = run_statistical_tests(df[["Column_Label"] + stat_metrics])
            st.dataframe(result_df)
            csv_out = os.path.join(tempfile.gettempdir(), "kruskal_results.csv")
            result_df.to_csv(csv_out, index=False)
            st.download_button("Download Stats CSV", open(csv_out, "rb"), "kruskal_results.csv")

    out_csv = os.path.join(tempfile.gettempdir(), "metrics_output.csv")
    df.to_csv(out_csv, index=False)
    st.download_button("📂 Download All Metrics", open(out_csv, "rb"), "indralux_metrics.csv")



