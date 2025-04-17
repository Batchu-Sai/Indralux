import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2, os, sys, tempfile

# Extend path for relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Core
from core.processor import process_with_breaks
from core.metrics import add_morphological_metrics, add_extended_metrics, add_ve_snr
from core.overlay import draw_colored_overlay_with_cv2
from core.plotting import plot_metric_trends_manual
from core.indralux_stats import run_statistical_tests

# Utils
from utils.pptx_extract import extract_clean_images_from_pptx
from utils.column_split_uniform import split_into_n_columns

# ─────────────────────────────────────────────
# Layout
# ─────────────────────────────────────────────
st.set_page_config(page_title="Indralux", page_icon="assets/favicon_32.png", layout="wide")
st.image("assets/indralux_final_logo.png", width=300)
st.markdown("<h2 style='text-align: center;'>Quantifying endothelial disruption — pixel by pixel</h2>", unsafe_allow_html=True)
st.markdown("---")

# ─────────────────────────────────────────────
# Batch analysis from .pptx
# ─────────────────────────────────────────────
if "batch_results" not in st.session_state:
    st.session_state.batch_results = {}

if st.checkbox("📂 Upload .pptx for Batch Analysis"):
    pptx_file = st.file_uploader("Upload your .pptx file", type=["pptx"])

    if pptx_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pptx") as tmp:
            tmp.write(pptx_file.read())
            pptx_path = tmp.name

        extract_dir = os.path.join(tempfile.gettempdir(), "pptx_clean_images")
        os.makedirs(extract_dir, exist_ok=True)
        slide_imgs = extract_clean_images_from_pptx(pptx_path, extract_dir)

        if not slide_imgs:
            st.error("❌ No valid slide images found.")
        else:
            selected = st.selectbox("📸 Select slide image to analyze:", slide_imgs)

            if selected:
                img_path = os.path.join(extract_dir, selected)
                st.image(img_path, caption=f"Slide Preview: {selected}", use_container_width=True)

                label_key = f"labels_{selected}"
                run_key = f"run_{selected}"
                if label_key not in st.session_state:
                    st.session_state[label_key] = "Control,5,15,30"

                n_cols = st.number_input("Number of panels in this slide:", min_value=1, max_value=12, value=4, key=f"ncols_{selected}")
                col_labels = st.text_input("Column labels:", value=st.session_state[label_key], key=label_key).split(",")
                col_labels = [l.strip() for l in col_labels]

                if st.button("▶️ Run Analysis", key=f"btn_{selected}"):
                    split_dir = os.path.join(tempfile.gettempdir(), "split_columns")
                    os.makedirs(split_dir, exist_ok=True)
                    col_paths = split_into_n_columns(img_path, split_dir, n_cols)

                    results = []
                    for idx, col_path in enumerate(col_paths):
                        try:
                            label = col_labels[idx] if idx < len(col_labels) else f"Col{idx+1}"
                            df, labels, img_rgb = process_with_breaks(col_path, n_columns=1, column_labels=[label])

                            morph_df = add_morphological_metrics(df, labels).drop(columns=["Column_Label"], errors="ignore")
                            morph_df = morph_df[[col for col in morph_df.columns if col not in df.columns or col == "Cell_ID"]]
                            df = pd.merge(df, morph_df, on="Cell_ID", how="left")

                            ext_df = add_extended_metrics(df, labels).drop(columns=["Column_Label"], errors="ignore")
                            ext_df = ext_df[[col for col in ext_df.columns if col not in df.columns or col == "Cell_ID"]]
                            df = pd.merge(df, ext_df, on="Cell_ID", how="left")

                            df = add_ve_snr(df, labels, img_rgb[:, :, 1] if img_rgb.ndim == 3 else img_rgb)

                            df["Slide_Image"] = selected
                            df["Panel_Label"] = label
                            results.append(df)
                        except Exception as e:
                            st.warning(f"⚠️ Skipped column {idx+1}: {e}")

                    if results:
                        result_df = pd.concat(results, ignore_index=True)
                        st.session_state.batch_results[selected] = result_df
                        st.success("Metrics extracted.")
                        st.dataframe(result_df.head())

                        # Plot
                        numeric = [c for c in result_df.columns if result_df[c].dtype in ['float64', 'int64']]
                        defaults = [m for m in ["DAPI_Intensity", "VE_Ratio", "Disruption_Index"] if m in numeric]
                        selected_metrics = st.multiselect("📈 Plot metrics:", numeric, default=defaults, key=f"plot_{selected}")

                        if selected_metrics:
                            fig_path = os.path.join(tempfile.gettempdir(), f"{selected}_plot.png")
                            plot_metric_trends_manual(result_df, selected_metrics, fig_path)
                            st.image(fig_path, caption="Trend Plot", use_container_width=True)

                        # Stats
                        stat_defaults = [m for m in ["VE_Ratio", "Disruption_Index"] if m in numeric]
                        stat_metrics = st.multiselect("📊 Run stats on:", numeric, default=stat_defaults, key=f"stats_{selected}")

                        if stat_metrics:
                            stats_df = run_statistical_tests(result_df[["Column_Label"] + stat_metrics])
                            st.dataframe(stats_df)
                            stats_csv = os.path.join(tempfile.gettempdir(), f"{selected}_stats.csv")
                            stats_df.to_csv(stats_csv, index=False)
                            st.download_button("⬇ Download Stats", open(stats_csv, "rb"), f"{selected}_stats.csv")

                        # Export CSV
                        csv_out = os.path.join(tempfile.gettempdir(), f"{selected}_metrics.csv")
                        result_df.to_csv(csv_out, index=False)
                        st.download_button("⬇ Download CSV", open(csv_out, "rb"), f"{selected}_metrics.csv")

# Global batch export
if st.session_state.batch_results:
    all_df = pd.concat(st.session_state.batch_results.values(), ignore_index=True)
    all_csv = os.path.join(tempfile.gettempdir(), "all_batch_metrics.csv")
    all_df.to_csv(all_csv, index=False)
    st.download_button("📦 Export All Batch Metrics", open(all_csv, "rb"), "indralux_batch_all.csv")

# ─────────────────────────────────────────────
# Single Image Upload
# ─────────────────────────────────────────────
st.markdown("## 📸 Single Image Mode")
single_file = st.file_uploader("Upload microscopy image", type=["png", "jpg", "jpeg"])

if single_file:
    col_labels = st.text_input("Enter labels (comma-separated):", "Control,5,15,30").split(",")
    col_labels = [c.strip() for c in col_labels]

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(single_file.read())
        img_path = tmp.name

    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else img

    st.image(img_rgb, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Processing..."):
        try:
            df, labels, img_rgb = process_with_breaks(img_path, n_columns=len(col_labels), column_labels=col_labels)
            morph = add_morphological_metrics(df, labels).drop(columns=["Column_Label"], errors="ignore")
            ext = add_extended_metrics(df, labels).drop(columns=["Column_Label"], errors="ignore")
            df = pd.merge(df, morph, on="Cell_ID", how="left")
            df = pd.merge(df, ext, on="Cell_ID", how="left")
            df = add_ve_snr(df, labels, img_rgb[:, :, 1] if img_rgb.ndim == 3 else img_rgb)
            st.success("Segmentation complete.")
        except Exception as e:
            st.error(f"Error processing image: {e}")
            st.stop()

    st.dataframe(df.head())

    if st.checkbox("Overlay Labels"):
        overlay = draw_colored_overlay_with_cv2(img_rgb, labels, df)
        path_overlay = os.path.join(tempfile.gettempdir(), "overlay.png")
        cv2.imwrite(path_overlay, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        st.image(path_overlay, caption="Overlay", use_container_width=True)

    if st.checkbox("Trend Plot"):
        available = [c for c in df.columns if df[c].dtype in ['float64', 'int64']]
        defaults = [m for m in ["DAPI_Intensity", "VE_Ratio", "Disruption_Index"] if m in available]
        selected_metrics = st.multiselect("Metrics to plot:", available, default=defaults)
        if selected_metrics:
            fig_path = os.path.join(tempfile.gettempdir(), "trend_single.png")
            plot_metric_trends_manual(df, selected_metrics, fig_path)
            st.image(fig_path, caption="Trend", use_container_width=True)

    if st.checkbox("Run Stats"):
        stat_metrics = st.multiselect("Metrics to test:", available, default=["VE_Ratio", "Disruption_Index"])
        if stat_metrics and "Column_Label" in df.columns:
            stat_df = run_statistical_tests(df[["Column_Label"] + stat_metrics])
            st.dataframe(stat_df)

    out_path = os.path.join(tempfile.gettempdir(), "single_metrics.csv")
    df.to_csv(out_path, index=False)
    st.download_button("📥 Download Metrics CSV", open(out_path, "rb"), "indralux_metrics.csv")

