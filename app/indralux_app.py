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
from utils.pptx_extract import extract_images_from_pptx
from utils.column_split_improved import split_columns_improved

# ——— PAGE HEADER ———
st.set_page_config(page_title="Indralux", page_icon="assets/favicon_32.png", layout="wide")
st.image("assets/indralux_final_logo.png", width=300)
st.markdown("<h2 style='text-align: center;'>Quantifying endothelial disruption — pixel by pixel</h2>", unsafe_allow_html=True)
st.markdown("---")

# ——— BATCH MODE (PPTX) ———
if st.checkbox("Upload PowerPoint (.pptx) for Batch Ingest"):
    pptx_file = st.file_uploader("Upload your .pptx file", type=["pptx"])

    if pptx_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pptx") as tmp_pptx:
            tmp_pptx.write(pptx_file.read())
            pptx_path = tmp_pptx.name

        extract_dir = os.path.join(tempfile.gettempdir(), "pptx_images")
        split_dir = os.path.join(tempfile.gettempdir(), "pptx_split")
        os.makedirs(extract_dir, exist_ok=True)
        os.makedirs(split_dir, exist_ok=True)

        st.info("Extracting images from slides...")
        extracted = extract_images_from_pptx(pptx_path, extract_dir) or []
        if not extracted:
            st.error("No slide images extracted.")
        else:
            st.success(f"{len(extracted)} slides extracted.")

            with st.expander("🖼 Preview slide images"):
                for i in range(0, len(extracted), 4):
                    cols = st.columns(4)
                    for j, col in enumerate(cols):
                        if i + j < len(extracted):
                            path = os.path.join(extract_dir, extracted[i + j])
                            col.image(path, caption=extracted[i + j], use_container_width=True)

            selected_slide_files = st.multiselect("Select slide images to analyze:", extracted, default=extracted)

            st.info("Splitting selected slide images into columns...")
            for file in selected_slide_files:
                img_path = os.path.join(extract_dir, file)
                try:
                    split_columns_improved(img_path, split_dir)
                except Exception as e:
                    st.warning(f"Failed to split {file}: {e}")

            split_files = sorted(os.listdir(split_dir))
            split_files = [f for f in split_files if any(slide in f for slide in selected_slide_files)]

            if split_files:
                st.success(f"{len(split_files)} column images created.")
                with st.expander("🧪 Column previews"):
                    for i in range(0, len(split_files), 5):
                        cols = st.columns(5)
                        for j, col in enumerate(cols):
                            if i + j < len(split_files):
                                path = os.path.join(split_dir, split_files[i + j])
                                col.image(path, caption=split_files[i + j], use_container_width=True)
            else:
                st.error("No column images created.")

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

    if st.checkbox("📊 Show metric trends"):
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

    if st.checkbox("📊 Run statistical tests"):
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



