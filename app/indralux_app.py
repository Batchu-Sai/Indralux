import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import tempfile, os, cv2
import sys

# Enable parent directory access
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import core modules
from core.processor import process_with_breaks
from core.metrics import add_morphological_metrics, add_extended_metrics, add_ve_snr
from core.overlay import draw_colored_overlay_with_cv2
from core.plotting import plot_spatial_disruption_map, plot_metric_trends_manual
from core.indralux_stats import run_statistical_tests

# Import pptx utilities
from utils.pptx_extract import extract_images_from_pptx
from utils.column_split_improved import split_columns_improved

# ─────────────────────────────────────────────
# PAGE CONFIG & HEADER
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Indralux",
    page_icon="assets/favicon_32.png",
    layout="centered"
)

st.image("assets/indralux_final_logo.png", width=300)
st.markdown(
    "<h2 style='text-align: center; margin-top: -10px;'>Quantifying endothelial disruption — pixel by pixel</h2>",
    unsafe_allow_html=True
)
st.markdown("---")

# ─────────────────────────────────────────────
# 🆕 PPTX UPLOAD FOR BATCH MODE
# ─────────────────────────────────────────────
if st.checkbox("Upload PowerPoint (.pptx) for Batch Ingest"):
    pptx_file = st.file_uploader("Upload your .pptx file", type=["pptx"])

    if pptx_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pptx") as tmp_pptx:
            tmp_pptx.write(pptx_file.read())
            pptx_path = tmp_pptx.name
            st.success("✅ PPTX uploaded!")

        extract_dir = os.path.join(tempfile.gettempdir(), "pptx_images")
        split_dir = os.path.join(tempfile.gettempdir(), "pptx_split")

        st.info("Extracting images from slides...")
        extract_images_from_pptx(pptx_path, extract_dir)

        extracted = sorted(os.listdir(extract_dir))
        if not extracted:
            st.error("❌ No slides were extracted. Check if the PowerPoint contains image-based slides.")
        else:
            st.success(f"✅ {len(extracted)} slides extracted.")
            st.write("Extracted slide images:")
            st.image([os.path.join(extract_dir, f) for f in extracted[:5]], caption=extracted[:5], width=150)

        st.info("Splitting slide images into columns...")
        for file in extracted:
            img_path = os.path.join(extract_dir, file)
            try:
                split_columns_improved(img_path, split_dir)
            except Exception as e:
                st.warning(f"⚠️ Failed to split {file}: {e}")

        split_files = os.listdir(split_dir)
        if not split_files:
            st.error("❌ No column images were generated from slides.")
        else:
            st.success(f"✅ {len(split_files)} column images created.")
            st.image([os.path.join(split_dir, f) for f in split_files[:5]], caption=split_files[:5], width=120)


# ─────────────────────────────────────────────
# 📷 SINGLE IMAGE ANALYSIS
# ─────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload a fluorescent microscopy image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    column_labels = st.text_input("Enter column labels separated by commas (e.g., Control,5,15,30)", "Control,5,15,30")
    column_labels = [label.strip() for label in column_labels.split(",")]

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        img_path = tmp.name

    st.image(img_path, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Processing image..."):
        try:
            df, labels, img_rgb = process_with_breaks(img_path, n_columns=len(column_labels), column_labels=column_labels)
            df = pd.merge(df, add_morphological_metrics(df, labels), on="Cell_ID", how="left", suffixes=("", "_morph"))
            df = pd.merge(df, add_extended_metrics(df, labels), on="Cell_ID", how="left", suffixes=("", "_ext"))
            df = add_ve_snr(df, labels, img_rgb[:, :, 1])
            st.success("Segmentation and analysis complete.")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"❌ Failed to process image: {e}")
            st.stop()

    # ─────────────────────────────────────────────
    # 🔲 OVERLAY
    # ─────────────────────────────────────────────
    if st.checkbox("Show overlay with cell labels"):
        overlay = draw_colored_overlay_with_cv2(img_rgb, labels, df)
        overlay_path = os.path.join(tempfile.gettempdir(), "overlay.png")
        cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        st.image(overlay_path, caption="Overlay", use_container_width=True)

    # ─────────────────────────────────────────────
    # 📈 METRIC TRENDS
    # ─────────────────────────────────────────────
    if st.checkbox("Show trend plots"):
        available_metrics = [col for col in df.columns if df[col].dtype in ['float64', 'int64'] and col not in ['Column_ID', 'Cell_ID']]
        selected_metrics = st.multiselect("Select metrics to plot:", options=available_metrics, default=["DAPI_Intensity", "VE_Ratio", "Disruption_Index"])

        if "Column_Label" not in df.columns:
            st.error("⚠️ 'Column_Label' not found. Ensure labels are correctly assigned.")
        elif not selected_metrics:
            st.warning("Please select at least one metric.")
        else:
            fig_path = os.path.join(tempfile.gettempdir(), "trend_plot.png")
            plot_metric_trends_manual(df, selected_metrics, fig_path)
            st.image(fig_path, caption="Metric Trends", use_container_width=True)

    # ─────────────────────────────────────────────
    # 📊 STATISTICAL TESTING
    # ─────────────────────────────────────────────
    if st.checkbox("Run statistics"):
        numeric_cols = [col for col in df.columns if df[col].dtype in ['float64', 'int64'] and col not in ['Column_ID', 'Cell_ID']]
        stat_metrics = st.multiselect("Select metrics to test:", options=numeric_cols, default=["VE_Ratio", "Disruption_Index"])

        if not stat_metrics:
            st.warning("Please select at least one metric to analyze.")
        else:
            result_df = run_statistical_tests(df[["Column_Label"] + stat_metrics])
            st.dataframe(result_df)

            kruskal_path = os.path.join(tempfile.gettempdir(), "kruskal_results.csv")
            result_df.to_csv(kruskal_path, index=False)
            st.download_button("Download Statistics CSV", open(kruskal_path, "rb"), "kruskal_results.csv")

    # ─────────────────────────────────────────────
    # 💾 EXPORT METRICS
    # ─────────────────────────────────────────────
    csv_path = os.path.join(tempfile.gettempdir(), "metrics_output.csv")
    df.to_csv(csv_path, index=False)
    st.download_button("Download Full Metrics CSV", open(csv_path, "rb"), "indralux_metrics.csv")



