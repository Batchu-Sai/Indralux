import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from processor import process_with_breaks
from metrics import add_morphological_metrics, add_extended_metrics, add_ve_snr
from overlay import draw_colored_overlay_with_cv2
from plotting import plot_spatial_disruption_map, plot_metric_trends_manual
from indralux_stats import run_statistical_tests
import cv2
import tempfile
import os

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

uploaded_file = st.file_uploader("Upload a fluorescent microscopy image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    column_labels = st.text_input("Enter column labels separated by commas (e.g., Control,5,15,30)", "Control,5,15,30")
    column_labels = [label.strip() for label in column_labels.split(",")]

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        img_path = tmp.name

    st.image(img_path, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Processing image..."):
        df, labels, img_rgb = process_with_breaks(img_path, n_columns=len(column_labels), column_labels=column_labels)
        df = pd.merge(df, add_morphological_metrics(df, labels), on="Cell_ID", how="left", suffixes=("", "_morph"))
        df = pd.merge(df, add_extended_metrics(df, labels), on="Cell_ID", how="left", suffixes=("", "_ext"))
        df = add_ve_snr(df, labels, img_rgb[:, :, 1])

    st.success("Segmentation and analysis complete.")
    st.dataframe(df.head())

    if st.checkbox("Show overlay with cell labels"):
        overlay = draw_colored_overlay_with_cv2(img_rgb, labels, df)
        overlay_path = os.path.join(tempfile.gettempdir(), "overlay.png")
        cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        st.image(overlay_path, caption="Overlay", use_container_width=True)

    if st.checkbox("Show trend plots"):
        available_metrics = [col for col in df.columns if df[col].dtype in ['float64', 'int64'] and col not in ['Column_ID', 'Cell_ID']]
        selected_metrics = st.multiselect("Select metrics to plot:", options=available_metrics, default=["DAPI_Intensity", "VE_Ratio", "Disruption_Index"])

        if "Column_Label" not in df.columns:
            st.error("⚠️ 'Column_Label' not found. Ensure labels are correctly assigned.")
            st.write("Available columns:", df.columns.tolist())
        elif not selected_metrics:
            st.warning("Please select at least one metric.")
        else:
            fig_path = os.path.join(tempfile.gettempdir(), "trend_plot.png")
            plot_metric_trends_manual(df, selected_metrics, fig_path)
            st.image(fig_path, caption="Metric Trends", use_container_width=True)

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

    # Always offer full download
    csv_path = os.path.join(tempfile.gettempdir(), "metrics_output.csv")
    df.to_csv(csv_path, index=False)
    st.download_button("Download Full Metrics CSV", open(csv_path, "rb"), "indralux_metrics.csv")


