
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile
import os
import matplotlib.pyplot as plt
from processor import process_with_breaks
from metrics import add_morphological_metrics, add_extended_metrics, add_ve_snr
from overlay import draw_colored_overlay_with_cv2
from plotting import plot_spatial_disruption_map

st.set_page_config(layout="wide")
st.title("Indralux: Endothelial Barrier Quantification")

st.sidebar.header("Step 1: Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose an image file", type=["jpg", "jpeg", "png", "tif", "tiff"])

st.sidebar.header("Step 2: Settings")
threshold_method = st.sidebar.selectbox("Thresholding Method", ["otsu", "local", "percentile"])
n_columns = st.sidebar.number_input("Number of Columns", min_value=1, max_value=20, value=4)
column_labels = st.sidebar.text_input("Column Labels (comma-separated)", "Control,5,10,15")

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    if st.sidebar.button("Run Indralux"):
        with st.spinner("Processing image..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            labels = [label.strip() for label in column_labels.split(",")]
            df, seg_labels, rgb_img = process_with_breaks(tmp_path, n_columns=n_columns, column_labels=labels)
            morph_df = add_morphological_metrics(df, seg_labels)
            df = pd.merge(df, morph_df, on="Cell_ID")
            ext_df = add_extended_metrics(df, seg_labels)
            df = pd.merge(df, ext_df, on="Cell_ID")

            ve_img = cv2.cvtColor(cv2.imread(tmp_path), cv2.COLOR_BGR2RGB)[:, :, 1]
            df = add_ve_snr(df, seg_labels, ve_img)

            overlay = draw_colored_overlay_with_cv2(rgb_img, seg_labels, df)
            overlay_path = tmp_path.replace(".png", "_overlay.png")
            cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            df_path = tmp_path.replace(".png", "_metrics.csv")
            df.to_csv(df_path, index=False)
            map_path = tmp_path.replace(".png", "_disruption_map.png")
            plot_spatial_disruption_map(df, map_path)

            st.subheader("Results Overview")
            st.dataframe(df.head())

            st.subheader("Overlay Image")
            st.image(overlay_path, use_column_width=True)

            st.subheader("Disruption Index Map")
            st.image(map_path, use_column_width=True)

            st.download_button("Download Metrics CSV", data=open(df_path, "rb").read(), file_name="indralux_metrics.csv")
            st.download_button("Download Overlay Image", data=open(overlay_path, "rb").read(), file_name="overlay_labeled.png")
            st.download_button("Download Disruption Map", data=open(map_path, "rb").read(), file_name="disruption_map.png")
