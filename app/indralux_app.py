import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tempfile, os, sys, cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Core modules
from core.processor import process_with_breaks
from core.metrics import add_morphological_metrics, add_extended_metrics, add_ve_snr
from core.overlay import draw_colored_overlay_with_cv2
from core.plotting import plot_metric_trends_manual
from core.indralux_stats import run_statistical_tests

# Utils
from utils.pptx_extract import extract_clean_images_from_pptx
from utils.column_split_uniform import split_into_n_columns

st.set_page_config(page_title="Indralux", page_icon="assets/favicon_32.png", layout="wide")
st.sidebar.image("assets/indralux_final_logo.png", width=240)
st.sidebar.title("Indralux: Endothelial Disruption")

if "batch_results" not in st.session_state:
    st.session_state.batch_results = {}

mode = st.sidebar.radio("Select mode", ["Batch PPTX Upload", "Single Image Analysis"])
channel_map = st.sidebar.text_input("Channel indices (F-actin, VE-cadherin, DAPI)", value="0,1,2")
channel_indices = [int(i) for i in channel_map.strip().split(",") if i.strip().isdigit()]
if len(channel_indices) != 3:
    st.sidebar.error("Must provide exactly 3 channel indices.")
    channel_indices = [0, 1, 2]

def preprocess_image(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Image loading failed.")
    if len(img.shape) == 2 or img.shape[2] == 1:
        img = np.stack([img]*3, axis=-1)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ----------------- Batch Mode -----------------
if mode == "Batch PPTX Upload":
    pptx_file = st.sidebar.file_uploader("Upload .pptx file", type=["pptx"])

    if pptx_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pptx") as tmp:
            tmp.write(pptx_file.read())
            pptx_path = tmp.name

        extract_dir = os.path.join(tempfile.gettempdir(), "pptx_clean_images")
        os.makedirs(extract_dir, exist_ok=True)
        clean_imgs = extract_clean_images_from_pptx(pptx_path, extract_dir)

        if clean_imgs:
            selected = st.selectbox("Select slide image to analyze:", clean_imgs)
            img_path = os.path.join(extract_dir, selected)
            st.image(img_path, caption=selected, use_container_width=True)

            label_key = f"labels_{selected}"
            run_key = f"run_{selected}"

            if label_key not in st.session_state:
                st.session_state[label_key] = "Control,5,10,15"
            if run_key not in st.session_state:
                st.session_state[run_key] = False

            n_cols = st.number_input("Number of panels", 1, 12, 4, key=f"ncols_{selected}")
            col_labels_input = st.text_input("Column labels (comma-separated):", key=label_key)
            col_labels = [l.strip() for l in col_labels_input.split(",")]

            if st.button("Run analysis", key=f"runbtn_{selected}"):
                st.session_state[run_key] = True

            if st.session_state[run_key]:
                split_dir = os.path.join(tempfile.gettempdir(), "split_columns")
                os.makedirs(split_dir, exist_ok=True)
                col_paths = split_into_n_columns(img_path, split_dir, n_cols)

                per_col_data = []
                for idx, col_path in enumerate(col_paths):
                    try:
                        label = col_labels[idx] if idx < len(col_labels) else f"Col{idx+1}"
                        df, labels, img_rgb = process_with_breaks(col_path, 1, [label], channel_indices)
                        morph = add_morphological_metrics(df, labels).drop(columns=["Column_Label"], errors="ignore")
                        ext = add_extended_metrics(df, labels).drop(columns=["Column_Label"], errors="ignore")
                        df = pd.merge(df, morph, on="Cell_ID", how="left")
                        df = pd.merge(df, ext, on="Cell_ID", how="left")
                        df = add_ve_snr(df, labels, img_rgb[:, :, channel_indices[1]])
                        df["Slide_Image"] = selected
                        df["Panel_Label"] = label
                        per_col_data.append(df)
                    except Exception as e:
                        st.warning(f"Failed to analyze panel {idx+1}: {e}")

                if per_col_data:
                    result_df = pd.concat(per_col_data, ignore_index=True)
                    st.session_state.batch_results[selected] = result_df
                    st.success("Analysis complete.")
                    st.dataframe(result_df.head())

                    metric_cols = [col for col in result_df.columns if result_df[col].dtype in ['float64', 'int64']]
                    chosen_metrics = st.multiselect("Plot metrics:", metric_cols, default=["VE_Ratio", "Disruption_Index"], key=f"plot_{selected}")
                    if chosen_metrics:
                        fig_path = os.path.join(tempfile.gettempdir(), f"plot_{selected}.png")
                        plot_metric_trends_manual(result_df, chosen_metrics, fig_path)
                        st.image(fig_path, caption="Metric Trends", use_container_width=True)

                    stat_cols = st.multiselect("Run stats on:", metric_cols, default=["VE_Ratio"], key=f"stats_{selected}")
                    if stat_cols and "Column_Label" in result_df.columns:
                        stats_df = run_statistical_tests(result_df[["Column_Label"] + stat_cols])
                        st.dataframe(stats_df)

# ----------------- Single Image Mode -----------------
elif mode == "Single Image Analysis":
    uploaded_file = st.sidebar.file_uploader("Upload microscopy image", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        column_labels = st.sidebar.text_input("Column labels (comma-separated):", "Control,5,15,30")
        column_labels = [label.strip() for label in column_labels.split(",")]

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.read())
            img_path = tmp.name

        st.image(img_path, caption="Uploaded Image", use_container_width=True)

        with st.spinner("Processing image..."):
            try:
                df, labels, img_rgb = process_with_breaks(img_path, len(column_labels), column_labels, channel_indices)
                morph_df = add_morphological_metrics(df, labels).drop(columns=["Column_Label"], errors="ignore")
                ext_df = add_extended_metrics(df, labels).drop(columns=["Column_Label"], errors="ignore")
                df = pd.merge(df, morph_df, on="Cell_ID", how="left")
                df = pd.merge(df, ext_df, on="Cell_ID", how="left")
                df = add_ve_snr(df, labels, img_rgb[:, :, channel_indices[1]])
                st.success("Analysis complete.")
            except Exception as e:
                st.error(f"Failed to process image: {e}")
                st.stop()

        st.dataframe(df.head())

        if st.checkbox("Show overlay with labels"):
            overlay = draw_colored_overlay_with_cv2(img_rgb, labels, df)
            overlay_path = os.path.join(tempfile.gettempdir(), "overlay.png")
            cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            st.image(overlay_path, caption="Overlay", use_container_width=True)

        if st.checkbox("Plot trends"):
            metrics = [col for col in df.columns if df[col].dtype in ['float64', 'int64']]
            selected_metrics = st.multiselect("Select metrics to plot:", options=metrics, default=["VE_Ratio", "Disruption_Index"])
            if "Column_Label" not in df.columns:
                st.error("Column_Label missing from DataFrame.")
            elif selected_metrics:
                fig_path = os.path.join(tempfile.gettempdir(), "trend_plot.png")
                plot_metric_trends_manual(df, selected_metrics, fig_path)
                st.image(fig_path, caption="Metric Trends", use_container_width=True)

        if st.checkbox("Run statistical tests"):
            numeric_cols = [col for col in df.columns if df[col].dtype in ['float64', 'int64']]
            stat_metrics = st.multiselect("Select metrics for analysis:", options=numeric_cols, default=["VE_Ratio", "Disruption_Index"])
            if stat_metrics and "Column_Label" in df.columns:
                result_df = run_statistical_tests(df[["Column_Label"] + stat_metrics])
                st.dataframe(result_df)
