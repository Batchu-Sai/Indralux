import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import tempfile, os, cv2, sys
from PIL import Image

# Enable parent directory access
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Core logic
from core.processor import process_with_breaks
from core.metrics import add_morphological_metrics, add_extended_metrics, add_ve_snr
from core.overlay import draw_colored_overlay_with_cv2
from core.plotting import plot_metric_trends_manual
from core.indralux_stats import run_statistical_tests

# Utilities
from utils.pptx_extract import extract_clean_images_from_pptx
from utils.column_split_uniform import split_into_n_columns

# Page Configuration
st.set_page_config(page_title="Fluorescent microscopy image analyzer", layout="wide")

# Session State
if "batch_results" not in st.session_state:
    st.session_state.batch_results = {}

# Mode Selection
mode = st.sidebar.radio("Select mode", ["Batch PPTX Upload", "Single Image Analysis"], key="mode_switch")

st.sidebar.markdown("**Note:** Images must be 3-channel RGB. If using custom markers, map the channels below.")

# Marker Channel Mapping
marker_f1 = st.sidebar.selectbox("Marker in Channel 1 (Red)", ["F-Actin", "VE-Cadherin", "DAPI", "Other"], index=0, key="marker_red")
marker_f2 = st.sidebar.selectbox("Marker in Channel 2 (Green)", ["VE-Cadherin", "F-Actin", "DAPI", "Other"], index=0, key="marker_green")
marker_f3 = st.sidebar.selectbox("Marker in Channel 3 (Blue)", ["DAPI", "F-Actin", "VE-Cadherin", "Other"], index=0, key="marker_blue")
marker_channel_map = {marker_f1: 0, marker_f2: 1, marker_f3: 2}

# Batch Mode
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
            try:
                Image.open(img_path).convert("RGB").save(img_path)
                img = cv2.imread(img_path)
                if img is None:
                    raise ValueError(f"OpenCV cannot read image: {img_path}")
                st.image(img_path, caption=selected, use_column_width=True)
            except Exception as e:
                st.error(f"Failed to load image: {e}")
                st.stop()

            label_key = f"labels_{selected}"
            run_key = f"run_{selected}"

            if label_key not in st.session_state:
                st.session_state[label_key] = "Control,5,10,15"
            if run_key not in st.session_state:
                st.session_state[run_key] = False

            n_cols = st.number_input("Number of panels", min_value=1, max_value=12, value=4, key=f"ncols_{selected}")
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
                        if not isinstance(col_path, str) or not os.path.exists(col_path):
                            raise FileNotFoundError(f"Panel path invalid: {col_path}")
                        Image.open(col_path).convert("RGB").save(col_path)
                        img = cv2.imread(col_path)
                        if img is None:
                            raise ValueError(f"Failed to load image: {col_path}")

                        label = col_labels[idx] if idx < len(col_labels) else f"Col{idx+1}"
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        channel_map = {k: v for k, v in marker_channel_map.items() if k != "Other"}

                        df, labels, _ = process_with_breaks(img_rgb, n_columns=1, column_labels=[label], channel_map=channel_map)
                        morph = add_morphological_metrics(df, labels).drop(columns=["Column_Label"], errors="ignore")
                        ext = add_extended_metrics(df, labels).drop(columns=["Column_Label"], errors="ignore")
                        df = pd.merge(df, morph, on="Cell_ID", how="left")
                        df = pd.merge(df, ext, on="Cell_ID", how="left")
                        ve_ch = channel_map.get("VE-Cadherin", None)
                        df = add_ve_snr(df, labels, img_rgb[:, :, ve_ch]) if ve_ch is not None else df.assign(VE_SNR=None)
                        df["Slide_Image"] = selected
                        df["Panel_Label"] = label
                        per_col_data.append(df)
                    except Exception as e:
                        st.warning(f"Panel {idx+1} failed: {e}")

                if per_col_data:
                    result_df = pd.concat(per_col_data, ignore_index=True)
                    st.session_state.batch_results[selected] = result_df
                    st.success("Analysis complete.")
                    st.dataframe(result_df.head())

# Single Image Mode
elif mode == "Single Image Analysis":
    uploaded_file = st.sidebar.file_uploader("Upload microscopy image", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        column_labels = st.sidebar.text_input("Enter column labels (comma-separated):", "Control,5,15,30")
        column_labels = [label.strip() for label in column_labels.split(",")]

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(uploaded_file.read())
            img_path = tmp.name

        try:
            Image.open(img_path).convert("RGB").save(img_path)
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"cv2 could not load the image at: {img_path}")
            st.image(img_path, caption="Uploaded Image", use_column_width=True)
        except Exception as e:
            st.error(f"Failed to load image: {e}")
            st.stop()

        with st.spinner("Processing image..."):
            try:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                channel_map = {k: v for k, v in marker_channel_map.items() if k != "Other"}

                df, labels, _ = process_with_breaks(img_rgb, n_columns=len(column_labels), column_labels=column_labels, channel_map=channel_map)
                morph_df = add_morphological_metrics(df, labels).drop(columns=["Column_Label"], errors="ignore")
                ext_df = add_extended_metrics(df, labels).drop(columns=["Column_Label"], errors="ignore")
                df = pd.merge(df, morph_df, on="Cell_ID", how="left")
                df = pd.merge(df, ext_df, on="Cell_ID", how="left")
                ve_ch = channel_map.get("VE-Cadherin", None)
                df = add_ve_snr(df, labels, img_rgb[:, :, ve_ch]) if ve_ch is not None else df.assign(VE_SNR=None)

                st.success("Analysis complete.")
                st.dataframe(df.head())
            except Exception as e:
                st.error(f"Failed to process image: {e}")
                st.stop()
