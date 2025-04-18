import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import tempfile, os, cv2, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.processor import process_with_breaks
from core.metrics import add_morphological_metrics, add_extended_metrics, add_ve_snr
from core.overlay import draw_colored_overlay_with_cv2
from core.plotting import plot_metric_trends_manual
from core.indralux_stats import run_statistical_tests

from utils.pptx_extract import extract_clean_images_from_pptx
from utils.column_split_uniform import split_into_n_columns

st.set_page_config(page_title="Fluorescent Microscopy Analyzer", layout="wide")

if "batch_results" not in st.session_state:
    st.session_state.batch_results = {}

# --- SIDEBAR CONTROLS ---
st.sidebar.title("Settings")
mode = st.sidebar.radio("Select mode", ["Batch PPTX Upload", "Single Image Analysis"], key="mode_select")

channel_mode = st.sidebar.radio("Image Type", ["Color (RGB)", "Grayscale"], key="channel_mode", help="Use grayscale for single-channel markers only.")

if channel_mode == "Color (RGB)":
    marker_f1 = st.sidebar.selectbox("Marker in Channel 1 (Red)", ["F-Actin", "VE-Cadherin", "DAPI", "Other"], key="marker_1")
    marker_f2 = st.sidebar.selectbox("Marker in Channel 2 (Green)", ["VE-Cadherin", "F-Actin", "DAPI", "Other"], key="marker_2")
    marker_f3 = st.sidebar.selectbox("Marker in Channel 3 (Blue)", ["DAPI", "F-Actin", "VE-Cadherin", "Other"], key="marker_3")

    marker_channel_map = {
        marker_f1: 0,
        marker_f2: 1,
        marker_f3: 2
    }

    st.sidebar.markdown("**Assigned Channels:**")
    for marker, channel in marker_channel_map.items():
        if marker != "Other":
            st.sidebar.markdown(f"- **{marker}** → Channel {channel}")
else:
    marker_channel_map = {
        "F-Actin": 0,
        "VE-Cadherin": 0,
        "DAPI": 0
    }
    st.sidebar.info("Grayscale mode: All markers assumed on channel 0.")


def handle_image(img_path, column_labels):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")

    if channel_mode == "Grayscale":
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if len(img.shape) == 2 else img
        channel_map = {k: 0 for k in ["VE-Cadherin", "F-Actin", "DAPI"]}
    else:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        channel_map = {k: v for k, v in marker_channel_map.items() if k != "Other"}

    df, labels, _ = process_with_breaks(img_rgb, n_columns=len(column_labels), column_labels=column_labels, channel_map=channel_map)
    morph = add_morphological_metrics(df, labels).drop(columns=["Column_Label"], errors="ignore")
    ext = add_extended_metrics(df, labels).drop(columns=["Column_Label"], errors="ignore")
    df = pd.merge(df, morph, on="Cell_ID", how="left")
    df = pd.merge(df, ext, on="Cell_ID", how="left")

    ve_ch = channel_map.get("VE-Cadherin")
    df = add_ve_snr(df, labels, img_rgb[:, :, ve_ch]) if ve_ch is not None else df.assign(VE_SNR=None)
    return df, labels, img_rgb

# --- MODE HANDLERS ---

if mode == "Single Image Analysis":
    uploaded_file = st.sidebar.file_uploader("Upload microscopy image", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        column_labels = st.sidebar.text_input("Column labels (comma-separated)", "Control,5,15,30").split(",")

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.read())
            img_path = tmp.name

        st.image(img_path, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Processing..."):
            try:
                df, labels, img_rgb = handle_image(img_path, column_labels)
                st.success("Analysis complete.")
            except Exception as e:
                st.error(f"Processing error: {e}")
                st.stop()

        st.dataframe(df.head())

        if st.checkbox("Show overlay"):
            overlay = draw_colored_overlay_with_cv2(img_rgb, labels, df)
            overlay_path = os.path.join(tempfile.gettempdir(), "overlay.png")
            cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            st.image(overlay_path, caption="Overlay", use_column_width=True)

        if st.checkbox("Plot trends"):
            metric_cols = [c for c in df.columns if df[c].dtype in ['float64', 'int64']]
            selected = st.multiselect("Select metrics", metric_cols, default=["DAPI_Intensity", "VE_Ratio", "Disruption_Index"])
            if selected and "Column_Label" in df:
                fig_path = os.path.join(tempfile.gettempdir(), "trend.png")
                plot_metric_trends_manual(df, selected, fig_path)
                st.image(fig_path, caption="Metric Trends", use_column_width=True)

        if st.checkbox("Run statistical tests"):
            stat_metrics = st.multiselect("Metrics for stats", metric_cols, default=["VE_Ratio", "Disruption_Index"])
            if stat_metrics and "Column_Label" in df:
                stats_df = run_statistical_tests(df[["Column_Label"] + stat_metrics])
                st.dataframe(stats_df)

elif mode == "Batch PPTX Upload":
    pptx_file = st.sidebar.file_uploader("Upload PPTX", type=["pptx"])
    if pptx_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pptx") as tmp:
            tmp.write(pptx_file.read())
            pptx_path = tmp.name

        extract_dir = os.path.join(tempfile.gettempdir(), "pptx_clean_images")
        os.makedirs(extract_dir, exist_ok=True)
        slides = extract_clean_images_from_pptx(pptx_path, extract_dir)

        if slides:
            selected = st.selectbox("Select slide to analyze", slides)
            img_path = os.path.join(extract_dir, selected)
            st.image(img_path, caption=selected, use_column_width=True)

            n_panels = st.number_input("Number of panels", 1, 12, 4, key="batch_panels")
            col_labels = st.text_input("Column labels (comma-separated)", "Control,5,10,15").split(",")

            if st.button("Run analysis", key="batch_run"):
                split_dir = os.path.join(tempfile.gettempdir(), "batch_columns")
                col_paths = split_into_n_columns(img_path, split_dir, n_panels)

                results = []
                for i, path in enumerate(col_paths):
                    try:
                        label = col_labels[i] if i < len(col_labels) else f"Col{i+1}"
                        df, labels, img_rgb = handle_image(path, [label])
                        df["Slide_Image"] = selected
                        df["Panel_Label"] = label
                        results.append(df)
                    except Exception as e:
                        st.warning(f"Panel {i+1} failed: {e}")

                if results:
                    combined = pd.concat(results, ignore_index=True)
                    st.session_state.batch_results[selected] = combined
                    st.success("Batch analysis complete.")
                    st.dataframe(combined.head())

                    metric_cols = [c for c in combined.columns if combined[c].dtype in ['float64', 'int64']]
                    selected = st.multiselect("Metrics to plot", metric_cols, default=["DAPI_Intensity", "VE_Ratio", "Disruption_Index"])
                    if selected:
                        fig_path = os.path.join(tempfile.gettempdir(), f"batch_plot.png")
                        plot_metric_trends_manual(combined, selected, fig_path)
                        st.image(fig_path, caption="Batch Metric Trends", use_column_width=True)

                    stat_metrics = st.multiselect("Metrics for stats", metric_cols, default=["VE_Ratio", "Disruption_Index"])
                    if stat_metrics:
                        stats_df = run_statistical_tests(combined[["Column_Label"] + stat_metrics])
                        st.dataframe(stats_df)
