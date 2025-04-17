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
from core.plotting import plot_metric_trends_manual
from core.indralux_stats import run_statistical_tests

# Utilities
from utils.pptx_extract import extract_clean_images_from_pptx
from utils.column_split_uniform import split_into_n_columns

# Page Configuration
st.set_page_config(page_title="Fluorescent microscopy image analyzer", page_icon="assets/favicon_32.png", layout="wide")
#st.sidebar.image("assets/indralux_final_logo.png", width=240)
#st.sidebar.title("Fluorescent microscopy image analyzer")

# Results tracking
if "batch_results" not in st.session_state:
    st.session_state.batch_results = {}

# Sidebar Mode Switch
mode = st.sidebar.radio("Select mode", ["Batch PPTX Upload", "Single Image Analysis"], key="mode_switch")

# Channel marker configuration with tooltips
marker_f1 = st.sidebar.selectbox(
    "Marker in Channel 1 (Red)",
    ["F-Actin", "VE-Cadherin", "DAPI", "Other"],
    index=0,
    help="Using standard markers? Leave as is. For custom stains (e.g., FITC, Alexa Fluor), specify mapping manually.",
    key="marker_red"
)

marker_f2 = st.sidebar.selectbox(
    "Marker in Channel 2 (Green)",
    ["VE-Cadherin", "F-Actin", "DAPI", "Other"],
    index=0,
    help="Using standard markers? Leave as is. For custom stains (e.g., FITC, Alexa Fluor), specify mapping manually.",
    key="marker_green"
)

marker_f3 = st.sidebar.selectbox(
    "Marker in Channel 3 (Blue)",
    ["DAPI", "F-Actin", "VE-Cadherin", "Other"],
    index=0,
    help="Using standard markers? Leave as is. For custom stains (e.g., FITC, Alexa Fluor), specify mapping manually.",
    key="marker_blue"
)

# Build marker-channel mapping
marker_channel_map = {
    marker_f1: 0,
    marker_f2: 1,
    marker_f3: 2
}

# Just for debugging / confirmation
st.sidebar.markdown("**Assigned Channels:**")
for marker, channel in marker_channel_map.items():
    if marker != "Other":
        st.sidebar.markdown(f"- **{marker}** → Channel {channel}")


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
            st.image(img_path, caption=selected, use_column_width=True)

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
                        label = col_labels[idx] if idx < len(col_labels) else f"Col{idx+1}"
                        df, labels, img_rgb = process_with_breaks(col_path, n_columns=1, column_labels=[label])
                        morph = add_morphological_metrics(df, labels).drop(columns=["Column_Label"], errors="ignore")
                        morph = morph[[col for col in morph.columns if col not in df.columns or col == "Cell_ID"]]
                        df = pd.merge(df, morph, on="Cell_ID", how="left")
                        ext = add_extended_metrics(df, labels).drop(columns=["Column_Label"], errors="ignore")
                        ext = ext[[col for col in ext.columns if col not in df.columns or col == "Cell_ID"]]
                        df = pd.merge(df, ext, on="Cell_ID", how="left")
                        df = add_ve_snr(df, labels, img_rgb[:, :, 1])  # Still assumes VE is in green
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
                    safe_defaults = [m for m in ["DAPI_Intensity", "VE_Ratio", "Disruption_Index"] if m in metric_cols]
                    chosen_metrics = st.multiselect("Plot metrics:", metric_cols, default=safe_defaults, key=f"plot_{selected}")

                    if chosen_metrics:
                        fig_path = os.path.join(tempfile.gettempdir(), f"plot_{selected}.png")
                        plot_metric_trends_manual(result_df, chosen_metrics, fig_path)
                        st.image(fig_path, caption="Metric Trends", use_column_width=True)

                    safe_stat_defaults = [m for m in ["VE_Ratio", "Disruption_Index"] if m in metric_cols]
                    stat_cols = st.multiselect("Run stats on:", metric_cols, default=safe_stat_defaults, key=f"stats_{selected}")

                    if stat_cols:
                        stats_df = run_statistical_tests(result_df[["Column_Label"] + stat_cols])
                        st.dataframe(stats_df)

# Single Image Mode
elif mode == "Single Image Analysis":
    uploaded_file = st.sidebar.file_uploader("Upload microscopy image", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        column_labels = st.sidebar.text_input("Enter column labels (comma-separated):", "Control,5,15,30")
        column_labels = [label.strip() for label in column_labels.split(",")]

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.read())
            img_path = tmp.name

        st.image(img_path, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Processing image..."):
            try:
                df, labels, img_rgb = process_with_breaks(img_path, n_columns=len(column_labels), column_labels=column_labels)
                morph_df = add_morphological_metrics(df, labels).drop(columns=["Column_Label"], errors="ignore")
                ext_df = add_extended_metrics(df, labels).drop(columns=["Column_Label"], errors="ignore")
                df = pd.merge(df, morph_df, on="Cell_ID", how="left")
                df = pd.merge(df, ext_df, on="Cell_ID", how="left")
                df = add_ve_snr(df, labels, img_rgb[:, :, 1])
                st.success("Analysis complete.")
            except Exception as e:
                st.error(f"Failed to process image: {e}")
                st.stop()

        st.dataframe(df.head())

        if st.checkbox("Show overlay with labels"):
            overlay = draw_colored_overlay_with_cv2(img_rgb, labels, df)
            overlay_path = os.path.join(tempfile.gettempdir(), "overlay.png")
            cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            st.image(overlay_path, caption="Overlay", use_column_width=True)

        if st.checkbox("Plot trends"):
            metrics = [col for col in df.columns if df[col].dtype in ['float64', 'int64'] and col not in ['Column_ID', 'Cell_ID']]
            defaults = [m for m in ["DAPI_Intensity", "VE_Ratio", "Disruption_Index"] if m in metrics]
            selected_metrics = st.multiselect("Select metrics to plot:", options=metrics, default=defaults)

            if "Column_Label" not in df.columns:
                st.error("Column_Label missing from DataFrame.")
            elif selected_metrics:
                fig_path = os.path.join(tempfile.gettempdir(), "trend_plot.png")
                plot_metric_trends_manual(df, selected_metrics, fig_path)
                st.image(fig_path, caption="Metric Trends", use_column_width=True)

        if st.checkbox("Run statistical tests"):
            numeric_cols = [col for col in df.columns if df[col].dtype in ['float64', 'int64'] and col not in ['Column_ID', 'Cell_ID']]
            stat_metrics = st.multiselect("Select metrics for analysis:", options=numeric_cols, default=[m for m in ["VE_Ratio", "Disruption_Index"] if m in numeric_cols])
            if stat_metrics and "Column_Label" in df.columns:
                result_df = run_statistical_tests(df[["Column_Label"] + stat_metrics])
                st.dataframe(result_df)

