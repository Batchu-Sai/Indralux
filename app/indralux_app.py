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

if "batch_results" not in st.session_state:
    st.session_state.batch_results = {}

mode = st.sidebar.radio("Select mode", ["Batch PPTX Upload", "Single Image Analysis"], key="mode_switch")

# Marker Channel Mapping (customizable in future)
marker_f1 = st.sidebar.selectbox("Marker in Channel 1 (Red)", ["F-Actin", "VE-Cadherin", "DAPI", "Other"], index=0)
marker_f2 = st.sidebar.selectbox("Marker in Channel 2 (Green)", ["VE-Cadherin", "F-Actin", "DAPI", "Other"], index=1)
marker_f3 = st.sidebar.selectbox("Marker in Channel 3 (Blue)", ["DAPI", "F-Actin", "VE-Cadherin", "Other"], index=2)
marker_channel_map = {marker_f1: 0, marker_f2: 1, marker_f3: 2}

# ──────────────── BATCH MODE ────────────────
if mode == "Batch PPTX Upload":
    pptx_file = st.sidebar.file_uploader("Upload .pptx file", type=["pptx"], key="pptx_uploader")

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
            st.image(img_path, caption=f"Preview: {selected}", use_column_width=True)

            label_key = f"labels_{selected}"
            run_key = f"run_{selected}"

            if label_key not in st.session_state:
                st.session_state[label_key] = "Control,5,10,15"

            n_cols = st.number_input("How many panels?", 1, 12, value=4, key=f"ncols_{selected}")
            col_labels_input = st.text_input("Column labels (comma-separated):", key=label_key)
            col_labels = [l.strip() for l in col_labels_input.split(",")]

            if st.button("▶️ Run analysis", key=f"runbtn_{selected}"):
                split_dir = os.path.join(tempfile.gettempdir(), "split_columns")
                os.makedirs(split_dir, exist_ok=True)
                col_paths = split_into_n_columns(img_path, split_dir, n_cols)

                per_col_data = []
                for idx, col_path in enumerate(col_paths):
                    try:
                        label = col_labels[idx] if idx < len(col_labels) else f"Col{idx+1}"
                        df, labels, img_rgb = process_with_breaks(col_path, n_columns=1, column_labels=[label])

                        morph = add_morphological_metrics(df, labels).drop(columns=["Column_Label", "Slide_Image", "Panel_Label"], errors="ignore")
                        morph = morph[[c for c in morph.columns if c not in df.columns or c == "Cell_ID"]]
                        df = pd.merge(df, morph, on="Cell_ID", how="left")

                        ext = add_extended_metrics(df, labels).drop(columns=["Column_Label", "Slide_Image", "Panel_Label"], errors="ignore")
                        ext = ext[[c for c in ext.columns if c not in df.columns or c == "Cell_ID"]]
                        df = pd.merge(df, ext, on="Cell_ID", how="left")

                        df = add_ve_snr(df, labels, img_rgb[:, :, marker_channel_map.get("VE-Cadherin", 1)])
                        df["Slide_Image"] = selected
                        df["Panel_Label"] = label
                        per_col_data.append(df)
                    except Exception as e:
                        st.warning(f"⚠️ Failed on column {idx+1}: {e}")

                if per_col_data:
                    result_df = pd.concat(per_col_data, ignore_index=True)
                    st.session_state.batch_results[selected] = result_df
                    st.success(f"✅ Metrics complete for {selected}")
                    st.dataframe(result_df.head())

                    metric_cols = [c for c in result_df.columns if result_df[c].dtype in ['float64', 'int64']]
                    safe_defaults = [m for m in ["DAPI_Intensity", "VE_Ratio", "Disruption_Index"] if m in metric_cols]

                    chosen_metrics = st.multiselect("📈 Plot metrics:", metric_cols, default=safe_defaults, key=f"plot_{selected}")
                    if chosen_metrics:
                        fig_path = os.path.join(tempfile.gettempdir(), f"plot_{selected}.png")
                        plot_metric_trends_manual(result_df, chosen_metrics, fig_path)
                        st.image(fig_path, caption="Metric Trends", use_column_width=True)

                    stat_metrics = st.multiselect("📊 Run statistics on:", metric_cols, default=["VE_Ratio", "Disruption_Index"], key=f"stats_{selected}")
                    if stat_metrics:
                        stats_df = run_statistical_tests(result_df[["Column_Label"] + stat_metrics])
                        st.dataframe(stats_df)
                        stat_path = os.path.join(tempfile.gettempdir(), f"stats_{selected}.csv")
                        stats_df.to_csv(stat_path, index=False)
                        st.download_button("⬇ Download Stats CSV", open(stat_path, "rb"), f"{selected}_stats.csv")

                    csv_out = os.path.join(tempfile.gettempdir(), f"{selected}_metrics.csv")
                    result_df.to_csv(csv_out, index=False)
                    st.download_button("⬇ Download Slide Metrics", open(csv_out, "rb"), f"{selected}_metrics.csv")

# Global export for batch
if st.session_state.batch_results:
    all_df = pd.concat(st.session_state.batch_results.values(), ignore_index=True)
    full_csv = os.path.join(tempfile.gettempdir(), "indralux_batch_all.csv")
    all_df.to_csv(full_csv, index=False)
    st.download_button("📦 Download All Batch Metrics", open(full_csv, "rb"), "indralux_batch_all.csv")

# ──────────────── SINGLE IMAGE MODE ────────────────
if mode == "Single Image Analysis":
    st.markdown("## 📸 Upload Single Microscopy Image")
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"], key="single_upload")

    if uploaded_file:
        labels_str = st.text_input("Column labels (comma-separated):", "Control,5,15,30", key="single_labels")
        labels = [l.strip() for l in labels_str.split(",")]

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.read())
            img_path = tmp.name

        st.image(img_path, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Processing image..."):
            try:
                df, labels_map, img_rgb = process_with_breaks(img_path, n_columns=len(labels), column_labels=labels)
                morph_df = add_morphological_metrics(df, labels_map).drop(columns=["Column_Label"], errors="ignore")
                ext_df = add_extended_metrics(df, labels_map).drop(columns=["Column_Label"], errors="ignore")
                df = pd.merge(df, morph_df, on="Cell_ID", how="left")
                df = pd.merge(df, ext_df, on="Cell_ID", how="left")
                df = add_ve_snr(df, labels_map, img_rgb[:, :, marker_channel_map.get("VE-Cadherin", 1)])
                st.success("Segmentation and metrics complete.")
                st.dataframe(df.head())
            except Exception as e:
                st.error(f"❌ Failed to process image: {e}")
                st.stop()

        if st.checkbox("Show overlay"):
            overlay = draw_colored_overlay_with_cv2(img_rgb, labels_map, df)
            path = os.path.join(tempfile.gettempdir(), "overlay.png")
            cv2.imwrite(path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            st.image(path, caption="Overlay", use_column_width=True)

        if st.checkbox("📈 Show metric trends", key="plot_trends_single"):
            numeric_cols = [c for c in df.columns if df[c].dtype in ['float64', 'int64']]
            defaults = [m for m in ["DAPI_Intensity", "VE_Ratio", "Disruption_Index"] if m in numeric_cols]
            selected_metrics = st.multiselect("Select metrics to plot:", numeric_cols, default=defaults, key="plot_metrics_single")

            if selected_metrics:
                fig_path = os.path.join(tempfile.gettempdir(), "trend_plot_single.png")
                plot_metric_trends_manual(df, selected_metrics, fig_path)
                st.image(fig_path, caption="Metric Trends", use_column_width=True)

        if st.checkbox("📊 Run statistics", key="stats_single"):
            numeric_cols = [col for col in df.columns if df[col].dtype in ['float64', 'int64']]
            stat_metrics = st.multiselect("Select metrics for stats:", options=numeric_cols, default=["VE_Ratio", "Disruption_Index"], key="stat_metrics_single")
            if "Column_Label" in df.columns and stat_metrics:
                stat_df = run_statistical_tests(df[["Column_Label"] + stat_metrics])
                st.dataframe(stat_df)
                stat_csv = os.path.join(tempfile.gettempdir(), "single_stats.csv")
                stat_df.to_csv(stat_csv, index=False)
                st.download_button("⬇ Download Stats CSV", open(stat_csv, "rb"), "single_stats.csv")

        csv_out = os.path.join(tempfile.gettempdir(), "metrics_output_single.csv")
        df.to_csv(csv_out, index=False)
        st.download_button("📂 Download All Metrics", open(csv_out, "rb"), "indralux_metrics_single.csv")
