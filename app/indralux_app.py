import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import tempfile, os, cv2, sys

# Enable parent directory access
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Core modules
from core.processor import process_with_breaks
from core.metrics import add_morphological_metrics, add_extended_metrics, add_ve_snr
from core.overlay import draw_colored_overlay_with_cv2
from core.plotting import plot_spatial_disruption_map, plot_metric_trends_manual
from core.indralux_stats import run_statistical_tests

# Utility modules
from utils.pptx_extract import extract_clean_images_from_pptx
from utils.column_split_uniform import split_into_n_columns

# ——— PAGE CONFIG ———
st.set_page_config(page_title="Indralux", page_icon="assets/favicon_32.png", layout="wide")
st.image("assets/indralux_final_logo.png", width=300)
st.markdown("<h2 style='text-align: center;'>Quantifying endothelial disruption — pixel by pixel</h2>", unsafe_allow_html=True)
st.markdown("---")

# ——— STATE INIT ———
if "batch_results" not in st.session_state:
    st.session_state.batch_results = {}

# ——— BATCH PPT ANALYSIS ———
if st.checkbox("📂 Upload .pptx for Batch Analysis"):
    pptx_file = st.file_uploader("Upload your .pptx file", type=["pptx"])

    if pptx_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pptx") as tmp:
            tmp.write(pptx_file.read())
            pptx_path = tmp.name

        extract_dir = os.path.join(tempfile.gettempdir(), "pptx_clean_images")
        os.makedirs(extract_dir, exist_ok=True)
        clean_imgs = extract_clean_images_from_pptx(pptx_path, extract_dir)

        if not clean_imgs:
            st.error("❌ No images extracted from .pptx")
        else:
            selected = st.selectbox("📸 Select slide to analyze:", clean_imgs)
            img_path = os.path.join(extract_dir, selected)
            st.image(img_path, caption=f"Preview: {selected}", use_container_width=True)

            n_key = f"ncols_{selected}"
            l_key = f"labels_{selected}"
            r_key = f"run_{selected}"

            if l_key not in st.session_state:
                st.session_state[l_key] = "Control,5,10,15"

            st.session_state[r_key] = st.button("▶️ Run Analysis", key=f"btn_{selected}")

            n_cols = st.number_input("Number of columns (panels)", min_value=1, max_value=12, value=4, key=n_key)
            col_labels = st.text_input("Column labels (comma-separated)", value=st.session_state[l_key], key=l_key)
            col_labels = [c.strip() for c in col_labels.split(",")]

            if st.session_state[r_key]:
                split_dir = os.path.join(tempfile.gettempdir(), "split_columns")
                os.makedirs(split_dir, exist_ok=True)
                col_paths = split_into_n_columns(img_path, split_dir, n_cols)

                per_col_data = []
                for idx, col_path in enumerate(col_paths):
                    try:
                        label = col_labels[idx] if idx < len(col_labels) else f"Col{idx+1}"
                        df, labels, img_rgb = process_with_breaks(col_path, n_columns=1, column_labels=[label])
                        morph = add_morphological_metrics(df, labels).drop(columns=["Column_Label"], errors="ignore")
                        ext = add_extended_metrics(df, labels).drop(columns=["Column_Label"], errors="ignore")

                        morph = morph[[c for c in morph.columns if c not in df.columns or c == "Cell_ID"]]
                        ext = ext[[c for c in ext.columns if c not in df.columns or c == "Cell_ID"]]

                        df = pd.merge(df, morph, on="Cell_ID", how="left")
                        df = pd.merge(df, ext, on="Cell_ID", how="left")
                        df = add_ve_snr(df, labels, img_rgb[:, :, 1])

                        df["Slide_Image"] = selected
                        df["Panel_Label"] = label
                        per_col_data.append(df)
                    except Exception as e:
                        st.warning(f"⚠️ Error in {col_path}: {e}")

                if per_col_data:
                    result_df = pd.concat(per_col_data, ignore_index=True)
                    st.session_state.batch_results[selected] = result_df
                    st.success("✅ Analysis complete")
                    st.dataframe(result_df.head())

                    metric_cols = [col for col in result_df.columns if result_df[col].dtype in [float, int]]
                    defaults = [m for m in ["DAPI_Intensity", "VE_Ratio", "Disruption_Index"] if m in metric_cols]

                    selected_metrics = st.multiselect("📈 Metrics to plot:", metric_cols, default=defaults, key=f"plot_{selected}")
                    if selected_metrics:
                        fig_path = os.path.join(tempfile.gettempdir(), f"{selected}_trend.png")
                        plot_metric_trends_manual(result_df, selected_metrics, fig_path)
                        st.image(fig_path, caption="Metric Trends", use_container_width=True)

                    stat_metrics = st.multiselect("📊 Metrics for stats:", metric_cols, default=["VE_Ratio", "Disruption_Index"], key=f"stat_{selected}")
                    if stat_metrics:
                        if "Column_Label" not in result_df.columns and "Panel_Label" in result_df.columns:
                            result_df["Column_Label"] = result_df["Panel_Label"]

                        stat_df = run_statistical_tests(result_df[["Column_Label"] + stat_metrics])
                        st.dataframe(stat_df)

                        stat_csv = os.path.join(tempfile.gettempdir(), f"{selected}_stats.csv")
                        stat_df.to_csv(stat_csv, index=False)
                        st.download_button("⬇ Download Stats CSV", open(stat_csv, "rb"), f"{selected}_stats.csv")

                    # Save CSV
                    out_csv = os.path.join(tempfile.gettempdir(), f"{selected}_metrics.csv")
                    result_df.to_csv(out_csv, index=False)
                    st.download_button("⬇ Download Metrics CSV", open(out_csv, "rb"), f"{selected}_metrics.csv")

# Full batch export
if st.session_state.batch_results:
    full_df = pd.concat(st.session_state.batch_results.values(), ignore_index=True)
    full_csv = os.path.join(tempfile.gettempdir(), "indralux_full_batch.csv")
    full_df.to_csv(full_csv, index=False)
    st.download_button("📦 Download All Batch Metrics", open(full_csv, "rb"), "indralux_batch_all.csv")

# ——— SINGLE IMAGE ANALYSIS ———
st.markdown("## 📸 Single Image Upload")
uploaded_file = st.file_uploader("Upload a fluorescent microscopy image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    col_text = st.text_input("Column labels (comma-separated):", "Control,5,15,30")
    column_labels = [label.strip() for label in col_text.split(",")]

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        img_path = tmp.name

    st.image(img_path, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Processing image..."):
        try:
            df, labels, img_rgb = process_with_breaks(img_path, n_columns=len(column_labels), column_labels=column_labels)
            morph_df = add_morphological_metrics(df, labels).drop(columns=["Column_Label"], errors="ignore")
            ext_df = add_extended_metrics(df, labels).drop(columns=["Column_Label"], errors="ignore")

            morph_df = morph_df[[c for c in morph_df.columns if c not in df.columns or c == "Cell_ID"]]
            ext_df = ext_df[[c for c in ext_df.columns if c not in df.columns or c == "Cell_ID"]]

            df = pd.merge(df, morph_df, on="Cell_ID", how="left")
            df = pd.merge(df, ext_df, on="Cell_ID", how="left")
            df = add_ve_snr(df, labels, img_rgb[:, :, 1])
            st.success("✅ Processing complete")
        except Exception as e:
            st.error(f"❌ Failed to process: {e}")
            st.stop()

    st.dataframe(df.head())

    if st.checkbox("Show overlay with cell labels"):
        overlay = draw_colored_overlay_with_cv2(img_rgb, labels, df)
        overlay_path = os.path.join(tempfile.gettempdir(), "overlay.png")
        cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        st.image(overlay_path, caption="Overlay", use_container_width=True)

    if st.checkbox("Plot metric trends"):
        metric_cols = [col for col in df.columns if df[col].dtype in [float, int]]
        defaults = [m for m in ["DAPI_Intensity", "VE_Ratio", "Disruption_Index"] if m in metric_cols]
        selected_metrics = st.multiselect("Select metrics:", metric_cols, default=defaults)
        if selected_metrics:
            fig_path = os.path.join(tempfile.gettempdir(), "trend_plot.png")
            plot_metric_trends_manual(df, selected_metrics, fig_path)
            st.image(fig_path, caption="Metric Trends", use_container_width=True)

    if st.checkbox("Run statistical tests"):
        numeric_cols = [col for col in df.columns if df[col].dtype in [float, int]]
        selected_stats = st.multiselect("Stats metrics:", numeric_cols, default=["VE_Ratio", "Disruption_Index"])
        if selected_stats and "Column_Label" in df.columns:
            stats = run_statistical_tests(df[["Column_Label"] + selected_stats])
            st.dataframe(stats)
            csv_path = os.path.join(tempfile.gettempdir(), "kruskal_results.csv")
            stats.to_csv(csv_path, index=False)
            st.download_button("Download Stats CSV", open(csv_path, "rb"), "kruskal_results.csv")

    out_csv = os.path.join(tempfile.gettempdir(), "metrics_output.csv")
    df.to_csv(out_csv, index=False)
    st.download_button("📥 Download Metrics CSV", open(out_csv, "rb"), "indralux_metrics.csv")
