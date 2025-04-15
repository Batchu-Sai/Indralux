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
from utils.pptx_extract import extract_clean_images_from_pptx
from utils.column_split_uniform import split_into_n_columns

# ——— PAGE HEADER ———
st.set_page_config(page_title="Indralux", page_icon="assets/favicon_32.png", layout="wide")
st.image("assets/indralux_final_logo.png", width=300)
st.markdown("<h2 style='text-align: center;'>Quantifying endothelial disruption — pixel by pixel</h2>", unsafe_allow_html=True)
st.markdown("---")


# ——— BATCH PPT ANALYSIS ———
# Track results
if "batch_results" not in st.session_state:
    st.session_state.batch_results = {}

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
            st.error("❌ No clean images found.")
        else:
            selected = st.selectbox("📸 Select slide image to analyze:", clean_imgs)

            if selected:
                # Show preview
                img_path = os.path.join(extract_dir, selected)
                st.image(img_path, caption=f"Preview: {selected}", use_container_width=True)

                # Setup state for this image
                label_key = f"labels_{selected}"
                run_key = f"run_{selected}"

                if label_key not in st.session_state:
                    st.session_state[label_key] = "Control,5,10,15"
                if run_key not in st.session_state:
                    st.session_state[run_key] = False

                n_cols = st.number_input("How many panels?", min_value=1, max_value=12, value=4, key=f"ncols_{selected}")
                col_labels_input = st.text_input("Column labels (comma-separated):", key=label_key)
                col_labels = [l.strip() for l in col_labels_input.split(",")]

                if st.button("▶️ Run analysis", key=f"runbtn_{selected}"):
                    st.session_state[run_key] = True

                # Run analysis if triggered
                if st.session_state[run_key]:
                    split_dir = os.path.join(tempfile.gettempdir(), "split_columns")
                    os.makedirs(split_dir, exist_ok=True)
                    col_paths = split_into_n_columns(img_path, split_dir, n_cols)

                    per_col_data = []
                    for idx, col_path in enumerate(col_paths):
                        try:
                            label = col_labels[idx] if idx < len(col_labels) else f"Col{idx+1}"
                            df, labels, img_rgb = process_with_breaks(col_path, n_columns=1, column_labels=[label])
                            morph = add_morphological_metrics(df, labels)
                            morph = morph.drop(columns=[col for col in ['Column_Label', 'Slide_Image', 'Panel_Label'] if col in morph.columns])
                            df = pd.merge(df, morph, on="Cell_ID", how="left")
                            ext = add_extended_metrics(df, labels)
                            ext = ext.drop(columns=[col for col in ['Column_Label', 'Slide_Image', 'Panel_Label'] if col in ext.columns])
                            df = pd.merge(df, ext, on="Cell_ID", how="left")
                            df = add_ve_snr(df, labels, img_rgb[:, :, 1])
                            df["Slide_Image"] = selected
                            df["Panel_Label"] = label
                            per_col_data.append(df)
                        except Exception as e:
                            st.warning(f"⚠️ Failed: {e}")

                    if per_col_data:
                        result_df = pd.concat(per_col_data, ignore_index=True)
                        st.session_state.batch_results[selected] = result_df
                        st.success(f"✅ Metrics for {selected}")
                        st.dataframe(result_df.head())

                        # Plots
                        metric_cols = [col for col in result_df.columns if result_df[col].dtype in ['float64', 'int64']]
                        safe_defaults = [m for m in ["DAPI_Intensity", "VE_Ratio", "Disruption_Index"] if m in metric_cols]
                        chosen_metrics = st.multiselect("📈 Plot metrics:", metric_cols, default=safe_defaults, key=f"plot_{selected}")

                        if chosen_metrics:
                            fig_path = os.path.join(tempfile.gettempdir(), f"plot_{selected}.png")
                            plot_metric_trends_manual(result_df, chosen_metrics, fig_path)
                            st.image(fig_path, caption="Trend Plot", use_container_width=True)

                        # Stats
                        safe_stat_defaults = [m for m in ["VE_Ratio", "Disruption_Index"] if m in metric_cols]
                        stat_cols = st.multiselect("📊 Run stats on:", metric_cols, default=safe_stat_defaults, key=f"stats_{selected}")

                        if stat_cols:
                            stats_df = run_statistical_tests(result_df[["Column_Label"] + stat_cols])
                            st.dataframe(stats_df)
                            stats_csv = os.path.join(tempfile.gettempdir(), f"stats_{selected}.csv")
                            stats_df.to_csv(stats_csv, index=False)
                            st.download_button("⬇ Download Stats CSV", open(stats_csv, "rb"), f"{selected}_stats.csv")

                        # Per-slide export
                        out_csv = os.path.join(tempfile.gettempdir(), f"{selected}_metrics.csv")
                        result_df.to_csv(out_csv, index=False)
                        st.download_button("⬇ Download Slide CSV", open(out_csv, "rb"), f"{selected}_metrics.csv")

# Global export
if st.session_state.batch_results:
    all_df = pd.concat(st.session_state.batch_results.values(), ignore_index=True)
    full_csv = os.path.join(tempfile.gettempdir(), "indralux_full_batch.csv")
    all_df.to_csv(full_csv, index=False)
    st.download_button("📦 Download All Metrics CSV", open(full_csv, "rb"), "indralux_batch_all.csv")



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

    if st.checkbox("Show metric trends"):
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

    if st.checkbox("Run statistical tests"):
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



