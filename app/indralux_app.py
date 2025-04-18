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

# ─── BATCH ANALYSIS ──────────────────────────────
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

                        # --- Morphological Metrics ---
                        morph_df = add_morphological_metrics(df, labels)
                        morph_df = morph_df.drop(columns=["Column_Label", "Slide_Image", "Panel_Label"], errors="ignore")

                        # Remove overlapping columns except the join key
                        morph_df = morph_df[[col for col in morph_df.columns if col not in df.columns or col == "Cell_ID"]]
                        df = pd.merge(df, morph_df, on="Cell_ID", how="left")
                    
                        # --- Extended Metrics ---
                        ext_df = add_extended_metrics(df, labels)
                        ext_df = ext_df.drop(columns=["Column_Label", "Slide_Image", "Panel_Label"], errors="ignore")
                        ext_df = ext_df[[col for col in ext_df.columns if col not in df.columns or col == "Cell_ID"]]
                        df = pd.merge(df, ext_df, on="Cell_ID", how="left")
                    
                        # --- VE SNR ---
                        df = add_ve_snr(df, labels, img_rgb[:, :, 1])
                    
                        # Annotate
                        df["Slide_Image"] = selected
                        df["Panel_Label"] = label
                        df["Column_Label"] = label
                    
                        per_col_data.append(df)

                    
                    except Exception as e:
                        st.warning(f"⚠️ Failed to process column {idx + 1} of {selected}: {e}")
                st.session_state["results_single"] = per_col_data
                # Store results for this slide in session (outside of try loop)
                st.session_state[f"results_{selected}"] = per_col_data

                    

                if per_col_data:
                    result_df = pd.concat(per_col_data, ignore_index=True)
                    result_df["Column_Label"] = result_df["Panel_Label"]
                    st.session_state.batch_results[selected] = result_df
                    st.success("✅ Analysis complete")
                    st.dataframe(result_df.head())

                    metric_cols = [col for col in result_df.columns if result_df[col].dtype in ['float64', 'int64']]
                    safe_defaults = [m for m in ["DAPI_Intensity", "VE_Ratio", "Disruption_Index"] if m in metric_cols]
                    chosen_metrics = st.multiselect("📈 Plot metrics:", metric_cols, default=safe_defaults, key=f"plot_{selected}")

                    if chosen_metrics:
                        fig_path = os.path.join(tempfile.gettempdir(), f"plot_{selected}.png")
                        plot_metric_trends_manual(result_df, chosen_metrics, fig_path)
                        st.image(fig_path, caption="Metric Trends", use_column_width=True)

                    stat_defaults = [m for m in ["VE_Ratio", "Disruption_Index"] if m in metric_cols]
                    stat_cols = st.multiselect("📊 Stats:", metric_cols, default=stat_defaults, key=f"stats_{selected}")
                    if stat_cols:
                        stats_df = run_statistical_tests(result_df[["Column_Label"] + stat_cols])
                        st.dataframe(stats_df)
                        csv_path = os.path.join(tempfile.gettempdir(), f"{selected}_stats.csv")
                        stats_df.to_csv(csv_path, index=False)
                        st.download_button("⬇ Download Stats", open(csv_path, "rb"), f"{selected}_stats.csv")

                    out_csv = os.path.join(tempfile.gettempdir(), f"{selected}_metrics.csv")
                    result_df.to_csv(out_csv, index=False)
                    st.download_button("⬇ Download Slide CSV", open(out_csv, "rb"), f"{selected}_metrics.csv")

# Final CSV
if st.session_state.batch_results:
    all_df = pd.concat(st.session_state.batch_results.values(), ignore_index=True)
    full_csv = os.path.join(tempfile.gettempdir(), "indralux_batch_all.csv")
    all_df.to_csv(full_csv, index=False)
    st.download_button("📦 Download All Metrics CSV", open(full_csv, "rb"), "indralux_batch_all.csv")

# ─── SINGLE IMAGE ANALYSIS ──────────────────────────────
if mode == "Single Image Analysis":
    uploaded_file = st.sidebar.file_uploader("Upload a single fluorescent microscopy image", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        column_labels = st.text_input("Enter column labels:", "Control,5,15,30").split(",")

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.read())
            img_path = tmp.name

        st.image(img_path, caption="Uploaded Image", use_column_width=True)
        with st.spinner("Processing..."):
            try:
                df, labels, img_rgb = process_with_breaks(img_path, n_columns=len(column_labels), column_labels=column_labels)
    
                morph_df = add_morphological_metrics(df, labels).drop(columns=["Column_Label"], errors="ignore")
                morph_df = morph_df[[col for col in morph_df.columns if col not in df.columns or col == "Cell_ID"]]
                df = pd.merge(df, morph_df, on="Cell_ID", how="left")
    
                ext_df = add_extended_metrics(df, labels).drop(columns=["Column_Label"], errors="ignore")
                ext_df = ext_df[[col for col in ext_df.columns if col not in df.columns or col == "Cell_ID"]]
                df = pd.merge(df, ext_df, on="Cell_ID", how="left")
    
                df = add_ve_snr(df, labels, img_rgb[:, :, 1])
    
                st.success("Segmentation and metrics complete.")
            except Exception as e:
                st.error(f"Failed to process image: {e}")
                st.stop()
        
        st.dataframe(df.head())
        if st.checkbox("Overlay", key='cb_overlay'):
            overlay = draw_colored_overlay_with_cv2(img_rgb, labels, df)
            overlay_path = os.path.join(tempfile.gettempdir(), "overlay.png")
            cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            st.image(overlay_path, caption="Overlay", use_column_width=True)
        
        if st.checkbox("Trend plots", key='cb_plot'):
            metric_cols = [col for col in df.columns if df[col].dtype in ['float64', 'int64']]
            defaults = [m for m in ["DAPI_Intensity", "VE_Ratio", "Disruption_Index"] if m in metric_cols]
            selected = st.multiselect("Metrics:", metric_cols, default=defaults)
            if selected:
                fig_path = os.path.join(tempfile.gettempdir(), "trend_plot.png")
                plot_metric_trends_manual(df, selected, fig_path)
                st.image(fig_path, caption="Metric Trends", use_column_width=True)
        
        if st.checkbox("Statistics", key='cb_stats'):
            metric_cols = [col for col in df.columns if df[col].dtype in ['float64', 'int64']]
            selected = st.multiselect("Run stats on:", metric_cols, default=[m for m in ["VE_Ratio", "Disruption_Index"] if m in metric_cols])
            if selected and "Column_Label" in df.columns:
                result_df = run_statistical_tests(df[["Column_Label"] + selected])
                st.dataframe(result_df)
                csv_path = os.path.join(tempfile.gettempdir(), "kruskal_results.csv")
                result_df.to_csv(csv_path, index=False)
                st.download_button("Download Stats CSV", open(csv_path, "rb"), "kruskal_results.csv")
        
        final_csv = os.path.join(tempfile.gettempdir(), "metrics_output.csv")
        df.to_csv(final_csv, index=False)
        st.download_button("📂 Download Metrics", open(final_csv, "rb"), "indralux_metrics.csv")







# ─── METRIC SELECTION & PLOTTING ──────────────────────────────
if mode == "Batch PPTX Upload":
    if 'pptx_file' in locals() and pptx_file and f"results_{selected}" in st.session_state:
        df_list = st.session_state[f"results_{selected}"]
        full_df = pd.concat(df_list, ignore_index=True)

        numeric_cols = full_df.select_dtypes(include="number").columns.tolist()
        excluded = ["Cell_ID"]
        metric_options = [col for col in numeric_cols if col not in excluded]

        selected_metric = st.selectbox("Select metric to graph or run stats on:", metric_options)

        if selected_metric:
            st.line_chart(full_df.groupby("Panel_Label")[selected_metric].mean())

            if st.button("Run stats on selected metric"):
                st.write("Running statistical tests...")
                stat_result = run_statistical_tests(full_df, metric=selected_metric)
                st.write(stat_result)

elif mode == "Single Image Analysis":
    if "results_single" in st.session_state:
        df_list = st.session_state["results_single"]
        full_df = pd.concat(df_list, ignore_index=True)

        numeric_cols = full_df.select_dtypes(include="number").columns.tolist()
        excluded = ["Cell_ID"]
        metric_options = [col for col in numeric_cols if col not in excluded]

        selected_metric = st.selectbox("Select metric to graph or run stats on (Single Image):", metric_options)

        if selected_metric:
            st.line_chart(full_df.groupby("Panel_Label")[selected_metric].mean())

            if st.button("Run stats on selected metric (Single Image)"):
                st.write("Running statistical tests...")
                stat_result = run_statistical_tests(full_df, metric=selected_metric)
                st.write(stat_result)
