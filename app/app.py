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

            n_cols = st.number_input("How many columns?", 1, 12, value=4, key=f"ncols_{selected}")
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


# ─── SINGLE IMAGE ANALYSIS ──────────────────────────────
elif mode == "Single Image Analysis":
    uploaded_image = st.sidebar.file_uploader("Upload a single RGB image", type=["png", "jpg", "jpeg", "tif", "tiff"])
    label_key = "label_single"
    if label_key not in st.session_state:
        st.session_state[label_key] = "Sample1"

    label = st.sidebar.text_input("Label for image:", key=label_key)
    n_cols = st.sidebar.number_input("How many columns?", 1, 12, value=1, key="ncols_single")
    col_labels_input = st.sidebar.text_input("Column labels (comma-separated):", value="Control,5,10,15")
    col_labels = [l.strip() for l in col_labels_input.split(",")]

    if uploaded_image:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(uploaded_image.read())
            img_path = tmp.name

        st.image(img_path, caption="Uploaded Image", use_column_width=True)

        if st.sidebar.button("▶️ Start Analysis", key="run_single"):
            split_dir = os.path.join(tempfile.gettempdir(), "split_columns_single")
            os.makedirs(split_dir, exist_ok=True)
            col_paths = split_into_n_columns(img_path, split_dir, n_cols)

            per_col_data = []
            all_labels = []
            all_imgs = []

            for idx, col_path in enumerate(col_paths):
                try:
                    df, labels, img_rgb = process_with_breaks(col_path, n_columns=1, column_labels=[label])
                    morph_df = add_morphological_metrics(df, labels)
                    ext_df = add_extended_metrics(df, labels)
                    df = add_ve_snr(df, labels, img_rgb[:, :, 1])

                    # Drop duplicates
                    morph_df = morph_df.drop(columns=["Column_Label", "Slide_Image", "Panel_Label"], errors="ignore")
                    morph_df = morph_df[[col for col in morph_df.columns if col not in df.columns or col == "Cell_ID"]]
                    df = pd.merge(df, morph_df, on="Cell_ID", how="left")

                    ext_df = ext_df.drop(columns=["Column_Label", "Slide_Image", "Panel_Label"], errors="ignore")
                    ext_df = ext_df[[col for col in ext_df.columns if col not in df.columns or col == "Cell_ID"]]
                    df = pd.merge(df, ext_df, on="Cell_ID", how="left")

                    col_label = col_labels[idx] if idx < len(col_labels) else f"Col{idx+1}"
                    df["Slide_Image"] = "SingleUpload"
                    df["Column_Label"] = col_label

                    per_col_data.append(df)
                    all_labels.append(labels)
                    all_imgs.append(img_rgb)
                except Exception as e:
                    st.warning(f"Error processing column {idx+1}: {e}")

            st.session_state["results_single"] = {
                "data": per_col_data,
                "labels": all_labels,
                "img_rgb": all_imgs
            }

    if "results_single" in st.session_state:
        per_col_data = st.session_state["results_single"]["data"]
        all_labels = st.session_state["results_single"]["labels"]
        all_imgs = st.session_state["results_single"]["img_rgb"]
        result_df = pd.concat(per_col_data, ignore_index=True)

        st.success("✅ Analysis complete.")
        st.dataframe(result_df.head())

        metric_cols = [col for col in result_df.columns if result_df[col].dtype in ['float64', 'int64']]

        if st.checkbox("Overlay"):
            for idx, (labels, img_rgb, df) in enumerate(zip(all_labels, all_imgs, per_col_data)):
                overlay = draw_colored_overlay_with_cv2(img_rgb, labels, df)
                overlay_path = os.path.join(tempfile.gettempdir(), f"overlay_col{idx+1}.png")
                cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                st.image(overlay_path, caption=f"Overlay Column {idx+1}", use_column_width=True)

        if st.checkbox("Trend plots"):
            default_metrics = ["DAPI_Intensity", "VE_Ratio", "Disruption_Index"]
            safe_defaults = [m for m in default_metrics if m in metric_cols]
            selected = st.multiselect("Metrics to plot:", metric_cols, default=safe_defaults, key="metrics_to_plot_single")
            if selected:
                fig_path = os.path.join(tempfile.gettempdir(), "trend_plot.png")
                plot_metric_trends_manual(result_df, selected, fig_path)
                st.image(fig_path, caption="Metric Trends", use_column_width=True)

        if st.checkbox("Statistics"):
            default_metrics = ["DAPI_Intensity", "VE_Ratio", "Disruption_Index"]
            safe_defaults = [m for m in default_metrics if m in metric_cols]
            selected = st.multiselect("Run stats on:", metric_cols, default=safe_defaults, key="metrics_for_stats_single")
            if selected and "Column_Label" in result_df.columns:
                stats_df = run_statistical_tests(result_df[["Column_Label"] + selected])
                st.dataframe(stats_df)
                stats_path = os.path.join(tempfile.gettempdir(), "kruskal_results.csv")
                stats_df.to_csv(stats_path, index=False)
                st.download_button("Download Stats CSV", open(stats_path, "rb"), "kruskal_results.csv")

        final_csv = os.path.join(tempfile.gettempdir(), "metrics_output.csv")
        result_df.to_csv(final_csv, index=False)
        st.download_button("📂 Download Metrics", open(final_csv, "rb"), "indralux_metrics.csv")

        if st.sidebar.button("🔁 Reset Analysis"):
            st.session_state.pop("results_single", None)
