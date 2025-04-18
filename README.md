# Fluorescence Image Analyzer

A lightweight image analysis tool for fluorescence microscopy data, built as a side project for in-house workflow.

## Features

- Single and batch image analysis  
- Column-wise splitting and labeling  
- Morphological + fluorescence metrics  
- Statistical tests (Kruskal-Wallis, Dunn's)  
- Visual overlays, trend plots, CSV export  
- Streamlit-powered interactive UI

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

<p align="center">
  <a href="https://indralux.streamlit.app/">Live App</a> •
  <a href="https://github.com/Batchu-Sai/Indralux">GitHub Repo</a> •
  <a href="docs/metrics_table.md">Full Metrics Table</a> •
  <a href="#how-it-works">How It Works</a>
</p>

---

If you use this tool in your work, please cite:

fluorescence-image-analyzer (2025)

Usage
	•	Upload RGB microscopy images
	•	Define number of columns and labels
	•	Click “▶️ Start Analysis” to process
	•	Use the sidebar to enable overlay, stats, and plotting
	•	Export metrics as CSV

Notes

This was a side project built for internal use — there’s minimal polish and no guarantees. Still, if it helps your workflow, you’re welcome to use or adapt it.

---

## Contributions

Built using open-source components: `scikit-image`, `OpenCV`, `numpy`, `pandas`, `matplotlib`.
