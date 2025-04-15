
# Indralux
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Quantifying endothelial barrier disruption with image-based metrics.

Indralux is an advanced image analysis pipeline purpose-built to quantify blood-brain barrier (BBB) disruption, particularly in response to agents like mannitol. It combines biologically grounded quantification with metrics inspired by structural engineering and image science.

---

## Key Features

- Segment individual endothelial cells from immunofluorescent images
- Compute per-cell and per-image metrics for:
  - VE-cadherin localization and signal-to-noise
  - Actin cytoskeleton integrity
  - Nuclear compaction (DAPI)
  - Junctional fragmentation (periphery breaks)
  - Cell shape (circularity, solidity, aspect ratio)
  - Intercellular space (gap area, density, max gap)
  - Curvature irregularity of junctions
- Composite Disruption Index to track BBB damage
- Overlay images with contours and cell IDs
- Multi-metric summary plots and spatial heatmaps
- Export results as CSV, Markdown, LaTeX
- CLI support and Jupyter demos

---

## Example Metrics

| Metric | Description |
|--------|-------------|
| VE_Intensity_Ratio | Ratio of VE-cadherin at the membrane vs cytoplasm |
| F_Intensity_Ratio | Ratio of cortical actin to cytoplasmic actin |
| Disruption_Index | Composite score based on VE, F-actin, DAPI, and breaks |
| Intercellular_Area | Total area not occupied by cells (gaps) |
| Max_Gap_Size | Largest contiguous paracellular space |
| JCI (Curvature Irregularity) | Measures smoothness of junction outlines |
| Inflection Count | Count of curvature reversals on cell boundary |
| DAPI_Intensity | Proxy for nuclear stress / compaction |
| Circularity | Shape roundness (1 = perfect circle) |
| Solidity | Compactness vs. convex hull |
| Packing Density | Cell count per 100 µm² |
| Zonal Entropy | Texture randomness in actin or nucleus |

---

## Literature Basis

This tool is designed around biological literature and physical modeling of BBB disruption:
- Rapoport SI (1990), [DOI: 10.1227/00006123-199003000-00008]
- Abbott NJ et al. (2008), [DOI: 10.1016/j.tins.2008.05.003]
- Sweeney MD et al. (2017), [DOI: 10.3389/fnins.2017.00521]

---

## Citation

If you use this tool in your work, please cite:

> Indralux: Engineering-Inspired Metrics for Quantifying Endothelial Barrier Disruption (2024)

---

## Installation

```bash
pip install .
```

Or use via CLI:
```bash
python cli.py --input path/to/image.tif --output results/ --n_columns 7 --column_labels Control 5 10 15 20 30 40
```

---

## Contributions

Built using open-source components: `scikit-image`, `OpenCV`, `numpy`, `pandas`, `matplotlib`.

Please see `indralux_full_metric_catalog.csv` for full metric descriptions.
