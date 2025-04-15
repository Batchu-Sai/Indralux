
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

The following peer-reviewed works informed the metrics, methodology, and biological rationale used in Indralux:

| Reference | Title | Journal | DOI | Relevance |
|-----------|-------|---------|-----|-----------|
| Rapoport SI (1990) | Osmotic opening of the blood-brain barrier | Neurosurgery | [Link](https://doi.org/10.1227/00006123-199003000-00008) | Mannitol-induced BBB disruption; foundational background |
| Abbott NJ et al. (2010) | Structure and function of the blood–brain barrier | Trends in Neurosciences | [Link](https://doi.org/10.1016/j.tins.2010.05.003) | Overview of endothelial junctions and barrier integrity |
| Sweeney MD et al. (2017) | Blood–brain barrier: from physiology to disease and back | Frontiers in Neuroscience | [Link](https://doi.org/10.3389/fnins.2017.00521) | Mechanisms of barrier breakdown in pathology |
| Haralick RM et al. (1973) | Textural Features for Image Classification | IEEE Transactions on Systems, Man, and Cybernetics | [Link](https://doi.org/10.1109/TSMC.1973.4309314) | Basis for entropy, texture, and spatial randomness metrics |
| Russ JC (2016) | The Image Processing Handbook, 7th ed. | CRC Press | [Link](https://doi.org/10.1201/b18708) | Reference for signal-to-noise and image quality assessment |
| Sobolev O et al. (2014) | Mechanosensitive junction remodeling in endothelium | Nature Cell Biology | [Link](https://doi.org/10.1038/ncb2939) | Biological precedent for curvature-based metrics |
| Zhou W et al. (2022) | Stress fiber dynamics and force transmission in endothelial monolayers | Molecular Cell | [Link](https://doi.org/10.1016/j.molcel.2022.03.001) | Actin pullback and inflection dynamics |

---

## Citation

If you use this tool in your work, please cite:

> Indralux: Engineering-Inspired Metrics for Quantifying Endothelial Barrier Disruption (2025)

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
