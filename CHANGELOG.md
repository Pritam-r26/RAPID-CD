# Changelog

All notable changes to RAPID-CD are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

\---

## \[1.0.0] — 2026

### Initial Public Release

#### General Analysis

* Automatic Δε / mdeg header detection from JASCO `YUNITS` field
* Three-way unit conversion (mdeg ↔ MRE ↔ Δε) across all modules
* Blank subtraction and smoothing (Savitzky-Golay, LOWESS)
* Multi-sample overlay with publication-style formatting
* Separate-panel view with optional base-curve overlay
* Auto-positioned multi-panel axis labels
* Statistical comparison panel: Pearson correlation heatmap, dendrogram, spectral indicators
* Secondary structure screening via NNLS with optional Chen chain-length correction
* 2×2 batch statistics panel export
* BeStSel and DichroWeb export formatter

#### Thermal Analysis

* Multi-column JASCO thermal file parsing and discrete-file mode
* Temperature filter in overlay tab (select subset of temperatures per sample)
* Multi-panel comparison by temperature with custom titles
* λ–T Spectromap (heatmap)
* λ Peak Tracking tab: dual-axis position + intensity plots, multi-sample comparison, combined 2×2 overview
* Secondary structure vs temperature (NNLS + empirical)
* Thermodynamics: two-state van't Hoff model with sloping baselines, fit quality (R², RMSE), confidence flags, residuals plot
* Apparent ΔG and ΔΔG with experiment-type context warnings (thermal / SDS / chemical denaturation / ligand)
* Thermodynamic spectral simulation (linear and cubic spline interpolation)
* Peak search windows: restrict λ min/max search range to avoid noise artefacts
* Melting wavelength input accepts decimal values

#### Reversibility Analysis

* Pre-melt vs post-refolding spectral comparison
* RMSD and Pearson correlation with interpretation ranges
* Peak statistics table with configurable search windows
* Plot customisation: axis ranges, line colours, grid toggle, zero reference line

#### Infrastructure

* Privacy-centric: all computation local, no external data transmission
* Modular sidebar navigation with mode switching
* Analysis log export (txt) for reproducibility
* Comprehensive in-app methodological notes and citation guidance

