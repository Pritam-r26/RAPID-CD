# Changelog

All notable changes to RAPID-CD are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [1.2.0] — 2026

### All Modules — Performance
- **Navigation tabs replaced with `st.radio()` buttons** — the previous `st.tabs()` implementation recomputed all tab content on every interaction. Switching to radio-based navigation means only the active panel is rendered, giving a noticeably faster UI response especially for multi-sample datasets.

### All Modules — Dark Mode Support
- Added a dark-background CSS media query. When the user's system is set to dark mode, the custom radio-tab header and radio panel backgrounds switch to dark navy rather than light blue, preventing the washed-out appearance that occurred previously.

### General Analysis & Reversibility — Sequence Parser
- **Added smart peptide sequence input field** in the sidebar for each sample (General Analysis) and for the refolded spectrum (Reversibility).
- User types or pastes the one-letter amino acid sequence (e.g. `ALYFWCG`); the software automatically:
  - Counts residues and pre-fills the residue number field
  - Calculates the molar extinction coefficient at 280 nm (ε₂₈₀) using the standard formula: ε₂₈₀ = (W × 5500) + (Y × 1490) + (C_SS × 125), where C_SS = number of disulfide bonds estimated as ⌊C/2⌋
  - Estimates approximate molecular weight from residue count (MW ≈ n_res × 110 + 18 Da)
  - Flags sequences with high aromatic content (Trp + Tyr > 15%) with a warning that near-UV signal may distort the 222 nm helical band
- Manual residue entry remains available as fallback when no sequence is provided.

### General Analysis — Manual Colour Per Sample
- Updated the per-sample colour picker in the sidebar so that each sample's trace colour can be changed manually and the change is immediately reflected in all overlay plots without needing to reprocess.

### Thermal Analysis — HT and Absorbance Channel Fix
- **Root cause identified and fixed:** JASCO multi-column thermal files store CD in Channel 1, HT (voltage) in Channel 2, and Absorbance in Channel 3. The previous implementation read only Channel 1 regardless of the selected output metric, which meant HT and Abs values were silently wrong (returning CD data instead).
- Added `_read_thermal_channel_cached()` and `read_thermal_channel()` functions that parse the correct channel from the JASCO file header based on the selected metric.
- When metric = HT, Channel 2 is read; when metric = Abs, Channel 3 is read; when metric = CD-derived (MRE, mdeg, Δε), Channel 1 is read as before.
- Blank subtraction is correctly skipped for HT and Abs channels (it only applies to CD-derived signals).

---

## [1.1.0] — 2026

### General Analysis — Tab 2 (Separate Panels)
- Added persistent dashed zero line (y = 0) across all subplots — always visible regardless of grid setting.
- Fixed subplot title / axis overlap: expanded top and bottom margins and applied `yshift=12` to annotation labels.
- Pushed the global X-axis "Wavelength (nm)" label further below tick marks for cleaner PNG/PDF exports.

### General Analysis — Tab 6 (replaced)
- **Removed:** "📡 Spectral Indicators" tab (Θ₂₂₂/Θ₂₀₈ ratio and 310-helix flag). This metric is only valid for α-helical peptides and produced scientifically invalid output for other structural classes.
- **Added:** "🗺️ Multi-Sample Spectral Projection" — a composite ridgeline + discrete heatmap figure for comparing multiple samples. Requires ≥ 2 processed samples.

### Thermal Analysis — Tab 3 (rebuilt)
- **Removed:** Continuous 2D heatmap and WebGL-dependent 3D surface plot.
- **Added:** Composite "Discrete Spectral Projection" — stacked ridgeline (top) + discrete strip heatmap (bottom). Strips represent only measured temperature points, eliminating interpolation artefacts. Fully 2D — eliminates WebGL PDF export bugs.

### Thermal Analysis — Tab 1 & Tab 3
- **Added:** Temperature Selection multiselect — filter down to a subset of recorded temperatures.
- **Added:** "✏️ Rename Temperature Labels" expander — clean up instrument decimal rounding in legends and CSV exports.

---

## [1.0.0] — 2026

### Initial Public Release
- Automatic Δε / mdeg header detection from JASCO `YUNITS` field
- Three-way unit conversion (mdeg ↔ MRE ↔ Δε) across all modules
- Blank subtraction, Savitzky-Golay and LOWESS smoothing
- Multi-sample overlay, separate-panel view, statistical comparison
- NNLS secondary structure screening with Chen chain-length correction
- BeStSel and DichroWeb export formatter
- Thermal Analysis: multi-temperature melt processing, λ Peak Tracking, two-state van't Hoff thermodynamics with fit quality metrics, spectral simulation
- Reversibility Analysis: RMSD and Pearson correlation with interpretation thresholds
