# RAPID-CD User Manual

## Rapid Analysis Pipeline for Interpreting Dichroism — v1.0

**Author:** Pritam Roy, Sorbonne University, Paris, France  
**Last updated:** 2026

\---

## Table of Contents

1. [Overview \& Scientific Philosophy](#1-overview--scientific-philosophy)
2. [Installation](#2-installation)
3. [Launching the Application](#3-launching-the-application)
4. [Home Screen](#4-home-screen)
5. [Module 1 — General Analysis](#5-module-1--general-analysis)
6. [Module 2 — Thermal Analysis](#6-module-2--thermal-analysis)
7. [Module 3 — Reversibility Analysis](#7-module-3--reversibility-analysis)
8. [Input File Format](#8-input-file-format)
9. [Unit Conversion Reference](#9-unit-conversion-reference)
10. [Thermodynamics — Scientific Notes](#10-thermodynamics--scientific-notes)
11. [Export \& Download Guide](#11-export--download-guide)
12. [Troubleshooting](#12-troubleshooting)
13. [Citation Guide](#13-citation-guide)

\---

## 1\. Overview \& Scientific Philosophy

RAPID-CD is a privacy-centric, locally-executed web application for the processing, visualisation, and preliminary analysis of circular dichroism (CD) spectroscopy data from peptides and small proteins.

**Key principle:** All computation is performed locally on the user's machine. No experimental data is transmitted to any external server.

RAPID-CD is designed as an end-to-end preprocessing and visualisation pipeline that bridges raw JASCO instrument output and submission-ready figures, while maintaining scientific rigour through explicit model assumptions and uncertainty labelling.

### What RAPID-CD is for

* Processing and comparing CD spectra from multiple samples
* Thermal melt analysis including overlay, peak tracking, and preliminary thermodynamics
* Reversibility assessment after thermal denaturation
* Generating publication-quality figures ready for journals
* Formatting data for external deconvolution servers (BeStSel, DichroWeb)

### What RAPID-CD is NOT for

* Replacing established deconvolution algorithms (BeStSel, CDSSTR) for quantitative secondary structure determination
* Absolute thermodynamic measurements without a sigmoidal melting transition
* Analysing data that has not been properly baseline-corrected for buffer absorption

\---

## 2\. Installation

### System Requirements

* Operating system: Windows 10/11, macOS 10.14+, or Linux (Ubuntu 18.04+)
* Python: 3.9 or higher
* RAM: 4 GB minimum, 8 GB recommended for large datasets
* Browser: Chrome, Firefox, Edge, or Safari (modern versions)

### Step-by-Step Installation

**Step 1: Install Python**

Download Python from https://www.python.org/downloads/ if not already installed. During installation on Windows, tick "Add Python to PATH".

**Step 2: Download RAPID-CD**

```bash
git clone https://github.com/YOUR-USERNAME/RAPID-CD.git
cd RAPID-CD
```

Or download the ZIP from GitHub and extract it.

**Step 3: Create a virtual environment (recommended)**

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\\Scripts\\activate
```

**Step 4: Install dependencies**

```bash
pip install -r requirements.txt
```

If any package fails, install it individually:

```bash
pip install streamlit pandas numpy plotly scipy statsmodels kaleido
```

**Step 5: Verify installation**

```bash
python -c "import streamlit, pandas, numpy, plotly, scipy, statsmodels, kaleido; print('All OK')"
```

\---

## 3\. Launching the Application

```bash
streamlit run rapid\_cd.py
```

The application will open automatically in your browser at `http://localhost:8501`.

To stop the application, press `Ctrl+C` in the terminal.

**Note:** Keep the terminal window open while using the app. Closing the terminal will stop the server.

### Troubleshooting launch issues

* If the browser does not open automatically, manually navigate to `http://localhost:8501`
* If port 8501 is in use, Streamlit will automatically use 8502 or 8503
* On macOS, you may need to use `python3` instead of `python`

\---

## 4\. Home Screen

The home screen presents three module cards:

|Card|Module|Purpose|
|-|-|-|
|🧪 Green|General Analysis|Single or multi-spectrum analysis at a fixed temperature|
|🔥 Orange|Thermal Analysis|Multi-temperature melt experiments|
|🔄 Blue|Reversibility|Pre-melt vs post-refolding comparison|

Click a card button to enter the corresponding module. Use the **🏠 Return to Home** button in the sidebar to go back at any time.

The sidebar also provides a **mode switcher** (radio buttons) to move between modules without returning home.

\---

## 5\. Module 1 — General Analysis

### 5.1 Sidebar Setup

**Experimental Setup panel:**

|Control|Description|
|-|-|
|Num Samples|Number of spectra to load (1–10)|
|Blanking|Individual (separate blank per sample) or Common (single blank for all)|

**Per-sample settings (expand each sample):**

|Field|Description|
|-|-|
|Name|Label shown in plots and exports|
|Conc (µM)|Peptide concentration in micromolar|
|Residues|Number of amino acid residues (for MRE conversion)|
|D-amino acid checkbox|Tick if the peptide uses D-amino acids — inverts signal for secondary structure estimation only|
|Input Data Format|Auto-detected from file header; override if needed|
|File uploader|`.txt` file (JASCO format)|
|Blank file|Optional buffer spectrum for subtraction|
|Colour|Line colour in overlay plots|

**Processing panel:**

|Control|Description|
|-|-|
|Output Metric|MRE (recommended), Raw (mdeg), Δε, HT, Abs|
|Pathlength (cm)|Cuvette pathlength (e.g. 0.1 cm for a 1 mm cell)|
|Apply Smoothing|Enable Savitzky-Golay or LOWESS smoothing|
|Method|Savitzky-Golay (default, fast) or LOWESS (robust, slower)|
|Frac/Window|LOWESS fraction (0.01–0.30) or SG window size (5–51, odd numbers)|

### 5.2 Plot Customization Expander

|Control|Description|
|-|-|
|Show Grid Lines|Toggle grid lines|
|Line Width|Trace thickness (0.5–5.0)|
|Title Font Size|Font size for plot titles|
|Axis Label Size|Font size for axis labels and tick numbers|
|Legend Size|Font size for legend entries|
|Min / Max WL|X-axis wavelength range|
|Y Min / Y Max|Manual Y-axis limits|

### 5.3 Peak Search Range Expander

Use this if noise near 195–200 nm is being incorrectly identified as the primary minimum or maximum.

|Control|Description|
|-|-|
|Restrict minimum search|Enable a custom wavelength window for finding λ\_min|
|Min search start/end (nm)|Default 205–230 nm — captures 208 \& 222 nm helical bands|
|Restrict maximum search|Enable a custom window for λ\_max|
|Max search start/end (nm)|Default 185–200 nm — captures the 192 nm positive band|

**Tip for α-helix:** Set minimum search to 205–230 nm and maximum search to 185–200 nm.  
**Tip for β-sheet:** Set minimum search to 210–225 nm.

### 5.4 Analysis Tabs

#### Tab 1 — Overlay

Shows all uploaded spectra on a single plot.

* Select/deselect individual curves using the multiselect widget
* Zero reference dashed line is always shown
* D-amino acid samples are flagged with an annotation

#### Tab 2 — Separate Panels

Displays each sample in its own subplot panel.

* Optional: show a common base curve in all panels
* X-axis label positioning is automatic

#### Tab 3 — Statistics

Comparative statistics across all samples:

* λ Min 1, λ Min 2 (if dual minima detected), λ Max per sample
* Bar charts for lambda minima and maxima comparison
* Intensity bar charts
* Downloadable 2×2 batch statistics figure

#### Tab 4 — Secondary Structure

Internal NNLS deconvolution using basis spectra:

* **NNLS estimation**: non-negative least squares fit to α-helix, β-sheet, random coil, PPII basis spectra
* **Empirical estimates**: single-point estimation at 222 nm (helix) and 217 nm (sheet)
* **Chen chain-length correction**: appropriate for short peptides (< 30 residues)
* **310-helix check**: if the 222/208 nm ratio < 0.6, a 310-helix warning is shown
* Results are qualitative — use BeStSel or DichroWeb for publication-quality deconvolution

#### Tab 5 — Spectral Indicators (diagnostic)

* 222/208 nm ratio plot — values near 1.0 suggest coiled-coil; < 0.6 suggests 310-helix
* Signal at 260 nm — values > ±1.0 indicate possible contamination or baseline issues

#### Tab 6 — BeStSel / DichroWeb Export

Formats data for direct upload to external servers:

* Select a sample and download a two-column tab-delimited `.txt` file
* Data is sorted in descending wavelength order as required by both servers
* Unit-specific warning for MRE data (must change data units on BeStSel website)

\---

## 6\. Module 2 — Thermal Analysis

### 6.1 Sidebar Setup

**Experimental Setup:**

|Control|Description|
|-|-|
|Num Samples|Number of samples (1–10)|
|Blanking|Individual or Common blank mode|
|Data Format|Multi-Column File (JASCO) or Discrete Files (one file per temperature)|

**Per-sample settings:**

|Field|Description|
|-|-|
|Name|Sample label|
|Conc (µM)|Concentration|
|Residues|Number of residues|
|JASCO file|Multi-column thermal `.txt` file (contains all temperatures)|
|OR: Discrete files|Upload one `.txt` per temperature point, specifying each temperature|
|Blank|Individual blank `.txt`|
|Input format|Auto-detected; override if needed|
|D-amino acid|Inverts signal for secondary structure tabs only|

**Processing panel:**

|Control|Description|
|-|-|
|Output Metric|MRE, mdeg, Δε, HT, Abs|
|Pathlength (cm)|Cuvette pathlength|
|Apply Smoothing|Savitzky-Golay or LOWESS|

### 6.2 Plot Customization Expander

Same controls as General Analysis plus:

|Control|Description|
|-|-|
|Melting WL (nm)|Wavelength for melting curve extraction; accepts decimals (e.g. 222.5 nm)|

### 6.3 Peak Search Range Expander

Identical to General Analysis peak search range (see Section 5.3). Applied across all thermal tabs that compute λ min/max.

### 6.4 Overlay Tab (Tab 1 — 🌈)

**Temperature Selection panel:**

* For a single sample: one multiselect showing all available temperatures
* For multiple samples: one multiselect per sample — independently filter which temperatures are plotted
* Colour mode: Auto gradient (blue = cold → red = hot) or Manual per-temperature colour pickers

**Smoothing Diagnostics:**
When smoothing is active, a second figure shows raw (dotted) vs smoothed (solid) traces for quality checking.

**Export:**
Downloads only the selected (filtered) temperatures, not the full dataset.

### 6.5 Multi-Panel Tab (Tab 2 — 🔲)

Each temperature gets its own subplot panel.

|Control|Description|
|-|-|
|Select Samples|Compare multiple samples side by side|
|Temp. match tolerance (°C)|Merge temperatures from different samples within this window|
|Axis font size|Font size for all tick labels and axis titles|
|Customise panel titles|Override the default "15.0 °C" label with custom text|

A zero reference line (y = 0) is always shown regardless of grid setting.

### 6.6 λ–T Spectromap (Tab 3 — 🗺️)

A 2D heatmap with wavelength on x-axis, temperature on y-axis, and CD signal as colour.

|Control|Description|
|-|-|
|Colour scale|RdBu\_r (recommended), Spectral\_r, RdYlBu\_r|
|Symmetric colour scale|Forces ±Z\_max so zero is always the midpoint colour|
|Figure height (px)|Resize the heatmap|

### 6.7 λ Peak Tracking (Tab 4 — 📊)

Tracks how the wavelength position and CD intensity of spectral peaks change with temperature.

**Sample selector:** Choose one or multiple samples to overlay.

**Section 1 — Dual-Axis Plots:**
Each plot shows λ min (or λ max) position AND signal intensity on the same figure:

* Left Y-axis (solid circles) = wavelength position (nm)
* Right Y-axis (dashed diamonds) = CD signal intensity

**Axis Scale expanders** (one for λ Min, one for λ Max):

* X Min/Max — Temperature axis range
* Left Y Min/Max — Wavelength axis range
* Right Y Min/Max — Intensity axis range

**Section 2 — Multi-Sample Comparison (Separate Axes):**
Four individual panels: λ Min position, λ Max position, Intensity at λ Min, Intensity at λ Max.

**Section 3 — Combined 2×2 Overview:**
All four metrics in a single exportable publication figure with sample names as subtitle.

### 6.8 Secondary Structure vs Temperature (Tab 5 — 🧩)

Tracks NNLS-estimated secondary structure content across the temperature series.

|Control|Description|
|-|-|
|Select samples|One or more samples|
|Chen correction|Apply chain-length correction (< 30 residues)|
|View|All components per sample OR compare one component across samples|

Results are qualitative indicators — see Section 10 for interpretation guidance.

### 6.9 Thermodynamics (Tab 6 — ⚗️)

#### Experiment Type \& Model Assumptions (mandatory — expand first)

|Experiment type|When to select|Key disclaimer|
|-|-|-|
|Thermal unfolding|Standard heat-induced melt|Valid ΔG only if sigmoidal transition present|
|Solvent/membrane mimic|SDS, TFE, lipid vesicles|ΔG reflects apparent micellar stabilisation, not intrinsic folding|
|Chemical denaturation|Urea, GdnHCl titrations|Temperature is not the correct x-axis; values are qualitative|
|Ligand binding|Ligand-induced conformational change|Not a true folding ΔG|

|Calculation method|When to use|
|-|-|
|**Two-state (van't Hoff)**|Sigmoidal melting curve present; data spans at least the beginning and end of the transition; R² ≥ 0.90|
|**Qualitative / no model**|Monotonic (non-sigmoidal) signal change; transition is incomplete; SDS or chemical denaturation experiment|

**How to know which to use:**
Ask: *"Does my melting curve have a flat plateau at low temperature AND a flat plateau at high temperature?"*

* Yes → Two-state is appropriate
* No (just drifts) → Use Qualitative mode

**Warning:** If the fitted Tm falls outside your measurement range by more than 20 °C, the fit is extrapolating and ΔG values should be treated as approximate.

#### Fit Quality Table

Appears only for Two-state mode. Shows per sample:

* **Tm (°C)** — midpoint of the fitted transition
* **ΔH\_vH (kcal/mol)** — van't Hoff enthalpy (cooperativity measure)
* **R²** — fit quality: ≥ 0.98 = ✅ Valid; 0.90–0.98 = 🟡 Caution; < 0.90 = 🔴 Invalid
* **RMSE** — root mean square residual
* **Confidence** — automatic flag

#### Plots

1. **Melting curve** — raw data points + fitted curve (dashed) for each sample
2. **Apparent ΔG vs Temperature** — ΔG profiles for all selected samples
3. **Fit Residuals** — appears when two-state fit is used; residuals should be randomly scattered around zero

#### ΔΔG Builder

Compare ΔG between two conditions (e.g., buffer vs SDS) at each temperature point:

* Select Reference and Target samples
* Set temperature match tolerance (adjust if samples were measured at slightly different temperatures)
* ΔΔG = ΔG\_target − ΔG\_reference

### 6.10 Spectral Simulation (Tab 7 — 🔮)

Mathematical interpolation to predict the CD spectrum at any temperature:

* **Linear**: safe for interpolation and extrapolation
* **Cubic Spline**: smoother curves for interpolation (avoid for extrapolation)
* Warning shown automatically if extrapolating outside measured range

\---

## 7\. Module 3 — Reversibility Analysis

Compares a spectrum extracted from the thermal melt file with a separately measured refolded spectrum.

### 7.1 Sidebar Setup

|Upload|Description|
|-|-|
|Thermal start / melt file|Multi-column JASCO thermal file|
|Refolded spectrum|Single `.txt` spectrum recorded after cooling|
|Buffer / blank|Optional blank for blank subtraction of refolded spectrum|

Then select:

* Output metric, pathlength, concentration, residues
* Input data format for melt and refolded files (auto-detected, override if needed)
* Smoothing options

### 7.2 Main Panel Controls

1. **Select Temperature from Melt** — choose which temperature from the melt to compare against the refolded spectrum
2. **Plot Customisation expander:**

   * X and Y axis limits
   * Line colours for each spectrum
   * Line width
   * Grid toggle
3. **Peak Search Range expander** — restrict wavelength window for min/max detection

### 7.3 Peak Statistics Table

Appears below the export buttons. Shows λ\_min, min intensity, λ\_max, max intensity for both spectra, computed within the user-specified search windows.

### 7.4 Statistical Analysis

|Metric|Description|
|-|-|
|Correlation|Pearson r between the two spectra across all wavelengths|
|Abs. RMSD|Average absolute deviation between spectra|
|Normalized RMSD (%)|RMSD expressed as a percentage of the signal amplitude range|

**Interpretation thresholds:**

|Result|Criterion|
|-|-|
|🟢 Excellent Reversibility|r ≥ 0.98 AND NRMSD < 5%|
|🔵 Good Reversibility|r ≥ 0.90|
|🟡 Partial Reversibility|r ≥ 0.70|
|🔴 Irreversible|r < 0.70|

\---

## 8\. Input File Format

### Standard JASCO Single-Spectrum File (.txt)

```
TITLE         MyPeptide\_25C
DATE          2025-01-15
YUNITS        CD\[mdeg]
XUNITS        Wavelength\[nm]
...
XYDATA
260.0    -0.123    456.7    0.001
259.0    -0.145    457.1    0.002
...
190.0    -5.234    512.3    0.021
```

The `YUNITS` line is used for automatic format detection. Supported values:

* `CD\[mdeg]` or `MDEG` → mdeg
* `MOL. CD`, `MOLAR CD`, `DELTA EPSILON`, `DELTAE` → Molar Δε
* `MEAN RESIDUE`, `MRE`, `\[THETA]` → Mean Residue Δε

### JASCO Multi-Temperature Thermal File

Contains multiple signal columns, one per temperature point. RAPID-CD reads the `Channel 1` header line to extract temperature values automatically.

### Acceptable Column Structure

The parser expects columns in the order: `Wavelength, CD, HT, Abs`. Extra columns beyond these four are ignored. Files with fewer columns are handled automatically.

\---

## 9\. Unit Conversion Reference

|From|To MRE|To Δε|
|-|-|-|
|mdeg|sig / (10 × path × conc\_M × nres) / 1000|(sig/fac/1000) / 3298.2 × 1000|
|Molar Δε|(sig × 3298.2 / nres) / 1000|sig / nres|
|Mean Residue Δε|(sig × 3298.2) / 1000|not applicable|

Where:

* `fac = 10 × path\_cm × conc\_mol/L × nres`
* `3298.2` is the conversion factor between Δε and MRE (deg·cm²·dmol⁻¹)

**Correct inputs are essential.** An error in concentration or pathlength of 10% propagates directly to a 10% error in all calculated MRE values.

\---

## 10\. Thermodynamics — Scientific Notes

### When ΔG is valid

The apparent ΔG is calculated using a two-state van't Hoff model:

```
y(T) = \[(aF + bF·T) + (aU + bU·T)·exp(-ΔG/RT)] / \[1 + exp(-ΔG/RT)]
ΔG(T) = ΔH\_vH · (1 - T/Tm)
```

This model is valid when:

1. The melting curve shows a sigmoidal (S-shaped) profile
2. R² ≥ 0.90 for the fitted curve
3. The Tm falls within or close to your measurement range
4. The system approximates a two-state (folded ↔ unfolded) equilibrium

### When ΔG is NOT valid

* Monotonically decreasing curves without a transition (common for short peptides in buffer)
* Experiments in SDS, TFE, or other membrane-mimetics (ΔG reflects micellar stabilisation, not intrinsic folding)
* Chemical denaturation experiments (use denaturant concentration, not temperature)
* Data with R² < 0.90 for the two-state fit

### Interpreting ΔG in SDS vs buffer

A negative ΔG means the unfolded state is thermodynamically favoured. A positive ΔG means the folded/helical state is favoured. The ΔΔG between two conditions quantifies the stabilisation energy provided by one environment relative to the other.

For SDS experiments specifically: report as "apparent ΔG in the micellar environment" and note that direct comparison to aqueous ΔG values is not thermodynamically valid.

### Why the qualitative (min-max) approach gives identical curves for different conditions

Min-max normalisation forces every dataset to span exactly 0→1. If two samples unfold over the same temperature range, their normalised α(T) profiles are identical regardless of absolute signal amplitude, and therefore their ΔG profiles are identical. This is an artefact of normalisation, not a real biochemical result. The two-state model avoids this by fitting the absolute signal values.

\---

## 11\. Export \& Download Guide

Every plot and dataset in RAPID-CD can be exported. Downloads appear as buttons below each figure.

|Format|Button label|Best for|
|-|-|-|
|PNG|📸 PNG|Journal submission, presentations|
|PDF|📄 PDF|Vector graphics, further editing in Illustrator/Inkscape|
|CSV|💾 CSV|Further analysis in Excel, GraphPad, Origin|

**PNG scale:** All PNG exports use scale=3 (3× the screen resolution), producing approximately 300 dpi figures suitable for most journals.

**PDF:** Vector format; lines and text remain sharp at any zoom level.

\---

## 12\. Troubleshooting

### "Column 'CD' not found"

The parser could not find the CD signal column. Check that your file has a standard JASCO header and data columns in the expected order (Wavelength, CD, HT, Abs).

### "Error reading files. Please check the formats."

The melt file could not be parsed as a multi-column JASCO thermal file. Verify that the file contains `Channel 1` header and numeric temperature values.

### Export buttons produce no file / give an error

This is usually a kaleido issue. Run `pip install kaleido --upgrade`. On some systems you may also need `pip install kaleido==0.2.1` (the older version is more stable).

### Smoothing produces a flat line or artefacts

For Savitzky-Golay: the window size must be odd and less than the number of data points. Reduce the window slider. For LOWESS: reduce the fraction slider.

### Two-state fit fails

The optimiser could not converge. This usually means no sigmoidal transition is present. Switch to Qualitative mode or reduce the temperature range to focus on the transition region.

### Tm falls far outside the measurement range

The curve does not contain a detectable transition in the measured window. The fitted Tm is an extrapolation and should not be reported as a measured value. Use Qualitative mode.

### "Could not save session" warning

This is a non-fatal warning. The session download only saves numeric and text widget states, not uploaded files. This feature is provided as a convenience reference.

\---

## 13\. Citation Guide

If you use RAPID-CD in published research, cite the software:

> Roy, P. (2026). \*RAPID-CD: Rapid Analysis Pipeline for Interpreting Dichroism\*. Sorbonne University. https://github.com/YOUR-USERNAME/RAPID-CD

Additionally, cite underlying methods based on what you used:

**Secondary structure (internal NNLS):**

> Brahms, S., \& Brahms, J. G. (1980). Journal of Molecular Biology, 138(2), 149–178.
> Greenfield, N. J., \& Fasman, G. D. (1969). Biochemistry, 8(10), 4108–4116.

**Chain-length correction (Chen):**

> Chen, Y. H., Yang, J. T., \& Chau, K. H. (1974). Biochemistry, 13(16), 3350–3359.

**BeStSel server:**

> Micsonai, A., et al. (2025). Nucleic Acids Research, 53, W73–83.
> Micsonai, A., et al. (2015). PNAS, 112(24), E3095–103.

**DichroWeb server:**

> Whitmore, L., \& Wallace, B. A. (2004). Nucleic Acids Research, 32, W668–673.

\---

*RAPID-CD User Manual v1.0 — Pritam Roy, Sorbonne University, 2026*

