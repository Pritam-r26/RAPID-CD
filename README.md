# RAPID-CD 🧬

### Rapid Analysis Pipeline for Interpreting Dichroism

[!\[Python 3.9+](https://img.shields.io/badge/python-3.9%25252B-blue.svg)](https://www.python.org/)
[!\[Streamlit](https://img.shields.io/badge/built%252520with-Streamlit-red)](https://streamlit.io)
[!\[License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[!\[Version](https://img.shields.io/badge/version-1.0-green)](CHANGELOG.md)

**RAPID-CD** is a locally-executed, privacy-centric web application for the processing, visualisation, and preliminary analysis of circular dichroism (CD) spectroscopy data from peptides and small proteins.

> All computation is performed locally on your machine. No experimental data is transmitted to any external server.

\---

## Key Features

|Module|Description|
|-|-|
|**General Analysis**|Single and multi-spectrum processing, unit conversion, smoothing, secondary structure screening, and statistical comparison|
|**Thermal Analysis**|Multi-temperature melt processing, thermal overlay, multi-panel comparison, spectromap, λ peak tracking, secondary structure vs temperature, thermodynamics (apparent ΔG, ΔΔG), and spectral simulation|
|**Reversibility**|Pre-melt vs post-refolding spectral comparison with RMSD and Pearson correlation|

### Capabilities at a glance

* **Automatic format detection** from JASCO file headers (mdeg / Molar Δε / Mean Residue Δε) with manual override
* **Unit conversion pipeline**: mdeg ↔ MRE ↔ Δε across all three modules
* **Smoothing**: Savitzky-Golay and LOWESS
* **Secondary structure screening** using internal NNLS basis spectra (α-helix, β-sheet, random coil, PPII)
* **Two-state van't Hoff thermodynamics** with sloping baselines, fit quality metrics (R², RMSE), and confidence indicators
* **Peak search windows** — restrict λ min/max search to avoid noise at 195–200 nm
* **Publication-grade export**: PNG, PDF, CSV for every plot and dataset
* **BeStSel and DichroWeb formatter** — direct export for external deconvolution servers

\---

## Installation

### Requirements

* Python 3.9 or higher
* pip

### Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/YOUR-USERNAME/RAPID-CD.git
cd RAPID-CD

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\\\\\\\\Scripts\\\\\\\\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the application
streamlit run rapid\\\\\\\_cd.py
```

The app will open automatically in your default web browser at `http://localhost:8501`.

### Dependencies

```
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.23.0
plotly>=5.15.0
scipy>=1.10.0
statsmodels>=0.14.0
kaleido>=0.2.1
```

> \\\\\\\*\\\\\\\*Note on kaleido:\\\\\\\*\\\\\\\* Required for PNG/PDF export of plots. If export buttons fail, run `pip install kaleido` separately.

\---

## Usage

1. Launch the app with `streamlit run rapid\\\\\\\_cd.py`
2. Select a module from the home screen (General Analysis / Thermal Analysis / Reversibility)
3. Upload your `.txt` data files (JASCO format) via the sidebar
4. Set concentration, pathlength, and number of residues
5. Select your output metric (MRE recommended for publication)
6. Explore the analysis tabs and download results

See the full **User Manual** (`MANUAL.md`) for detailed instructions on every feature.

\---

## Input File Format

RAPID-CD accepts standard JASCO `.txt` exports. The software auto-detects the data unit from the `YUNITS` line in the file header. Files must contain at minimum a `Wavelength` and `CD` column. Multi-temperature thermal files (JASCO multi-column format) are supported natively.

\---

## Citation

If you use RAPID-CD in published research, please cite:

> Roy, P. (2026). \\\\\\\*RAPID-CD: Rapid Analysis Pipeline for Interpreting Dichroism\\\\\\\*. Sorbonne University, Paris, France. https://github.com/YOUR-USERNAME/RAPID-CD

Additionally cite the underlying methods as appropriate:

* **NNLS deconvolution basis spectra:** Brahms \& Brahms (1980); Greenfield \& Fasman (1969)
* **BeStSel:** Micsonai et al. (2022), *Nucleic Acids Research*
* **DichroWeb:** Whitmore \& Wallace (2004), *Nucleic Acids Research*
* **Chain-length correction:** Chen et al. (1974), *Biochemistry*

Full citation list is available in-app under **General Analysis → Secondary Structure → Citations**.

\---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

\---

## Contributing \& Updates

Contributions, bug reports, and feature requests are welcome via [GitHub Issues](https://github.com/YOUR-USERNAME/RAPID-CD/issues).

To update your local installation after new releases:

```bash
git pull origin main
pip install -r requirements.txt
```

\---

## Author

**Pritam Roy**  
Sorbonne University, Paris, France  
Contact: pritsam56@gmail.com or pritam.roy@sorbonne-universite.fr

\---

## Acknowledgements

RAPID-CD was developed as part of research at Sorbonne University. The secondary structure basis spectra are derived from the foundational empirical datasets of Brahms \& Brahms (1980) and Greenfield \& Fasman (1969).

