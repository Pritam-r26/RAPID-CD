"""
RAPID-CD: Rapid Analysis Pipeline for Interpreting Dichroism
=============================================================
Version  : 1.0
Author   : Pritam Roy
Institute: Sorbonne University, Paris, France
Repository: https://github.com/pritam-r26/RAPID-CD

Description
-----------
RAPID-CD is a locally-executed, privacy-centric web application for the
processing, visualisation and preliminary analysis of circular dichroism (CD)
spectroscopy data from peptides and small proteins.

Key capabilities:
  - Automatic input-format detection from JASCO file headers (mdeg / Molar Δε /
    Mean Residue Δε) with manual override
  - Full unit conversion pipeline: mdeg ↔ MRE ↔ Δε across all three modules
  - Blank subtraction, Savitzky-Golay and LOWESS smoothing
  - Multi-sample overlay, separate-panel, and statistical comparison views
  - Internal NNLS secondary structure screening (α-helix, β-sheet, random coil)
  - Thermal melt processing: Tm, ΔG, ΔΔG, 3D landscape, spectral simulation
  - Reversibility quantification via RMSD and Pearson correlation
  - Export formatter for BeStSel (Micsonai et al., 2022) and DichroWeb
    (Whitmore & Wallace, 2004)

All computation is performed locally. No data is transmitted externally.

Modules
-------
  Module 1 — General Analysis  : single-spectrum processing and comparison
  Module 2 — Thermal Analysis  : multi-temperature melt analysis
  Module 3 — Reversibility     : pre-melt vs post-refolding comparison

Dependencies
------------
  streamlit, pandas, numpy, plotly, scipy, statsmodels, kaleido

  Install via:
      pip install streamlit pandas numpy plotly scipy statsmodels kaleido

Usage
-----
  streamlit run rapid_cd.py

Citation
--------
  If you use RAPID-CD in published research, please cite:
  Roy, P. (2026). RAPID-CD: Rapid Analysis Pipeline for Interpreting Dichroism.
  Sorbonne University. https://github.com/pritam-r26/RAPID-CD

  Additionally cite the underlying methods as appropriate (see the in-app
  References section under General Analysis → Secondary Structure).

License
-------
  MIT License. See LICENSE file for details.

Changelog
---------
  v1.0  (2025) — Initial public release.
                 Automatic Δε / mdeg header detection.
                 Three-way unit conversion across all modules.
                 Optional base-curve in Separate panel.
                 Auto-positioned multi-panel axis labels.
                 2×2 batch statistics panel export.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Standard library and third-party imports
# ─────────────────────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import statsmodels.api as sm
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter, find_peaks
from scipy.optimize import nnls, curve_fit
import io
import datetime

# ── SECTION 1: CONFIGURATION, PAGE SETUP & REFERENCE DATA ───────────────────
st.set_page_config(page_title="RAPID-CD v1.0", layout="wide", page_icon="🧬")

st.markdown("""
<style>
    .block-container {padding-top: 1rem;}
    div[data-testid="stExpander"] div[role="button"] p {
        font-weight: bold;
        color: #2e7bcf;
        font-size: 16px;
    }
    div[data-testid="stNumberInput"] input { padding: 0px 5px; }

    /* ── Home Screen Styling ── */
    .home-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 10px;
        min-height: 240px;
    }
    .home-icon  { font-size: 50px; margin-bottom: 10px; }
    .rapid-title {
        text-align: center;
        color: #1a1a2e;
    }
    @media (prefers-color-scheme: dark) {
        .rapid-title { color: #fafafa; }
    }
    .home-title { font-size: 20px; font-weight: bold; color: #333; }
    .home-desc  { font-size: 14px; color: #666; margin-bottom: 20px; }

    /* Dark-mode overrides for home cards */
    @media (prefers-color-scheme: dark) {
        .home-card  { background-color: #1e2530; }
        .home-title { color: #f0f0f0; }
        .home-desc  { color: #a0a8b8; }
    }

    /* ── Analysis-view tab selector bar ── */
    /* Header banner (the blue strip rendered via st.markdown) */
    .rapid-tab-header {
        background: linear-gradient(90deg, #1565c0, #1976d2);
        color: #ffffff;
        padding: 9px 18px;
        border-radius: 10px 10px 0 0;
        font-weight: 700;
        font-size: 13px;
        letter-spacing: 0.4px;
        margin-bottom: 0px;
        border: 1.5px solid #1565c0;
        border-bottom: none;
        line-height: 1.4;
    }

    /* Radio-group container — only targets horizontal radio groups
       (our two tab selectors) by detecting the flex-row layout.
       Vertical radios in sidebars are unaffected. */
    div[data-testid="stRadio"]:has([role="radiogroup"][style*="row"]) {
        background-color: #e8f0fe;
        border: 1.5px solid #1565c0;
        border-top: none;
        border-radius: 0 0 10px 10px;
        padding: 10px 16px 12px 16px;
        margin-top: 0px;
        margin-bottom: 12px;
    }
    /* Radio option label text */
    div[data-testid="stRadio"]:has([role="radiogroup"][style*="row"]) label {
        font-size: 13px;
        font-weight: 500;
    }

    @media (prefers-color-scheme: dark) {
        .rapid-tab-header {
            background: linear-gradient(90deg, #0d47a1, #1565c0);
            border-color: #0d47a1;
        }
        div[data-testid="stRadio"]:has([role="radiogroup"][style*="row"]) {
            background-color: #0a1929;
            border-color: #0d47a1;
        }
        div[data-testid="stRadio"]:has([role="radiogroup"][style*="row"]) label {
            color: #c8d8f0;
        }
    }
</style>
""", unsafe_allow_html=True)

COLORS = ["black", "red", "blue", "green", "purple", "orange", "cyan", "magenta", "brown", "gray"]

# Reference basis spectra calibrated to 190-260 nm (1 nm step, 71 points)
# Values derived from foundational empirical sets:
# 1. Greenfield, N. J., & Fasman, G. D. (1969). Biochemistry, 8(10), 4108-4116.
# 2. Brahms, S., & Brahms, J. G. (1980). J. Mol. Biol., 138(2), 149-178.
REF_WL = np.arange(190, 261, 1)

# 1. Alpha Helix (Max @ 192nm, Minima @ 208nm and 222nm)
REF_HELIX = np.array([
    45.0, 60.0, 70.0, 75.0, 65.0, 48.0, 32.0, 18.0, 8.0, 0.0, -7.0, -14.0, -21.0, -26.0, 
    -30.0, -32.5, -34.0, -34.5, -34.5, -33.5, -32.0, -29.0, -26.0, -24.0, -23.5, -24.0, 
    -25.5, -28.0, -31.0, -33.5, -35.0, -36.0, -36.0, -35.0, -33.0, -30.0, -26.0, -22.0, 
    -18.0, -14.0, -10.0, -7.0, -5.0, -3.0, -2.0, -1.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
])

# 2. Beta Sheet (Max @ 195nm, Min @ 216nm)
REF_SHEET = np.array([
    -10.0, -2.0, 7.0, 15.0, 21.0, 25.0, 25.0, 23.0, 19.0, 14.0, 9.0, 5.0, 1.0, -2.0, 
    -5.0, -8.0, -11.0, -13.5, -15.0, -16.5, -17.5, -18.0, -18.5, -18.5, -18.0, -17.5, 
    -16.5, -15.0, -13.5, -12.0, -10.0, -8.5, -7.0, -5.5, -4.0, -3.0, -2.0, -1.0, -0.5, 
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
])

# 3. Random Coil (Deep Min @ 198nm)
REF_COIL = np.array([
    -20.0, -25.0, -30.0, -35.0, -38.0, -40.0, -40.0, -38.0, -34.0, -28.0, -20.0, -12.0, 
    -5.0, -1.0, 1.0, 2.0, 3.0, 4.0, 4.5, 4.5, 4.0, 3.0, 2.0, 1.0, 0.5, 0.0, 0.0, 0.0, 
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
])

# 4. Polyproline II (PPII) (Min @ 198nm, Max @ 218nm)
REF_PPII = np.array([
    -15.0, -20.0, -25.0, -30.0, -34.0, -36.0, -38.0, -38.0, -36.0, -32.0, -26.0, -20.0, 
    -14.0, -8.0, -3.0, 1.0, 4.0, 6.0, 7.5, 8.5, 9.0, 9.5, 9.8, 10.0, 10.0, 9.5, 8.5, 
    7.5, 6.0, 4.5, 3.0, 2.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
])

def pad_to_71(arr):
    target_len = 71
    if len(arr) < target_len:
        return np.concatenate((arr, np.zeros(target_len - len(arr))))
    return arr[:target_len]

# Apply padding
REF_HELIX = pad_to_71(REF_HELIX)
REF_SHEET = pad_to_71(REF_SHEET)
REF_COIL  = pad_to_71(REF_COIL)
REF_PPII  = pad_to_71(REF_PPII)

# Create the Basis Matrix
REF_MATRIX = np.vstack([REF_HELIX, REF_SHEET, REF_COIL, REF_PPII]).T

# ── SECTION 2: CORE HELPER FUNCTIONS ─────────────────────────────────────────
def get_theme_colors():
    """
    Detects whether Streamlit is in dark or light mode and returns
    appropriate colours for all plot elements.
    Returns a dict with keys: bg, paper, text, grid, line, axis
    """
    try:
        theme = st.get_option("theme.base")
        is_dark = (theme == "dark")
    except Exception:
        is_dark = False

    if is_dark:
        return {
            "bg":        "#0e1117",   # Streamlit dark background
            "paper":     "#0e1117",
            "text":      "#fafafa",   # near-white text
            "grid":      "#2d2d2d",   # subtle dark grid
            "line":      "#aaaaaa",   # light grey axes
            "axis":      "#fafafa",
            "template":  "plotly_dark",
            "legend_border": "#555555",
        }
    else:
        return {
            "bg":        "white",
            "paper":     "white",
            "text":      "black",
            "grid":      "lightgray",
            "line":      "black",
            "axis":      "black",
            "template":  "simple_white",
            "legend_border": "black",
        }
# ── Module-level theme colours (re-evaluated on every Streamlit rerun) ──
_c = get_theme_colors()

def apply_publication_style(fig, title, x_label, y_label, show_grid=True, width=None, height=500, plot_mode='lines'):
    """
    Applies publication style — automatically adapts to dark or light mode.
    """
    c = get_theme_colors()

    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", font=dict(family="Arial", size=22, color=c["text"]), x=0.5, xanchor='center'),
        template=c["template"],
        paper_bgcolor=c["paper"],
        plot_bgcolor=c["bg"],
        width=width, height=height,
        font=dict(family="Arial", size=16, color=c["text"]),
        xaxis=dict(
            title=dict(text=f"<b>{x_label}</b>", font=dict(size=18, family="Arial", color=c["axis"])),
            tickfont=dict(size=14, family="Arial", color=c["axis"]),
            showgrid=show_grid,
            gridwidth=1,
            gridcolor=c["grid"],
            showline=True,
            linewidth=2,
            linecolor=c["line"],
            mirror=True
        ),
        yaxis=dict(
            title=dict(text=f"<b>{y_label}</b>", font=dict(size=18, family="Arial", color=c["axis"])),
            tickfont=dict(size=14, family="Arial", color=c["axis"]),
            showgrid=show_grid,
            gridwidth=1,
            gridcolor=c["grid"],
            showline=True,
            linewidth=2,
            linecolor=c["line"],
            mirror=True
        ),
        legend=dict(
            font=dict(size=14, family="Arial", color=c["text"]),
            bordercolor=c["legend_border"],
            borderwidth=1
        ),
        margin=dict(r=50, l=60, t=60, b=60)
    )

    is_bar = False
    if fig.data:
        if isinstance(fig.data[0], go.Bar):
            is_bar = True

    if not is_bar:
        if plot_mode == 'lines+markers':
            fig.update_traces(
                mode='lines+markers',
                marker=dict(size=9, line=dict(width=1, color=c["line"]), symbol='circle')
            )
        elif plot_mode == 'lines':
            fig.update_traces(mode='lines')
        else:
            fig.update_traces(mode=plot_mode)

    return fig

def render_plot_editor(key_prefix, default_title="Plot", default_color="#0000FF"):
    """Creates a standardized sidebar/expander for editing plot parameters."""
    with st.expander(f"🎨 Figure Editor: {default_title}", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            p_width = st.number_input("Width (px)", 300, 2000, 600, step=50, key=f"{key_prefix}_w")
            p_height = st.number_input("Height (px)", 200, 1500, 450, step=50, key=f"{key_prefix}_h")
            show_grid = st.checkbox("Show Grid", True, key=f"{key_prefix}_grid")
        with c2:
            font_size = st.number_input("Font Size", 8, 30, 14, key=f"{key_prefix}_fs")
            line_width = st.slider("Marker Size", 1.0, 15.0, 8.0, key=f"{key_prefix}_lw")
            p_color = st.color_picker("Main Color", default_color, key=f"{key_prefix}_col")
        with c3:
            st.markdown("**Axis Ranges (0=Auto)**")
            x_min = st.number_input("X Min", value=0.0, key=f"{key_prefix}_xmin")
            x_max = st.number_input("X Max", value=0.0, key=f"{key_prefix}_xmax")
            y_min = st.number_input("Y Min", value=0.0, key=f"{key_prefix}_ymin")
            y_max = st.number_input("Y Max", value=0.0, key=f"{key_prefix}_ymax")
            
    return {
        "width": p_width, "height": p_height, "grid": show_grid,
        "font_size": font_size, "marker_size": line_width, "color": p_color,
        "x_range": [x_min, x_max] if x_max > x_min else None,
        "y_range": [y_min, y_max] if y_max != y_min else None
    }

def apply_plot_style_custom(fig, style_dict, title, xtitle, ytitle, plot_mode='lines+markers'):
    """Applies CUSTOM style from the editor expander — dark mode aware."""
    c = get_theme_colors()

    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", font=dict(size=style_dict["font_size"]+4, color=c["text"]), x=0.5),
        width=style_dict["width"],
        height=style_dict["height"],
        template=c["template"],
        paper_bgcolor=c["paper"],
        plot_bgcolor=c["bg"],
        font=dict(size=style_dict["font_size"], color=c["text"], family="Arial"),
        xaxis=dict(
            title=f"<b>{xtitle}</b>",
            showgrid=style_dict["grid"],
            showline=True,
            linewidth=2,
            linecolor=c["line"],
            mirror=True,
            gridcolor=c["grid"],
            tickfont=dict(color=c["axis"])
        ),
        yaxis=dict(
            title=f"<b>{ytitle}</b>",
            showgrid=style_dict["grid"],
            showline=True,
            linewidth=2,
            linecolor=c["line"],
            mirror=True,
            gridcolor=c["grid"],
            tickfont=dict(color=c["axis"])
        ),
        legend=dict(
            font=dict(color=c["text"]),
            bordercolor=c["legend_border"],
            borderwidth=1
        ),
        margin=dict(r=50, l=60, t=60, b=60)
    )

    if style_dict["x_range"]: fig.update_xaxes(range=style_dict["x_range"])
    if style_dict["y_range"]: fig.update_yaxes(range=style_dict["y_range"])

    is_bar = False
    if fig.data and isinstance(fig.data[0], go.Bar):
        is_bar = True

    if not is_bar:
        if plot_mode == 'lines+markers':
            fig.update_traces(
                mode='lines+markers',
                line=dict(width=2.5),
                marker=dict(size=style_dict["marker_size"], line=dict(width=1, color=c["line"]), symbol='circle')
            )
        else:
            fig.update_traces(
                mode=plot_mode,
                line=dict(width=style_dict["marker_size"])
            )

    if len(fig.data) == 1:
        fig.update_traces(marker=dict(color=style_dict["color"]), marker_color=style_dict["color"])
    return fig

def generate_log_file(mode, samples, settings):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_txt = f"Peptide CD Analysis Log\n"
    log_txt += f"Date: {timestamp}\n"
    log_txt += f"Mode: {mode}\n"
    log_txt += "="*30 + "\n\n"
    log_txt += "Global Processing Settings:\n"
    for k, v in settings.items():
        log_txt += f"  {k}: {v}\n"
    log_txt += "\n"
    log_txt += "Sample Details:\n"
    for s in samples:
        log_txt += f"- Name: {s.get('name', 'N/A')}\n"
        if "conc" in s: log_txt += f"  Conc: {s['conc']} uM\n"
        if "nres" in s: log_txt += f"  Residues: {s['nres']}\n"
        if "format" in s: log_txt += f"  Format: {s['format']}\n"
        if "discrete_files" in s and s["discrete_files"]:
            log_txt += f"  Discrete Files: {len(s['discrete_files'])}\n"
        log_txt += "\n"
    return log_txt

@st.cache_data(show_spinner=False)
def _detect_yunits_cached(file_bytes: bytes) -> str:
    """Cached inner: receives raw bytes so Streamlit can hash them."""
    try:
        text = file_bytes.decode("latin-1")
    except Exception:
        return "unknown"
    for line in text.split("\n")[:60]:
        upper = line.upper().strip()
        if not upper.startswith("YUNITS"):
            continue
        if any(k in upper for k in ["CD [MDEG]", "CD[MDEG]", "MDEG", "[MDEG]"]):
            return "mdeg"
        if any(k in upper for k in ["MOL. CD", "MOL CD", "MOLAR CD",
                                     "DELTA EPSILON", "DELTA-EPSILON",
                                     "DELTAE", "Δε", "MOLAR ELLIPTICITY"]):
            return "delta_eps_molar"
        if any(k in upper for k in ["MEAN RESIDUE", "MRE", "DEG CM2 DMOL",
                                     "DEG.CM2/DMOL", "[THETA]"]):
            return "delta_eps_residue"
    return "unknown"


def detect_yunits(uploaded_file):
    """
    Public wrapper: extracts bytes from the UploadedFile object and
    delegates to the cached implementation. Returns one of:
    'mdeg', 'delta_eps_molar', 'delta_eps_residue', 'unknown'.
    Safe to call before read_cd_file — does not advance the file pointer.
    """
    if uploaded_file is None:
        return "unknown"
    try:
        return _detect_yunits_cached(uploaded_file.getvalue())
    except Exception:
        return "unknown"


@st.cache_data(show_spinner=False)
def _read_cd_file_cached(file_bytes: bytes):
    """Cached inner: parses a single-spectrum CD file from raw bytes."""
    try:
        text = file_bytes.decode("latin-1")
    except Exception:
        return None
    lines = text.split("\n")
    start_idx = 0
    for i, line in enumerate(lines):
        if "XYDATA" in line:
            start_idx = i + 1
            break
    if start_idx == 0:
        for i, line in enumerate(lines):
            if len(line.strip()) > 0 and line.strip()[0].isdigit():
                start_idx = i
                break
    parsed_data = []
    for line in lines[start_idx:]:
        if not line.strip():
            continue
        clean_line = line.replace(",", ".")
        parts = clean_line.split()
        try:
            nums = [float(p) for p in parts]
            if len(nums) >= 2:
                parsed_data.append(nums)
        except ValueError:
            continue
    if not parsed_data:
        return None
    df = pd.DataFrame(parsed_data)
    cols = ["Wavelength", "CD", "HT", "Abs"]
    if len(df.columns) > len(cols):
        df = df.iloc[:, :len(cols)]
        df.columns = cols
    else:
        df.columns = cols[: len(df.columns)]
    return df.sort_values("Wavelength")


def read_cd_file(uploaded_file):
    """Public wrapper: extracts bytes then delegates to the cached parser."""
    if uploaded_file is None:
        return None
    try:
        return _read_cd_file_cached(uploaded_file.getvalue())
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def _read_thermal_file_cached(file_bytes: bytes):
    """Cached inner: parses a JASCO multi-temperature thermal file from raw bytes."""
    try:
        text = file_bytes.decode("latin-1")
    except Exception:
        return None, None
    lines = text.split("\n")
    start_idx = -1
    temps = []
    for i, line in enumerate(lines):
        if "XYDATA" in line:
            for j in range(i, min(i + 10, len(lines))):
                if "Channel 1" in lines[j]:
                    temp_line = lines[j + 1].strip().replace(",", ".")
                    temps = [float(t) for t in temp_line.split()]
                    start_idx = j + 2
                    break
            break
    if start_idx == -1 or not temps:
        return None, None
    data_rows = []
    for line in lines[start_idx:]:
        if not line.strip() or "Channel" in line:
            break
        clean_line = line.replace(",", ".")
        parts = clean_line.split()
        try:
            nums = [float(p) for p in parts]
            if len(nums) == len(temps) + 1:
                data_rows.append(nums)
        except ValueError:
            continue
    if not data_rows:
        return None, None
    df = pd.DataFrame(data_rows, columns=["Wavelength"] + [f"{t}" for t in temps])
    return df.sort_values("Wavelength"), temps


def read_thermal_file(uploaded_file):
    """Public wrapper: extracts bytes then delegates to the cached parser."""
    if uploaded_file is None:
        return None, None
    try:
        return _read_thermal_file_cached(uploaded_file.getvalue())
    except Exception:
        return None, None
@st.cache_data(show_spinner=False)
def _read_thermal_channel_cached(file_bytes: bytes, channel: int):
    """
    Parse a specific channel from a JASCO multi-temperature multicolumn file.
      channel=1 → CD (mdeg)   matches YUNITS
      channel=2 → HT (V)      matches Y2UNITS
      channel=3 → Absorbance  matches Y3UNITS
    Returns (DataFrame, temps) or (None, None).
    """
    try:
        text = file_bytes.decode("latin-1")
    except Exception:
        return None, None
    lines = text.split("\n")
    target = f"Channel {channel}"
    start_idx = -1
    temps = []
    for i, line in enumerate(lines):
        if target in line:
            temp_line = lines[i + 1].strip().replace(",", ".")
            try:
                temps = [float(t) for t in temp_line.split()]
            except ValueError:
                return None, None
            start_idx = i + 2
            break
    if start_idx == -1 or not temps:
        return None, None
    data_rows = []
    for line in lines[start_idx:]:
        if not line.strip() or "Channel" in line:
            break
        clean_line = line.replace(",", ".")
        parts = clean_line.split()
        try:
            nums = [float(p) for p in parts]
            if len(nums) == len(temps) + 1:
                data_rows.append(nums)
        except ValueError:
            continue
    if not data_rows:
        return None, None
    df = pd.DataFrame(data_rows, columns=["Wavelength"] + [f"{t}" for t in temps])
    return df.sort_values("Wavelength"), temps


def read_thermal_channel(uploaded_file, channel: int):
    """
    Public wrapper: read a specific channel from a JASCO multicolumn thermal file.
    channel=1 → CD, channel=2 → HT, channel=3 → Abs.
    Falls back to read_thermal_file (channel 1) if channel data is missing.
    """
    if uploaded_file is None:
        return None, None
    try:
        return _read_thermal_channel_cached(uploaded_file.getvalue(), channel)
    except Exception:
        return None, None



@st.cache_data(show_spinner=False)
def deconvolve_signal(sample_wl, sample_sig):
    common_start = max(190, min(sample_wl))
    common_end = min(260, max(sample_wl))
    if common_end - common_start < 10: return None 
    calc_grid = np.arange(np.ceil(common_start), np.floor(common_end)+1, 1)
    f_sample = interp1d(sample_wl, sample_sig, kind='linear')
    y_vec = f_sample(calc_grid)
    idx_start = int(calc_grid[0] - 190)
    idx_end = int(calc_grid[-1] - 190) + 1
    A_mat = REF_MATRIX[idx_start:idx_end, :]
    coeffs, resid = nnls(A_mat, y_vec)
    total = np.sum(coeffs)
    if total == 0: return [0,0,0,0]
    return (coeffs / total) * 100

@st.cache_data(show_spinner=False)
def check_310_helix(wl, sig):
    f = interp1d(wl, sig, bounds_error=False, fill_value="extrapolate")
    val_222 = f(222)
    val_208 = f(208)
    if abs(val_208) < 0.1: return False
    ratio = val_222 / val_208
    if ratio < 0.6 and val_208 < -5: return True
    return False

@st.cache_data(show_spinner=False)
def get_min_max(wl, sig, r_min, r_max):
    mask = (wl >= r_min) & (wl <= r_max)
    if not np.any(mask): return None
    mwl, msig = wl[mask], sig[mask]
    
    # 1. Absolute Max
    idx_max = np.argmax(msig)
    max_wl = mwl[idx_max]
    max_val = msig[idx_max]

    # 2. Absolute Min (Fallback)
    idx_min = np.argmin(msig)
    min_wl = mwl[idx_min]
    min_val = msig[idx_min]

    # 3. Smart Local Minima Detection (For 208/222nm Helices)
    # We invert the signal so valleys become peaks
    peaks, _ = find_peaks(-msig, distance=8, prominence=0.5)
    
    sec_min_wl = None
    sec_min_val = None
    
    if len(peaks) > 1:
        # Sort detected peaks by actual signal depth
        sorted_peaks = sorted(peaks, key=lambda i: msig[i])
        best_peak = sorted_peaks[0]
        second_best_peak = sorted_peaks[1]
        
        # Ensure the secondary peak is distinctly separated (avoiding noise on a single broad trough)
        if abs(mwl[best_peak] - mwl[second_best_peak]) > 8:
            sec_min_wl = mwl[second_best_peak]
            sec_min_val = msig[second_best_peak]
            
            # Re-align primary min to the best peak found
            min_wl = mwl[best_peak]
            min_val = msig[best_peak]

    return {
        "Lambda Min 1 (nm)": round(min_wl, 1), 
        "Min 1 Value": round(min_val, 2),
        "Lambda Min 2 (nm)": round(sec_min_wl, 1) if sec_min_wl else None,
        "Min 2 Value": round(sec_min_val, 2) if sec_min_val else None,
        "Lambda Max (nm)": round(max_wl, 1), 
        "Max Value": round(max_val, 2)
    }

def format_axis_text(text, bold, italic):
    if bold: text = f"<b>{text}</b>"
    if italic: text = f"<i>{text}</i>"
    return text

@st.cache_data(show_spinner=False)
def apply_smoothing(wl, sig, method, val, poly=2):
    if method == "LOWESS (Match R)":
        x_range = max(wl) - min(wl)
        lowess = sm.nonparametric.lowess(sig, wl, frac=val, it=3, delta=0.01*x_range)
        return lowess[:, 0], lowess[:, 1]
    else:
        if len(sig) > val:
            return wl, savgol_filter(sig, window_length=int(val), polyorder=poly)
        else:
            return wl, sig

@st.cache_data(show_spinner=False)
def calculate_delG_raw(temps, signals):
    R = 1.9872
    T_K = np.array(temps) + 273.15
    y_vals = np.array(signals)
    y_min = np.min(y_vals)
    y_max = np.max(y_vals)
    if y_max - y_min == 0: return np.zeros_like(y_vals), np.zeros_like(y_vals), np.zeros_like(y_vals)
    if abs(y_vals[-1] - y_vals[0]) > 0:
        alpha = (y_vals - y_vals[0]) / (y_vals[-1] - y_vals[0])
    else:
        alpha = (y_vals - y_min) / (y_max - y_min)
    alpha_clipped = np.clip(alpha, 0.001, 0.999)
    K_eq = alpha_clipped / (1 - alpha_clipped)
    delG_cal = -R * T_K * np.log(K_eq)
    delG_kcal = delG_cal / 1000.0
    return alpha, K_eq, delG_kcal

@st.cache_data(show_spinner=False)
def _compute_ss_for_sample(
    wl_tuple, sig_tuple, temp_tuple,
    nres: int, is_d_peptide: bool,
    apply_chen: bool, metric: str
):
    """
    Compute NNLS and empirical secondary structure at every temperature for one
    thermal sample.  All inputs are plain Python tuples / scalars so
    st.cache_data can hash them without issue.

    Returns two lists of dicts: (nnls_rows, emp_rows).
    """
    from scipy.optimize import nnls as _nnls
    nnls_rows = []
    emp_rows  = []

    for wl_arr, sig_arr, temp in zip(wl_tuple, sig_tuple, temp_tuple):
        wl_np  = np.array(wl_arr)
        sig_np = np.array(sig_arr)
        _sig   = -sig_np if is_d_peptide else sig_np
        f_s    = interp1d(wl_np, _sig, bounds_error=False, fill_value=0)

        # NNLS
        dyn_helix = REF_HELIX.copy()
        if apply_chen and 4 <= nres < 30:
            dyn_helix = REF_HELIX * (1.0 - 2.57 / nres)
        dyn_mat  = np.vstack([dyn_helix, REF_SHEET, REF_COIL]).T
        sig_grid = f_s(REF_WL)
        x, _     = _nnls(dyn_mat, sig_grid)
        total    = np.sum(x)
        fracs    = (x / total * 100) if total > 0 else [0.0, 0.0, 0.0]

        # Empirical — back-convert to full MRE for thresholds
        v222 = float(f_s(222))
        v217 = float(f_s(217))
        if metric == "MRE":
            mre222, mre217 = v222 * 1000, v217 * 1000
        elif "Δε" in metric or "delta" in metric.lower():
            mre222, mre217 = v222 * 3298.2, v217 * 3298.2
        else:
            mre222, mre217 = v222, v217

        emp_h = max(0.0, min(100.0, (mre222 / -33000.0) * 100.0))
        emp_s = max(0.0, min(100.0, (mre217 / -30000.0) * 100.0))

        nnls_rows.append({
            "Temperature (°C)": temp,
            "α-Helix (%)":      round(fracs[0], 1),
            "β-Sheet (%)":      round(fracs[1], 1),
            "Random Coil (%)":  round(fracs[2], 1),
        })
        emp_rows.append({
            "Temperature (°C)":     temp,
            "Emp. Helix 222nm (%)": round(emp_h, 1),
            "Emp. Sheet 217nm (%)": round(emp_s, 1),
        })

    return nnls_rows, emp_rows


# ── NAVIGATION STATE ─────────────────────────────────────────────────────────
if 'page' not in st.session_state:
    st.session_state.page = "Home"

def go_home():
    st.session_state.page = "Home"


# ── SECTION 3: HOME / LANDING PAGE ───────────────────────────────────────────
if st.session_state.page == "Home":
    
    # 1. CENTERED TITLE
    _home_muted= "#a0a8b8" if st.get_option("theme.base") == "dark" else "#555555"
    st.markdown("<h1 class='rapid-title'>RAPID-CD</h1>", unsafe_allow_html=True)
    
    # Load banner image from local file if available, else skip
    import base64
    import os
    
    def get_base64_image(file_path, fallback_url):
        """Helper function to load local images or use a web fallback."""
        if os.path.exists(file_path):
            with open(file_path, "rb") as img_file:
                b64_string = base64.b64encode(img_file.read()).decode()
            return f"data:image/png;base64,{b64_string}"
        return fallback_url

    # Define your images here (You can use local files or web links)
    # Just change the "Cover_Image_1.png" etc. to whatever your files are named!
    img_1 = get_base64_image("Cover_Image_2.png", "Cover_Image_2.png")
    
    st.markdown(f"""
    <style>
        /* This container puts the images side-by-side */
        .banner-container {{
            display: flex;
            gap: 15px; /* Space between the images */
            margin-bottom: 10px;
            width: 100%;
        }}
        .youtube-banner {{
            flex: 1; /* Forces all images to be the exact same width */
            height: 250px; 
            object-fit: cover; 
            border-radius: 12px; 
        }}
    </style>
    
    <div class="banner-container">
        <img src="{img_1}" class="youtube-banner">
    </div>
    
    <div style="text-align: center; color: {_home_muted}; font-size: 14px; margin-bottom: 25px;">
        Rapid Analysis Pipeline for Interpreting Dichroism
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Choose an analysis module to begin:")

    # ... (The rest of your button code follows here unchanged) ...
    
    # Button colour overrides for each home-screen module card
    st.markdown("""
    <style>
        /* 1st Column: General Analysis (Green) */
        div[data-testid="stColumn"]:nth-child(1) button,
        div[data-testid="column"]:nth-child(1) button {
            background-color: #28a745 !important; 
            border: 1px solid #28a745 !important; 
        }
        div[data-testid="stColumn"]:nth-child(1) button *,
        div[data-testid="column"]:nth-child(1) button * {
            color: white !important;
            font-weight: bold !important;
        }

        /* 2nd Column: Thermal Analysis (Orange) */
        div[data-testid="stColumn"]:nth-child(2) button,
        div[data-testid="column"]:nth-child(2) button {
            background-color: #fd7e14 !important; 
            border: 1px solid #fd7e14 !important; 
        }
        div[data-testid="stColumn"]:nth-child(2) button *,
        div[data-testid="column"]:nth-child(2) button * {
            color: white !important;
            font-weight: bold !important;
        }

        /* 3rd Column: Reversibility (Blue) */
        div[data-testid="stColumn"]:nth-child(3) button,
        div[data-testid="column"]:nth-child(3) button {
            background-color: #007bff !important; 
            border: 1px solid #007bff !important; 
        }
        div[data-testid="stColumn"]:nth-child(3) button *,
        div[data-testid="column"]:nth-child(3) button * {
            color: white !important;
            font-weight: bold !important;
        }
        
        /* Smooth Hover Effects */
        div[data-testid="stColumn"]:nth-child(1) button:hover, div[data-testid="column"]:nth-child(1) button:hover { background-color: #218838 !important; border-color: #218838 !important; }
        div[data-testid="stColumn"]:nth-child(2) button:hover, div[data-testid="column"]:nth-child(2) button:hover { background-color: #e85e0c !important; border-color: #e85e0c !important; }
        div[data-testid="stColumn"]:nth-child(3) button:hover, div[data-testid="column"]:nth-child(3) button:hover { background-color: #0056b3 !important; border-color: #0056b3 !important; }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div class="home-card"><div class="home-icon">🧪</div><div class="home-title">General Analysis</div><div class="home-desc">Single-spectrum CD pipeline. Supports <b>mdeg</b> and <b>Δε</b> inputs with automatic header detection. Features blank subtraction, MRE/Δε conversion, smoothing, NNLS structural screening, statistics, and publication-grade export for BeStSel and DichroWeb.</div></div>""", unsafe_allow_html=True)
        if st.button("Start General Analysis", use_container_width=True):
            st.session_state.page = "General Analysis"
            st.rerun()
    with col2:
        st.markdown("""<div class="home-card"><div class="home-icon">🔥</div><div class="home-title">Thermal Analysis</div><div class="home-desc">Analyze multi-temperature melts.<br>Calculate Tm, Thermodynamics (ΔG, ΔΔG), and visualize transitions.</div></div>""", unsafe_allow_html=True)
        if st.button("Start Thermal Analysis", use_container_width=True):
            st.session_state.page = "Thermal Analysis"
            st.rerun()
    with col3:
        st.markdown("""<div class="home-card"><div class="home-icon">🔄</div><div class="home-title">Reversibility</div><div class="home-desc">Compare pre-melt and post-refolding spectra.<br>Calculate RMSD and Correlation scores to assess recovery.</div></div>""", unsafe_allow_html=True)
        if st.button("Start Reversibility Check", use_container_width=True):
            st.session_state.page = "Reversibility Analysis"
            st.rerun()

    st.divider()
    
    with st.expander("📖 Quick Guide to Start"):
        st.markdown("""
        1. **Select a Module:** Click a button above based on your experiment type.
        2. **Upload Data:** Use the sidebar to upload raw `.txt` files (Jasco format supported).
        3. **Set Parameters:** Input concentration and pathlength carefully for accurate MRE calculations.
        4. **Analyze:** Use the tabs to view Overlays, Statistics, and Structure estimations.
        5. **Download:** Export your processed plots and high-resolution publication figures.
        6. **For Reversibility Analysis:** Upload a single multi-panel file containing both pre-melt and post-melt spectra recorded at the same temperature.
        """)
        
    with st.expander("📚 How to Cite & References"):
        st.markdown("""
        **If you use this software in your research, please cite the application and the implemented biophysical methods:**
        
        * **Software:** Pritam Roy [Sorbonne University], "RAPID-CD: Peptide CD Analysis Suite", (2026).
        * **Thermodynamics & Melt Analysis:** Greenfield, N. J. (2006). Using circular dichroism spectra to estimate protein secondary structure. *Nature Protocols*, 1(6), 2876-2890.
        * **Internal NNLS Deconvolution:** Sreerama, N., & Woody, R. W. (2000). Estimation of protein secondary structure from circular dichroism spectra. *Analytical Biochemistry*, 287(2), 252-260.
        * **Basis Spectra (Helix/Sheet/Coil):** Brahms, S., & Brahms, J. G. (1980). Determination of protein secondary structure in solution by vacuum ultraviolet circular dichroism. *Journal of Molecular Biology*, 138(2), 149-178.
        * **Computed CD Foundations:** Greenfield, N. J., & Fasman, G. D. (1969). Computed circular dichroism spectra for the evaluation of protein conformation. *Biochemistry*, 8(10), 4108-4116.
        * **Chain-Length Correction:** Chen, Y. H., Yang, J. T., & Chau, K. H. (1974). Determination of the secondary structures of proteins by circular dichroism and optical rotatory dispersion. *Biochemistry*, 13(16), 3350-3359.
        
        **If you utilize the external BeStSel export tool, please ensure you cite their server:**
        * **BeStSel Webserver:** Micsonai, A., et al. (2025). BeStSel webserver: update and expansion of the secondary structure estimation method. *Nucleic Acids Research*, 53, W73-83.
        * **BeStSel Algorithm:** Micsonai, A., et al. (2015). Accurate secondary structure prediction and fold recognition for circular dichroism spectroscopy. *PNAS*, 112(24), E3095-103.
        """)
        
    with st.expander("ℹ️ About this Software"):
        st.markdown("""
        **RAPID-CD** (Rapid Analysis Pipeline for Interpreting Dichroism) is a privacy-centric, locally-executed tool designed for biophysicists and protein engineers working with circular dichroism spectroscopy data.

        All processing is performed on the user's own machine — no data is transmitted to external servers, ensuring full confidentiality of unpublished experimental data. RAPID-CD is designed as an end-to-end preprocessing and visualisation pipeline, bridging the gap between raw instrument output and submission-ready formatting for established deconvolution servers including BeStSel and DichroWeb.

        **Developed by:** Pritam Roy, Sorbonne University, Paris, France.
        **Contact / Citation:** Please refer to the documentation for citation guidance.
        """)
        
    st.markdown("---")
    st.caption("**RAPID-CD v1.0 | Pritam Roy, Sorbonne University | Rapid Analysis Pipeline for Interpreting Dichroism | Compatible with JASCO & Standard Text Data**")

# ── SECTION 4: ANALYSIS MODULES (General / Thermal / Reversibility) ──────────
else:
    with st.sidebar:
        if st.button("🏠 Return to Home"):
            go_home()
            st.rerun()
        st.markdown("---")
        st.markdown("---")
        st.title("Analysis Mode")
        modes = ["General Analysis", "Thermal Analysis", "Reversibility Analysis"]
        try: curr_index = modes.index(st.session_state.page)
        except: curr_index = 0
        selected_mode = st.radio("Select Mode", modes, index=curr_index, label_visibility="collapsed")
        if selected_mode != st.session_state.page:
            st.session_state.page = selected_mode
            st.rerun()
        st.markdown("---")

    mode = st.session_state.page

    # ── MODULE 1: GENERAL ANALYSIS ───────────────────────────────────────────────
    if mode == "General Analysis":
        with st.sidebar:
            st.markdown("## 🧪 Experimental Setup")
            c_g1, c_g2 = st.columns(2)
            num_samples = c_g1.number_input("Num Samples", 1, 10, 2)
            blank_mode = c_g2.radio("Blanking", ["Common", "Individual"], label_visibility="collapsed")
            if blank_mode == "Common":
                st.info("ℹ️ **Common Blank Mode:** Files uploaded below will use this SINGLE blank.")
                common_blank = st.file_uploader("💧 Upload Common Buffer/Blank File (.txt)", type=["txt"])
            else:
                st.info("ℹ️ **Individual Blank Mode:** Upload a separate blank for EACH sample.")
                common_blank = None

            st.divider()
            samples = []
            for i in range(num_samples):
                with st.expander(f"Sample {i+1}", expanded=(i==0)):
                    c1, c2 = st.columns([2, 1.2])
                    n = c1.text_input("Name", f"Sample {i+1}", key=f"n{i}", placeholder="Name")
                    c = c2.number_input("Conc (µM)", value=50.0, key=f"c{i}")
                    
                    # --- SMART SEQUENCE PARSER (General) ---
                    seq_input = st.text_input("Peptide Sequence (Optional)", key=f"seq_{i}", placeholder="e.g., ALYFWC...")
                    clean_seq = "".join([char.upper() for char in seq_input if char.isalpha()])
                    
                    if clean_seq:
                        r = len(clean_seq)
                        num_W = clean_seq.count('W')
                        num_Y = clean_seq.count('Y')
                        num_C = clean_seq.count('C')
                        ext_coeff = (num_W * 5500) + (num_Y * 1490) + (int(num_C / 2) * 125)
                        mw_est = (r * 110) + 18
                        
                        st.info(f"✅ **Auto-counted: {r} residues**")
                        c_seq1, c_seq2 = st.columns(2)
                        c_seq1.caption(f"🧮 **ε₂₈₀:** {ext_coeff} M⁻¹cm⁻¹")
                        c_seq2.caption(f"⚖️ **MW:** ~{mw_est} Da")
                        
                        aromatic_pct = ((num_W + num_Y) / r) * 100
                        if aromatic_pct > 15:
                            st.warning(f"⚠️ **High Aromatic Content ({aromatic_pct:.1f}%):** Trp/Tyr signals may distort the 222 nm helical band.")
                    else:
                        r = st.number_input("Or enter number of residues manually:", value=6, key=f"r_manual_{i}")
                    # ---------------------------------------

                    # File uploader must precede detect_yunits() — f must be defined first
                    st.markdown("---")
                    c4, c5 = st.columns(2)
                    with c4: f = st.file_uploader("🧬 Upload Sample File (.txt)", key=f"f{i}", type=["txt"])
                    b = None
                    if blank_mode == "Individual":
                        with c5: b = st.file_uploader("💧 Upload Blank File (.txt)", key=f"b{i}", type=["txt"])

                    # Auto-detect input format from YUNITS header, allow manual override
                    FORMAT_OPTIONS = [
                        "mdeg (Raw CD Signal)",
                        "Δε — Molar CD (per molecule)",
                        "Δε — Mean Residue (per residue)",
                    ]

                    # Auto-detect from the uploaded file header (f is now defined)
                    detected_tag = detect_yunits(f)
                    tag_to_index = {
                        "mdeg":               0,
                        "delta_eps_molar":    1,
                        "delta_eps_residue":  2,
                        "unknown":            0,   # fall back to mdeg if not found
                    }
                    detected_index = tag_to_index.get(detected_tag, 0)

                    # Badge shown once a file is uploaded
                    badge_map = {
                        "mdeg":               "✅ Auto-detected: **mdeg**",
                        "delta_eps_molar":    "✅ Auto-detected: **Mol. CD (Δε per molecule)**",
                        "delta_eps_residue":  "✅ Auto-detected: **Mean Residue Δε**",
                        "unknown":            "⚠️ Format not found in header — please select manually.",
                    }
                    if f is not None:
                        st.caption(badge_map.get(detected_tag, ""))

                    input_fmt = st.selectbox(
                        "📥 Input Data Format" + (" (override if needed)" if detected_tag != "unknown" and f is not None else ""),
                        FORMAT_OPTIONS,
                        index=detected_index,
                        key=f"fmt{i}",
                        help=(
                            "**Auto-detected** from the `YUNITS` line in your file header (JASCO format). "
                            "You can override this manually if needed.\n\n"
                            "**mdeg:** Standard raw instrument output.\n\n"
                            "**Δε Molar:** Collaborator file normalised per molecule — "
                            "needs pathlength + concentration + residues for MRE.\n\n"
                            "**Δε Mean Residue:** Already per residue — "
                            "needs pathlength + concentration only."
                        )
                    )

                    # D-amino acid peptide flag — stored per sample.
                    # Affects ONLY secondary structure calculations (signal is
                    # internally inverted before NNLS/empirical formulas).
                    # The displayed spectrum is always left as measured.
                    is_d_peptide = st.checkbox(
                        "🔄 D-amino acid peptide",
                        value=False,
                        key=f"dpep{i}",
                        help=(
                            "Tick this if the peptide is composed of D-amino acids (mirror-image "
                            "enantiomer of natural L-peptides).\n\n"
                            "**Effect on the plot:** None — the measured signal is always displayed "
                            "as recorded. For a D-helix this will appear as a positive spectrum, "
                            "which is the correct experimental result.\n\n"
                            "**Effect on secondary structure (Sec. Structure tab):** The signal is "
                            "multiplied by −1 internally before NNLS deconvolution and empirical "
                            "estimates, so the L-peptide basis spectra and thresholds apply correctly. "
                            "The inversion is NOT applied to the exported plot or CSV."
                        )
                    )
                    samples.append({"name": n, "conc": c, "nres": r, "file": f, "blank": b, "input_fmt": input_fmt, "is_d_peptide": is_d_peptide})
                    
            st.divider()
            st.markdown("### ⚙️ Processing")
            metric = st.selectbox(
                "Output Metric",
                ["MRE", "Raw (mdeg)", "Δε (M⁻¹cm⁻¹)", "HT", "Abs"],
                help=(
                    "**MRE:** Mean Residue Ellipticity — standard for publication.\n\n"
                    "**Raw (mdeg):** Raw instrument signal (or back-calculated if input is Δε).\n\n"
                    "**Δε:** Molar CD — displayed directly if input is Δε, or converted from mdeg.\n\n"
                    "**HT / Abs:** Diagnostic instrument channels."
                )
            )
            path_cm = st.number_input("Pathlength (cm)", value=0.1)
            
            # Smoothing controls
            apply_smooth = st.checkbox("Apply Smoothing", value=True, help="Toggle to apply or remove all mathematical smoothing.")
            if apply_smooth:
                smooth_method = st.radio("Method", ["Savitzky-Golay (Recommended)", "LOWESS (Match R)"])
                smooth_val = st.slider("Smoothing Fraction (f)", 0.01, 0.30, 0.10) if smooth_method == "LOWESS (Match R)" else st.slider("Window Length", 5, 51, 11, step=2)
            else:
                smooth_method = "None"
                smooth_val = 0

        #Processing
        processed_curves = []
        c_blank_df = read_cd_file(common_blank)

        # Conversion constant: MRE [θ] (deg·cm²·dmol⁻¹·res⁻¹) = Δε × 3298.2
        DELTA_EPS_TO_MRE = 3298.2

        for i, s in enumerate(samples):
            if not s["file"]: continue
            df = read_cd_file(s["file"])
            if df is None: continue
            
            b_df = c_blank_df
            if blank_mode == "Individual" and s["blank"]: b_df = read_cd_file(s["blank"])
            
            # Select the appropriate data column based on metric
            y_col = "CD"
            if metric == "HT": y_col = "HT"
            elif metric == "Abs": y_col = "Abs"
            if y_col not in df.columns: continue
            
            wl, sig = df["Wavelength"].values, df[y_col].values

            # Retrieve input format tag for this sample
            input_fmt = s.get("input_fmt", "mdeg (Raw CD Signal)")
            is_mdeg            = "mdeg" in input_fmt
            is_delta_molar     = "Molar" in input_fmt      # Δε per molecule
            is_delta_residue   = "Mean Residue" in input_fmt  # Δε already per residue

            # Blank subtraction (before unit conversion, skipped for HT/Abs)
            if metric not in ["HT", "Abs"] and b_df is not None:
                f_b = interp1d(b_df["Wavelength"], b_df["CD"], bounds_error=False, fill_value="extrapolate")
                sig = sig - f_b(wl)

            # ----------------------------------------------------------------
            # CONVERSION BLOCK
            # Relation constants:
            #   mdeg → MRE:        [θ] = mdeg / (10 × l × c_M × N_res)   [stored ×10⁻³]
            #   Δε_molar → MRE:    [θ] = Δε × 3298.2 / N_res             [stored ×10⁻³]
            #   Δε_residue → MRE:  [θ] = Δε × 3298.2                     [stored ×10⁻³]
            # ----------------------------------------------------------------
            fac_mdeg = 10 * path_cm * (s["conc"] * 1e-6) * s["nres"]  # for mdeg conversions
            nres     = s["nres"] if s["nres"] > 0 else 1

            if is_mdeg:
                if metric == "MRE":
                    if fac_mdeg != 0: sig = sig / fac_mdeg / 1000
                elif metric == "Δε (M⁻¹cm⁻¹)":
                    # mdeg → MRE → Δε_residue: Δε = [θ] / 3298.2
                    if fac_mdeg != 0: sig = (sig / fac_mdeg / 1000) / 3298.2 * 1000
                # Raw (mdeg) / HT / Abs: no conversion needed

            elif is_delta_molar:
                # Input is Δε per whole molecule
                if metric == "MRE":
                    sig = (sig * 3298.2 / nres) / 1000          # stored ×10⁻³
                elif metric == "Raw (mdeg)":
                    sig = sig * 3298.2 / nres * fac_mdeg
                elif metric == "Δε (M⁻¹cm⁻¹)":
                    sig = sig / nres                              # convert to per-residue Δε

            elif is_delta_residue:
                # Input is Δε already per residue (mean residue Δε)
                if metric == "MRE":
                    sig = (sig * 3298.2) / 1000                  # stored ×10⁻³
                elif metric == "Raw (mdeg)":
                    sig = sig * 3298.2 * fac_mdeg
                # "Δε (M⁻¹cm⁻¹)" → display as-is (already per residue)
            # ----------------------------------------------------------------
                
            # Preserve unsmoothed signal for diagnostic panel
            raw_sig = sig.copy() 
            
            if apply_smooth:
                smooth_wl, smooth_sig = apply_smoothing(wl, sig, smooth_method, smooth_val)
            else:
                smooth_wl, smooth_sig = wl, sig
                
            processed_curves.append({
                "name": s["name"], 
                "wl": smooth_wl, 
                "sig": smooth_sig, 
                "raw_sig": raw_sig, 
                "color": COLORS[i % len(COLORS)], 
                "nres": s["nres"],
                "input_fmt": input_fmt,
                "is_d_peptide": s.get("is_d_peptide", False)
            })

        if processed_curves:
            with st.sidebar:
                st.markdown("---")
                log_data = generate_log_file("General Analysis", samples, {"Metric": metric, "Path": path_cm})
                st.download_button("💾 Download Analysis Log", log_data, file_name="analysis_log.txt")

        st.title("🧬 General Analysis")
        if processed_curves:
            with st.expander("🛠️ Plot Customization & Ranges", expanded=True):
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    show_grid = st.checkbox("Show Grid Lines", True) 
                    line_width = st.slider("Line Width", 0.5, 5.0, 2.5, 0.5)
                with col_b:
                    st.markdown("**Axis Styling**")
                    fs_title = st.number_input("Title Font Size", 10, 30, 16)
                    fs_num = st.number_input("Axis Label Size", 8, 24, 12)
                    fs_leg = st.number_input("Legend Size", 8, 24, 12)
                with col_c:
                    st.markdown("**Manual Axis Limits**")
                    wl_min = st.number_input("Min WL", value=190)
                    wl_max = st.number_input("Max WL", value=260)
                    y_min = st.number_input("Y Min", value=-30)
                    y_max = st.number_input("Y Max", value=30)

            # ── PEAK SEARCH RANGE (optional) ───────────────────
            with st.expander("🔍 Peak Search Range (optional)", expanded=False):
                st.caption(
                    "By default the software searches for maxima and minima "
                    "across the full visible wavelength range. "
                    "Restrict the window here if noise at 195-200 nm is being "
                    "picked up instead of the true helical bands at 208/222 nm."
                )
                _ga_psr_c1, _ga_psr_c2 = st.columns(2)
                with _ga_psr_c1:
                    st.markdown("**↓ Minimum search window**")
                    _ga_use_min = st.checkbox("Restrict minimum search", value=False, key="ga_use_min_range")
                    _ga_min_lo = st.number_input("Min search: start (nm)", 170.0, 350.0, 205.0, step=1.0, key="ga_min_srch_lo", disabled=not _ga_use_min)
                    _ga_min_hi = st.number_input("Min search: end (nm)",   170.0, 350.0, 230.0, step=1.0, key="ga_min_srch_hi", disabled=not _ga_use_min)
                with _ga_psr_c2:
                    st.markdown("**↑ Maximum search window**")
                    _ga_use_max = st.checkbox("Restrict maximum search", value=False, key="ga_use_max_range")
                    _ga_max_lo = st.number_input("Max search: start (nm)", 170.0, 350.0, 185.0, step=1.0, key="ga_max_srch_lo", disabled=not _ga_use_max)
                    _ga_max_hi = st.number_input("Max search: end (nm)",   170.0, 350.0, 205.0, step=1.0, key="ga_max_srch_hi", disabled=not _ga_use_max)
                st.info(
                    "💡 **Tip:** For α-helix, set minimum: 205–230 nm (208 & 222 nm bands), "
                    "maximum: 185–200 nm (192 nm positive band). For β-sheet: minimum 210–225 nm."
                )
            # Resolved GA search ranges
            _ga_min_r = (_ga_min_lo, _ga_min_hi) if _ga_use_min else (wl_min, wl_max)
            _ga_max_r = (_ga_max_lo, _ga_max_hi) if _ga_use_max else (wl_min, wl_max)

            # Set y-axis label from selected output metric
            if metric == "MRE":
                y_axis_label = "MRE [θ] (x10³ deg cm² dmol⁻¹ res⁻¹)"
            elif metric == "Raw (mdeg)":
                y_axis_label = "CD (mdeg)"
            elif metric == "Δε (M⁻¹cm⁻¹)":
                y_axis_label = "Δε (M⁻¹ cm⁻¹)"
            elif metric == "HT":
                y_axis_label = "HT (Volts)"
            else: 
                y_axis_label = "Absorbance (OD)"
            final_curves = []
            stat_grid = np.arange(wl_min, wl_max + 1, 1)
            
            for p in processed_curves:
                f_final = interp1d(p["wl"], p["sig"], kind='linear', fill_value="extrapolate")
                f_raw = interp1d(p["wl"], p["raw_sig"], kind='linear', fill_value="extrapolate") # <-- NEW
            
                v_start, v_end = max(min(p["wl"]), wl_min), min(max(p["wl"]), wl_max)
                vis_grid, vis_sig, vis_raw = np.array([]), np.array([]), np.array([]) # <-- NEW
                stat_sig = np.zeros(len(stat_grid))
                struct_pct, is_310 = None, False
                if v_end > v_start:
                    vis_grid = np.unique(np.concatenate(([wl_min], p["wl"][(p["wl"] >= v_start) & (p["wl"] <= v_end)], [wl_max])))
                    vis_sig = f_final(vis_grid)
                    vis_raw = f_raw(vis_grid) # <-- NEW
                    stat_sig = f_final(stat_grid)
                    if metric == "MRE":
                        struct_pct = deconvolve_signal(p["wl"], p["sig"])
                        is_310 = check_310_helix(p["wl"], p["sig"])
                final_curves.append({"name": p["name"], "wl": vis_grid, "sig": vis_sig, "raw_sig": vis_raw, "stat_sig": stat_sig, "color": p["color"], "structure": struct_pct, "is_310": is_310, "nres": p["nres"], "is_d_peptide": p.get("is_d_peptide", False)})

            
            st.markdown(
                '<div class="rapid-tab-header">'  
                '📋 &nbsp; General Analysis — Select View'
                '</div>',
                unsafe_allow_html=True
            )
            _ga_tab = st.radio(
                "Select View",
                ["📊 Overlay", "🔲 Separate", "📝 Statistics",
                 "🧩 Sec. Structure", "🔗 Similarity", "🗺️ Spectral Projection"],
                horizontal=True, key="ga_tab_radio",
                label_visibility="collapsed"
            )
            
            if _ga_tab == "📊 Overlay":
                # Show format-detection summary banner if mixed inputs detected
                fmt_summary = {}
                for p in final_curves:
                    fmt = p.get("input_fmt", "mdeg (Raw CD Signal)")
                    fmt_summary.setdefault(fmt, []).append(p["name"])
                if len(fmt_summary) > 1:
                    st.warning(
                        "⚠️ **Mixed input formats detected across samples.** All signals have been converted to the "
                        f"selected output metric (**{metric}**), but mixing mdeg and Δε sources may introduce "
                        "systematic offsets if concentrations are not accurately defined. Verify your parameters carefully.\n\n"
                        + "\n".join([f"- **{fmt}:** {', '.join(names)}" for fmt, names in fmt_summary.items()])
                    )
                elif "Δε" in list(fmt_summary.keys())[0]:
                    st.info(
                        f"ℹ️ **All samples uploaded as Δε (M⁻¹cm⁻¹).** "
                        f"Displayed as: **{metric}**. "
                        "Conversion applied: [θ] = Δε × 3298.2 (for MRE output) or displayed directly (for Δε output)."
                    )

                # Main multi-sample overlay
                st.markdown("##### Final Processed Overlay")
                
                # --- NEW: COLOUR MODE SELECTION ---
                c_mode1, c_mode2 = st.columns([2, 1])
                with c_mode1:
                    sel = st.multiselect("Select Curves:", [p["name"] for p in final_curves], default=[p["name"] for p in final_curves])
                with c_mode2:
                    ga_colour_mode = st.radio(
                        "Colour mode",
                        ["Auto (Default)", "Manual per-sample"],
                        key="ga_ov_colour_mode",
                        help="Auto: Uses Plotly's default palette. Manual: Pick a specific colour for each sample."
                    )

                # Manual colour pickers
                manual_colors_ga = {}
                if ga_colour_mode == "Manual per-sample":
                    with st.expander(f"🎨 Pick colours for {len(sel)} selected sample(s)", expanded=True):
                        DEFAULT_PALETTE = [
                            "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
                            "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"
                        ]
                        cols_cp = st.columns(min(5, max(1, len(sel))))
                        for i, sname in enumerate(sel):
                            with cols_cp[i % 5]:
                                manual_colors_ga[sname] = st.color_picker(
                                    f"{sname}",
                                    value=DEFAULT_PALETTE[i % len(DEFAULT_PALETTE)],
                                    key=f"ga_ov_col_{i}"
                                )
                # ----------------------------------

                fig = go.Figure()
                for p in final_curves:
                    if p["name"] in sel and len(p["wl"]) > 0:
                        # --- NEW: APPLY SELECTED COLOUR ---
                        line_col = manual_colors_ga[p["name"]] if ga_colour_mode == "Manual per-sample" else p["color"]
                        # ----------------------------------
                        fig.add_trace(go.Scatter(x=p["wl"], y=p["sig"], mode='lines', name=p["name"], line=dict(color=line_col, width=line_width)))
                
                fig = apply_publication_style(fig, "CD Spectra Overlay", "Wavelength (nm)", y_axis_label, show_grid, plot_mode='lines')
                fig.update_xaxes(range=[wl_min, wl_max])
                fig.update_yaxes(range=[y_min, y_max])
                fig.add_hline(y=0, line_dash="dash", line_color=_c["text"], line_width=1, opacity=0.5)
                # Add a D-peptide label on the plot for any flagged sample
                d_peptide_names = [p["name"] for p in final_curves if p.get("is_d_peptide") and p["name"] in sel]
                if d_peptide_names:
                    fig.add_annotation(
                        text="⚠ D-amino acid peptide(s): " + ", ".join(d_peptide_names) +
                             " — signal displayed as measured (positive = expected for D-helix)",
                        xref="paper", yref="paper", x=0.0, y=1.04,
                        showarrow=False,
                        font=dict(size=11, color="darkorange", family="Arial"),
                        align="left"
                    )
                st.plotly_chart(fig, use_container_width=True)
                
                # Smoothing diagnostics: raw vs smoothed side-by-side
                if apply_smooth:
                    st.divider()
                    st.markdown("##### 🔍 Smoothing Diagnostics (Raw vs. Smooth)")
                    st.caption("Verify that the mathematical smoothing has not distorted the biophysical features of your raw data.")
                    
                    fig_diag = go.Figure()
                    for p in final_curves:
                        if p["name"] in sel and len(p["wl"]) > 0:
                            # --- NEW: APPLY SELECTED COLOUR TO DIAGNOSTICS ---
                            line_col = manual_colors_ga[p["name"]] if ga_colour_mode == "Manual per-sample" else p["color"]
                            # ----------------------------------
                            
                            # Plot Raw Data (Faded/Dotted)
                            fig_diag.add_trace(go.Scatter(
                                x=p["wl"], y=p["raw_sig"], 
                                mode='lines', 
                                name=f"{p['name']} (Raw)",
                                line=dict(color=line_col, width=1.5, dash='dot'),
                                opacity=0.6
                            ))
                            # Plot Smoothed Data (Solid/Bold)
                            fig_diag.add_trace(go.Scatter(
                                x=p["wl"], y=p["sig"], 
                                mode='lines', 
                                name=f"{p['name']} (Smoothed)",
                                line=dict(color=line_col, width=line_width)
                            ))
                            
                    fig_diag = apply_publication_style(fig_diag, "Raw vs. Smoothed Comparison", "Wavelength (nm)", y_axis_label, show_grid, plot_mode='lines')
                    fig_diag.update_xaxes(range=[wl_min, wl_max])
                    fig_diag.update_yaxes(range=[y_min, y_max])
                    fig_diag.add_hline(y=0, line_dash="dash", line_color=_c["text"], line_width=1, opacity=0.5)
                    st.plotly_chart(fig_diag, use_container_width=True)

                # Build export DataFrame (raw + smoothed columns when smoothing active)
                df_export = pd.DataFrame()
                for p in final_curves:
                    if p["name"] in sel and len(p["wl"]) > 0:
                        if apply_smooth:
                            # If smoothing is on, give them side-by-side Raw and Smooth columns!
                            temp_df = pd.DataFrame({
                                "Wavelength (nm)": p["wl"], 
                                f"{p['name']} (Raw)": p["raw_sig"],
                                f"{p['name']} (Smoothed)": p["sig"]
                            })
                        else:
                            temp_df = pd.DataFrame({"Wavelength (nm)": p["wl"], p["name"]: p["sig"]})
                            
                        if df_export.empty: df_export = temp_df
                        else: df_export = pd.merge(df_export, temp_df, on="Wavelength (nm)", how="outer")
                
                if not df_export.empty: df_export = df_export.sort_values("Wavelength (nm)").round(4)
                
                # Export buttons
                st.markdown("##### 📥 Export Data & Plots")
                try: 
                    st.download_button("💾 Download Combined Data (CSV)", df_export.to_csv(index=False), "overlay_data.csv")
                except: pass
                
                # Create separate columns for the different plot downloads
                c_dl1, c_dl2 = st.columns(2)
                
                with c_dl1:
                    st.caption("**Top Panel: Final Processed Overlay**")
                    try: st.download_button("📸 DL Processed Plot (PNG)", fig.to_image(format="png", scale=3), "processed_overlay.png", "image/png", key="dl_main_png")
                    except: pass
                    try: st.download_button("📄 DL Processed Plot (PDF)", fig.to_image(format="pdf"), "processed_overlay.pdf", "application/pdf", key="dl_main_pdf")
                    except: pass
                
                if apply_smooth:
                    with c_dl2:
                        st.caption("**Bottom Panel: Raw vs. Smooth Comparison**")
                        try: st.download_button("📸 DL Raw & Smooth Plot (PNG)", fig_diag.to_image(format="png", scale=3), "raw_vs_smooth.png", "image/png", key="dl_diag_png")
                        except: pass
                        try: st.download_button("📄 DL Raw & Smooth Plot (PDF)", fig_diag.to_image(format="pdf"), "raw_vs_smooth.pdf", "application/pdf", key="dl_diag_pdf")
                        except: pass

            if _ga_tab == "🔲 Separate":
                # ── 1. CONTROLS ──────────────────────────────────────────────
                c_sel1, c_sel2, c_sel3 = st.columns([1.5, 1, 1])
                with c_sel1:
                    base_options = ["None"] + [p["name"] for p in final_curves]
                    base_name = st.selectbox(
                        "Base Curve (optional)",
                        base_options,
                        index=0,
                        help="Select a curve to overlay as a faint grey dashed reference in every subplot. Choose \'None\' for clean individual plots."
                    )
                    base = next((p for p in final_curves if p["name"] == base_name), None)
                with c_sel2:
                    n_cols_mp = st.slider("Number of Columns", 1, 5, 2, key="mp_ncols")
                with c_sel3:
                    mp_font = st.number_input("Label Font Size", 8, 26, 14, key="mp_font",
                                              help="Font size for axis labels and subplot titles.")

                # ── 2. AUTO-LAYOUT CALCULATION ───────────────────────────────
                # Margins and label positions are derived mathematically so the
                # output is correct regardless of grid size — no manual sliders.
                n_plots   = len(final_curves)
                n_rows_mp = max(1, (n_plots + n_cols_mp - 1) // n_cols_mp)

                cell_w = 420
                cell_h = 370
                fig_w  = cell_w * n_cols_mp
                fig_h  = cell_h * n_rows_mp

                # Left margin scales with y-axis label length
                mg_l = min(160, max(100, 85 + int(len(y_axis_label) * 1.8)))
                mg_b = 100
                mg_t = 85
                mg_r = 30

                # Spacing shrinks as the grid grows
                v_sp = max(0.06, min(0.22, 0.55 / n_rows_mp))
                h_sp = max(0.04, min(0.14, 0.40 / n_cols_mp))

                # ── 3. LABEL POSITION FORMULA ────────────────────────────────
                # In Plotly paper-coords, x=0 is the LEFT edge of the PLOT AREA.
                # The left/bottom margins lie at negative coords.
                # We place each label at the centre of its respective margin:
                #
                #   plot_area_w = fig_w - mg_l - mg_r  (pixels)
                #   Y-label x   = -(mg_l/2) / plot_area_w
                #   X-label y   = -(mg_b/2) / plot_area_h
                plot_area_w = max(1, fig_w - mg_l - mg_r)
                plot_area_h = max(1, fig_h - mg_t - mg_b)
                y_lbl_x = -(mg_l * 0.50) / plot_area_w
                x_lbl_y = -(mg_b * 0.75) / plot_area_h

                # ── 4. BUILD COMBINED FIGURE ─────────────────────────────────
                fig_all = make_subplots(
                    rows=n_rows_mp, cols=n_cols_mp,
                    subplot_titles=[p["name"] for p in final_curves],
                    vertical_spacing=v_sp,
                    horizontal_spacing=h_sp
                )
                for i, p in enumerate(final_curves):
                    r_idx = (i // n_cols_mp) + 1
                    c_idx = (i % n_cols_mp) + 1
                    if base:
                        fig_all.add_trace(
                            go.Scatter(x=base["wl"], y=base["sig"], showlegend=False,
                                       line=dict(color=_c["grid"], width=1.5, dash="dash")),
                            row=r_idx, col=c_idx
                        )
                    fig_all.add_trace(
                        go.Scatter(x=p["wl"], y=p["sig"], showlegend=False,
                                   line=dict(color=p["color"], width=line_width)),
                        row=r_idx, col=c_idx
                    )

                # ── 5. APPLY LAYOUT ──────────────────────────────────────────
                fig_all.update_layout(
                    height=fig_h, width=fig_w,
                    template=_c["template"],
                        paper_bgcolor=_c["paper"],
                        plot_bgcolor=_c["bg"],
                    showlegend=False,
                    title_text=f"Separate Analysis \u2014 {n_rows_mp}\u00d7{n_cols_mp} grid",
                    title_font=dict(size=mp_font + 2, family="Arial", color=_c["text"]),
                    margin=dict(l=mg_l, r=mg_r, t=mg_t, b=mg_b),
                    font=dict(family="Arial", size=mp_font - 2, color=_c["text"])
                )
                fig_all.update_xaxes(
                    range=[wl_min, wl_max],
                    showgrid=show_grid, gridcolor=_c["grid"],
                    showline=True, linewidth=1.5, linecolor=_c["line"], mirror=True,
                    tickfont=dict(size=mp_font - 2, family="Arial", color=_c["text"])
                )
                fig_all.update_yaxes(
                    range=[y_min, y_max],
                    showgrid=show_grid, gridcolor=_c["grid"],
                    showline=True, linewidth=1.5, linecolor=_c["line"], mirror=True,
                    tickfont=dict(size=mp_font - 2, family="Arial", color=_c["text"])
                )
                for ann in fig_all.layout.annotations:
                    ann.font = dict(size=mp_font, family="Arial", color=_c["text"])
                    ann.update(yshift=12)

                # ── 6. SHARED AXIS LABELS (auto-positioned) ──────────────────
                fig_all.add_annotation(
                    text=f"<b>{y_axis_label}</b>",
                    x=y_lbl_x, y=0.5,
                    xref="paper", yref="paper",
                    textangle=-90, showarrow=False,
                    font=dict(size=mp_font, family="Arial", color=_c["text"])
                )
                fig_all.add_annotation(
                    text="<b>Wavelength (nm)</b>",
                    x=0.5, y=x_lbl_y,
                    xref="paper", yref="paper",
                    showarrow=False,
                    font=dict(size=mp_font, family="Arial", color=_c["text"])
                )

                # ── 7. PREVIEW & DOWNLOAD ────────────────────────────────────
                st.info("\U0001f447 Preview and download the combined panel below.")
                with st.expander("\U0001f5bc\ufe0f Combined Panel Preview & Download", expanded=True):
                    fig_all.add_hline(y=0, line_dash="dash", line_color=_c["text"], line_width=1, opacity=0.4)  #Zero reference line applied to all subplots in the grid
                    st.plotly_chart(fig_all, use_container_width=False)
                    c_dl1, c_dl2 = st.columns(2)
                    try:
                        c_dl1.download_button(
                            "\U0001f4e5 Download PNG (High-Res)",
                            fig_all.to_image(format="png", scale=3),
                            f"multipanel_{n_rows_mp}x{n_cols_mp}.png", "image/png",
                            key="mp_dl_png"
                        )
                    except: pass
                    try:
                        c_dl2.download_button(
                            "\U0001f4c4 Download PDF (Vector)",
                            fig_all.to_image(format="pdf"),
                            f"multipanel_{n_rows_mp}x{n_cols_mp}.pdf", "application/pdf",
                            key="mp_dl_pdf"
                        )
                    except: pass

                st.divider()

                # ── 8. INDIVIDUAL PLOTS ───────────────────────────────────────
                st.subheader("Individual Curve Details")
                cols_ui = st.columns(n_cols_mp)
                for i, p in enumerate(final_curves):
                    with cols_ui[i % n_cols_mp]:
                        f_ind = go.Figure()
                        if base:
                            f_ind.add_trace(go.Scatter(
                                x=base["wl"], y=base["sig"],
                                name=base["name"] + " (base)",
                                line=dict(color=_c["grid"], width=2, dash="dash")
                            ))
                        f_ind.add_trace(go.Scatter(
                            x=p["wl"], y=p["sig"],
                            name=p["name"],
                            line=dict(color=p["color"], width=line_width)
                        ))
                        f_ind = apply_publication_style(
                            f_ind, p["name"],
                            "Wavelength (nm)", y_axis_label,
                            show_grid, height=320, plot_mode="lines"
                        )
                        f_ind.update_xaxes(range=[wl_min, wl_max])
                        f_ind.update_yaxes(range=[y_min, y_max])
                        f_ind.add_hline(y=0, line_dash="dash", opacity=0.3)
                        st.plotly_chart(f_ind, use_container_width=True)
                        c1, c2 = st.columns(2)
                        try:
                            c1.download_button("PNG", f_ind.to_image(format="png", scale=3), p["name"] + ".png", "image/png", key=f"p_{i}")
                            c2.download_button("PDF", f_ind.to_image(format="pdf"), p["name"] + ".pdf", "application/pdf", key=f"f_{i}")
                        except: pass


            if _ga_tab == "📝 Statistics":
                stats_rows = []
                for p in final_curves:
                    if len(p["wl"]) > 0:
                        _ga_lo = min(_ga_min_r[0], _ga_max_r[0])
                        _ga_hi = max(_ga_min_r[1], _ga_max_r[1])
                        st_dict = get_min_max(p["wl"], p["sig"], _ga_lo, _ga_hi)
                        if st_dict and _ga_use_min:
                            _d2 = get_min_max(p["wl"], p["sig"], _ga_min_r[0], _ga_min_r[1])
                            if _d2: st_dict["Lambda Min 1 (nm)"] = _d2["Lambda Min 1 (nm)"]; st_dict["Min 1 Value"] = _d2["Min 1 Value"]
                        if st_dict and _ga_use_max:
                            _d3 = get_min_max(p["wl"], p["sig"], _ga_max_r[0], _ga_max_r[1])
                            if _d3: st_dict["Lambda Max (nm)"] = _d3["Lambda Max (nm)"]; st_dict["Max Value"] = _d3["Max Value"]
                        if st_dict: st_dict["Sample"] = p["name"]; stats_rows.append(st_dict)
                
                if stats_rows: 
                    df_st = pd.DataFrame(stats_rows).set_index("Sample")
                    st.dataframe(df_st)
                    
                    # The bar charts will now ALWAYS show, even for 1 sample
                    
                    # Lambda minima bar charts
                    st.markdown("##### 📉 Lambda Minima Comparison")
                    c_s1, c_s2 = st.columns(2)
                    with c_s1:
                        style_min_wl = render_plot_editor("stat_min_g", "Lambda Minima", "#0000FF") 
                        fig_stat1 = go.Figure(data=[go.Bar(
                            x=df_st.index, 
                            y=df_st["Lambda Min 1 (nm)"],
                            text=df_st["Lambda Min 1 (nm)"].astype(str) + " nm", # Adds text to bars
                            textposition='auto'
                        )])
                        fig_stat1 = apply_plot_style_custom(fig_stat1, style_min_wl, "Primary Lambda Minima", "Sample", "Wavelength (nm)")
                        if not style_min_wl["y_range"]: fig_stat1.update_yaxes(range=[180, 240])
                        st.plotly_chart(fig_stat1)
                        try: st.download_button("DL Lambda Min (JPG)", fig_stat1.to_image(format="jpg", scale=3), "lambda_min.jpg", "image/jpeg", key="dl_min_wl")
                        except: pass

                    with c_s2:
                        style_min_v = render_plot_editor("stat_min_v", "Min Intensity", "#FF0000")
                        fig_stat2 = go.Figure(data=[go.Bar(
                            x=df_st.index, 
                            y=df_st["Min 1 Value"],
                            text=df_st["Min 1 Value"].astype(str), # Adds text to bars
                            textposition='auto'
                        )])
                        fig_stat2 = apply_plot_style_custom(fig_stat2, style_min_v, "Primary Min Intensity", "Sample", "Signal")
                        st.plotly_chart(fig_stat2)
                        try: st.download_button("DL Intensity Plot (JPG)", fig_stat2.to_image(format="jpg", scale=3), "min_intensity.jpg", "image/jpeg", key="dl_min_v")
                        except: pass
                        
                    st.divider()
                    
                    # Lambda maxima bar charts
                    st.markdown("##### 📈 Lambda Maxima Comparison")
                    c_m1, c_m2 = st.columns(2)
                    with c_m1:
                        style_max_wl = render_plot_editor("stat_max_wl", "Lambda Maxima", "#008000") 
                        fig_max1 = go.Figure(data=[go.Bar(
                            x=df_st.index, 
                            y=df_st["Lambda Max (nm)"],
                            text=df_st["Lambda Max (nm)"].astype(str) + " nm", # Adds text to bars
                            textposition='auto'
                        )])
                        fig_max1 = apply_plot_style_custom(fig_max1, style_max_wl, "Lambda Maxima", "Sample", "Wavelength (nm)")
                        if not style_max_wl["y_range"]: fig_max1.update_yaxes(range=[180, 260])
                        st.plotly_chart(fig_max1)
                        try: st.download_button("DL Lambda Max (JPG)", fig_max1.to_image(format="jpg", scale=3), "lambda_max.jpg", "image/jpeg", key="dl_max_wl")
                        except: pass

                    with c_m2:
                        style_max_v = render_plot_editor("stat_max_v", "Max Intensity", "#FFA500")
                        fig_max2 = go.Figure(data=[go.Bar(
                            x=df_st.index, 
                            y=df_st["Max Value"],
                            text=df_st["Max Value"].astype(str), # Adds text to bars
                            textposition='auto'
                        )])
                        fig_max2 = apply_plot_style_custom(fig_max2, style_max_v, "Max Intensity", "Sample", "Signal")
                        st.plotly_chart(fig_max2)
                        try: st.download_button("DL Max Intensity (JPG)", fig_max2.to_image(format="jpg", scale=3), "max_intensity.jpg", "image/jpeg", key="dl_max_v")
                        except: pass

                    # Batch export: 2x2 combined statistics panel
                    st.divider()
                    st.markdown("##### 📦 Download All Four Spectra as a Combined Panel")
                    st.caption("Generates a single high-resolution 2×2 panel combining Lambda Minima, Min Intensity, Lambda Maxima, and Max Intensity.")

                    try:
                        from plotly.subplots import make_subplots as _make_subplots_stats

                        fig_all_stats = _make_subplots_stats(
                            rows=2, cols=2,
                            subplot_titles=[
                                "<b>Primary Lambda Minima</b>",
                                "<b>Primary Min Intensity</b>",
                                "<b>Lambda Maxima</b>",
                                "<b>Max Intensity</b>"
                            ],
                            vertical_spacing=0.20,
                            horizontal_spacing=0.14
                        )

                        # --- Row 1 Col 1: Lambda Min ---
                        fig_all_stats.add_trace(go.Bar(
                            x=df_st.index, y=df_st["Lambda Min 1 (nm)"],
                            text=(df_st["Lambda Min 1 (nm)"].astype(str) + " nm"),
                            textposition='auto',
                            marker_color="#0000FF", showlegend=False
                        ), row=1, col=1)

                        # --- Row 1 Col 2: Min Intensity ---
                        fig_all_stats.add_trace(go.Bar(
                            x=df_st.index, y=df_st["Min 1 Value"],
                            text=df_st["Min 1 Value"].round(2).astype(str),
                            textposition='auto',
                            marker_color="#FF0000", showlegend=False
                        ), row=1, col=2)

                        # --- Row 2 Col 1: Lambda Max ---
                        fig_all_stats.add_trace(go.Bar(
                            x=df_st.index, y=df_st["Lambda Max (nm)"],
                            text=(df_st["Lambda Max (nm)"].astype(str) + " nm"),
                            textposition='auto',
                            marker_color="#008000", showlegend=False
                        ), row=2, col=1)

                        # --- Row 2 Col 2: Max Intensity ---
                        fig_all_stats.add_trace(go.Bar(
                            x=df_st.index, y=df_st["Max Value"],
                            text=df_st["Max Value"].round(2).astype(str),
                            textposition='auto',
                            marker_color="#FFA500", showlegend=False
                        ), row=2, col=2)

                        # Y-axis labels
                        fig_all_stats.update_yaxes(title_text="Wavelength (nm)", row=1, col=1)
                        fig_all_stats.update_yaxes(title_text=y_axis_label, row=1, col=2)
                        fig_all_stats.update_yaxes(title_text="Wavelength (nm)", row=2, col=1)
                        fig_all_stats.update_yaxes(title_text=y_axis_label, row=2, col=2)

                        # Shared styling
                        fig_all_stats.update_layout(
                            height=700, width=900,
                            template=_c["template"],
                        paper_bgcolor=_c["paper"],
                        plot_bgcolor=_c["bg"],
                            title=dict(
                                text="<b>Spectral Statistics Summary</b>",
                                x=0.5, xanchor="center",
                                font=dict(family="Arial", size=20, color=_c["text"])
                            ),
                            font=dict(family="Arial", size=13, color=_c["text"]),
                            margin=dict(l=70, r=30, t=80, b=60)
                        )
                        fig_all_stats.update_xaxes(showline=True, linewidth=1.5, linecolor=_c["line"], mirror=True, tickangle=-30)
                        fig_all_stats.update_yaxes(showline=True, linewidth=1.5, linecolor=_c["line"], mirror=True, showgrid=True, gridcolor=_c["grid"])

                        # Preview
                        st.plotly_chart(fig_all_stats, use_container_width=True)

                        # Download buttons
                        c_batch1, c_batch2 = st.columns(2)
                        try: c_batch1.download_button("📸 Download All 4 Plots (PNG)", fig_all_stats.to_image(format="png", scale=3), "stats_all4_panel.png", "image/png", key="dl_all4_png")
                        except: pass
                        try: c_batch2.download_button("📄 Download All 4 Plots (PDF)", fig_all_stats.to_image(format="pdf"), "stats_all4_panel.pdf", "application/pdf", key="dl_all4_pdf")
                        except: pass

                    except Exception as e:
                        st.warning(f"Could not generate combined panel: {e}")

                de = pd.DataFrame({"Wavelength": stat_grid})
                for p in final_curves: de[p["name"]] = p["stat_sig"]
                st.download_button("Download Full CSV", de.to_csv(index=False).encode('utf-8'), "cd_data.csv", "text/csv")
                
                # Literature discovery: generate Google Scholar query from primary minimum
                if stats_rows:
                    st.divider()
                    st.markdown("##### 📚 Literature Discovery")
                    st.caption("Automatically search Google Scholar for peptides with similar primary spectral minima.")
                    
                    search_target = st.selectbox("Select sample to query:", [s["Sample"] for s in stats_rows], key="lit_search_sel")
                    target_data = next(s for s in stats_rows if s["Sample"] == search_target)
                    
                    min_wl = target_data.get("Lambda Min 1 (nm)")
                    
                    if min_wl is not None and not pd.isna(min_wl):
                        low_bound = int(min_wl - 3)
                        high_bound = int(min_wl + 3)
                        
                        st.info(f"Target Primary Minima: **{min_wl} nm**. Searching literature for range: **{low_bound} nm to {high_bound} nm**.")
                        
                        wl_terms = " OR ".join([f'"{wl} nm"' for wl in range(low_bound, high_bound + 1)])
                        query = f'"circular dichroism" peptide ({wl_terms})'
                        
                        import urllib.parse
                        safe_query = urllib.parse.quote(query)
                        url = f"https://scholar.google.com/scholar?q={safe_query}"
                        
                        st.markdown(
                            f'<a href="{url}" target="_blank" style="display: inline-block; padding: 0.6em 1.2em; color: white; background-color: #4CAF50; border-radius: 5px; text-decoration: none; font-weight: bold; text-align: center;">🔍 Search Google Scholar for Similar Spectra</a>', 
                            unsafe_allow_html=True
                        )
            
            if _ga_tab == "🧩 Sec. Structure":
                st.subheader("🧩 Secondary Structure Estimation")
                
                st.warning("""
                **⚠️ Qualitative Screening Tool:**
                This module provides two methods of estimation:
                1. **NNLS Deconvolution:** A whole-spectrum fit against basis reference spectra.
                2. **Empirical Estimation:** A quick single-point mathematical approximation based on 222 nm (Helix) and 217 nm (Sheet).
                """)

                if metric == "Δε (M⁻¹cm⁻¹)":
                    st.info("ℹ️ **Δε input detected.** For secondary structure estimation the software automatically converts Δε → MRE using the relation [θ] = Δε × 3298.2 before applying empirical formulas and NNLS basis spectra.")
                elif metric not in ["MRE", "Δε (M⁻¹cm⁻¹)"]:
                    st.warning("⚠️ Secondary structure estimation is most meaningful when the output metric is **MRE** or **Δε**. Results in raw mdeg / HT / Abs may be unreliable.")
                
                apply_chen = st.checkbox(
                    "Apply Chen Chain-Length Correction to NNLS", 
                    value=False, 
                    help="Only appropriate for short helical peptides (< 30 residues)."
                )
                
                nnls_results = []
                empirical_results = []
                
                for p in final_curves:
                    if len(p["wl"]) > 0:
                        nres = p.get("nres", 20)
                        
                        from scipy.interpolate import interp1d
                        from scipy.optimize import nnls
                        
                        # For D-amino acid peptides the measured signal has opposite sign
                        # to L-peptide basis spectra. Flip internally for structure maths only.
                        # The displayed plot (p["sig"]) is never modified.
                        _struct_sig = -p["sig"] if p.get("is_d_peptide", False) else p["sig"]
                        f_int = interp1d(p["wl"], _struct_sig, bounds_error=False, fill_value=0)
                        
                        # Full-spectrum NNLS deconvolution against basis matrix
                        dynamic_helix = REF_HELIX.copy()
                        if apply_chen and 4 <= nres < 30:
                            correction_factor = 1.0 - (2.57 / nres)
                            dynamic_helix = REF_HELIX * correction_factor
                        
                        dynamic_matrix = np.vstack([dynamic_helix, REF_SHEET, REF_COIL]).T
                        sig_interp = f_int(REF_WL)
                        
                        x, rnorm = nnls(dynamic_matrix, sig_interp)
                        total = np.sum(x)
                        fracs = (x / total) * 100 if total > 0 else [0, 0, 0]
                        
                        nnls_results.append({
                            "Sample": p["name"],
                            "Residues": nres,
                            "Alpha Helix (%)": round(fracs[0], 1),
                            "Beta Sheet (%)": round(fracs[1], 1),
                            "Random Coil (%)": round(fracs[2], 1)
                        })
                        
                        # Single-point empirical estimates (222 nm helix, 217 nm sheet)
                        val_222 = f_int(222)
                        val_217 = f_int(217)
                        
                        # All empirical thresholds (-33 000, -30 000) are in full MRE units.
                        # We must back-convert from the current stored scale.
                        if metric == "MRE":
                            # Stored as ×10⁻³ → restore
                            true_mre_222 = val_222 * 1000
                            true_mre_217 = val_217 * 1000
                        elif metric == "Δε (M⁻¹cm⁻¹)":
                            # Per-residue Δε → MRE full units
                            true_mre_222 = val_222 * 3298.2
                            true_mre_217 = val_217 * 3298.2
                        else:
                            # Raw mdeg / HT / Abs — not meaningful, pass through
                            true_mre_222 = val_222
                            true_mre_217 = val_217
                            
                        # Standard Alpha Helix Formula (Theoretical Max: -33,000 at 222nm)
                        empirical_helix = (true_mre_222 / -33000.0) * 100.0
                        
                        # Standard Beta Sheet Formula (Theoretical Max: ~ -30,000 at 217nm)
                        empirical_sheet = (true_mre_217 / -30000.0) * 100.0
                        
                        # Safety boundaries (Cap between 0% and 100%)
                        empirical_helix = max(0.0, min(100.0, empirical_helix))
                        empirical_sheet = max(0.0, min(100.0, empirical_sheet))
                        
                        empirical_results.append({
                            "Sample": p["name"],
                            "Empirical Helix (222nm %)": round(empirical_helix, 1),
                            "Empirical Sheet (217nm %)": round(empirical_sheet, 1)
                        })
                        
                if nnls_results and empirical_results:
                    # Render NNLS Table
                    st.markdown("##### 🧬 Table 1: Full-Spectrum Deconvolution (NNLS)")
                    df_nnls = pd.DataFrame(nnls_results).set_index("Sample")
                    st.dataframe(df_nnls, use_container_width=True)
                    
                    # Render Empirical Table
                    st.markdown("##### 🧮 Table 2: Single-Point Empirical Approximations")
                    st.caption("*Note: Single-point beta-sheet approximations are highly qualitative and easily skewed by overlapping helical signals.*")
                    df_emp = pd.DataFrame(empirical_results).set_index("Sample")
                    st.dataframe(df_emp, use_container_width=True)
                    
                    # Bar Chart (Using NNLS data as it is more accurate)
                    # Render structural composition bar charts
                    st.markdown("##### 📊 Structural Distribution Plots")
                    
                    # Create two columns for the side-by-side graphs
                    col_plot1, col_plot2 = st.columns(2)
                    
                    # 1. NNLS Plot (Stacked Bar)
                    fig_nnls = go.Figure()
                    colors_nnls = ['#1f77b4', '#ff7f0e', '#2ca02c']
                    columns_nnls = ["Alpha Helix (%)", "Beta Sheet (%)", "Random Coil (%)"]
                    
                    for idx, col in enumerate(columns_nnls):
                        fig_nnls.add_trace(go.Bar(
                            name=col.replace(" (%)", ""), 
                            x=df_nnls.index, 
                            y=df_nnls[col],
                            marker_color=colors_nnls[idx],
                            text=df_nnls[col],
                            texttemplate='%{text:.1f}%', # Adds the % sign to the number
                            textposition='auto',        # Smart placement (inside or outside)
                            textfont=dict(size=14)
                        ))
                        
                    fig_nnls.update_layout(
                        title=dict(text="<b>Full-Spectrum NNLS Deconvolution</b>", x=0.5, y=0.98, xanchor='center'),
                        barmode='stack', 
                        template="plotly_white",
                        # Fix Legend Overlap: Centered and pushed down slightly while margin is increased
                        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),
                        margin=dict(l=60, r=20, t=100, b=40), 
                        font=dict(family="Arial", size=14, color=_c["text"]),
                        plot_bgcolor=_c["bg"]
                    )
                    
                    # Professional Scientific Borders (NNLS)
                    fig_nnls.update_xaxes(
                        showline=True, linewidth=2, linecolor=_c["line"], mirror=True, 
                        tickfont=dict(size=14, family="Arial", color=_c["text"])
                    )
                    fig_nnls.update_yaxes(
                        title_text="<b>Percentage (%)</b>", 
                        title_font=dict(size=16, family="Arial", color=_c["text"]),
                        showline=True, linewidth=2, linecolor=_c["line"], mirror=True,
                        showgrid=True, gridcolor=_c["grid"], range=[0, 105],
                        tickfont=dict(size=14, family="Arial", color=_c["text"])
                    )
                    
                    # 2. Empirical Plot (Grouped Bar)
                    fig_emp = go.Figure()
                    colors_emp = ['#9467bd', '#8c564b']
                    columns_emp = ["Empirical Helix (222nm %)", "Empirical Sheet (217nm %)"]
                    
                    for idx, col in enumerate(columns_emp):
                        # Clean up the names for the legend
                        clean_name = col.replace("Empirical ", "").replace(" %)", "").replace("(", "")
                        fig_emp.add_trace(go.Bar(
                            name=clean_name, 
                            x=df_emp.index, 
                            y=df_emp[col],
                            marker_color=colors_emp[idx],
                            text=df_emp[col],
                            texttemplate='%{text:.1f}%', # Adds the % sign to the number
                            textposition='auto',        # Smart placement
                            textfont=dict(size=14)
                        ))
                        
                    fig_emp.update_layout(
                        title=dict(text="<b>Single-Point Empirical Estimation</b>", x=0.5, y=0.98, xanchor='center'),
                        barmode='group', 
                        template="plotly_white",
                        # Fix Legend Overlap
                        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),
                        margin=dict(l=60, r=20, t=100, b=40), 
                        font=dict(family="Arial", size=14, color=_c["text"]),
                        plot_bgcolor=_c["bg"]
                    )
                    
                    # Professional Scientific Borders (Empirical)
                    fig_emp.update_xaxes(
                        showline=True, linewidth=2, linecolor=_c["line"], mirror=True,
                        tickfont=dict(size=14, family="Arial", color=_c["text"])
                    )
                    
                    # Dynamically calculate the maximum Y-value for the Empirical plot to ensure labels fit
                    emp_max_val = df_emp.max().max() if not df_emp.empty else 100
                    y_max_emp = min(110, emp_max_val + 15) # Gives 15% breathing room above the tallest bar for the text label
                    
                    fig_emp.update_yaxes(
                        title_text="<b>Percentage (%)</b>",
                        title_font=dict(size=16, family="Arial", color=_c["text"]),
                        showline=True, linewidth=2, linecolor=_c["line"], mirror=True,
                        showgrid=True, gridcolor=_c["grid"], range=[0, y_max_emp],
                        tickfont=dict(size=14, family="Arial", color=_c["text"])
                    )
                    
                    # Render Plots & Download Buttons
                    with col_plot1:
                        st.plotly_chart(fig_nnls, use_container_width=True)
                        st.download_button("💾 Download NNLS Data (CSV)", df_nnls.to_csv().encode('utf-8'), "nnls_structure.csv", "text/csv")
                        try:
                            c1, c2 = st.columns(2)
                            c1.download_button("📸 DL NNLS PNG", fig_nnls.to_image(format="png", scale=3), "nnls_plot.png")
                            c2.download_button("📄 DL NNLS PDF", fig_nnls.to_image(format="pdf"), "nnls_plot.pdf")
                        except: pass

                    with col_plot2:
                        st.plotly_chart(fig_emp, use_container_width=True)
                        st.download_button("💾 Download Empirical Data (CSV)", df_emp.to_csv().encode('utf-8'), "empirical_structure.csv", "text/csv")
                        try:
                            c1, c2 = st.columns(2)
                            c1.download_button("📸 DL Empirical PNG", fig_emp.to_image(format="png", scale=3), "empirical_plot.png")
                            c2.download_button("📄 DL Empirical PDF", fig_emp.to_image(format="pdf"), "empirical_plot.pdf")
                        except: pass

                # Export formatter for BeStSel and DichroWeb external servers
                st.divider()
                st.markdown("##### 🌐 Publication-Grade Export (BeStSel & DichroWeb)")
                st.markdown("""
                Top-tier journals often require deconvolution via established algorithms. Both **BeStSel** and **DichroWeb** require a strict two-column text format, sorted in descending wavelength order. Use this tool to instantly format your data for upload.
                """)
                
                c_ext1, c_ext2 = st.columns([1, 1])
                
                with c_ext1:
                    st.info("**Step 1: Download Formatted Data**")
                    if final_curves:
                        export_target = st.selectbox("Select sample to format:", [p["name"] for p in final_curves if len(p["wl"]) > 0], key="bestsel_sel")
                        
                        if export_target:
                            p_target = next(p for p in final_curves if p["name"] == export_target)
                            
                            # Zip the wavelengths and signals together, then sort by wavelength (x[0]) in reverse
                            sorted_data = sorted(zip(p_target["wl"], p_target["sig"]), key=lambda x: x[0], reverse=True)
                            
                            bestsel_str = ""
                            for w, s in sorted_data:
                                # Multiply by 1000 ONLY if the user is in MRE mode
                                export_val = s * 1000 if metric == "MRE" else s
                                bestsel_str += f"{w:.2f}\t{export_val:.4f}\n"
                                
                            st.download_button(
                                label=f"⬇️ Download {export_target} File",
                                data=bestsel_str.encode('utf-8'),
                                file_name=f"{export_target}_Export.txt",
                                mime="text/plain",
                                help="Perfectly formatted for DichroWeb (Free Format) or BeStSel."
                            )
                            
                            if sorted_data:
                                wl_max = sorted_data[0][0]  
                                wl_min = sorted_data[-1][0] 
                                n_points = len(sorted_data)
                                st.caption(f"**Export Info:** Wavelength range {wl_min:.1f} to {wl_max:.1f} nm ({n_points} points).")
                
                with c_ext2:
                    st.success("**Step 2: Upload to Server**")
                    st.markdown("Click a link below to open your preferred server, then upload the `.txt` file.")
                    
                    if metric == "MRE":
                        st.warning("⚠️ **CRITICAL FOR BESTSEL:** BeStSel's default unit is `mdeg`. Because your exported file contains standard MRE data, you **MUST** change the 'Data units' dropdown on their website to **Mean residue ellipticity** before hitting submit!")
                    elif metric in ["Raw (mdeg)", "CD"]:
                        st.info("ℹ️ **BESTSEL UNIT:** You are exporting raw CD data. You can safely leave the BeStSel 'Data units' dropdown on its default setting (`mdeg`).")
                    
                    st.markdown(
                        '<a href="https://bestsel.elte.hu/index.php" target="_blank" style="display: inline-block; padding: 0.5em 1em; color: white; background-color: #007BFF; border-radius: 5px; text-decoration: none; font-weight: bold; margin-bottom: 10px;">🔗 Open BeStSel Server</a>', 
                        unsafe_allow_html=True
                    )
                    
                    st.markdown(
                        '<a href="http://dichroweb.cryst.bbk.ac.uk/html/home.shtml" target="_blank" style="display: inline-block; padding: 0.5em 1em; color: white; background-color: #28a745; border-radius: 5px; text-decoration: none; font-weight: bold;">🔗 Open DichroWeb Server</a>', 
                        unsafe_allow_html=True
                    )
                    st.caption("*Note: DichroWeb requires a free academic account. Select 'Free Format'.*")
                
                st.divider()
                with st.expander("📚 Citations — Required if using these servers in publication"):
                    st.markdown("""
                    **CRITICAL - Internal Basis Spectra Source:**
                    * Brahms, S., & Brahms, J. G. (1980). Determination of protein secondary structure in solution by vacuum ultraviolet circular dichroism. *Journal of Molecular Biology*, 138(2), 149-178.
                    * Greenfield, N. J., & Fasman, G. D. (1969). Computed circular dichroism spectra for the evaluation of protein conformation. *Biochemistry*, 8(10), 4108-4116.

                    **If you use the optional Length-Corrected NNLS:**
                    * Chen, Y. H., Yang, J. T., & Chau, K. H. (1974). Determination of the secondary structures of proteins by circular dichroism and optical rotatory dispersion. *Biochemistry*, 13(16), 3350-3359.
                    
                    **If you export to BeStSel:**
                    * Micsonai, A., et al. (2022). BeStSel web server: secondary structure and fold prediction for circular dichroism spectroscopy. *Nucleic Acids Research*, 50(W1), W90-W98.
                    """)
            if _ga_tab == "🔗 Similarity":
                st.subheader("📈 Statistical & Similarity Analysis")

                if len(final_curves) < 2:
                    st.warning("⚠️ You need at least 2 samples to perform correlation and clustering analysis.")
                else:
                    # 1. Prepare standardized data matrix
                    common_wl = np.arange(190, 261, 1)
                    sample_names = []
                    sig_matrix = []
                    
                    from scipy.interpolate import interp1d
                    for p in final_curves:
                        if len(p["wl"]) > 0:
                            f_int = interp1d(p["wl"], p["sig"], bounds_error=False, fill_value=0)
                            sig_matrix.append(f_int(common_wl))
                            sample_names.append(p["name"])
                            
                    sig_matrix = np.array(sig_matrix)
                    
                    if len(sample_names) >= 2:
                        c1, c2 = st.columns(2)
                        
                        # Pearson correlation heatmap
                        with c1:
                            st.markdown("##### Pearson Correlation Heatmap")
                            corr_matrix = np.corrcoef(sig_matrix)
                            
                            fig_corr = go.Figure(data=go.Heatmap(
                                z=corr_matrix,
                                x=sample_names,
                                y=sample_names,
                                colorscale='RdBu',
                                zmin=-1, zmax=1,
                                text=np.round(corr_matrix, 2),
                                texttemplate="<b>%{text}</b>",
                                hoverinfo="z",
                                showscale=True
                            ))
                            
                            # Apply strict publication styling (Bold borders, black fonts)
                            fig_corr.update_layout(
                                template="plotly_white", 
                                margin=dict(l=20, r=20, t=30, b=20),
                                width=400, height=400,
                                font=dict(family="Arial", size=14, color=_c["text"])
                            )
                            fig_corr.update_xaxes(showline=True, linewidth=2, linecolor=_c["line"], mirror=True, tickfont=dict(color=_c["text"], weight='bold'))
                            fig_corr.update_yaxes(showline=True, linewidth=2, linecolor=_c["line"], mirror=True, tickfont=dict(color=_c["text"], weight='bold'))
                            
                            st.plotly_chart(fig_corr, use_container_width=True)
                            
                        # Hierarchical clustering dendrogram
                        with c2:
                            st.markdown("##### Hierarchical Clustering (Dendrogram)")
                            import plotly.figure_factory as ff
                            fig_dendro = ff.create_dendrogram(sig_matrix, labels=sample_names, orientation='left')
                            
                            # Apply strict publication styling
                            fig_dendro.update_layout(
                                template="plotly_white",
                                width=400, height=400,
                                margin=dict(l=20, r=20, t=30, b=20),
                                xaxis_title="<b>Distance (Dissimilarity)</b>",
                                yaxis_title="<b>Samples</b>",
                                font=dict(family="Arial", size=14, color=_c["text"])
                            )
                            fig_dendro.update_xaxes(showline=True, linewidth=2, linecolor=_c["line"], mirror=True, tickfont=dict(color=_c["text"]))
                            fig_dendro.update_yaxes(showline=True, linewidth=2, linecolor=_c["line"], mirror=True, tickfont=dict(color=_c["text"], weight='bold'))
                            
                            st.plotly_chart(fig_dendro, use_container_width=True)
                            
                        st.markdown("<br>", unsafe_allow_html=True) # Add a little spacing
                        
                        try:
                            # Generate high-res image bytes using Kaleido
                            img_corr = fig_corr.to_image(format="png", scale=3)
                            img_dendro = fig_dendro.to_image(format="png", scale=3)
                            
                            col_btn1, col_btn2 = st.columns(2)
                            with col_btn1:
                                st.download_button(label="📸 Download Heatmap (High-Res PNG)", data=img_corr, file_name="Correlation_Heatmap.png", mime="image/png", use_container_width=True)
                            with col_btn2:
                                st.download_button(label="📸 Download Dendrogram (High-Res PNG)", data=img_dendro, file_name="Clustering_Dendrogram.png", mime="image/png", use_container_width=True)
                                
                        except Exception as e:
                            st.info("💡 *Note: To enable direct image download buttons, please install the image exporter by running `pip install kaleido` in your terminal.*")

                        st.divider()
                        with st.expander("💡 How to interpret these charts"):
                            st.info("""
                            **Pearson Correlation (Heatmap):** Measures the linear similarity of the overall curve shapes.
                            * **1.0**: Perfect structural match (identical shape).
                            * **> 0.9**: Highly similar spectral signatures (e.g., both are predominantly alpha-helical).
                            * **~ 0.0**: No linear similarity between the structures.
                            * **Negative (< 0)**: The curves are inverted relative to one another.
                              
                            **Dendrogram (Hierarchical Clustering):** Groups your samples into "families" based on mathematical distance across the whole spectrum.
                            * **Short branches** (samples connected very close to the left side) indicate samples that are highly structurally similar.
                            * **Distinct major clusters** (branches that split far to the right) indicate major structural differences. This is highly useful for proving that a specific subset of mutants or solvent conditions forces the peptide into a completely different conformational state.
                            """)


            if _ga_tab == "🗺️ Spectral Projection":
                st.subheader("🗺️ Multi-Sample Spectral Projection")
                st.markdown(
                    "This composite visualization compares multiple samples using the discrete projection style. "
                    "The top panel displays stacked, baseline-resolved CD spectra, while the bottom panel "
                    "projects the exact same data as discrete color-coded intensity strips. "
                    "**Highly recommended for comparing wild-type vs mutants, or varying solvent conditions.**"
                )

                # Filter samples that actually have data
                proj_samples = [p for p in final_curves if len(p["wl"]) > 0]
                
                if len(proj_samples) < 2:
                    st.info("👈 Please select and process at least 2 samples to generate a multi-sample projection.")
                else:
                    # 1. UI CONTROLS
                    st.markdown("##### 🛠️ Plot Controls")
                    c_p1, c_p2, c_p3, c_p4 = st.columns(4)
                    with c_p1:
                        colorscale_ga = st.selectbox(
                            "Bottom Color Scale", 
                            ["RdBu_r", "PRGn_r", "PiYG_r", "PuOr_r", "Viridis"], 
                            index=0, key="ga_proj_cs",
                            help="Select 'PRGn_r' for the classic Purple/Green publication aesthetic."
                        )
                    with c_p2:
                        symmetric_ga = st.checkbox("Symmetric Range (±)", value=True, key="ga_proj_sym")
                    with c_p3:
                        # Calculate amplitude for dynamic spacing
                        max_sigs = [np.max(p["sig"]) for p in proj_samples]
                        min_sigs = [np.min(p["sig"]) for p in proj_samples]
                        amplitude_ga = max(max_sigs) - min(min_sigs) if max_sigs else 10
                        spacing_ga = st.slider("Top Panel Spacing", 0.0, float(amplitude_ga), float(amplitude_ga * 0.4), step=float(amplitude_ga*0.05), key="ga_proj_spc")
                    with c_p4:
                        default_ht_ga = max(600, 200 + len(proj_samples) * 45)
                        fig_height_ga = st.number_input("Figure Height (px)", 500, 2000, default_ht_ga, step=50, key="ga_proj_ht")

                    # 2. DATA INTERPOLATION (Map all samples to a common wavelength grid)
                    common_wl_grid = np.arange(wl_min, wl_max + 1, 1)
                    z_matrix_ga = []
                    sample_names_ga = []
                    
                    # Reverse the list so the first sample appears at the TOP of the stack
                    for p in reversed(proj_samples):
                        f_int = interp1d(p["wl"], p["sig"], bounds_error=False, fill_value=0)
                        z_matrix_ga.append(f_int(common_wl_grid))
                        sample_names_ga.append(p["name"])
                        
                    z_arr_ga = np.array(z_matrix_ga)

                    # 3. BUILD THE COMPOSITE FIGURE
                    fig_comp_ga = make_subplots(
                        rows=2, cols=1, 
                        shared_xaxes=True, 
                        row_heights=[0.7, 0.3], 
                        vertical_spacing=0.03
                    )
                    
                    tick_vals_ga = []
                    
                    # Color Logic (Match PRGn if selected)
                    pos_col_line = 'rgba(214, 39, 40, 1)'   # Red
                    pos_col_fill = 'rgba(214, 39, 40, 0.4)'
                    neg_col_line = 'rgba(31, 119, 180, 1)'  # Blue
                    neg_col_fill = 'rgba(31, 119, 180, 0.4)'
                    
                    if colorscale_ga == "PRGn":
                        pos_col_line = 'rgba(118, 42, 131, 1)'   # Purple
                        pos_col_fill = 'rgba(118, 42, 131, 0.5)'
                        neg_col_line = 'rgba(27, 120, 55, 1)'    # Green
                        neg_col_fill = 'rgba(27, 120, 55, 0.5)'

                    # TOP PANEL: Ridgeline Plot
                    for i, s_name in enumerate(sample_names_ga):
                        offset = i * spacing_ga
                        tick_vals_ga.append(offset)
                        
                        sig_grid = z_arr_ga[i]
                        y_pos = np.where(sig_grid > 0, sig_grid, 0) + offset
                        y_neg = np.where(sig_grid < 0, sig_grid, 0) + offset
                        
                        # Fills
                        fig_comp_ga.add_trace(go.Scatter(x=common_wl_grid, y=np.full_like(common_wl_grid, offset), mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'), row=1, col=1)
                        fig_comp_ga.add_trace(go.Scatter(x=common_wl_grid, y=y_pos, mode='lines', line=dict(color=pos_col_line, width=1.5), fill='tonexty', fillcolor=pos_col_fill, name=f"{s_name} (+)", showlegend=False), row=1, col=1)
                        
                        fig_comp_ga.add_trace(go.Scatter(x=common_wl_grid, y=np.full_like(common_wl_grid, offset), mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'), row=1, col=1)
                        fig_comp_ga.add_trace(go.Scatter(x=common_wl_grid, y=y_neg, mode='lines', line=dict(color=neg_col_line, width=1.5), fill='tonexty', fillcolor=neg_col_fill, name=f"{s_name} (-)", showlegend=False), row=1, col=1)
                        
                        # Baseline
                        fig_comp_ga.add_trace(go.Scatter(x=[common_wl_grid[0], common_wl_grid[-1]], y=[offset, offset], mode='lines', line=dict(color=_c["text"], width=1, dash='solid'), opacity=0.2, showlegend=False, hoverinfo='skip'), row=1, col=1)

                    # BOTTOM PANEL: Discrete Heatmap
                    zmin_val = float(np.nanmin(z_arr_ga))
                    zmax_val = float(np.nanmax(z_arr_ga))
                    if symmetric_ga:
                        z_abs = max(abs(zmin_val), abs(zmax_val))
                        zmin_val, zmax_val = -z_abs, z_abs

                    fig_comp_ga.add_trace(go.Heatmap(
                        z=z_arr_ga, x=common_wl_grid, y=sample_names_ga,
                        colorscale=colorscale_ga, zmin=zmin_val, zmax=zmax_val,
                        colorbar=dict(
                            title=dict(text=f"<b>{y_axis_label.split('(')[0].strip()}</b>", side="right", font=dict(size=12, family="Arial", color=_c["text"])),
                            tickfont=dict(size=11, family="Arial", color=_c["text"]),
                            thickness=15, len=0.35, y=0.15 
                        ),
                        xgap=0, ygap=3 
                    ), row=2, col=1)

                    # STYLING & LAYOUT
                    fig_comp_ga.update_layout(
                        title=dict(text="<b>Multi-Sample Spectral Projection</b>", x=0.5, font=dict(family="Arial", size=18, color=_c["text"])),
                        template=_c["template"],
                        paper_bgcolor=_c["paper"],
                        height=fig_height_ga, margin=dict(l=100, r=40, t=60, b=60)
                    )
                    
                    fig_comp_ga.update_xaxes(showline=True, linewidth=2, linecolor=_c["line"], mirror=True, showgrid=False, range=[wl_min, wl_max], row=1, col=1)
                    fig_comp_ga.update_xaxes(title_text="<b>Wavelength (nm)</b>", title_font=dict(size=16, family="Arial", color=_c["text"]), tickfont=dict(size=14, family="Arial", color=_c["text"]), showline=True, linewidth=2, linecolor=_c["line"], mirror=True, showgrid=False, range=[wl_min, wl_max], row=2, col=1)
                    
                    fig_comp_ga.update_yaxes(title_text="<b>Sample</b>", tickmode='array', tickvals=tick_vals_ga, ticktext=sample_names_ga, showline=True, linewidth=2, linecolor=_c["line"], mirror=True, showgrid=False, zeroline=False, tickfont=dict(size=12, family="Arial", color=_c["text"]), row=1, col=1)
                    fig_comp_ga.update_yaxes(showline=True, linewidth=2, linecolor=_c["line"], mirror=True, showgrid=False, tickfont=dict(size=12, family="Arial", color=_c["text"]), row=2, col=1)

                    st.plotly_chart(fig_comp_ga, use_container_width=True, key="tab6_comp")

                    # EXPORT BUTTONS
                    st.markdown("##### 📥 Export Publication Plot")
                    try:
                        c_dl1, c_dl2, c_dl3 = st.columns(3)
                        c_dl1.download_button("📸 Download Plot (PNG)", fig_comp_ga.to_image(format="png", scale=3), "general_projection.png", "image/png", key="ga_cp_dl_png")
                        c_dl2.download_button("📄 Download Plot (PDF)", fig_comp_ga.to_image(format="pdf"), "general_projection.pdf", "application/pdf", key="ga_cp_dl_pdf")
                        
                        # Fix: Make sure sample names in CSV match the matrix order exactly
                        matrix_df = pd.DataFrame(z_arr_ga, index=sample_names_ga, columns=common_wl_grid.round(2))
                        c_dl3.download_button("💾 Download Matrix (CSV)", matrix_df.to_csv(), "general_projection_matrix.csv", key="ga_cp_dl_csv")
                    except: pass

    # ── MODULE 2: THERMAL ANALYSIS ───────────────────────────────────────────────
    elif mode == "Thermal Analysis":
        with st.sidebar:
            st.markdown("## 🔥 Experimental Setup")
            c_t1, c_t2 = st.columns(2)
            num_thermal = c_t1.number_input("Num Samples", 1, 10, 1)
            blank_mode_t = c_t2.radio("Blanking", ["Individual", "Common"], label_visibility="collapsed")
            
            if blank_mode_t == "Common":
                st.info("ℹ️ **Common Blank Mode:** Files uploaded below will use this SINGLE blank.")
                common_blank_t = st.file_uploader("💧 Upload Common Buffer/Blank File (.txt)", type=["txt"])
            else:
                st.info("ℹ️ **Individual Blank Mode:** Upload a separate blank for EACH sample.")
                common_blank_t = None
                
            st.divider()
            thermal_samples = []
            for i in range(num_thermal):
                with st.expander(f"Sample {i+1}", expanded=(i==0)):
                    c1, c2 = st.columns([2, 1.2])
                    n = c1.text_input("Name", f"Sample {i+1}", key=f"nt{i}")
                    c = c2.number_input("Conc (µM)", value=50.0, key=f"ct{i}")
                    
                    # --- SMART SEQUENCE PARSER (Thermal) ---
                    seq_input = st.text_input("Peptide Sequence (Optional)", key=f"seq_t_{i}", placeholder="e.g., ALYFWC...")
                    clean_seq = "".join([char.upper() for char in seq_input if char.isalpha()])
                    
                    if clean_seq:
                        r = len(clean_seq)
                        num_W = clean_seq.count('W')
                        num_Y = clean_seq.count('Y')
                        num_C = clean_seq.count('C')
                        ext_coeff = (num_W * 5500) + (num_Y * 1490) + (int(num_C / 2) * 125)
                        mw_est = (r * 110) + 18
                        
                        st.info(f"✅ **Auto-counted: {r} residues**")
                        c_seq1, c_seq2 = st.columns(2)
                        c_seq1.caption(f"🧮 **ε₂₈₀:** {ext_coeff} M⁻¹cm⁻¹")
                        c_seq2.caption(f"⚖️ **MW:** ~{mw_est} Da")
                        
                        aromatic_pct = ((num_W + num_Y) / r) * 100
                        if aromatic_pct > 15:
                            st.warning(f"⚠️ **High Aromatic Content ({aromatic_pct:.1f}%):** Trp/Tyr signals may distort the 222 nm helical band.")
                    else:
                        r = st.number_input("Or enter number of residues manually:", value=6, key=f"rt_manual_{i}")
                    # ---------------------------------------
                    st.markdown("---")
                    data_format = st.radio("Data Format", ["Multi-Column File (Jasco)", "Discrete Files (One per Temp)"], key=f"fmt{i}")
                    f, discrete_files, b = None, [], None
                    if data_format == "Multi-Column File (Jasco)":
                        f = st.file_uploader("🧬 Upload Jasco File (.txt)", key=f"ft{i}", type=["txt"])
                    else:
                        st.caption("Upload separate files for each temperature point.")
                        num_temps = st.number_input("Number of Temp Points", 2, 50, 5, key=f"ntemps{i}")
                        for j in range(num_temps):
                            c_d1, c_d2 = st.columns([1, 2.5])
                            t_val = c_d1.number_input(f"T{j+1} (°C)", value=20+j*10, key=f"tval{i}_{j}")
                            f_val = c_d2.file_uploader(f"File {j+1}", key=f"fval{i}_{j}", label_visibility="collapsed", type=["txt"])
                            if f_val: discrete_files.append({"temp": t_val, "file": f_val})
                    if blank_mode_t == "Individual": b = st.file_uploader("💧 Upload Blank File (.txt)", key=f"bt{i}", type=["txt"])
                    # ── INPUT FORMAT: AUTO-DETECT + MANUAL OVERRIDE ──────────
                    # For multi-column files detect from the thermal file itself;
                    # for discrete files detect from the first uploaded file.
                    THERMAL_FORMAT_OPTIONS = [
                        "mdeg (Raw CD Signal)",
                        "\u0394\u03b5 — Molar CD (per molecule)",
                        "\u0394\u03b5 — Mean Residue (per residue)",
                    ]
                    _detect_file = f if data_format == "Multi-Column File (Jasco)" else (discrete_files[0]["file"] if discrete_files else None)
                    t_detected_tag = detect_yunits(_detect_file)
                    t_tag_to_idx = {"mdeg": 0, "delta_eps_molar": 1, "delta_eps_residue": 2, "unknown": 0}
                    t_detected_idx = t_tag_to_idx.get(t_detected_tag, 0)
                    t_badge_map = {
                        "mdeg":             "\u2705 Auto-detected: **mdeg**",
                        "delta_eps_molar":  "\u2705 Auto-detected: **Mol. CD (\u0394\u03b5 per molecule)**",
                        "delta_eps_residue":"\u2705 Auto-detected: **Mean Residue \u0394\u03b5**",
                        "unknown":          "\u26a0\ufe0f Format not found in header — please select manually.",
                    }
                    if _detect_file is not None:
                        st.caption(t_badge_map.get(t_detected_tag, ""))
                    t_input_fmt = st.selectbox(
                        "\U0001f4e5 Input Data Format" + (" (override if needed)" if t_detected_tag != "unknown" and _detect_file is not None else ""),
                        THERMAL_FORMAT_OPTIONS,
                        index=t_detected_idx,
                        key=f"tfmt{i}",
                        help=(
                            "Auto-detected from the YUNITS line in your file header.\n\n"
                            "**mdeg:** Standard JASCO raw output.\n\n"
                            "**\u0394\u03b5 Molar:** Normalised per molecule (needs N_res for MRE).\n\n"
                            "**\u0394\u03b5 Mean Residue:** Already per residue."
                        )
                    )
                    # D-amino acid / R-chirality inversion flag.
                    # Only affects the Sec. Structure tab calculations.
                    # Overlay and all other plots are never modified.
                    t_is_d_peptide = st.checkbox(
                        "🔄 D-amino acid / inverted chirality (R-peptide)",
                        value=False,
                        key=f"t_dpep{i}",
                        help=(
                            "Tick this if the peptide uses D-amino acids or has R-chirality "
                            "that inverts the CD spectrum relative to standard L-peptide "
                            "basis spectra.\n\n"
                            "**Effect on overlay / melt plots:** None — the measured signal "
                            "is always displayed as recorded.\n\n"
                            "**Effect on Sec. Structure tab:** The signal is multiplied "
                            "by −1 internally before NNLS and empirical estimates, so the "
                            "L-peptide basis spectra and thresholds apply correctly. "
                            "The inversion is NOT applied to any exported plot or CSV."
                        )
                    )
                    thermal_samples.append({"name": n, "conc": c, "nres": r, "format": data_format,
                                            "file": f, "discrete_files": discrete_files, "blank": b,
                                            "input_fmt": t_input_fmt,
                                            "is_d_peptide": t_is_d_peptide})
            
            st.divider()
            st.markdown("### ⚙️ Processing")
            metric = st.selectbox("Output Metric", ["MRE", "Raw (mdeg)", "\u0394\u03b5 (M\u207b\u00b9cm\u207b\u00b9)", "HT", "Abs"],
                help="MRE: standard publication metric. Raw (mdeg): no normalisation. \u0394\u03b5: molar CD (auto-converted). HT/Abs: diagnostic channels.")
            path_cm = st.number_input("Pathlength (cm)", value=0.1)
            
            # Smoothing controls
            apply_smooth = st.checkbox("Apply Smoothing", value=True, help="Toggle to apply or remove all mathematical smoothing.")
            if apply_smooth:
                smooth_method = st.radio("Method", ["Savitzky-Golay", "LOWESS"], index=0)
                smooth_val = st.slider("Frac/Window", 0.01 if smooth_method == "LOWESS" else 5, 0.30 if smooth_method == "LOWESS" else 51, step=0.01 if smooth_method == "LOWESS" else 2)
            else:
                smooth_method = "None"
                smooth_val = 0

        st.title("🔥 Thermal Analysis")
        processed_datasets = {}
        c_blank_df_t = read_cd_file(common_blank_t)
        
        for s in thermal_samples:
            proc_curves, temps, wl_ref = [], [], None
            
            # 1. Multi-Column Logic (Jasco)
            if s["format"] == "Multi-Column File (Jasco)" and s["file"]:
                # Select the correct JASCO channel based on the requested metric:
                #   Channel 1 = CD (mdeg), Channel 2 = HT (V), Channel 3 = Absorbance
                _ch = 2 if metric == "HT" else (3 if metric == "Abs" else 1)
                if _ch > 1:
                    raw_df, temps = read_thermal_channel(s["file"], _ch)
                else:
                    raw_df, temps = read_thermal_file(s["file"])
                if raw_df is not None:
                    b_df = c_blank_df_t if blank_mode_t == "Common" else (read_cd_file(s["blank"]) if s["blank"] else None)
                    # Blank subtraction only applies to CD-derived metrics, not HT/Abs
                    if b_df is not None and metric not in ["HT", "Abs"]:
                        f_b = interp1d(b_df["Wavelength"], b_df["CD"], bounds_error=False, fill_value="extrapolate")
                        blank_sig = f_b(raw_df["Wavelength"])
                        for t in temps: raw_df[f"{t}"] -= blank_sig
                    wl = raw_df["Wavelength"].values
                    wl_ref = wl
                    for t in temps:
                        sig = raw_df[f"{t}"].values
                        # ── UNIT CONVERSION (multi-column) ───────────────────
                        _t_fmt   = s.get("input_fmt", "mdeg (Raw CD Signal)")
                        _is_mdeg = "mdeg" in _t_fmt
                        _is_dm   = "Molar" in _t_fmt
                        _is_dr   = "Mean Residue" in _t_fmt
                        _nres    = s["nres"] if s["nres"] > 0 else 1
                        _fac     = 10 * path_cm * (s["conc"] * 1e-6) * s["nres"]
                        if not metric in ["HT", "Abs"]:
                            if _is_mdeg:
                                if metric == "MRE":
                                    if s["conc"] == 0: continue
                                    sig = sig / _fac / 1000
                                elif metric == "\u0394\u03b5 (M\u207b\u00b9cm\u207b\u00b9)":
                                    if _fac != 0: sig = (sig / _fac / 1000) / 3298.2 * 1000
                            elif _is_dm:
                                if metric == "MRE":
                                    sig = (sig * 3298.2 / _nres) / 1000
                                elif metric == "Raw (mdeg)":
                                    sig = sig * 3298.2 / _nres * _fac
                                elif metric == "\u0394\u03b5 (M\u207b\u00b9cm\u207b\u00b9)":
                                    sig = sig / _nres
                            elif _is_dr:
                                if metric == "MRE":
                                    sig = (sig * 3298.2) / 1000
                                elif metric == "Raw (mdeg)":
                                    sig = sig * 3298.2 * _fac
                        # Preserve unsmoothed signal
                        raw_sig = sig.copy()
                        
                        if apply_smooth:
                            if smooth_method == "LOWESS": _, s_sig = apply_smoothing(wl, sig, "LOWESS (Match R)", smooth_val)
                            else: _, s_sig = apply_smoothing(wl, sig, "Savitzky-Golay", smooth_val)
                        else:
                            s_sig = sig
                            
                        proc_curves.append({"temp": t, "sig": s_sig, "raw_sig": raw_sig, "wl": wl})
                        
            # 2. Discrete Logic (One file per temp)
            elif s["format"] == "Discrete Files (One per Temp)" and s["discrete_files"]:
                b_df = c_blank_df_t if blank_mode_t == "Common" else (read_cd_file(s["blank"]) if s["blank"] else None)
                sorted_files = sorted(s["discrete_files"], key=lambda x: x["temp"])
                for item in sorted_files:
                    df = read_cd_file(item["file"])
                    if df is not None:
                        y_col = "CD"
                        if metric == "HT" and "HT" in df.columns: y_col = "HT"
                        elif metric == "Abs" and "Abs" in df.columns: y_col = "Abs"
                        elif y_col not in df.columns: continue
                        
                        wl, sig = df["Wavelength"].values, df[y_col].values
                        # Blank subtraction (applies to all signal metrics)
                        if b_df is not None and metric not in ["HT", "Abs"]:
                            f_b = interp1d(b_df["Wavelength"], b_df["CD"], bounds_error=False, fill_value="extrapolate")
                            sig -= f_b(wl)
                        # ── UNIT CONVERSION (discrete files) ────────────────
                        _t_fmt   = s.get("input_fmt", "mdeg (Raw CD Signal)")
                        _is_mdeg = "mdeg" in _t_fmt
                        _is_dm   = "Molar" in _t_fmt
                        _is_dr   = "Mean Residue" in _t_fmt
                        _nres    = s["nres"] if s["nres"] > 0 else 1
                        _fac     = 10 * path_cm * (s["conc"] * 1e-6) * s["nres"]
                        if not metric in ["HT", "Abs"]:
                            if _is_mdeg:
                                if metric == "MRE":
                                    if s["conc"] == 0: continue
                                    sig = sig / _fac / 1000
                                elif metric == "\u0394\u03b5 (M\u207b\u00b9cm\u207b\u00b9)":
                                    if _fac != 0: sig = (sig / _fac / 1000) / 3298.2 * 1000
                            elif _is_dm:
                                if metric == "MRE":
                                    sig = (sig * 3298.2 / _nres) / 1000
                                elif metric == "Raw (mdeg)":
                                    sig = sig * 3298.2 / _nres * _fac
                                elif metric == "\u0394\u03b5 (M\u207b\u00b9cm\u207b\u00b9)":
                                    sig = sig / _nres
                            elif _is_dr:
                                if metric == "MRE":
                                    sig = (sig * 3298.2) / 1000
                                elif metric == "Raw (mdeg)":
                                    sig = sig * 3298.2 * _fac
                        # Preserve unsmoothed signal
                        raw_sig = sig.copy()
                        
                        if apply_smooth:
                            if smooth_method == "LOWESS": s_wl, s_sig = apply_smoothing(wl, sig, "LOWESS (Match R)", smooth_val)
                            else: s_wl, s_sig = apply_smoothing(wl, sig, "Savitzky-Golay", smooth_val)
                        else:
                            s_wl, s_sig = wl, sig
                            
                        proc_curves.append({"temp": item["temp"], "sig": s_sig, "raw_sig": raw_sig, "wl": s_wl})
                        temps.append(item["temp"])
                        wl_ref = s_wl
                        
            if proc_curves: processed_datasets[s["name"]] = {"curves": proc_curves, "temps": temps, "wl_ref": wl_ref}

        if processed_datasets:
            with st.sidebar:
                st.markdown("---")
                log_data = generate_log_file("Thermal Analysis", thermal_samples, {"Metric": metric, "Path": path_cm})
                st.download_button("💾 Download Analysis Log", log_data, file_name="thermal_log.txt")

            st.markdown("### 📂 Select Sample")
            selected_name = st.selectbox("Choose File to Analyze", list(processed_datasets.keys()))
            curr_data = processed_datasets[selected_name]["curves"]
            curr_temps = processed_datasets[selected_name]["temps"]
            curr_wl = processed_datasets[selected_name]["wl_ref"]

            # --- CUSTOMIZATION ---
            with st.expander("🛠️ Plot Customization", expanded=True):
                c_cust1, c_cust2, c_cust3 = st.columns(3)
                actual_min_wl, actual_max_wl = int(min(curr_wl)), int(max(curr_wl))
                with c_cust1:
                    show_grid = st.checkbox("Show Grid Lines", True) 
                    line_width = st.slider("Line Width", 0.5, 5.0, 2.5, 0.5)
                with c_cust2:
                    st.markdown("**Axis Styling**")
                    fs_title = st.number_input("Title Font Size", 10, 30, 20)
                    fs_num = st.number_input("Axis Label Size", 8, 24, 16)
                    fs_leg = st.number_input("Legend Size", 8, 24, 14)
                with c_cust3:
                    wl_min = st.number_input("Min WL", 170, 350, actual_min_wl)
                    wl_max = st.number_input("Max WL", 170, 350, actual_max_wl)
                    y_auto = st.checkbox("Auto Y-Scale", True)
                    if not y_auto:
                        y_min = st.number_input("Y Min", -100, 100, -20)
                        y_max = st.number_input("Y Max", -100, 100, 20)
                    melt_wl = st.number_input("Melting WL (nm)", 170.0, 350.0, 222.0, step=0.5,
                                               key="melt_wl_input",
                                               help="Accepts decimals (e.g. 222.5). "
                                                    "Set to the Smart Suggested wavelength for best results.")

            # ── PEAK SEARCH RANGE (optional) ───────────────────
            # Sits outside the Customization expander so it is always visible.
            with st.expander("🔍 Peak Search Range (optional)", expanded=False):
                st.caption(
                    "By default the software searches for ↑ maxima and ↓ minima "
                    "across the entire visible wavelength range. If noise spikes near "
                    "195–200 nm are being picked up instead of the true 208/222 nm "
                    "helical bands, restrict the search window here."
                )
                _psr_c1, _psr_c2 = st.columns(2)
                with _psr_c1:
                    st.markdown("**↓ Minimum search window**")
                    _use_min_range = st.checkbox("Restrict minimum search", value=False, key="t_use_min_range")
                    _min_srch_lo = st.number_input("Min search: start (nm)", 170.0, 350.0, 205.0, step=1.0, key="t_min_srch_lo", disabled=not _use_min_range)
                    _min_srch_hi = st.number_input("Min search: end (nm)",   170.0, 350.0, 230.0, step=1.0, key="t_min_srch_hi", disabled=not _use_min_range)
                with _psr_c2:
                    st.markdown("**↑ Maximum search window**")
                    _use_max_range = st.checkbox("Restrict maximum search", value=False, key="t_use_max_range")
                    _max_srch_lo = st.number_input("Max search: start (nm)", 170.0, 350.0, 185.0, step=1.0, key="t_max_srch_lo", disabled=not _use_max_range)
                    _max_srch_hi = st.number_input("Max search: end (nm)",   170.0, 350.0, 205.0, step=1.0, key="t_max_srch_hi", disabled=not _use_max_range)
                st.info(
                    "💡 **Tip:** For a typical α-helix peptide set **minimum search: 205–230 nm** "
                    "(captures 208 & 222 nm bands) and **maximum search: 185–200 nm** "
                    "(captures the 192 nm positive band). "
                    "For β-sheet set minimum: 210–225 nm."
                )
            # Resolved search ranges used by get_min_max throughout this module
            _t_min_r = (_min_srch_lo, _min_srch_hi) if _use_min_range else (actual_min_wl, actual_max_wl)
            _t_max_r = (_max_srch_lo, _max_srch_hi) if _use_max_range else (actual_min_wl, actual_max_wl)

            # Set y-axis labels for selected metric
            y_lbl_full, y_lbl_short = "", ""
            if metric == "MRE":
                y_lbl_full = "MRE [θ] (x10³ deg cm² dmol⁻¹ res⁻¹)"
                y_lbl_short = "MRE [x10³]"
            elif metric == "Raw (mdeg)" or metric == "CD":
                y_lbl_full = "CD (mdeg)"
                y_lbl_short = "CD (mdeg)"
            elif metric == "Δε (M⁻¹cm⁻¹)":
                y_lbl_full = "Δε (M⁻¹ cm⁻¹)"
                y_lbl_short = "Δε"
            elif metric == "HT":
                y_lbl_full = "HT (Volts)"
                y_lbl_short = "HT (V)"
            else:
                y_lbl_full = "Absorbance (OD)"
                y_lbl_short = "Abs (OD)"

            xt = format_axis_text("Wavelength (nm)", True, False)
            yt = format_axis_text(y_lbl_full, True, False)

            st.markdown(
                '<div class="rapid-tab-header">'  
                '🌡️ &nbsp; Thermal Analysis — Select View'
                '</div>',
                unsafe_allow_html=True
            )
            _th_tab = st.radio(
                "Select View",
                ["🌈 Overlay", "🔲 Multi-Panel", "🗺️ λ–T Spectromap",
                 "📊 λ Peak Tracking", "🧩 Sec. Structure",
                 "⚗️ Thermodynamics", "🔮 Spectral Simulation"],
                horizontal=True, key="th_tab_radio",
                label_visibility="collapsed"
            )
        
            # ── TAB 1: OVERLAY (multi-sample + per-temperature colour) ──────
            if _th_tab == "🌈 Overlay":
                st.markdown("##### 🌈 Thermal CD Overlay")
                ov_col1, ov_col2 = st.columns([2, 1])
                with ov_col1:
                    ov_samples = st.multiselect(
                        "Samples to overlay",
                        list(processed_datasets.keys()),
                        default=[selected_name],
                        help=(
                            "Select one sample (standard thermal gradient view) or multiple "
                            "samples to compare them. Different line styles (solid/dashed/dotted) "
                            "distinguish samples; colour represents temperature within each sample."
                        )
                    )
                with ov_col2:
                    colour_mode = st.radio(
                        "Colour mode",
                        ["Auto gradient (blue→red)", "Manual per-temperature"],
                        key="ov_colour_mode",
                        help="Auto: blue=cold, red=hot. Manual: pick a colour for each temperature."
                    )

                if not ov_samples:
                    st.info("Select at least one sample above.")
                else:
                    # Collect all unique temperatures across selected samples
                    all_ov_temps = sorted(set(
                        d["temp"]
                        for sname in ov_samples
                        for d in processed_datasets[sname]["curves"]
                    ))

                    # ── Per-sample temperature filter ──────────────────────────
                    st.markdown("---")
                    st.markdown("##### 🌡️ Temperature Selection & Labelling")
                    st.caption(
                        "Choose which temperatures to display for each sample. "
                        "Useful when you only need to compare a subset of the recorded temperatures."
                    )

                    # Build per-sample temp lists for the filter UI
                    per_sample_temps = {
                        sname: sorted(d["temp"] for d in processed_datasets[sname]["curves"])
                        for sname in ov_samples
                    }

                    # If a single sample, show one row; if multiple, show one row per sample
                    selected_temps_per_sample = {}
                    if len(ov_samples) == 1:
                        sname = ov_samples[0]
                        available = per_sample_temps[sname]
                        chosen = st.multiselect(
                            f"Temperatures to show ({sname})",
                            options=available,
                            default=available,
                            key="ov_temp_filter_single",
                            format_func=lambda t: f"{t} °C"
                        )
                        selected_temps_per_sample[sname] = chosen if chosen else available
                    else:
                        for sname in ov_samples:
                            available = per_sample_temps[sname]
                            chosen = st.multiselect(
                                f"Temperatures — {sname}",
                                options=available,
                                default=available,
                                key=f"ov_temp_filter_{sname}",
                                format_func=lambda t: f"{t} °C"
                            )
                            selected_temps_per_sample[sname] = chosen if chosen else available

                    # Derive the union of selected temps (for colour mapping)
                    active_temps = sorted(set(
                        t
                        for sname in ov_samples
                        for t in selected_temps_per_sample[sname]
                    ))
                    
                    # --- NEW: TEMPERATURE RENAMING FOR OVERLAY ---
                    custom_labels_ov = {}
                    if active_temps:
                        with st.expander(f"✏️ Rename Temperature Labels ({len(active_temps)} active temperatures)", expanded=False):
                            st.caption("Edit the text below to cleanly format the temperatures for your plot legend and CSV export (e.g., change 14.9 to 15).")
                            ren_cols = st.columns(min(6, max(1, len(active_temps))))
                            for i, t in enumerate(active_temps):
                                with ren_cols[i % 6]:
                                    custom_labels_ov[t] = st.text_input(f"Orig: {t}", value=f"{t}", key=f"ov_lbl_rename_{t}")
                    
                    st.markdown("---")

                    # Manual colour pickers (shown in an expander to keep UI clean)
                    manual_colors = {}
                    if colour_mode == "Manual per-temperature":
                        with st.expander(
                            f"🎨 Pick colours for {len(active_temps)} selected temperature(s)",
                            expanded=True
                        ):
                            DEFAULT_PALETTE = [
                                "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
                                "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"
                            ]
                            cols_cp = st.columns(min(5, max(1, len(active_temps))))
                            for ti, t in enumerate(active_temps):
                                with cols_cp[ti % 5]:
                                    display_t = custom_labels_ov.get(t, t)
                                    manual_colors[t] = st.color_picker(
                                        f"{display_t}°C",
                                        value=DEFAULT_PALETTE[ti % len(DEFAULT_PALETTE)],
                                        key=f"ov_col_{ti}"
                                    )

                    dash_styles = ["solid", "dash", "dot", "dashdot", "longdash"]
                    fig_ov = go.Figure()

                    for s_idx, sname in enumerate(ov_samples):
                        s_data = processed_datasets[sname]["curves"]
                        # Filter to only selected temperatures for this sample
                        filtered_data = [d for d in s_data if d["temp"] in selected_temps_per_sample[sname]]
                        n_c    = len(filtered_data)
                        dash   = dash_styles[s_idx % len(dash_styles)]
                        for i, d in enumerate(filtered_data):
                            if colour_mode == "Manual per-temperature":
                                color_str = manual_colors.get(d["temp"], "#333333")
                            else:
                                frac      = i / max(1, n_c - 1)
                                r_col     = int(255 * frac)
                                b_col     = 255 - r_col
                                color_str = f"rgb({r_col}, 0, {b_col})"
                            
                            # Use custom label
                            display_temp = custom_labels_ov.get(d["temp"], d["temp"])
                            label = (f"{sname} — {display_temp}°C"
                                     if len(ov_samples) > 1 else f"{display_temp}°C")
                            
                            fig_ov.add_trace(go.Scatter(
                                x=d["wl"], y=d["sig"], mode="lines",
                                name=label,
                                line=dict(color=color_str, width=line_width, dash=dash),
                                legendgroup=sname,
                                legendgrouptitle_text=sname if len(ov_samples) > 1 else None
                            ))

                    title_str = " + ".join(ov_samples) if len(ov_samples) > 1 else ov_samples[0]
                    if len(ov_samples) > 1:
                        st.info(
                            "ℹ️ **Multi-sample overlay.** Line style distinguishes samples; "
                            "colour encodes temperature. Useful for D- vs L-peptide comparison."
                        )

                    fig_ov = apply_publication_style(
                        fig_ov, f"Thermal Overlay — {title_str}",
                        "Wavelength (nm)", y_lbl_full, show_grid
                    )
                    fig_ov.update_xaxes(range=[wl_min, wl_max])
                    if not y_auto: fig_ov.update_yaxes(range=[y_min, y_max])
                    fig_ov.add_hline(y=0, line_dash="dash", line_color=_c["text"],
                                     line_width=1, opacity=0.5)
                    st.plotly_chart(fig_ov, use_container_width=True, key="tab1_overlay")

                    if apply_smooth:
                        st.divider()
                        st.markdown("##### 🔍 Smoothing Diagnostics — Primary Sample Only")
                        st.caption(f"Showing raw vs smoothed for: **{selected_name}**")
                        fig_diag = go.Figure()
                        n_curves = len(curr_data)
                        for i, d in enumerate(curr_data):
                            # Only show selected temps for diagnostics to avoid clutter
                            if d["temp"] not in selected_temps_per_sample.get(selected_name, []):
                                continue
                                
                            r_col     = int(255 * (i / max(1, n_curves - 1)))
                            b_col     = 255 - r_col
                            color_str = f"rgb({r_col}, 0, {b_col})"
                            
                            display_temp = custom_labels_ov.get(d["temp"], d["temp"])
                            
                            fig_diag.add_trace(go.Scatter(
                                x=d["wl"], y=d["raw_sig"], mode="lines",
                                name=f"{display_temp}°C (Raw)",
                                line=dict(color=color_str, width=1.0, dash="dot"),
                                opacity=0.5
                            ))
                            fig_diag.add_trace(go.Scatter(
                                x=d["wl"], y=d["sig"], mode="lines",
                                name=f"{display_temp}°C (Smoothed)",
                                line=dict(color=color_str, width=line_width)
                            ))
                        fig_diag = apply_publication_style(
                            fig_diag, f"Raw vs. Smoothed ({selected_name})",
                            "Wavelength (nm)", y_lbl_full, show_grid, plot_mode="lines"
                        )
                        fig_diag.update_xaxes(range=[wl_min, wl_max])
                        if not y_auto: fig_diag.update_yaxes(range=[y_min, y_max])
                        fig_diag.add_hline(y=0, line_dash="dash", line_color=_c["text"],
                                           line_width=1, opacity=0.5)
                        st.plotly_chart(fig_diag, use_container_width=True, key="tab1_diag")

                    df_export = pd.DataFrame({"Wavelength (nm)": curr_wl})
                    # Export only the temperatures that are currently selected in the filter
                    _primary_selected_temps = selected_temps_per_sample.get(selected_name, [d["temp"] for d in curr_data])
                    for d in curr_data:
                        if d["temp"] not in _primary_selected_temps:
                            continue
                        f_sig = interp1d(d["wl"], d["sig"],     bounds_error=False, fill_value="extrapolate")
                        f_raw = interp1d(d["wl"], d["raw_sig"], bounds_error=False, fill_value="extrapolate")
                        
                        display_temp = custom_labels_ov.get(d["temp"], d["temp"])
                        
                        if apply_smooth:
                            df_export[f"{display_temp}°C (Raw)"]      = f_raw(curr_wl)
                            df_export[f"{display_temp}°C (Smoothed)"] = f_sig(curr_wl)
                        else:
                            df_export[f"{display_temp}°C"] = f_sig(curr_wl)
                    df_export = df_export.round(4)

                    st.markdown("##### 📥 Export")
                    try:
                        st.download_button("💾 Download Primary Sample Data (CSV)",
                                           df_export.to_csv(index=False),
                                           f"{selected_name}_thermal_data.csv")
                    except: pass
                    c_dl1, c_dl2 = st.columns(2)
                    with c_dl1:
                        st.caption("**Overlay Plot**")
                        try: c_dl1.download_button("📸 PNG", fig_ov.to_image(format="png", scale=3), f"{title_str}_overlay.png", "image/png", key="t_dl_main_png")
                        except: pass
                        try: c_dl1.download_button("📄 PDF", fig_ov.to_image(format="pdf"), f"{title_str}_overlay.pdf", "application/pdf", key="t_dl_main_pdf")
                        except: pass
                    if apply_smooth:
                        with c_dl2:
                            st.caption("**Diagnostics Plot**")
                            try: c_dl2.download_button("📸 PNG", fig_diag.to_image(format="png", scale=3), f"{selected_name}_diag.png", "image/png", key="t_dl_diag_png")
                            except: pass
                            try: c_dl2.download_button("📄 PDF", fig_diag.to_image(format="pdf"), f"{selected_name}_diag.pdf", "application/pdf", key="t_dl_diag_pdf")
                            except: pass

            # ── TAB 2: MULTI-PANEL ───────────────────────────────────────────
            if _th_tab == "🔲 Multi-Panel":
                st.markdown("##### 🔲 Multi-Panel Comparison by Temperature")

                col_mp1, col_mp2, col_mp3 = st.columns([2, 1, 1])
                with col_mp1:
                    sel_mp_samples = st.multiselect(
                        "Select Samples to Compare",
                        list(processed_datasets.keys()),
                        default=[selected_name],
                        key="mp_samples"
                    )
                with col_mp2:
                    temp_tol_mp = st.slider(
                        "Temp. match tolerance (°C)",
                        min_value=0.0, max_value=10.0, value=0.5, step=0.1,
                        key="mp_tol",
                        help="Temperatures within this window are shown in the same panel."
                    )
                with col_mp3:
                    mp_font_sz = st.number_input("Axis font size", 8, 20, 11, key="mp_fs",
                                                 help="Font size for tick labels and axis titles.")

                if not sel_mp_samples:
                    st.info("Select at least one sample above.")
                else:
                    sample_colors = {name: COLORS[i % len(COLORS)] for i, name in enumerate(sel_mp_samples)}

                    # Build temperature groups with tolerance
                    groups = []
                    for sname in sel_mp_samples:
                        for t in processed_datasets[sname]["temps"]:
                            matched = False
                            for g in groups:
                                if abs(t - g["rep"]) <= max(temp_tol_mp, 0.5):
                                    if sname not in g["actual"]:
                                        g["actual"][sname] = t
                                    matched = True
                                    break
                            if not matched:
                                groups.append({"rep": t, "actual": {sname: t}})
                    groups = sorted(groups, key=lambda g: g["rep"])

                    # Warn about merged temperatures
                    merged_warnings = []
                    for g in groups:
                        tv = list(g["actual"].values())
                        if len(tv) > 1 and (max(tv) - min(tv)) > 0.01:
                            merged_warnings.append(
                                f"**~{g['rep']:.1f} °C panel:** " +
                                ", ".join([f"{sn}={v:.2f}°C" for sn, v in g["actual"].items()])
                            )
                    if merged_warnings:
                        with st.expander(f"⚠️ {len(merged_warnings)} panel(s) contain merged temperatures", expanded=False):
                            st.caption("Spectra measured at slightly different temperatures, merged by tolerance setting.")
                            for w in merged_warnings:
                                st.markdown("• " + w)

                    if groups:
                        # ── Custom panel title editor ─────────────────────────
                        with st.expander("✏️ Customise panel titles (optional)", expanded=False):
                            st.caption(
                                "By default each panel is labelled with the measured temperature. "
                                "You can override any title here — e.g. write '15' instead of '15.2', "
                                "or add a condition label like '15 °C (native)'."
                            )
                            custom_titles = {}
                            cp_cols = st.columns(min(4, len(groups)))
                            for gi, g in enumerate(groups):
                                with cp_cols[gi % 4]:
                                    custom_titles[gi] = st.text_input(
                                        f"Panel {gi+1}",
                                        value=f"{g['rep']:.1f} °C",
                                        key=f"mp_title_{gi}"
                                    )

                        subplot_titles = [custom_titles.get(gi, f"{g['rep']:.1f} °C")
                                          for gi, g in enumerate(groups)]

                        cols_per_row = 3
                        rows_mp = int(np.ceil(len(groups) / cols_per_row))

                        # Cell sizing — wide enough to avoid label overlap
                        cell_w = 380
                        cell_h = 320
                        mg_l   = max(90, 60 + len(y_lbl_short) * 5)
                        mg_b   = 70
                        mg_t   = 50
                        mg_r   = 20

                        fig_sub = make_subplots(
                            rows=rows_mp, cols=cols_per_row,
                            subplot_titles=subplot_titles,
                            vertical_spacing=max(0.08, 0.45 / rows_mp),
                            horizontal_spacing=max(0.04, 0.35 / cols_per_row)
                        )

                        for idx, g in enumerate(groups):
                            r_idx = (idx // cols_per_row) + 1
                            c_idx = (idx % cols_per_row) + 1
                            for sname, actual_t in g["actual"].items():
                                curve_found = min(
                                    processed_datasets[sname]["curves"],
                                    key=lambda item: abs(item["temp"] - actual_t)
                                )
                                fig_sub.add_trace(
                                    go.Scatter(
                                        x=curve_found["wl"], y=curve_found["sig"],
                                        mode="lines", name=sname,
                                        legendgroup=sname, showlegend=(idx == 0),
                                        line=dict(color=sample_colors[sname], width=1.5)
                                    ),
                                    row=r_idx, col=c_idx
                                )

                        fig_sub.update_layout(
                            height=cell_h * rows_mp,
                            width=cell_w * cols_per_row,
                            template=_c["template"],
                        paper_bgcolor=_c["paper"],
                        plot_bgcolor=_c["bg"],
                            showlegend=True,
                            legend=dict(font=dict(size=mp_font_sz, family="Arial"),
                                        borderwidth=1, bordercolor=_c["legend_border"]),
                            margin=dict(l=mg_l, r=mg_r, t=mg_t, b=mg_b),
                            font=dict(family="Arial", size=mp_font_sz, color=_c["text"])
                        )

                        # Subplot titles font
                        # --- BOX SIZING & MARGINS---
                        cell_w = 380   # Width of each box
                        cell_h = 320   # Height of each box
                        mg_l   = max(90, 60 + len(y_lbl_short) * 5)
                        mg_b   = 70
                        mg_t   = 65    # INCREASED: Gives more room at the top of the whole figure
                        mg_r   = 20
                        
                        for ann in fig_sub.layout.annotations:
                            ann.font = dict(size=mp_font_sz + 1, family="Arial", color=_c["text"])
                            ann.update(yshift=10)  # NEW: Pushes the title 10 pixels UP, away from the box border!

                        # Y-axis: full label on leftmost column, short on others
                        for r_i in range(1, rows_mp + 1):
                            fig_sub.update_yaxes(
                                title_text=f"<b>{y_lbl_full}</b>",
                                title_font=dict(size=mp_font_sz, family="Arial", color=_c["text"]),
                                tickfont=dict(size=mp_font_sz - 1, family="Arial", color=_c["text"]),
                                showgrid=show_grid, gridcolor=_c["grid"],
                                showline=True, linewidth=1, linecolor=_c["line"],
                                mirror=True, row=r_i, col=1
                            )
                            # No y-title on other columns (avoids clutter)
                            for c_i in range(2, cols_per_row + 1):
                                fig_sub.update_yaxes(
                                    title_text="",
                                    tickfont=dict(size=mp_font_sz - 1, family="Arial", color=_c["text"]),
                                    showgrid=show_grid, gridcolor=_c["grid"],
                                    showline=True, linewidth=1, linecolor=_c["line"],
                                    mirror=True, row=r_i, col=c_i
                                )

                        fig_sub.update_xaxes(
                            range=[wl_min, wl_max], showgrid=show_grid,
                            showline=True, linewidth=1, linecolor=_c["line"], mirror=True,
                            tickfont=dict(size=mp_font_sz - 1, family="Arial", color=_c["text"]),
                            title_text="<b>Wavelength (nm)</b>",
                            title_font=dict(size=mp_font_sz, family="Arial", color=_c["text"])
                        )
                        # Only show x-title on bottom row
                        for r_i in range(1, rows_mp):
                            for c_i in range(1, cols_per_row + 1):
                                fig_sub.update_xaxes(title_text="", row=r_i, col=c_i)

                        if not y_auto:
                            fig_sub.update_yaxes(range=[y_min, y_max])

                        # Zero reference line always visible regardless of grid setting
                        fig_sub.add_hline(y=0, line_dash="dash", line_color=_c["text"],
                                          line_width=1, opacity=0.4)

                        st.plotly_chart(fig_sub, use_container_width=False,
                                        key="tab2_multipanel")

                        try:
                            c_mp1, c_mp2 = st.columns(2)
                            c_mp1.download_button("📸 PNG (High-res)",
                                                  fig_sub.to_image(format="png", scale=3),
                                                  "multipanel.png", "image/png",
                                                  key="mp_dl_png")
                            c_mp2.download_button("📄 PDF (Vector)",
                                                  fig_sub.to_image(format="pdf"),
                                                  "multipanel.pdf", "application/pdf",
                                                  key="mp_dl_pdf")
                        except: pass

            # ── TAB 3: SPECTRAL LANDSCAPE (COMPOSITE DISCRETE PROJECTION) ─────
            if _th_tab == "🗺️ λ–T Spectromap":
                st.subheader(f"🗺️ λ–T Spectromap — {selected_name}")
                st.markdown(
                    "This composite visualization replicates the discrete projection style used in advanced spectroscopic publications. "
                    "The top panel displays stacked, baseline-resolved CD spectra, while the bottom panel "
                    "projects the exact same data as discrete color-coded intensity bars."
                )

                unique_wl = np.sort(curr_wl)
                raw_temps = sorted(curr_temps)

                # --- NEW: TEMPERATURE SELECTION & RENAMING ---
                st.markdown("##### 🌡️ Temperature Selection & Labelling")
                
                selected_temps = st.multiselect(
                    "Select Temperatures to Include", 
                    options=raw_temps, 
                    default=raw_temps,
                    key=f"hm_tsel_{selected_name}",
                    help="Remove temperatures to reduce clutter, or select just 3-5 specific points to highlight."
                )
                
                unique_temps = sorted(selected_temps)
                custom_labels = {}
                
                if unique_temps:
                    with st.expander("✏️ Rename Temperature Labels (e.g., change 14.9 to 15)", expanded=False):
                        st.caption("Edit the text below to cleanly format the temperatures for your final publication figure.")
                        ren_cols = st.columns(min(6, len(unique_temps)))
                        for i, t in enumerate(unique_temps):
                            with ren_cols[i % 6]:
                                # Use text_input so they can write "15" or even "15 (Native)"
                                custom_labels[t] = st.text_input(f"Orig: {t}", value=f"{t}", key=f"lbl_{selected_name}_{t}")
                else:
                    st.warning("👈 Please select at least one temperature to display the plot.")
                
                # Only build the plot if at least one temperature is selected
                if unique_temps:
                    # Interpolate every curve onto a common uniform wavelength grid
                    wl_grid = np.linspace(unique_wl[0], unique_wl[-1], len(unique_wl))
                    z_matrix = []
                    for t in unique_temps:
                        curve = next((c for c in curr_data if c["temp"] == t), None)
                        if curve:
                            f_int = interp1d(curve["wl"], curve["sig"], bounds_error=False, fill_value=0)
                            z_matrix.append(f_int(wl_grid))
                        else:
                            z_matrix.append(np.zeros_like(wl_grid))
                    z_arr = np.array(z_matrix)

                    # --- UI CONTROLS ---
                    st.markdown("##### 🛠️ Plot Controls")
                    c_c1, c_c2, c_c3, c_c4 = st.columns(4)
                    with c_c1:
                        # Fixed the PRGn_r reversed colorscale here!
                        colorscale = st.selectbox(
                            "Bottom Color Scale", 
                            ["RdBu_r", "PRGn_r", "PiYG_r", "PuOr_r", "Viridis"], 
                            index=0, key=f"hm_cs_{selected_name}",
                            help="Tip: Select 'PRGn_r' to perfectly match the Purple/Green aesthetic of the Nature Communications paper!"
                        )
                    with c_c2:
                        symmetric = st.checkbox("Symmetric Range (±)", value=True, key=f"hm_sym_{selected_name}")
                    with c_c3:
                        sig_max = max([np.max(c["sig"]) for c in curr_data if c["temp"] in unique_temps])
                        sig_min = min([np.min(c["sig"]) for c in curr_data if c["temp"] in unique_temps])
                        amplitude = sig_max - sig_min if sig_max != sig_min else 10
                        spacing = st.slider("Top Panel Spacing", 0.0, float(amplitude), float(amplitude * 0.4), step=float(amplitude*0.05), key=f"rl_spc_{selected_name}")
                    with c_c4:
                        # Dynamically adjust default height based on number of temperatures
                        default_ht = max(400, 200 + len(unique_temps) * 45)
                        fig_height = st.number_input("Figure Height (px)", 300, 2000, default_ht, step=50, key=f"fig_ht_{selected_name}")

                    # --- BUILD THE COMPOSITE FIGURE ---
                    fig_comp = make_subplots(
                        rows=2, cols=1, 
                        shared_xaxes=True, 
                        row_heights=[0.7, 0.3], 
                        vertical_spacing=0.03   
                    )
                    
                    tick_vals = []
                    tick_text = []
                    
                    # Colors
                    pos_col_line = 'rgba(214, 39, 40, 1)'
                    pos_col_fill = 'rgba(214, 39, 40, 0.4)'
                    neg_col_line = 'rgba(31, 119, 180, 1)'
                    neg_col_fill = 'rgba(31, 119, 180, 0.4)'
                    
                    if colorscale == "PRGn_r":
                        pos_col_line = 'rgba(118, 42, 131, 1)'   # Purple
                        pos_col_fill = 'rgba(118, 42, 131, 0.5)'
                        neg_col_line = 'rgba(27, 120, 55, 1)'    # Green
                        neg_col_fill = 'rgba(27, 120, 55, 0.5)'

                    # 1. POPULATE TOP PANEL (Ridgeline)
                    for i, t in enumerate(unique_temps):
                        offset = i * spacing
                        tick_vals.append(offset)
                        
                        # Apply custom label from the user dictionary
                        display_label = f"{custom_labels[t]}°C"
                        tick_text.append(display_label)
                        
                        sig_grid = z_arr[i]
                        y_pos = np.where(sig_grid > 0, sig_grid, 0) + offset
                        y_neg = np.where(sig_grid < 0, sig_grid, 0) + offset
                        
                        fig_comp.add_trace(go.Scatter(x=wl_grid, y=np.full_like(wl_grid, offset), mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'), row=1, col=1)
                        fig_comp.add_trace(go.Scatter(x=wl_grid, y=y_pos, mode='lines', line=dict(color=pos_col_line, width=1.5), fill='tonexty', fillcolor=pos_col_fill, name=f"{display_label} (+)", showlegend=False), row=1, col=1)
                        
                        fig_comp.add_trace(go.Scatter(x=wl_grid, y=np.full_like(wl_grid, offset), mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'), row=1, col=1)
                        fig_comp.add_trace(go.Scatter(x=wl_grid, y=y_neg, mode='lines', line=dict(color=neg_col_line, width=1.5), fill='tonexty', fillcolor=neg_col_fill, name=f"{display_label} (-)", showlegend=False), row=1, col=1)
                        
                        fig_comp.add_trace(go.Scatter(x=[wl_grid[0], wl_grid[-1]], y=[offset, offset], mode='lines', line=dict(color=_c["text"], width=1, dash='solid'), opacity=0.2, showlegend=False, hoverinfo='skip'), row=1, col=1)

                    # 2. POPULATE BOTTOM PANEL (Discrete Heatmap)
                    zmin_val = float(np.nanmin(z_arr))
                    zmax_val = float(np.nanmax(z_arr))
                    if symmetric:
                        z_abs = max(abs(zmin_val), abs(zmax_val))
                        zmin_val, zmax_val = -z_abs, z_abs

                    fig_comp.add_trace(go.Heatmap(
                        z=z_arr, x=wl_grid, y=tick_text,
                        colorscale=colorscale, zmin=zmin_val, zmax=zmax_val,
                        colorbar=dict(
                            title=dict(text=f"<b>{y_lbl_short}</b>", side="right", font=dict(size=12, family="Arial", color=_c["text"])),
                            tickfont=dict(size=11, family="Arial", color=_c["text"]),
                            thickness=15, len=0.35, y=0.15 
                        ),
                        xgap=0, ygap=3 
                    ), row=2, col=1)

                    # --- STYLING & LAYOUT ---
                    fig_comp.update_layout(
                        title=dict(text=f"<b>Discrete Spectral Projection — {selected_name}</b>", x=0.5, font=dict(family="Arial", size=18, color=_c["text"])),
                        template=_c["template"],
                        paper_bgcolor=_c["paper"],
                        plot_bgcolor=_c["bg"],
                        height=fig_height,
                        margin=dict(l=80, r=40, t=60, b=60)
                    )
                    
                    fig_comp.update_xaxes(showline=True, linewidth=2, linecolor=_c["line"], mirror=True, showgrid=False, range=[wl_min, wl_max], row=1, col=1)
                    fig_comp.update_xaxes(title_text="<b>Wavelength (nm)</b>", title_font=dict(size=16, family="Arial", color=_c["text"]), tickfont=dict(size=14, family="Arial", color=_c["text"]), showline=True, linewidth=2, linecolor=_c["line"], mirror=True, showgrid=False, range=[wl_min, wl_max], row=2, col=1)
                    
                    fig_comp.update_yaxes(title_text="<b>Temperature (°C)</b>", tickmode='array', tickvals=tick_vals, ticktext=tick_text, showline=True, linewidth=2, linecolor=_c["line"], mirror=True, showgrid=False, zeroline=False, tickfont=dict(size=12, family="Arial", color=_c["text"]), row=1, col=1)
                    fig_comp.update_yaxes(showline=True, linewidth=2, linecolor=_c["line"], mirror=True, showgrid=False, tickfont=dict(size=12, family="Arial", color=_c["text"]), row=2, col=1)

                    st.plotly_chart(fig_comp, use_container_width=True, key=f"tab3_comp_{selected_name}")

                    # --- EXPORT BUTTONS ---
                    st.markdown("##### 📥 Export Publication Plot")
                    try:
                        c_dl1, c_dl2, c_dl3 = st.columns(3)
                        c_dl1.download_button("📸 Download Plot (PNG)", fig_comp.to_image(format="png", scale=3), f"{selected_name}_projection.png", "image/png", key=f"cp_dl_png_{selected_name}")
                        c_dl2.download_button("📄 Download Plot (PDF)", fig_comp.to_image(format="pdf"), f"{selected_name}_projection.pdf", "application/pdf", key=f"cp_dl_pdf_{selected_name}")
                        
                        # Apply custom labels to the CSV export index as well!
                        matrix_df = pd.DataFrame(z_arr, index=tick_text, columns=wl_grid.round(2))
                        c_dl3.download_button("💾 Download Matrix (CSV)", matrix_df.to_csv(), f"{selected_name}_projection_matrix.csv", key=f"cp_dl_csv_{selected_name}")
                    except: pass


            if _th_tab == "📊 λ Peak Tracking":
                st.subheader("📊 λ Peak Tracking")
                st.caption(
                    "Tracks how peak **position (wavelength)** and **intensity** shift with temperature. "
                    "Dual-axis plots overlay both on a single chart. Compare multiple samples side-by-side."
                )

                # ── Sample selector ───────────────────────────────────────────
                pt_samples = st.multiselect(
                    "Samples to analyse",
                    list(processed_datasets.keys()),
                    default=[selected_name],
                    key="pt_samples",
                    help="Select one sample for a single view, or multiple to compare peak shifts across samples."
                )

                if not pt_samples:
                    st.info("Select at least one sample above.")
                else:
                    # ── Build stats for every selected sample ─────────────────
                    all_peak_stats = {}   # {sname: df_tstats}
                    for pt_sname in pt_samples:
                        pt_data = processed_datasets[pt_sname]["curves"]
                        _pt_min_wl = int(min(processed_datasets[pt_sname]["wl_ref"]))
                        _pt_max_wl = int(max(processed_datasets[pt_sname]["wl_ref"]))
                        rows = []
                        for d in pt_data:
                            # Use restricted windows if set, else full sample range
                            _eff_lo = min(_t_min_r[0] if _use_min_range else _pt_min_wl,
                                          _t_max_r[0] if _use_max_range else _pt_min_wl)
                            _eff_hi = max(_t_min_r[1] if _use_min_range else _pt_max_wl,
                                          _t_max_r[1] if _use_max_range else _pt_max_wl)
                            mm = get_min_max(d["wl"], d["sig"], _eff_lo, _eff_hi)
                            if mm and _use_min_range:
                                _mm2 = get_min_max(d["wl"], d["sig"], _t_min_r[0], _t_min_r[1])
                                if _mm2: mm["Lambda Min 1 (nm)"] = _mm2["Lambda Min 1 (nm)"]; mm["Min 1 Value"] = _mm2["Min 1 Value"]
                            if mm and _use_max_range:
                                _mm3 = get_min_max(d["wl"], d["sig"], _t_max_r[0], _t_max_r[1])
                                if _mm3: mm["Lambda Max (nm)"] = _mm3["Lambda Max (nm)"]; mm["Max Value"] = _mm3["Max Value"]
                            if mm:
                                mm["Temperature"] = d["temp"]
                                rows.append(mm)
                        if rows:
                            all_peak_stats[pt_sname] = pd.DataFrame(rows).set_index("Temperature")

                    if not all_peak_stats:
                        st.warning("Data range insufficient for peak tracking.")
                    else:
                        # ── Fix 3: name the sample in the smart suggestion ────
                        _ps_name = list(all_peak_stats.keys())[0]
                        _ps_ref  = all_peak_stats[_ps_name]
                        s_min_wl = np.round(_ps_ref["Lambda Min 1 (nm)"].mean(), 1)
                        s_max_wl = np.round(_ps_ref["Lambda Max (nm)"].mean(), 1)
                        st.info(
                            f"**💡 Smart Suggestion (based on '{_ps_name}'):** "
                            f"Use **{s_min_wl} nm** (mean λ Min) or **{s_max_wl} nm** (mean λ Max) "
                            f"as the monitoring wavelength for Melting Curve analysis."
                        )

                        # Data table (for primary sample only to keep UI clean)
                        with st.expander(f"📋 Peak Stats Table — {_ps_name}", expanded=False):
                            st.dataframe(all_peak_stats[_ps_name])

                        # Colour palette — one colour per sample
                        _PT_PALETTE = [
                            "#1f77b4","#d62728","#2ca02c","#ff7f0e","#9467bd",
                            "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"
                        ]
                        pt_colors = {sn: _PT_PALETTE[i % len(_PT_PALETTE)]
                                     for i, sn in enumerate(pt_samples)}

                        _mk = dict(size=8, symbol="circle", line=dict(width=1, color=_c["line"]))

                        # ─────────────────────────────────────────────────────
                        # SECTION 1 — Dual-axis plots (λ position + intensity)
                        # X = Temperature | Left Y = Wavelength | Right Y = Intensity
                        # ─────────────────────────────────────────────────────
                        st.markdown("#### 📍 λ Position & Intensity — Dual Axis")
                        st.caption(
                            "Each plot carries **two y-axes**: left axis = peak wavelength position (nm), "
                            "right axis = CD signal intensity at that peak. "
                            "Solid lines = wavelength (left), dashed lines = intensity (right). "
                            "Use the **Axis Scale** expander under each plot to fix axis ranges."
                        )

                        # ── Fix 1: Axis scale controls — collected BEFORE the columns
                        # so Streamlit can render them across the full width ───────────
                        with st.expander("🔧 Axis Scale — λ Minimum dual-axis plot", expanded=False):
                            _dmin_r1, _dmin_r2, _dmin_r3 = st.columns(3)
                            with _dmin_r1:
                                st.markdown("**X axis (Temperature)**")
                                da_min_x0 = st.number_input("X Min (°C)", value=0.0, key="da_min_x0",
                                                            help="0 = auto")
                                da_min_x1 = st.number_input("X Max (°C)", value=0.0, key="da_min_x1",
                                                            help="0 = auto")
                            with _dmin_r2:
                                st.markdown("**Left Y (Wavelength, nm)**")
                                da_min_ly0 = st.number_input("Left Y Min", value=0.0, key="da_min_ly0",
                                                              help="0 = auto")
                                da_min_ly1 = st.number_input("Left Y Max", value=0.0, key="da_min_ly1",
                                                              help="0 = auto")
                            with _dmin_r3:
                                st.markdown("**Right Y (Intensity)**")
                                da_min_ry0 = st.number_input("Right Y Min", value=0.0, key="da_min_ry0",
                                                              help="0 = auto")
                                da_min_ry1 = st.number_input("Right Y Max", value=0.0, key="da_min_ry1",
                                                              help="0 = auto")
                        da_min_ranges = {
                            "x":  [da_min_x0,  da_min_x1]  if da_min_x1  > da_min_x0  else None,
                            "ly": [da_min_ly0, da_min_ly1] if da_min_ly1 > da_min_ly0 else None,
                            "ry": [da_min_ry0, da_min_ry1] if da_min_ry1 > da_min_ry0 else None,
                        }

                        with st.expander("🔧 Axis Scale — λ Maximum dual-axis plot", expanded=False):
                            _dmax_r1, _dmax_r2, _dmax_r3 = st.columns(3)
                            with _dmax_r1:
                                st.markdown("**X axis (Temperature)**")
                                da_max_x0 = st.number_input("X Min (°C)", value=0.0, key="da_max_x0",
                                                            help="0 = auto")
                                da_max_x1 = st.number_input("X Max (°C)", value=0.0, key="da_max_x1",
                                                            help="0 = auto")
                            with _dmax_r2:
                                st.markdown("**Left Y (Wavelength, nm)**")
                                da_max_ly0 = st.number_input("Left Y Min", value=0.0, key="da_max_ly0",
                                                              help="0 = auto")
                                da_max_ly1 = st.number_input("Left Y Max", value=0.0, key="da_max_ly1",
                                                              help="0 = auto")
                            with _dmax_r3:
                                st.markdown("**Right Y (Intensity)**")
                                da_max_ry0 = st.number_input("Right Y Min", value=0.0, key="da_max_ry0",
                                                              help="0 = auto")
                                da_max_ry1 = st.number_input("Right Y Max", value=0.0, key="da_max_ry1",
                                                              help="0 = auto")
                        da_max_ranges = {
                            "x":  [da_max_x0,  da_max_x1]  if da_max_x1  > da_max_x0  else None,
                            "ly": [da_max_ly0, da_max_ly1] if da_max_ly1 > da_max_ly0 else None,
                            "ry": [da_max_ry0, da_max_ry1] if da_max_ry1 > da_max_ry0 else None,
                        }

                        # ═══════════════════════════════════════════════════════
                        # DUAL-AXIS PLOT SPACING TUNING BLOCK
                        # ═══════════════════════════════════════════════════════
                        # All magic numbers that control the layout of the
                        # λ Min / λ Max dual-axis plots live here.
                        # Change one value, save, and rerun to see the effect.
                        #
                        # ┌─────────────────────────────────────────────────────┐
                        # │  MARGIN VARIABLES (all in pixels)                   │
                        # │                                                     │
                        # │  _DA_TITLE_PX  — vertical space reserved for the   │
                        # │      plot title above the legend.                   │
                        # │      Increase if title text is clipped at top.      │
                        # │      Default: 38                                    │
                        # │                                                     │
                        # │  _DA_ROW_PX    — height of one legend row           │
                        # │      (one row = one sample = 2 entries).            │
                        # │      Increase if legend rows are cramped.           │
                        # │      Default: 28                                    │
                        # │                                                     │
                        # │  _DA_LEG_PAD   — padding inside the legend border   │
                        # │      (top + bottom combined, px).                   │
                        # │      Default: 10                                    │
                        # │                                                     │
                        # │  _DA_GAP_PX    — clear gap (px) between the         │
                        # │      BOTTOM of the legend box and the TOP axis      │
                        # │      line of the plot.                              │
                        # │      Increase if legend still clips the axis.       │
                        # │      Default: 14                                    │
                        # │                                                     │
                        # │  _DA_B_MARGIN  — bottom margin (px): space below    │
                        # │      the x-axis for the axis title + tick labels.   │
                        # │      Increase if x-axis label is cropped.           │
                        # │      Default: 65                                    │
                        # │                                                     │
                        # │  _DA_L_MARGIN  — left margin (px): space for the   │
                        # │      left y-axis title + tick labels.               │
                        # │      Default: 75                                    │
                        # │                                                     │
                        # │  _DA_R_MARGIN  — right margin (px): space for the  │
                        # │      right (intensity) y-axis title + ticks.        │
                        # │      Default: 85                                    │
                        # │                                                     │
                        # │  _DA_BASE_H    — minimum plot height (px) before    │
                        # │      extra legend rows are added.                   │
                        # │      Default: 440                                   │
                        # └─────────────────────────────────────────────────────┘
                        _DA_TITLE_PX  = 100   # ← tune: title vertical space (px)
                        _DA_ROW_PX    = 28   # ← tune: legend row height (px)
                        _DA_LEG_PAD   = 10   # ← tune: legend border padding (px)
                        _DA_GAP_PX    = 14   # ← tune: gap between legend and top axis (px)
                        _DA_B_MARGIN  = 65   # ← tune: bottom margin (px)
                        _DA_L_MARGIN  = 75   # ← tune: left margin (px)
                        _DA_R_MARGIN  = 85   # ← tune: right margin (px)
                        _DA_BASE_H    = 440  # ← tune: base figure height (px)
                        # ═══════════════════════════════════════════════════════

                        def _make_dual_axis(wl_col, val_col, title, ranges):
                            """Build dual-axis figure supporting multiple samples.
                            ranges = {"x": [lo,hi]|None, "ly": ..., "ry": ...}
                            All layout constants come from the TUNING BLOCK above.
                            """
                            n_samp      = len([sn for sn in pt_samples if sn in all_peak_stats])
                            n_leg_rows  = max(1, n_samp)
                            legend_h_px = n_leg_rows * _DA_ROW_PX + _DA_LEG_PAD
                            # Top margin = title space + legend height + axis gap
                            t_margin    = _DA_TITLE_PX + legend_h_px + _DA_GAP_PX
                            fig_height  = _DA_BASE_H + n_leg_rows * _DA_ROW_PX

                            # Legend y in paper coords (0=bottom, 1=top of PLOT AREA).
                            # Values > 1.0 place the legend above the top axis line.
                            # Formula: gap_px expressed as a fraction of the plot pixel height.
                            plot_area_h = max(fig_height - t_margin - _DA_B_MARGIN, 1)
                            y_leg       = 1.0 + _DA_GAP_PX / plot_area_h

                            fig_da = make_subplots(specs=[[{"secondary_y": True}]])
                            for sn in pt_samples:
                                if sn not in all_peak_stats:
                                    continue
                                df_s = all_peak_stats[sn]
                                col  = pt_colors[sn]
                                # Left axis — wavelength (solid circles)
                                fig_da.add_trace(go.Scatter(
                                    x=df_s.index, y=df_s[wl_col],
                                    mode="lines+markers",
                                    name=f"{sn} — λ (nm)",
                                    legendgroup=sn,
                                    showlegend=True,
                                    line=dict(color=col, width=2.5, dash="solid"),
                                    marker=dict(**_mk, color=col)
                                ), secondary_y=False)
                                # Right axis — intensity (dashed diamonds)
                                fig_da.add_trace(go.Scatter(
                                    x=df_s.index, y=df_s[val_col],
                                    mode="lines+markers",
                                    name=f"{sn} — intensity",
                                    legendgroup=sn,
                                    showlegend=True,
                                    line=dict(color=col, width=2.5, dash="dash"),
                                    marker=dict(size=8, symbol="diamond",
                                                line=dict(width=1, color=_c["line"]),
                                                color=col)
                                ), secondary_y=True)

                            # Zero reference on intensity axis
                            fig_da.add_hline(y=0, line_dash="dot", line_color="gray",
                                             line_width=1, opacity=0.5)

                            fig_da.update_layout(
                                # Title: do NOT set y/yanchor/yref here.
                                # Plotly automatically centres the title in the top
                                # margin (t_margin) when those overrides are absent.
                                # Setting y=1.0 with yref="paper" puts the title at
                                # the TOP OF THE PLOT AREA — inside the box — which
                                # is the bug visible in the previous screenshot.
                                title=dict(
                                    text=f"<b>{title}</b>",
                                    x=0.5, xanchor="center",
                                    font=dict(size=18, family="Arial", color=_c["text"])
                                    # ↑ y / yanchor / yref intentionally omitted —
                                    #   Plotly places the title in t_margin space.
                                ),
                                template=_c["template"],
                        paper_bgcolor=_c["paper"],
                        plot_bgcolor=_c["bg"],
                                height=fig_height,
                                font=dict(family="Arial", size=13, color=_c["text"]),
                                # Legend: horizontal bar sitting above the top axis.
                                # y > 1.0 means above the plot area (plot area = 0–1).
                                # yanchor="bottom" anchors the legend BOX bottom at y,
                                # so the legend grows upward into the top margin.
                                legend=dict(
                                    orientation="h",
                                    x=0.5, xanchor="center",
                                    y=y_leg,        # ← computed above from _DA_GAP_PX
                                    yanchor="bottom",
                                    # No yref override — default paper coords work correctly
                                    font=dict(size=11, family="Arial"),
                                    bordercolor=_c["legend_border"],
                                    borderwidth=1,
                                    tracegroupgap=2
                                ),
                                margin=dict(
                                    l=_DA_L_MARGIN,   # ← left margin (tuning block)
                                    r=_DA_R_MARGIN,   # ← right margin (tuning block)
                                    t=t_margin,       # ← auto-computed from tuning block
                                    b=_DA_B_MARGIN    # ← bottom margin (tuning block)
                                )
                            )
                            # mirror=True closes the top edge of the plot as a full box
                            fig_da.update_xaxes(
                                title_text="<b>Temperature (°C)</b>",
                                showline=True, linewidth=2, linecolor=_c["line"],
                                mirror=True,
                                showgrid=True, gridcolor=_c["grid"],
                                tickfont=dict(size=12, family="Arial", color=_c["text"])
                            )
                            fig_da.update_yaxes(
                                title_text="<b>Wavelength (nm)</b>",
                                showline=True, linewidth=2, linecolor=_c["line"],
                                mirror=False,
                                showgrid=True, gridcolor=_c["grid"],
                                tickfont=dict(size=12, family="Arial", color=_c["text"]),
                                secondary_y=False
                            )
                            fig_da.update_yaxes(
                                title_text=f"<b>Intensity ({y_lbl_short})</b>",
                                showline=True, linewidth=2, linecolor=_c["line"],
                                mirror=False,
                                showgrid=False,
                                zeroline=False,
                                tickfont=dict(size=12, family="Arial", color=_c["text"]),
                                secondary_y=True
                            )
                            # Apply user-specified axis ranges from the scale expanders
                            if ranges.get("x"):
                                fig_da.update_xaxes(range=ranges["x"])
                            if ranges.get("ly"):
                                fig_da.update_yaxes(range=ranges["ly"], secondary_y=False)
                            if ranges.get("ry"):
                                fig_da.update_yaxes(range=ranges["ry"], secondary_y=True)
                            return fig_da

                        da_col1, da_col2 = st.columns(2)

                        with da_col1:
                            fig_da_min = _make_dual_axis(
                                "Lambda Min 1 (nm)", "Min 1 Value",
                                "λ Minimum — Position & Intensity",
                                da_min_ranges
                            )
                            st.plotly_chart(fig_da_min, use_container_width=True, key="da_min")
                            try:
                                _da1, _da2 = st.columns(2)
                                _da1.download_button("📸 PNG", fig_da_min.to_image(format="png", scale=3),
                                                     "lambda_min_dual.png", key="dl_da_min_png")
                                _da2.download_button("📄 PDF", fig_da_min.to_image(format="pdf"),
                                                     "lambda_min_dual.pdf", key="dl_da_min_pdf")
                            except: pass

                        with da_col2:
                            fig_da_max = _make_dual_axis(
                                "Lambda Max (nm)", "Max Value",
                                "λ Maximum — Position & Intensity",
                                da_max_ranges
                            )
                            st.plotly_chart(fig_da_max, use_container_width=True, key="da_max")
                            try:
                                _da3, _da4 = st.columns(2)
                                _da3.download_button("📸 PNG", fig_da_max.to_image(format="png", scale=3),
                                                     "lambda_max_dual.png", key="dl_da_max_png")
                                _da4.download_button("📄 PDF", fig_da_max.to_image(format="pdf"),
                                                     "lambda_max_dual.pdf", key="dl_da_max_pdf")
                            except: pass

                        # ─────────────────────────────────────────────────────
                        # SECTION 2 — Single-axis multi-sample comparison
                        # (λ position only, then intensity only — cleaner read)
                        # ─────────────────────────────────────────────────────
                        st.divider()
                        st.markdown("#### 🔀 Multi-Sample Comparison — Separate Axes")
                        st.caption(
                            "Same data as above split into four individual panels for cleaner "
                            "quantitative reading. Each panel shows all selected samples overlaid."
                        )

                        def _make_single_axis(y_col, title, y_label, dash="solid", symbol="circle"):
                            fig_s = go.Figure()
                            for sn in pt_samples:
                                if sn not in all_peak_stats:
                                    continue
                                df_s = all_peak_stats[sn]
                                col  = pt_colors[sn]
                                fig_s.add_trace(go.Scatter(
                                    x=df_s.index, y=df_s[y_col],
                                    mode="lines+markers", name=sn,
                                    line=dict(color=col, width=2.5, dash=dash),
                                    marker=dict(size=8, symbol=symbol,
                                                line=dict(width=1, color=_c["text"]), color=col)
                                ))
                            fig_s = apply_publication_style(
                                fig_s, title, "Temperature (°C)", y_label,
                                show_grid, height=380, plot_mode="lines+markers"
                            )
                            fig_s.add_hline(y=0, line_dash="dot", line_color="gray",
                                            line_width=1, opacity=0.5)
                            return fig_s

                        sa_c1, sa_c2 = st.columns(2)
                        with sa_c1:
                            fig_sa_wl_min = _make_single_axis(
                                "Lambda Min 1 (nm)", "λ Minimum Position", "Wavelength (nm)"
                            )
                            st.plotly_chart(fig_sa_wl_min, use_container_width=True, key="sa_wl_min")
                            try:
                                _s1, _s2 = st.columns(2)
                                _s1.download_button("📸 PNG", fig_sa_wl_min.to_image(format="png", scale=3),
                                                    "lam_min_pos.png", key="dl_sa_wlmin_png")
                                _s2.download_button("💾 CSV",
                                    pd.concat({sn: all_peak_stats[sn][["Lambda Min 1 (nm)"]]
                                               for sn in all_peak_stats}, axis=1).to_csv(),
                                    "lam_min_pos.csv", key="dl_sa_wlmin_csv")
                            except: pass

                        with sa_c2:
                            fig_sa_wl_max = _make_single_axis(
                                "Lambda Max (nm)", "λ Maximum Position", "Wavelength (nm)"
                            )
                            st.plotly_chart(fig_sa_wl_max, use_container_width=True, key="sa_wl_max")
                            try:
                                _s3, _s4 = st.columns(2)
                                _s3.download_button("📸 PNG", fig_sa_wl_max.to_image(format="png", scale=3),
                                                    "lam_max_pos.png", key="dl_sa_wlmax_png")
                                _s4.download_button("💾 CSV",
                                    pd.concat({sn: all_peak_stats[sn][["Lambda Max (nm)"]]
                                               for sn in all_peak_stats}, axis=1).to_csv(),
                                    "lam_max_pos.csv", key="dl_sa_wlmax_csv")
                            except: pass

                        sa_c3, sa_c4 = st.columns(2)
                        with sa_c3:
                            fig_sa_int_min = _make_single_axis(
                                "Min 1 Value", "Intensity at λ Minimum", y_lbl_full, dash="dash", symbol="diamond"
                            )
                            st.plotly_chart(fig_sa_int_min, use_container_width=True, key="sa_int_min")
                            try:
                                _s5, _s6 = st.columns(2)
                                _s5.download_button("📸 PNG", fig_sa_int_min.to_image(format="png", scale=3),
                                                    "int_min.png", key="dl_sa_intmin_png")
                                _s6.download_button("💾 CSV",
                                    pd.concat({sn: all_peak_stats[sn][["Min 1 Value"]]
                                               for sn in all_peak_stats}, axis=1).to_csv(),
                                    "int_min.csv", key="dl_sa_intmin_csv")
                            except: pass

                        with sa_c4:
                            fig_sa_int_max = _make_single_axis(
                                "Max Value", "Intensity at λ Maximum", y_lbl_full, dash="dash", symbol="diamond"
                            )
                            st.plotly_chart(fig_sa_int_max, use_container_width=True, key="sa_int_max")
                            try:
                                _s7, _s8 = st.columns(2)
                                _s7.download_button("📸 PNG", fig_sa_int_max.to_image(format="png", scale=3),
                                                    "int_max.png", key="dl_sa_intmax_png")
                                _s8.download_button("💾 CSV",
                                    pd.concat({sn: all_peak_stats[sn][["Max Value"]]
                                               for sn in all_peak_stats}, axis=1).to_csv(),
                                    "int_max.csv", key="dl_sa_intmax_csv")
                            except: pass

                        # ─────────────────────────────────────────────────────
                        # SECTION 3 — Combined 2×2 Overview (multi-sample)
                        # Fixed: increased vertical_spacing + height to prevent
                        # subplot title overlap with axis labels in the UI.
                        # ─────────────────────────────────────────────────────
                        st.divider()
                        st.markdown("#### 🗂️ Combined Peak Overview (2×2)")
                        st.caption(
                            "All four peak-tracking metrics in a single exportable figure. "
                            "Each selected sample appears as a separate coloured trace per panel."
                        )

                        fig_combo = make_subplots(
                            rows=2, cols=2,
                            subplot_titles=[
                                "λ Minimum Position (nm)",
                                "λ Maximum Position (nm)",
                                f"Intensity at λ Min ({y_lbl_short})",
                                f"Intensity at λ Max ({y_lbl_short})"
                            ],
                            # Increased spacing to prevent subplot titles overlapping axes in UI
                            vertical_spacing=0.28,
                            horizontal_spacing=0.14
                        )

                        _mk_combo = dict(size=7, symbol="circle", line=dict(width=1, color=_c["line"]))
                        _mk_combo_dia = dict(size=7, symbol="diamond", line=dict(width=1, color=_c["line"]))

                        for sn in pt_samples:
                            if sn not in all_peak_stats:
                                continue
                            df_s = all_peak_stats[sn]
                            col  = pt_colors[sn]
                            show_leg = True

                            fig_combo.add_trace(go.Scatter(
                                x=df_s.index, y=df_s["Lambda Min 1 (nm)"],
                                mode="lines+markers", name=sn,
                                legendgroup=sn, showlegend=show_leg,
                                line=dict(color=col, width=2),
                                marker=dict(**_mk_combo, color=col)
                            ), row=1, col=1)

                            fig_combo.add_trace(go.Scatter(
                                x=df_s.index, y=df_s["Lambda Max (nm)"],
                                mode="lines+markers", name=sn,
                                legendgroup=sn, showlegend=False,
                                line=dict(color=col, width=2),
                                marker=dict(**_mk_combo, color=col)
                            ), row=1, col=2)

                            fig_combo.add_trace(go.Scatter(
                                x=df_s.index, y=df_s["Min 1 Value"],
                                mode="lines+markers", name=sn,
                                legendgroup=sn, showlegend=False,
                                line=dict(color=col, width=2, dash="dash"),
                                marker=dict(**_mk_combo_dia, color=col)
                            ), row=2, col=1)

                            fig_combo.add_trace(go.Scatter(
                                x=df_s.index, y=df_s["Max Value"],
                                mode="lines+markers", name=sn,
                                legendgroup=sn, showlegend=False,
                                line=dict(color=col, width=2, dash="dash"),
                                marker=dict(**_mk_combo_dia, color=col)
                            ), row=2, col=2)

                        # Axis styling per panel
                        _ax_common = dict(showline=True, linewidth=2, linecolor=_c["line"],
                                          mirror=True, showgrid=True, gridcolor=_c["grid"],
                                          tickfont=dict(size=11, family="Arial", color=_c["text"]))
                        for _r, _col, _yt in [
                            (1, 1, "<b>Wavelength (nm)</b>"),
                            (1, 2, "<b>Wavelength (nm)</b>"),
                            (2, 1, f"<b>{y_lbl_short}</b>"),
                            (2, 2, f"<b>{y_lbl_short}</b>"),
                        ]:
                            fig_combo.update_yaxes(title_text=_yt, **_ax_common, row=_r, col=_col)
                        for _r in [1, 2]:
                            for _col in [1, 2]:
                                _xt = "<b>Temp (°C)</b>" if _r == 2 else ""
                                fig_combo.update_xaxes(title_text=_xt, **_ax_common, row=_r, col=_col)

                        # Shift subplot title annotations upward so they clear the axis area
                        for ann in fig_combo.layout.annotations:
                            ann.update(yshift=8, font=dict(size=13, family="Arial", color=_c["text"]))

                        sample_label = " vs ".join(pt_samples) if len(pt_samples) > 1 else pt_samples[0]
                        fig_combo.update_layout(
                            title=dict(
                                text=f"<b>Peak Tracking Summary — {sample_label}</b>",
                                x=0.5, xanchor="center",
                                font=dict(size=18, family="Arial", color=_c["text"])
                            ),
                            template=_c["template"],
                        paper_bgcolor=_c["paper"],
                        plot_bgcolor=_c["bg"],
                            # Taller figure gives each subplot more breathing room
                            height=820,
                            font=dict(family="Arial", size=12, color=_c["text"]),
                            showlegend=(len(pt_samples) > 1),
                            legend=dict(font=dict(size=12), bordercolor=_c["legend_border"], borderwidth=1,
                                        x=1.01, y=0.5, xanchor="left"),
                            # Large top margin keeps main title clear of first row subplot titles
                            margin=dict(l=75, r=120 if len(pt_samples) > 1 else 40, t=100, b=70)
                        )
                        st.plotly_chart(fig_combo, use_container_width=True, key="tab4_combo")
                        try:
                            cc1, cc2 = st.columns(2)
                            cc1.download_button(
                                "📸 Download Combined Overview (PNG)",
                                fig_combo.to_image(format="png", scale=3),
                                f"peak_overview_{'_vs_'.join(pt_samples)}.png",
                                "image/png", key="dl_combo_png"
                            )
                            cc2.download_button(
                                "📄 Download Combined Overview (PDF)",
                                fig_combo.to_image(format="pdf"),
                                f"peak_overview_{'_vs_'.join(pt_samples)}.pdf",
                                "application/pdf", key="dl_combo_pdf"
                            )
                        except: pass

                        # Full CSV export — all samples stacked
                        st.divider()
                        if len(all_peak_stats) == 1:
                            _csv_full = list(all_peak_stats.values())[0].to_csv()
                        else:
                            _csv_full = pd.concat(all_peak_stats, axis=0).to_csv()
                        st.download_button(
                            "💾 Download Full Peak Stats (CSV)",
                            _csv_full.encode("utf-8"),
                            "thermal_peak_stats.csv", "text/csv",
                            key="dl_full_stats_csv"
                        )
                
            # ── TAB 5: SECONDARY STRUCTURE vs TEMPERATURE ────────────────────
            if _th_tab == "🧩 Sec. Structure":
                st.subheader("🧩 Secondary Structure vs Temperature")
                st.info(
                    "Estimates secondary structure at each temperature using NNLS and "
                    "empirical methods (same as General Analysis). Tracks how helix / "
                    "sheet / coil content changes during unfolding."
                )
                st.warning(
                    "⚠️ **Qualitative screen.** Most reliable with MRE metric and "
                    "190–240 nm wavelength range."
                )

                if metric not in ["MRE", "Δε (M⁻¹cm⁻¹)"]:
                    st.error("Requires MRE or Δε output metric.")
                else:
                    # ── Controls ──────────────────────────────────────────────
                    ss_c1, ss_c2 = st.columns([2, 1])
                    with ss_c1:
                        ss_samples = st.multiselect(
                            "Select samples to analyse",
                            list(processed_datasets.keys()),
                            default=[selected_name],
                            key="ss_samp",
                            help=(
                                "Select one sample for a single-sample view, or multiple "
                                "samples to compare how secondary structure content changes "
                                "with temperature across different peptides."
                            )
                        )
                    with ss_c2:
                        apply_chen_t = st.checkbox(
                            "Chen chain-length correction",
                            value=False, key="ss_chen",
                            help="Appropriate for short helical peptides < 30 residues."
                        )

                    if not ss_samples:
                        st.info("Select at least one sample above.")
                    else:
                        # Compute NNLS + empirical for each selected sample
                        all_nnls   = {}   # {sname: DataFrame indexed by temp}
                        all_emp    = {}

                        for ss_sample in ss_samples:
                            ss_nres = next(
                                (s["nres"] for s in thermal_samples if s["name"] == ss_sample), 20
                            )
                            is_d_ss = next(
                                (s.get("is_d_peptide", False)
                                 for s in thermal_samples if s["name"] == ss_sample), False
                            )
                            ss_data = processed_datasets[ss_sample]["curves"]

                            # Convert to plain tuples so st.cache_data can hash them
                            _wl_tup  = tuple(tuple(d["wl"])  for d in ss_data)
                            _sig_tup = tuple(tuple(d["sig"]) for d in ss_data)
                            _tmp_tup = tuple(d["temp"]       for d in ss_data)

                            nnls_rows_s, emp_rows_s = _compute_ss_for_sample(
                                _wl_tup, _sig_tup, _tmp_tup,
                                ss_nres, bool(is_d_ss),
                                bool(apply_chen_t), metric
                            )

                            all_nnls[ss_sample] = pd.DataFrame(nnls_rows_s).set_index("Temperature (°C)")
                            all_emp[ss_sample]  = pd.DataFrame(emp_rows_s).set_index("Temperature (°C)")


                        # ── View selector ─────────────────────────────────────
                        ss_view = st.radio(
                            "View",
                            ["All components per sample",
                             "Compare one component across samples"],
                            horizontal=True, key="ss_view"
                        )

                        # Marker spec — circle with black border on each data point
                        _mk = dict(size=7, symbol="circle",
                                   line=dict(width=1, color=_c["line"]))

                        if ss_view == "All components per sample":
                            # One plot per sample, all 3 components, lines+markers
                            COMP_COLORS = {
                                "α-Helix (%)":    "#1f77b4",
                                "β-Sheet (%)":    "#ff7f0e",
                                "Random Coil (%)":"#2ca02c",
                            }
                            for ss_sample in ss_samples:
                                st.markdown(f"##### 🧬 NNLS — {ss_sample}")
                                st.dataframe(all_nnls[ss_sample], use_container_width=True)

                                fig_ss = go.Figure()
                                for comp, col in COMP_COLORS.items():
                                    fig_ss.add_trace(go.Scatter(
                                        x=all_nnls[ss_sample].index,
                                        y=all_nnls[ss_sample][comp],
                                        mode="lines+markers",
                                        name=comp,
                                        line=dict(color=col, width=2.5),
                                        marker=dict(**_mk, color=col)
                                    ))
                                fig_ss = apply_publication_style(
                                    fig_ss,
                                    f"NNLS Structure vs Temperature — {ss_sample}",
                                    "Temperature (°C)", "Content (%)",
                                    show_grid, height=420, plot_mode="lines+markers"
                                )
                                fig_ss.update_yaxes(range=[0, 105])
                                ss_key = f"ss_nnls_{ss_sample.replace(' ','_')}"
                                st.plotly_chart(fig_ss, use_container_width=True, key=ss_key)

                                try:
                                    c1s, c2s = st.columns(2)
                                    c1s.download_button(
                                        f"💾 CSV — {ss_sample}",
                                        all_nnls[ss_sample].to_csv().encode("utf-8"),
                                        f"{ss_sample}_nnls_vs_temp.csv", key=f"dl_nnls_{ss_sample}"
                                    )
                                    c2s.download_button(
                                        f"📸 PNG — {ss_sample}",
                                        fig_ss.to_image(format="png", scale=3),
                                        f"{ss_sample}_nnls.png", key=f"dl_nnls_png_{ss_sample}"
                                    )
                                except: pass

                                st.divider()

                        else:
                            # Compare one chosen component across all selected samples
                            comp_choice = st.selectbox(
                                "Component to compare",
                                ["α-Helix (%)", "β-Sheet (%)", "Random Coil (%)"],
                                key="ss_comp_choice"
                            )
                            comp_colors_samples = {
                                name: COLORS[i % len(COLORS)]
                                for i, name in enumerate(ss_samples)
                            }

                            st.markdown(f"##### 📊 {comp_choice} vs Temperature — all selected samples")
                            fig_comp = go.Figure()
                            for ss_sample in ss_samples:
                                df_s = all_nnls[ss_sample]
                                col  = comp_colors_samples[ss_sample]
                                fig_comp.add_trace(go.Scatter(
                                    x=df_s.index,
                                    y=df_s[comp_choice],
                                    mode="lines+markers",
                                    name=ss_sample,
                                    line=dict(color=col, width=2.5),
                                    marker=dict(**_mk, color=col)
                                ))
                            fig_comp = apply_publication_style(
                                fig_comp,
                                f"{comp_choice} vs Temperature",
                                "Temperature (°C)", comp_choice,
                                show_grid, height=460, plot_mode="lines+markers"
                            )
                            fig_comp.update_yaxes(range=[0, 105])
                            st.plotly_chart(fig_comp, use_container_width=True,
                                            key="ss_comp_comparison")

                            # Combined table
                            df_combined = pd.concat(
                                {sn: all_nnls[sn][comp_choice] for sn in ss_samples},
                                axis=1
                            )
                            df_combined.columns = ss_samples
                            st.dataframe(df_combined, use_container_width=True)

                            try:
                                c1c, c2c = st.columns(2)
                                c1c.download_button(
                                    "💾 Download Comparison CSV",
                                    df_combined.to_csv().encode("utf-8"),
                                    f"ss_comparison_{comp_choice[:5]}.csv",
                                    key="dl_comp_csv"
                                )
                                c2c.download_button(
                                    "📸 Download Comparison PNG",
                                    fig_comp.to_image(format="png", scale=3),
                                    f"ss_comparison_{comp_choice[:5]}.png",
                                    key="dl_comp_png"
                                )
                            except: pass

                        # ── Empirical estimates ───────────────────────────────
                        st.divider()
                        st.markdown("##### 🧮 Empirical Single-Point Estimates")
                        st.caption("Helix from 222 nm, sheet from 217 nm. Qualitative only.")
                        for ss_sample in ss_samples:
                            st.markdown(f"**{ss_sample}**")
                            st.dataframe(all_emp[ss_sample], use_container_width=True)
                            fig_emp = go.Figure()
                            fig_emp.add_trace(go.Scatter(
                                x=all_emp[ss_sample].index,
                                y=all_emp[ss_sample]["Emp. Helix 222nm (%)"],
                                mode="lines+markers", name="Empirical Helix",
                                line=dict(color="#9467bd", width=2.5),
                                marker=dict(**_mk, color="#9467bd")
                            ))
                            fig_emp.add_trace(go.Scatter(
                                x=all_emp[ss_sample].index,
                                y=all_emp[ss_sample]["Emp. Sheet 217nm (%)"],
                                mode="lines+markers", name="Empirical Sheet",
                                line=dict(color="#8c564b", width=2.5),
                                marker=dict(**_mk, color="#8c564b")
                            ))
                            fig_emp = apply_publication_style(
                                fig_emp,
                                f"Empirical Estimates vs Temperature — {ss_sample}",
                                "Temperature (°C)", "Content (%)",
                                show_grid, height=360, plot_mode="lines+markers"
                            )
                            emp_key = f"ss_emp_{ss_sample.replace(' ', '_')}"
                            st.plotly_chart(fig_emp, use_container_width=True, key=emp_key)
                            try:
                                c1e, c2e = st.columns(2)
                                c1e.download_button(
                                    f"💾 Emp. CSV — {ss_sample}",
                                    all_emp[ss_sample].to_csv().encode("utf-8"),
                                    f"{ss_sample}_emp_vs_temp.csv",
                                    key=f"dl_emp_{ss_sample}"
                                )
                                c2e.download_button(
                                    f"📸 Emp. PNG — {ss_sample}",
                                    fig_emp.to_image(format="png", scale=3),
                                    f"{ss_sample}_emp.png",
                                    key=f"dl_emp_png_{ss_sample}"
                                )
                            except: pass
                            st.divider()

            if _th_tab == "⚗️ Thermodynamics":
                st.subheader("⚗️ Apparent ΔG — Model-Dependent Thermodynamics")

                # ── EXPERIMENT TYPE & SCIENTIFIC CONTEXT ──────────────────────────────
                with st.expander("🧪 Experiment Context & Model Assumptions", expanded=True):
                    _exp_c1, _exp_c2 = st.columns(2)
                    with _exp_c1:
                        exp_type = st.radio(
                            "Experiment type",
                            ["Thermal unfolding", "Solvent/membrane mimic (SDS etc.)",
                             "Chemical denaturation", "Ligand binding"],
                            key="thermo_exp_type",
                            help="Controls which warnings are shown and how results are labelled."
                        )
                        model_type = st.radio(
                            "Calculation method",
                            ["Two-state (van't Hoff)", "Qualitative / no model"],
                            key="thermo_model",
                            help=(
                                "Two-state: fits a sigmoidal model with sloping baselines — "
                                "gives physically meaningful ΔG and distinguishes samples with "
                                "different cooperativity even at the same wavelength. "
                                "Qualitative: original min-max normalisation (fallback when "
                                "no sigmoidal transition is present)."
                            )
                        )
                    with _exp_c2:
                        st.markdown("**🔬 Scientific disclaimer for selected context:**")
                        if exp_type == "Thermal unfolding":
                            st.info(
                                "ΔG values are only meaningful if a cooperative sigmoidal transition "
                                "is present in your temperature range. If the melt is incomplete "
                                "(no plateau at high temperature), ΔG is unreliable. "
                                "All values are **apparent model-dependent ΔG**."
                            )
                        elif exp_type == "Solvent/membrane mimic (SDS etc.)":
                            st.warning(
                                "⚠️ **SDS / micellar environment:** ΔG here reflects **apparent "
                                "structural stabilisation in the micellar environment**, not intrinsic "
                                "thermodynamic folding free energy. Different melt profiles in buffer "
                                "vs SDS will yield different ΔG only if the thermal unfolding "
                                "cooperativity (shape of the sigmoidal transition) changes — not "
                                "simply because the absolute CD signal differs. "
                                "**ΔΔG comparison between conditions is scientifically more "
                                "informative than absolute ΔG in this context.**"
                            )
                        elif exp_type == "Chemical denaturation":
                            st.warning(
                                "⚠️ **Chemical denaturation:** The x-axis should represent "
                                "denaturant concentration, not temperature. ΔG extrapolation to zero "
                                "denaturant requires m-value linear extrapolation, which is not "
                                "implemented here. Treat values as qualitative only."
                            )
                        elif exp_type == "Ligand binding":
                            st.warning(
                                "⚠️ **Ligand binding:** ΔG here reflects apparent "
                                "binding-induced conformational change, not intrinsic folding. "
                                "For true binding thermodynamics, complement with ITC or SPR."
                            )

                # ── SAMPLE SELECTION ───────────────────────────────────────────────────
                sel_thermo_samples = st.multiselect("Select Samples", list(processed_datasets.keys()), default=[selected_name])

                results     = {}
                fit_results = {}  # {sname: {Tm_C, dH, R2, RMSE, fit_sig, residuals, conf_flag, conf_color}}

                if sel_thermo_samples:
                    for sname in sel_thermo_samples:
                        dataset = processed_datasets[sname]
                        temp_list, sig_list = [], []
                        for d in dataset["curves"]:
                            f_m = interp1d(d["wl"], d["sig"], kind='linear')
                            if min(d["wl"]) <= melt_wl <= max(d["wl"]):
                                sig_list.append(float(f_m(melt_wl)))
                                temp_list.append(d["temp"])

                        if len(temp_list) < 4:
                            st.warning(f"⚠️ {sname}: need at least 4 temperature points.")
                            continue

                        T_K   = np.array(temp_list) + 273.15
                        Y     = np.array(sig_list)
                        R_gas = 1.9872e-3  # kcal / mol / K

                        if model_type == "Two-state (van't Hoff)":
                            # ── TWO-STATE MODEL WITH SLOPING BASELINES ─────────────────────────
                            # y(T) = [(aF + bF*T) + (aU + bU*T)*exp(-dG/RT)] / [1 + exp(-dG/RT)]
                            # dG(T) = dH * (1 - T/Tm)
                            # Params: aF, bF, aU, bU, dH (kcal/mol), Tm (K)
                            def _two_state(T, aF, bF, aU, bU, dH, Tm):
                                dG = dH * (1.0 - T / Tm)
                                K  = np.exp(-dG / (R_gas * T))
                                return ((aF + bF * T) + (aU + bU * T) * K) / (1.0 + K)

                            try:
                                n      = len(T_K)
                                n_base = max(1, n // 5)
                                aF0    = float(np.mean(Y[:n_base]))
                                aU0    = float(np.mean(Y[-n_base:]))
                                Tm0    = float(T_K[n // 2])
                                p0     = [aF0, 0.0, aU0, 0.0, 50.0, Tm0]
                                bounds = ([-np.inf, -1.0, -np.inf, -1.0,   5.0, T_K[0]  - 50],
                                          [ np.inf,  1.0,  np.inf,  1.0, 500.0, T_K[-1] + 50])
                                popt, _ = curve_fit(_two_state, T_K, Y, p0=p0,
                                                    bounds=bounds, maxfev=10000)
                                aF_f, bF_f, aU_f, bU_f, dH_f, Tm_f = popt

                                fit_Y  = _two_state(T_K, *popt)
                                resid  = Y - fit_Y
                                ss_res = float(np.sum(resid ** 2))
                                ss_tot = float(np.sum((Y - np.mean(Y)) ** 2))
                                r2     = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
                                rmse   = float(np.sqrt(np.mean(resid ** 2)))
                                Tm_C   = Tm_f - 273.15

                                dG_fit    = dH_f * (1.0 - T_K / Tm_f)
                                K_fit     = np.exp(-dG_fit / (R_gas * T_K))
                                alpha_fit = K_fit / (1.0 + K_fit)

                                if r2 >= 0.98:
                                    conf_flag, conf_color = "✅ Valid", "success"
                                elif r2 >= 0.90:
                                    conf_flag, conf_color = "🟡 Caution", "warning"
                                else:
                                    conf_flag, conf_color = "🔴 Invalid — non-two-state", "error"

                                results[sname] = {
                                    "temp": temp_list, "sig": sig_list,
                                    "delG": dG_fit, "alpha": alpha_fit,
                                    "K_eq": K_fit, "Tm": round(Tm_C, 1)
                                }
                                fit_results[sname] = {
                                    "Tm_C": round(Tm_C, 1), "dH": round(dH_f, 2),
                                    "R2": round(r2, 4), "RMSE": round(rmse, 4),
                                    "fit_sig": fit_Y.tolist(), "residuals": resid.tolist(),
                                    "conf_flag": conf_flag, "conf_color": conf_color,
                                }
                            except Exception as _fit_err:
                                st.warning(
                                    f"⚠️ Two-state fit failed for **{sname}**: {_fit_err}. "
                                    "Switch to 'Qualitative' mode or check that your data "
                                    "spans a sigmoidal transition."
                                )
                        else:
                            # ── QUALITATIVE FALLBACK (original min-max) ─────────────────────────
                            alpha, K_eq, delG = calculate_delG_raw(temp_list, sig_list)
                            tm_est = "N/A"
                            try:
                                f_alpha = interp1d(alpha, temp_list, bounds_error=False)
                                tm_val  = f_alpha(0.5)
                                if not np.isnan(tm_val): tm_est = round(tm_val, 1)
                            except: pass
                            results[sname] = {
                                "temp": temp_list, "sig": sig_list,
                                "delG": delG, "alpha": alpha,
                                "K_eq": K_eq, "Tm": tm_est
                            }
                            st.caption(
                                f"🔵 **{sname}:** Qualitative mode — ΔG is purely descriptive "
                                "(min-max normalisation; different conditions may appear identical)."
                            )

                    # ── FIT QUALITY SUMMARY ──────────────────────────────────────────────
                    if fit_results:
                        st.markdown("##### 📊 Fit Quality & Confidence")
                        fq_rows = []
                        for sn, fr in fit_results.items():
                            fq_rows.append({
                                "Sample": sn, "Tm (°C)": fr["Tm_C"],
                                "ΔHᵥₕ (kcal/mol)": fr["dH"],
                                "R²": fr["R2"], "RMSE": fr["RMSE"],
                                "Confidence": fr["conf_flag"]
                            })
                        st.dataframe(pd.DataFrame(fq_rows).set_index("Sample"), use_container_width=True)
                        poor = [sn for sn, fr in fit_results.items() if fr["R2"] < 0.90]
                        if poor:
                            st.error(
                                f"🔴 **Non-two-state behaviour detected** in: {', '.join(poor)}. "
                                "ΔG values are likely invalid. Consider switching to 'Qualitative' "
                                "mode or inspect whether a sigmoidal transition is present."
                            )

                    # ── SUMMARY TABLE ────────────────────────────────────────────────────
                    st.markdown("##### 📄 Calculated Parameters")
                    summary_data = []
                    for sname, d in results.items():
                        for i in range(len(d["temp"])):
                            summary_data.append({
                                "Sample": sname,
                                "Temperature (°C)": d["temp"][i],
                                f"Signal @ {melt_wl:.1f} nm": round(d["sig"][i], 3),
                                "Keq": round(float(d["K_eq"][i]), 3),
                                f"Apparent ΔG (kcal/mol) @ {melt_wl:.1f} nm": round(float(d["delG"][i]), 3),
                                "Fraction Unfolded": round(float(d["alpha"][i]), 3)
                            })
                    if selected_name in results:
                        st.metric(f"Estimated Tm — {selected_name}", f"{results[selected_name]['Tm']} °C")
                    if summary_data:
                        df_summ = pd.DataFrame(summary_data)
                        st.dataframe(df_summ, use_container_width=True)
                        st.download_button("Download Thermo Data (CSV)", df_summ.to_csv(index=False), "thermodynamics.csv")

                    # ── MELTING CURVE + FIT + RESIDUALS ─────────────────────────────────
                    _COLORS_T = ["#1f77b4","#d62728","#2ca02c","#ff7f0e","#9467bd",
                                 "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"]
                    _n_cols   = 3 if fit_results else 2
                    _pcols    = st.columns(_n_cols)

                    with _pcols[0]:
                        style_tm = render_plot_editor("tm_plot", "Melting Curve", "#000000")
                        fig_mre  = go.Figure()
                        for _si, sname in enumerate(sel_thermo_samples):
                            if sname not in results: continue
                            _col = _COLORS_T[_si % len(_COLORS_T)]
                            fig_mre.add_trace(go.Scatter(
                                x=results[sname]["temp"], y=results[sname]["sig"],
                                mode='markers', name=f"{sname} data",
                                marker=dict(color=_col, size=8, symbol='circle',
                                            line=dict(width=1, color=_c["text"]))))
                            if sname in fit_results:
                                fig_mre.add_trace(go.Scatter(
                                    x=results[sname]["temp"],
                                    y=fit_results[sname]["fit_sig"],
                                    mode='lines', name=f"{sname} fit",
                                    line=dict(color=_col, width=2.5, dash='dash')))
                        fig_mre = apply_plot_style_custom(
                            fig_mre, style_tm,
                            "Melting Curve (data + two-state fit)" if fit_results else "Melting Curve",
                            "Temp (°C)", f"Signal @ {melt_wl:.1f} nm", plot_mode='lines+markers')
                        fig_mre.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.4)
                        st.plotly_chart(fig_mre, use_container_width=True)
                        try:
                            _dm1, _dm2 = st.columns(2)
                            _dm1.download_button("DL Melt PNG", fig_mre.to_image(format="png", scale=3), "melting_curve.png", key="dl_melt_png")
                            _dm2.download_button("DL Melt PDF", fig_mre.to_image(format="pdf"), "melting_curve.pdf", key="dl_melt_pdf")
                        except: pass

                    with _pcols[1]:
                        style_dg = render_plot_editor("dg_plot", "Apparent ΔG", "#000000")
                        fig_dg   = go.Figure()
                        for _si, sname in enumerate(sel_thermo_samples):
                            if sname not in results: continue
                            _col  = _COLORS_T[_si % len(_COLORS_T)]
                            d     = results[sname]
                            valid = ~np.isnan(np.array(d["delG"]).astype(float))
                            if np.sum(valid) > 1:
                                fig_dg.add_trace(go.Scatter(
                                    x=np.array(d["temp"])[valid],
                                    y=np.array(d["delG"]).astype(float)[valid],
                                    mode='lines+markers', name=sname,
                                    line=dict(color=_col, width=2.5),
                                    marker=dict(size=8, color=_col, symbol='circle',
                                                line=dict(width=1, color=_c["text"]))))
                        _dg_lbl = "Apparent ΔG" if model_type == "Two-state (van't Hoff)" else "Qualitative ΔG"
                        fig_dg = apply_plot_style_custom(
                            fig_dg, style_dg, f"{_dg_lbl} vs Temperature",
                            "Temp (°C)", f"{_dg_lbl} (kcal/mol) @ {melt_wl:.1f} nm",
                            plot_mode='lines+markers')
                        fig_dg.add_hline(y=0, line_dash="dash", line_color=_c["text"])
                        st.plotly_chart(fig_dg, use_container_width=True)
                        try:
                            _dd1, _dd2 = st.columns(2)
                            _dd1.download_button("DL ΔG PNG", fig_dg.to_image(format="png", scale=3), "delG_plot.png", key="dl_dg_png")
                            _dd2.download_button("DL ΔG PDF", fig_dg.to_image(format="pdf"), "delG_plot.pdf", key="dl_dg_pdf")
                        except: pass

                    if fit_results and _n_cols == 3:
                        with _pcols[2]:
                            st.markdown("###### Fit Residuals")
                            fig_res = go.Figure()
                            for _si, sname in enumerate(sel_thermo_samples):
                                if sname not in fit_results: continue
                                _col = _COLORS_T[_si % len(_COLORS_T)]
                                fig_res.add_trace(go.Scatter(
                                    x=results[sname]["temp"],
                                    y=fit_results[sname]["residuals"],
                                    mode='markers+lines', name=sname,
                                    line=dict(color=_col, width=1.5),
                                    marker=dict(size=6, color=_col)))
                            fig_res = apply_publication_style(
                                fig_res, "Fit Residuals",
                                "Temp (°C)", "Residual (data − fit)",
                                show_grid, height=380, plot_mode="lines+markers")
                            fig_res.add_hline(y=0, line_dash="dash", line_color=_c["text"], opacity=0.6)
                            st.plotly_chart(fig_res, use_container_width=True, key="tab6_residuals")
                            st.caption(
                                "Residuals should be randomly scattered around zero. "
                                "Systematic patterns indicate the two-state model is a poor fit."
                            )

                    # ── ΔΔG BUILDER ──────────────────────────────────────────────────────
                    st.divider()
                    st.subheader("⚖️ Custom ΔΔG Builder")
                    col_comp1, col_comp2 = st.columns(2)
                    n_comps  = col_comp1.number_input("Comparisons", 1, 10, 1)
                    temp_tol = col_comp2.number_input(
                        "Temp Match Tolerance (°C)", 0.01, 5.00, 0.50, step=0.10,
                        help="Adjust if CD machine records slightly different temperatures for different samples."
                    )
                    ddg_list  = []
                    fig_ddg   = go.Figure()
                    style_ddg = render_plot_editor("ddg_plot", "ΔΔG Plot", "#800080")
                    target_name, ref_name = "", ""
                    for i in range(n_comps):
                        c_ref, c_tgt = st.columns(2)
                        ref_name    = c_ref.selectbox(f"Reference {i+1}", list(results.keys()), key=f"ref_{i}")
                        target_name = c_tgt.selectbox(f"Target {i+1}",    list(results.keys()), key=f"tgt_{i}")
                        if ref_name and target_name:
                            ref_d = results[ref_name]
                            tgt_d = results[target_name]
                            common_temp_plot, ddg_values = [], []
                            for t_tgt in tgt_d["temp"]:
                                diffs    = [abs(t_tgt - t_ref) for t_ref in ref_d["temp"]]
                                if not diffs: continue
                                min_diff = min(diffs)
                                if min_diff <= temp_tol:
                                    idx_ref = diffs.index(min_diff)
                                    idx_tgt = tgt_d["temp"].index(t_tgt)
                                    dg_ref  = float(ref_d["delG"][idx_ref])
                                    dg_tgt  = float(tgt_d["delG"][idx_tgt])
                                    if not np.isnan(dg_ref) and not np.isnan(dg_tgt):
                                        val = dg_tgt - dg_ref
                                        common_temp_plot.append(t_tgt)
                                        ddg_values.append(val)
                                        ddg_list.append({
                                            "Comparison": f"{target_name} - {ref_name}",
                                            "Temperature (°C)": round(t_tgt, 1),
                                            f"ΔΔG (kcal/mol) @ {melt_wl:.1f} nm": round(val, 3)
                                        })
                            if common_temp_plot:
                                fig_ddg.add_trace(go.Scatter(
                                    x=common_temp_plot, y=ddg_values,
                                    mode='lines+markers',
                                    name=f"{target_name} − {ref_name}",
                                    marker=dict(size=style_ddg["marker_size"])))
                            else:
                                st.warning(f"⚠️ No matching temperatures for {target_name} vs {ref_name} within {temp_tol} °C.")

                    dynamic_title = (f"ΔΔG ({target_name} − {ref_name})"
                                     if n_comps == 1 and ddg_list else "ΔΔG Comparison")
                    fig_ddg = apply_plot_style_custom(
                        fig_ddg, style_ddg, dynamic_title,
                        "Temperature (°C)", f"ΔΔG (kcal/mol) @ {melt_wl:.1f} nm",
                        plot_mode='lines+markers')
                    fig_ddg.add_hline(y=0, line_dash="dash", line_color=_c["text"])
                    st.plotly_chart(fig_ddg)
                    if ddg_list:
                        df_ddg = pd.DataFrame(ddg_list)
                        st.markdown("##### ΔΔG Data Table")
                        st.dataframe(df_ddg, use_container_width=True)
                        st.download_button("💾 Download ΔΔG CSV", df_ddg.to_csv(index=False), "ddg_data.csv")
                    try:
                        col_d1, col_d2 = st.columns(2)
                        col_d1.download_button("DL ΔΔG PNG", fig_ddg.to_image(format="png", scale=3), "ddg_plot.png")
                        col_d2.download_button("DL ΔΔG PDF", fig_ddg.to_image(format="pdf"), "ddg_plot.pdf")
                    except: pass

                st.divider()
                with st.expander("📘 Methodological Note: Model & Equations", expanded=True):
                    st.markdown(r"""
                    **Two-State van't Hoff Model (default — recommended for sigmoidal melts):**

                    The CD signal at each temperature is fitted to a two-state model with independently
                    sloping pre- and post-transition baselines:

                    $$y(T) = \frac{(a_F + b_F T) + (a_U + b_U T)\,e^{-\Delta G(T)/RT}}{1 + e^{-\Delta G(T)/RT}}$$

                    where: $\Delta G(T) = \Delta H_{vH}\left(1 - \frac{T}{T_m}\right)$

                    Fitted parameters: $a_F, b_F$ (folded baseline), $a_U, b_U$ (unfolded baseline),
                    $\Delta H_{vH}$ (van't Hoff enthalpy, kcal/mol), $T_m$ (melting temperature, K).

                    **Why this is better than min-max normalisation:**
                    Min-max normalisation forces $\alpha$ to span 0→1 for every sample regardless of
                    signal amplitude, which is why buffer vs SDS conditions appear to give identical ΔG.
                    The two-state fit uses the actual signal values and cooperativity to extract ΔG,
                    so samples with different structural stability in different solvents will give
                    genuinely different ΔG profiles.

                    **Confidence criteria:** R² ≥ 0.98 = ✅ Valid | 0.90–0.98 = 🟡 Caution | < 0.90 = 🔴 Invalid

                    **SDS / non-aqueous conditions:** ΔG reflects apparent structural stabilisation
                    in the micellar environment, not intrinsic aqueous folding free energy.
                    Use ΔΔG to compare conditions at each temperature.
                    """)


            if _th_tab == "🔮 Spectral Simulation":
                st.subheader("🔮 Thermodynamic Spectral Simulation")
                st.markdown("""
                **Methodology:** This module uses continuous mathematical interpolation across the entire temperature gradient to accurately simulate the CD spectrum at unmeasured intermediate temperatures. 
                
                ⚠️ **Extrapolation Warning:** You can simulate temperatures outside your measured range. However, please note that mathematical extrapolation assumes the unfolding trend continues at its current rate. If your protein has a sudden, cooperative melting point that you did not physically measure, the math cannot predict that sudden cliff!
                """)
                
                c_sim1, c_sim2 = st.columns([1, 2])
                
                with c_sim1:
                    min_t = float(min(curr_temps))
                    max_t = float(max(curr_temps))
                    
                    st.info(f"**Measured Range:** {min_t}°C to {max_t}°C")
                    
                    # --- FIX 1: Removed min_value and max_value locks ---
                    target_temp = st.number_input(
                        "Target Temperature (°C)", 
                        value=80.0, 
                        step=0.1,
                        help="You can enter temperatures inside or outside your measured range!"
                    )
                    
                    # Choose interpolation method
                    sim_method = st.radio("Interpolation Math", ["Linear (Strict & Safe)", "Cubic Spline (Smoothest)"])
                    method_key = 'cubic' if sim_method == "Cubic Spline (Smoothest)" else 'linear'
                    
                    if target_temp > max_t or target_temp < min_t:
                        st.warning("You are extrapolating outside your measured data. 'Linear' math is highly recommended to prevent wild polynomial swings.")

                with c_sim2:
                    # Perform Wavelength-by-Wavelength Interpolation
                    simulated_sig = []
                    
                    # We look at every single wavelength one by one
                    for i, w in enumerate(curr_wl):
                        # Extract the signal for this specific wavelength across ALL measured temperatures
                        y_vals_at_w = [d["sig"][i] for d in curr_data]
                        
                        # --- FIX 2: Added `fill_value="extrapolate"` to unlock SciPy's boundary limits ---
                        f_t = interp1d(curr_temps, y_vals_at_w, kind=method_key, bounds_error=False, fill_value="extrapolate")
                        
                        # Calculate the exact signal at the user's target temperature
                        simulated_sig.append(float(f_t(target_temp)))
                        
                    # Build the plot
                    style_sim = render_plot_editor("sim_plot", f"Simulated Spectrum at {target_temp}°C", "#D95319")
                    
                    fig_sim = go.Figure()
                    
                    # Optional: Add boundary curves for context
                    fig_sim.add_trace(go.Scatter(x=curr_wl, y=curr_data[0]["sig"], mode='lines', name=f"Measured {min_t}°C", line=dict(color='blue', dash='dash', width=1)))
                    fig_sim.add_trace(go.Scatter(x=curr_wl, y=curr_data[-1]["sig"], mode='lines', name=f"Measured {max_t}°C", line=dict(color='red', dash='dash', width=1)))
                    
                    # Add the simulated curve
                    fig_sim.add_trace(go.Scatter(x=curr_wl, y=simulated_sig, mode='lines', name=f"Simulated {target_temp}°C", line=dict(color=style_sim["color"], width=3)))
                    
                    fig_sim = apply_plot_style_custom(fig_sim, style_sim, f"Simulated Spectrum ({target_temp} °C)", "Wavelength (nm)", y_lbl_full, plot_mode='lines')
                    fig_sim.add_hline(y=0, line_dash="dash", line_color=_c["text"])
                    fig_sim.update_xaxes(range=[wl_min, wl_max])
                    if not y_auto: fig_sim.update_yaxes(range=[y_min, y_max])
                    
                    st.plotly_chart(fig_sim, use_container_width=True)
                    
                    # Download the simulated data
                    df_sim = pd.DataFrame({
                        "Wavelength (nm)": curr_wl,
                        f"Simulated_Signal_at_{target_temp}C": simulated_sig
                    })
                    
                    col_dl1, col_dl2 = st.columns(2)
                    col_dl1.download_button("💾 Download Simulated Data (CSV)", df_sim.to_csv(index=False), f"Simulated_{target_temp}C.csv")
                    try: col_dl2.download_button("📸 Download Simulation Plot", fig_sim.to_image(format="png", scale=3), f"SimPlot_{target_temp}C.png")
                    except: pass
        else: st.info("👈 Set the number of samples and upload your files in the sidebar.")

    # ── MODULE 3: REVERSIBILITY ANALYSIS ─────────────────────────────────────────
    elif mode == "Reversibility Analysis":
        st.title("🔄 Thermal Reversibility Analysis")
        with st.sidebar:
            st.markdown("### 📂 Upload Files")

            # ── Melt file + name (kept together) ─────────────────────────────
            st.markdown("**Thermal start / melt file**")
            c_r1a, c_r1b = st.columns([2, 1.3])
            with c_r1a:
                f_melt = st.file_uploader(
                    "Upload (.txt)", type=["txt"], key="rev_melt",
                    label_visibility="collapsed"
                )
            with c_r1b:
                name_melt = st.text_input(
                    "Sample name", "Melt", key="nm_melt",
                    help="This name will appear in the plot legend and exported files."
                )

            # ── Refolded file + name ──────────────────────────────────────────
            st.markdown("**Refolded spectrum**")
            c_r2a, c_r2b = st.columns([2, 1.3])
            with c_r2a:
                f_single = st.file_uploader(
                    "Upload (.txt)", type=["txt"], key="rev_single",
                    label_visibility="collapsed"
                )
            with c_r2b:
                name_single = st.text_input(
                    "Sample name", "Refolded", key="nm_single",
                    help="This name will appear in the plot legend and exported files."
                )

            # ── Blank ─────────────────────────────────────────────────────────
            st.markdown("**Buffer / blank (optional)**")
            f_blank = st.file_uploader("Upload (.txt)", type=["txt"], key="rev_blank",
                                       label_visibility="collapsed")

            st.markdown("### ⚙️ Settings")
            metric_rev  = st.selectbox(
                "Output Metric",
                ["MRE", "Raw (mdeg)", "Δε (M⁻¹cm⁻¹)"],
                key="rev_metric",
                help="MRE: publication standard. Raw (mdeg): no normalisation. Δε: molar CD."
            )
            path_rev = st.number_input("Path (cm)", 0.1, key="rev_path")
            conc_rev = st.number_input("Conc (µM)", 50.0, key="rev_conc")
            # --- SMART SEQUENCE PARSER (Reversibility) ---
            seq_input_rev = st.text_input("Peptide Sequence (Optional)", key="rev_seq", placeholder="e.g., ALYFWC...")
            clean_seq_rev = "".join([char.upper() for char in seq_input_rev if char.isalpha()])
            
            if clean_seq_rev:
                res_rev = len(clean_seq_rev)
                num_W = clean_seq_rev.count('W')
                num_Y = clean_seq_rev.count('Y')
                num_C = clean_seq_rev.count('C')
                ext_coeff = (num_W * 5500) + (num_Y * 1490) + (int(num_C / 2) * 125)
                mw_est = (res_rev * 110) + 18
                
                st.info(f"✅ **Auto-counted: {res_rev} residues**")
                c_seq1, c_seq2 = st.columns(2)
                c_seq1.caption(f"🧮 **ε₂₈₀:** {ext_coeff} M⁻¹cm⁻¹")
                c_seq2.caption(f"⚖️ **MW:** ~{mw_est} Da")
            else:
                res_rev = st.number_input("Or enter number of residues manually:", value=6, key="rev_res_manual")
            # ---------------------------------------------


            # ── INPUT FORMAT: AUTO-DETECT for BOTH files ──────────────────
            REV_FMT_OPTIONS = [
                "mdeg (Raw CD Signal)",
                "Δε — Molar CD (per molecule)",
                "Δε — Mean Residue (per residue)",
            ]
            _rev_tag_to_idx = {"mdeg": 0, "delta_eps_molar": 1, "delta_eps_residue": 2, "unknown": 0}
            _rev_badge_map  = {
                "mdeg":             "✅ Auto-detected: **mdeg**",
                "delta_eps_molar":  "✅ Auto-detected: **Mol. CD (Δε per molecule)**",
                "delta_eps_residue":"✅ Auto-detected: **Mean Residue Δε**",
                "unknown":          "⚠️ Not found in header — select manually.",
            }

            st.markdown("**Thermal start file format:**")
            _melt_tag = detect_yunits(f_melt)
            if f_melt: st.caption(_rev_badge_map.get(_melt_tag, ""))
            fmt_melt = st.selectbox(
                "Melt Input Format" + (" (override)" if _melt_tag != "unknown" and f_melt else ""),
                REV_FMT_OPTIONS, index=_rev_tag_to_idx.get(_melt_tag, 0),
                key="rev_fmt_melt"
            )

            st.markdown("**Refolded file format:**")
            _single_tag = detect_yunits(f_single)
            if f_single: st.caption(_rev_badge_map.get(_single_tag, ""))
            fmt_single = st.selectbox(
                "Refolded Input Format" + (" (override)" if _single_tag != "unknown" and f_single else ""),
                REV_FMT_OPTIONS, index=_rev_tag_to_idx.get(_single_tag, 0),
                key="rev_fmt_single"
            )

            # Warn if the two files are in different formats
            if f_melt and f_single and fmt_melt != fmt_single:
                st.warning(
                    "⚠️ **Mixed formats detected.**  "
                    f"Melt: `{fmt_melt}` vs Refolded: `{fmt_single}`.  "
                    "Both will be converted to the selected output metric, but verify "
                    "that your concentration and residue inputs are correct for each file."
                )

            # Smoothing
            apply_smooth_r = st.checkbox("Apply Smoothing", value=True, key="rev_apply_smooth")
            if apply_smooth_r:
                smooth_method_r = st.radio("Method", ["Savitzky-Golay", "LOWESS"], index=0, key="rev_smooth")
                smooth_val_r = st.slider("Frac/Window", 0.01 if smooth_method_r == "LOWESS" else 5,
                                         0.30 if smooth_method_r == "LOWESS" else 51, key="rev_sw")
            else:
                smooth_method_r = "None"
                smooth_val_r = 0

        if f_melt and f_single:
            df_melt, melt_temps = read_thermal_file(f_melt)
            df_single = read_cd_file(f_single)
            df_blank  = read_cd_file(f_blank)
            if df_melt is not None and df_single is not None:
                selected_temp = st.selectbox("Select Temperature from Melt to Compare:", melt_temps)
                wl_m, sig_m = df_melt["Wavelength"].values, df_melt[f"{selected_temp}"].values
                if "CD" not in df_single.columns: st.error("Column 'CD' not found."); st.stop()
                wl_s, sig_s = df_single["Wavelength"].values, df_single["CD"].values

                # Blank subtraction on refolded spectrum only
                if df_blank is not None:
                    f_b = interp1d(df_blank["Wavelength"], df_blank["CD"], bounds_error=False, fill_value="extrapolate")
                    sig_s -= f_b(wl_s)

                # ── CONVERSION HELPER ─────────────────────────────────────
                def _rev_convert(sig, fmt, metric, path, conc_uM, nres):
                    """Apply the correct unit conversion for reversibility module."""
                    _nres = nres if nres > 0 else 1
                    _fac  = 10 * path * (conc_uM * 1e-6) * nres
                    _is_mdeg = "mdeg" in fmt
                    _is_dm   = "Molar" in fmt
                    _is_dr   = "Mean Residue" in fmt
                    if _is_mdeg:
                        if metric == "MRE":
                            return sig / _fac / 1000 if _fac != 0 else sig
                        elif metric == "Δε (M⁻¹cm⁻¹)":
                            return (sig / _fac / 1000) / 3298.2 * 1000 if _fac != 0 else sig
                    elif _is_dm:
                        if metric == "MRE":
                            return (sig * 3298.2 / _nres) / 1000
                        elif metric == "Raw (mdeg)":
                            return sig * 3298.2 / _nres * _fac
                        elif metric == "Δε (M⁻¹cm⁻¹)":
                            return sig / _nres
                    elif _is_dr:
                        if metric == "MRE":
                            return (sig * 3298.2) / 1000
                        elif metric == "Raw (mdeg)":
                            return sig * 3298.2 * _fac
                    return sig  # HT, Abs, or no conversion needed

                sig_m = _rev_convert(sig_m, fmt_melt,   metric_rev, path_rev, conc_rev, res_rev)
                sig_s = _rev_convert(sig_s, fmt_single, metric_rev, path_rev, conc_rev, res_rev)

                # Align refolded onto melt wavelength grid
                f_interp = interp1d(wl_s, sig_s, bounds_error=False, fill_value="extrapolate")
                sig_s_aligned = f_interp(wl_m)

                # Smoothing
                if apply_smooth_r:
                    if smooth_method_r == "LOWESS":
                        _, sig_m          = apply_smoothing(wl_m, sig_m,          "LOWESS (Match R)", smooth_val_r)
                        _, sig_s_aligned  = apply_smoothing(wl_m, sig_s_aligned,  "LOWESS (Match R)", smooth_val_r)
                    else:
                        _, sig_m          = apply_smoothing(wl_m, sig_m,          "Savitzky-Golay", int(smooth_val_r))
                        _, sig_s_aligned  = apply_smoothing(wl_m, sig_s_aligned,  "Savitzky-Golay", int(smooth_val_r))

                # ── PLOT CUSTOMISATION ────────────────────────────
                _wl_actual_min = int(np.floor(wl_m.min()))
                _wl_actual_max = int(np.ceil(wl_m.max()))
                with st.expander("🛠️ Plot Customisation", expanded=True):
                    _rc1, _rc2, _rc3 = st.columns(3)
                    with _rc1:
                        st.markdown("**Axis limits**")
                        _rev_xmin  = st.number_input("X Min (nm)", value=float(_wl_actual_min), step=1.0, key="rev_xmin")
                        _rev_xmax  = st.number_input("X Max (nm)", value=float(_wl_actual_max), step=1.0, key="rev_xmax")
                        _rev_yauto = st.checkbox("Auto Y-scale", value=True, key="rev_yauto")
                        if not _rev_yauto:
                            _rev_ymin = st.number_input("Y Min", value=-30.0, step=1.0, key="rev_ymin")
                            _rev_ymax = st.number_input("Y Max", value=30.0,  step=1.0, key="rev_ymax")
                        else:
                            _rev_ymin, _rev_ymax = None, None
                    with _rc2:
                        st.markdown("**Line colours & width**")
                        _col_melt = st.color_picker(f"Melt ({name_melt})", "#CC0000", key="rev_col_melt")
                        _col_ref  = st.color_picker(f"Refolded ({name_single})", "#0044CC", key="rev_col_ref")
                        _rev_lw   = st.slider("Line width", 0.5, 5.0, 2.5, 0.5, key="rev_lw")
                    with _rc3:
                        st.markdown("**Grid**")
                        _rev_grid = st.checkbox("Show grid lines", value=True, key="rev_grid")

                # ── PEAK SEARCH RANGE (optional) ───────────────────
                with st.expander("🔍 Peak Search Range (optional)", expanded=False):
                    st.caption(
                        "By default the software searches the full visible wavelength range. "
                        "Restrict the windows below if noise near 195–200 nm is being picked "
                        "up instead of the true helical bands at 208/222 nm. "
                        "Results appear as a table below the plot."
                    )
                    _rp1, _rp2 = st.columns(2)
                    with _rp1:
                        st.markdown("**↓ Minimum search window**")
                        _rev_use_min = st.checkbox("Restrict minimum search", value=False, key="rev_use_min")
                        _rev_min_lo = st.number_input("Min search start (nm)", 170.0, 350.0, 205.0, step=1.0, key="rev_min_lo", disabled=not _rev_use_min)
                        _rev_min_hi = st.number_input("Min search end (nm)",   170.0, 350.0, 230.0, step=1.0, key="rev_min_hi", disabled=not _rev_use_min)
                    with _rp2:
                        st.markdown("**↑ Maximum search window**")
                        _rev_use_max = st.checkbox("Restrict maximum search", value=False, key="rev_use_max")
                        _rev_max_lo = st.number_input("Max search start (nm)", 170.0, 350.0, 185.0, step=1.0, key="rev_max_lo", disabled=not _rev_use_max)
                        _rev_max_hi = st.number_input("Max search end (nm)",   170.0, 350.0, 205.0, step=1.0, key="rev_max_hi", disabled=not _rev_use_max)
                    st.info(
                        "💡 For α-helix: minimum 205–230 nm (208 & 222 nm bands), maximum 185–200 nm. "
                        "For β-sheet: minimum 210–225 nm."
                    )
                # Resolved ranges
                _rev_min_r = (_rev_min_lo, _rev_min_hi) if _rev_use_min else (_wl_actual_min, _wl_actual_max)
                _rev_max_r = (_rev_max_lo, _rev_max_hi) if _rev_use_max else (_wl_actual_min, _wl_actual_max)

                # Y-axis label
                _rev_ylbl = {"MRE": "MRE [θ] (x10³ deg cm² dmol⁻¹ res⁻¹)",
                             "Raw (mdeg)": "CD (mdeg)",
                             "Δε (M⁻¹cm⁻¹)": "Δε (M⁻¹ cm⁻¹)"}.get(metric_rev, "Signal")

                fig_rev = go.Figure()
                fig_rev.add_trace(go.Scatter(x=wl_m, y=sig_m, mode="lines",
                                             name=f"{name_melt} ({selected_temp}°C)",
                                             line=dict(color=_col_melt, width=_rev_lw)))
                fig_rev.add_trace(go.Scatter(x=wl_m, y=sig_s_aligned, mode="lines",
                                             name=name_single,
                                             line=dict(color=_col_ref, width=_rev_lw)))
                fig_rev = apply_publication_style(fig_rev, "Reversibility Check",
                                                  "Wavelength (nm)", _rev_ylbl, _rev_grid, plot_mode="lines")
                # Zero reference line always visible regardless of grid setting
                fig_rev.add_hline(y=0, line_dash="dash", line_color=_c["text"], line_width=1, opacity=0.5)
                # Apply axis ranges
                fig_rev.update_xaxes(range=[_rev_xmin, _rev_xmax])
                if not _rev_yauto:
                    fig_rev.update_yaxes(range=[_rev_ymin, _rev_ymax])
                st.plotly_chart(fig_rev, use_container_width=True, key="rev_main_chart")
                
                # --- EXPORT DATA & PLOTS ---
                st.markdown("##### 📥 Export Data & Plots")
                
                # 1. Prepare CSV Data
                df_rev_export = pd.DataFrame({
                    "Wavelength (nm)": wl_m,
                    f"{name_melt} ({selected_temp}°C)": sig_m,
                    f"{name_single} (Aligned)": sig_s_aligned
                }).round(4)
                
                c_dl1, c_dl2, c_dl3 = st.columns(3)
                
                try: c_dl1.download_button("💾 Download Data (CSV)", df_rev_export.to_csv(index=False), "reversibility_data.csv", "text/csv", key="rev_dl_csv")
                except: pass
                try: c_dl2.download_button("📸 Download Plot (PNG)", fig_rev.to_image(format="png", scale=3), "reversibility_plot.png", "image/png", key="rev_dl_png")
                except: pass
                try: c_dl3.download_button("📄 Download Plot (PDF)", fig_rev.to_image(format="pdf"), "reversibility_plot.pdf", "application/pdf", key="rev_dl_pdf")
                except: pass
                
                # ── PEAK STATISTICS TABLE ────────────────────────────
                st.divider()
                st.markdown("##### 📋 Peak Statistics")
                st.caption(
                    "Min and max values computed within the search windows set above. "
                    "Adjust the Peak Search Range expander to restrict the wavelength "
                    "window if noise is being picked up instead of the true bands."
                )
                _peak_rows = []
                for _sig_arr, _lbl in [
                    (sig_m,         f"{name_melt} ({selected_temp}°C)"),
                    (sig_s_aligned, name_single)
                ]:
                    # Min within restricted (or full) window
                    _mm_min = get_min_max(wl_m, _sig_arr, _rev_min_r[0], _rev_min_r[1])
                    # Max within restricted (or full) window
                    _mm_max = get_min_max(wl_m, _sig_arr, _rev_max_r[0], _rev_max_r[1])
                    _peak_rows.append({
                        "Spectrum": _lbl,
                        "λ Min 1 (nm)":     _mm_min["Lambda Min 1 (nm)"] if _mm_min else "N/A",
                        "Min 1 Intensity": _mm_min["Min 1 Value"]       if _mm_min else "N/A",
                        "λ Min 2 (nm)":     _mm_min["Lambda Min 2 (nm)"] if _mm_min else "N/A",
                        "Min 2 Intensity": _mm_min["Min 2 Value"]       if _mm_min else "N/A",
                        "λ Max (nm)":      _mm_max["Lambda Max (nm)"]  if _mm_max else "N/A",
                        "Max Intensity":  _mm_max["Max Value"]          if _mm_max else "N/A",
                    })
                if _peak_rows:
                    df_peaks = pd.DataFrame(_peak_rows).set_index("Spectrum")
                    st.dataframe(df_peaks, use_container_width=True)
                    try:
                        st.download_button(
                            "💾 Download Peak Stats (CSV)",
                            df_peaks.to_csv(),
                            "rev_peak_stats.csv", "text/csv",
                            key="rev_dl_peaks"
                        )
                    except: pass

                st.divider()
                
                # --- STATISTICS CALCULATIONS ---
                valid_mask = ~np.isnan(sig_m) & ~np.isnan(sig_s_aligned)
                rmsd = np.sqrt(np.mean((sig_m[valid_mask] - sig_s_aligned[valid_mask])**2))
                
                # Calculate signal range for normalization
                signal_range = np.max(sig_m[valid_mask]) - np.min(sig_m[valid_mask])
                nrmsd = (rmsd / signal_range) * 100 if signal_range != 0 else 0
                
                corr = np.corrcoef(sig_m[valid_mask], sig_s_aligned[valid_mask])[0,1]
                
                st.subheader("Statistical Analysis")
                c1, c2, c3 = st.columns(3)
                c1.metric("Correlation", f"{corr:.3f}")
                c2.metric("Abs. RMSD", f"{rmsd:.2f}")
                c3.metric("Normalized RMSD", f"{nrmsd:.1f} %")
                
                interp = "Irreversible"
                if corr >= 0.98 and nrmsd < 5.0: interp = "Excellent Reversibility"
                elif corr >= 0.90: interp = "Good Reversibility"
                elif corr >= 0.70: interp = "Partial Reversibility"
                st.info(f"📝 **Result:** {interp}")
                
                # --- EXPLANATION DROPDOWN ---
                with st.expander("💡 Understanding RMSD & Interpretation Ranges"):
                    st.markdown("""
                    **What is Absolute RMSD?**
                    Root Mean Square Deviation (RMSD) measures the average absolute mathematical distance between the Start and Refolded spectra across all wavelengths. A value of exactly 0.0 means perfect overlap.
                    
                    **What is Normalized RMSD (NRMSD %)?**
                    An Absolute RMSD of "2.0" might be terrible if your peptide has a very weak CD signal, but excellent if your peptide has a massive CD signal. To account for this, the software **normalizes the RMSD against the peak-to-trough range of the starting spectrum.** This gives you the error as a straightforward percentage of the total signal scale.
                    
                    **How does the software judge the result?**
                    * 🟢 **Excellent Reversibility:** Correlation ≥ 0.98 **AND** Normalized RMSD < 5.0%
                    * 🔵 **Good Reversibility:** Correlation ≥ 0.90
                    * 🟡 **Partial Reversibility:** Correlation ≥ 0.70
                    * 🔴 **Irreversible:** Correlation < 0.70
                    """)
            else: st.error("Error reading files. Please check the formats.")
        else: st.info("👈 Upload your thermal start and refolded spectra in the sidebar to begin.")
