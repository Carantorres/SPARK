import os
import re
import io
import inspect
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.signal import savgol_filter, find_peaks
from scipy.stats import linregress
from scipy.optimize import least_squares
import streamlit as st
import streamlit.components.v1 as components
from streamlit_sortables import sort_items

# ============================================================
# PAGE CONFIGURATION & WHITE THEME
# ============================================================
st.set_page_config(page_title="SPARK Analyzer", layout="wide")

# White (light) theme. The recommended way is the file .streamlit/config.toml shipped
# next to this script. This fallback also forces the light theme when the app runs
# without that file, even if the computer/browser is in dark mode.
WHITE_THEME = {
    "base": "light",
    "primaryColor": "#FF4B4B",
    "backgroundColor": "#FFFFFF",
    "secondaryBackgroundColor": "#F4F6F8",
    "textColor": "#1F2328",
}


def _force_light_theme() -> bool:
    try:
        from streamlit import config as _st_config
        if _st_config.get_option("theme.base") == "light":
            return False
        for _k, _v in WHITE_THEME.items():
            try:
                _st_config.set_option(f"theme.{_k}", _v)
            except Exception:
                pass
        return _st_config.get_option("theme.base") == "light"
    except Exception:
        return False


if not st.session_state.get("_spark_theme_checked", False):
    st.session_state["_spark_theme_checked"] = True
    if _force_light_theme():
        st.rerun()

st.markdown("""
<style>
section[data-testid="stSidebar"] { border-right: 1px solid rgba(49, 51, 63, 0.12); }
div[data-testid="stMetric"] { border: 1px solid rgba(49, 51, 63, 0.12); border-radius: 8px; padding: 8px 12px; }
</style>
""", unsafe_allow_html=True)

st.title("⚡ SPARK")
st.markdown("### System for Potentiostat Analysis & Research Knowledge")

st.markdown("""
**A comprehensive tool for automated chemical and electrochemical data analysis.**
Seamlessly process CV, LSV, CP (chronopotentiometry), EIS files from Gamry, Biologic, PalmSens and **Metrohm Autolab (NOVA)**, plus **Chemical Speciation Diagrams**. Features robust catalytic parameter extraction, noise-free potential window detection, and equivalent circuit fitting for impedance spectroscopy.
""")

with st.popover("📖 View Calculation Methods & Algorithms"):
    st.markdown("""
    **1. Physico-Chemical Corrections**
    *   **iR Drop Compensation:** $E_{corr} = E - I \\cdot R_u \\cdot (\\%comp/100)$, applied to voltammetry (CV/LSV). CP and EIS data are not iR-corrected.
    *   **RHE Scale Conversion:** $E_{RHE} = E_{ref} + E^0_{ref} + 0.0591 \\cdot pH$.
    *   **Current Density Normalization:** Raw current is scaled to geometric current density $j$ (mA/cm²).

    **2. ECSA & Capacitance Extraction (CV)**
    *   **$C_{dl}$ & Roughness Factor (RF):** $\\Delta j / 2$ is read at a user-defined non-faradaic potential on the anodic and cathodic sweeps (sweeps are located from the potential vertices, whatever the starting direction). Regression is forced through the origin ($y = C_{dl} \\cdot v$). $RF = C_{dl} / C_{dl,ref}$. Requires ≥ 2 different scan rates.
    *   **ECSA-Normalized Kinetics:** $j_{p,ECSA} = j_p / RF$.

    **3. Catalytic Parameter Extraction (LSV)**
    *   **Polarity:** detected automatically from the sign of the current at $\\vert{}I\\vert{}_{max}$ (cathodic → HER/reduction, anodic → OER/UOR/oxidation).
    *   **Onset Potential ($E_{onset}$):** walking from the low-current end of the curve towards the current maximum, first potential where $\\vert{}I\\vert{}$ reaches (and stays above, 3 points) 5% of $\\vert{}I\\vert{}_{max}$.
    *   **Overpotentials ($\\eta_j$):** $\\eta = \\vert{}E(j) - E_{rev}\\vert{}$ read on the same branch, with linear interpolation.
    *   **Robust Tafel Slope:** sliding window on the reaction branch (2–40% of $\\vert{}I\\vert{}_{max}$), maximizing $R^2$.

    **4. Scan Rate Kinetics ($b$-value)**
    *   A linear regression of $\\log_{10}(j_{pa})$ vs $\\log_{10}(v)$ calculates the slope $b$ (requires ≥ 2 different scan rates).

    **5. Electrochemical Impedance Spectroscopy (EIS)**
    *   **Equivalent Circuit Fitting:** Complex Non-Linear Least Squares (CNLS) with **Modulus Weighting** ($\\sigma = \\vert{}Z\\vert{}$). $R_s$ is fitted together with the other elements. Multi-start + hierarchical seeding, positive parameters fitted in log-scale.
    *   **Goodness of fit:** $\\chi^2 = \\sum[(\\Delta Z'^2 + \\Delta Z''^2)/\\vert{}Z\\vert{}^2] / (2N - p)$; parameter errors (%) from the Jacobian. Parameters stuck at a bound are flagged.
    *   **Effective Capacitance ($C_{eff}$, Hsu–Mansfeld):** $C_{eff} = Q^{1/n} \\cdot R^{(1-n)/n}$, where $R$ is the resistor in parallel with that CPE.
    *   **Multi-time-constant models:** elements are ordered by time constant (1 = highest frequency); the lowest-frequency $R$ is reported as $R_{ct}$. $R_p = Z'(\\omega \\to 0) - R_s$.
    *   **Area normalization:** $R_{ct} \\cdot A$ (Ω·cm²$_{geo}$) and $R_{ct} \\cdot A \\cdot RF$ (Ω·cm²$_{ECSA}$).
    *   **Polarization Sensitivity:** $R_{ct}$ ratios to evaluate potential-dependent activity.

    **6. Chemical Speciation (Medusa)**
    *   Features a custom reverse-engineering algorithm that reads raw `.plt` vector files and maps screen coordinates back to chemical thermodynamic data (Log Conc / Fraction vs pH) to generate high-quality plots.

    **7. Supported files (auto-detected)**
    *   Gamry `.DTA` · Biologic `.mpt` / EC-Lab text · PalmSens PSTrace `.csv` · **Metrohm Autolab NOVA ASCII `.txt`** (EIS, CV, LSV, CP) · Medusa `.plt`.
    """)

st.markdown("---")

# ============================================================
# SIDEBAR SETUP (ALL SETTINGS DECLARED FIRST)
# ============================================================
with st.sidebar:
    st.header("🎛️ Instrument Format")
    instrument = st.selectbox(
        "Default parser (used only if a file can't be identified):",
        [
            "Gamry 1010B (.DTA)",
            "Biologic SP-50e (.mpt)",
            "PalmSens PSTrace (.csv)",
            "Metrohm Autolab PGSTAT204 – NOVA (.txt)",
            "Hydra & Medusa Speciation (.plt / .txt)"
        ]
    )
    st.info("💡 Gamry, Biologic, PalmSens, Metrohm NOVA (.txt) and Medusa (.plt) files are auto-detected!")

    st.markdown("---")
    st.header("⚡ iR Drop Compensation")
    apply_ir = st.toggle("Apply iR Compensation", value=False, help="E_corr = E − I·Ru·(%/100). Applied to CV/LSV data (CP and EIS are not iR-corrected).")
    ru_ohms = st.number_input("Uncompensated Resistance (Ru) [Ohms]", value=10.0, step=1.0) if apply_ir else 0.0
    comp_percent = st.slider("Compensation Percentage (%)", 0, 100, 85, 1) if apply_ir else 0.0

    st.markdown("---")
    st.header("⚖️ Reference Electrode & RHE")
    ref_elec = st.selectbox("Reference Electrode", ["Ag/AgCl (sat. KCl)", "SCE (sat. KCl)", "Hg/HgO (1M KOH)", "Custom"])
    custom_ref_name = st.text_input("Custom Reference Label", value="Ref.") if ref_elec == "Custom" else ref_elec.split(" (")[0]

    convert_to_rhe = st.toggle("Convert E to RHE scale", value=True)
    if convert_to_rhe:
        e0_ref = st.number_input("Custom E0_Ref (V vs SHE)", value=0.000, step=0.01) if ref_elec == "Custom" else (0.197 if ref_elec.startswith("Ag") else (0.241 if ref_elec.startswith("SCE") else 0.098))
        if ref_elec != "Custom": st.info(f"Using Standard E₀ = {e0_ref} V")
        ph_val = st.number_input("pH of the solution", value=14.0, step=0.1)
        E_label_base = "E (V vs RHE)"
    else:
        e0_ref, ph_val = 0.0, 0.0
        E_label_base = f"E (V vs {custom_ref_name})"
    x_axis_label = E_label_base + (" [iR corrected]" if apply_ir else "")

    st.markdown("---")
    st.header("⚙️ Catalytic Parameters")
    electrode_area = st.number_input("Electrode Area (cm²)", min_value=0.00001, value=1.00000, step=0.001, format="%.5f")
    manual_scan_rate = st.number_input("Manual Scan Rate (mV/s) [Optional]", value=0.0, step=10.0, help="Overrides the scan rate stored in the files. A scan rate written in the file name or legend label (e.g. 'CV_50mVs', '50 mV/s') takes priority.")
    e_rev = st.number_input("Thermodynamic Potential E_rev (V, same scale as plots)", value=0.000, step=0.01, help="HER: 0.000 V vs RHE · OER: 1.229 V vs RHE · UOR: 0.37 V vs RHE")

    st.markdown("---")
    st.header("📏 ECSA & C_dl Parameters")
    c_dl_ref = st.number_input("Reference C_dl (mF/cm²)", min_value=0.001, value=0.040, step=0.001, format="%.3f")
    e_non_faradaic = st.number_input("Non-Faradaic E (V, same scale as plots) for C_dl", value=0.25, step=0.05)
    manual_rf = st.number_input("Roughness Factor RF for EIS (Optional)", value=1.0, min_value=0.0, step=0.1, help="If known from C_dl, the EIS table adds Rct·A·RF (resistance per cm² of electrochemically active area).")

    st.markdown("---")
    st.header("🔋 Peak Search (Kinetics)")
    limit_peak_search = st.toggle("Limit Peak Search Window", value=False)
    if limit_peak_search:
        c_min, c_max = st.columns(2)
        with c_min: peak_min_v = st.number_input("Min E (V)", value=0.20, step=0.05)
        with c_max: peak_max_v = st.number_input("Max E (V)", value=0.60, step=0.05)
    else:
        peak_min_v, peak_max_v = None, None

    st.markdown("---")
    st.header("✂️ EIS Frequency Cropping")
    crop_eis = st.toggle("Limit Frequency Range", value=False, help="Discard noisy data at very low or high frequencies before fitting (e.g. gas bubble noise at low Hz).")
    if crop_eis:
        c_fmin, c_fmax = st.columns(2)
        with c_fmin: eis_min_f = st.number_input("Min Freq (Hz)", value=0.05, format="%.3f")
        with c_fmax: eis_max_f = st.number_input("Max Freq (Hz)", value=100000.0, step=1000.0)
    else:
        eis_min_f, eis_max_f = 1e-9, 1e9

    st.markdown("---")
    st.header("✂️ Medusa Axis Cropping")
    crop_medusa = st.toggle("Limit X/Y Axes (Medusa Only)", value=False, help="Crop the pH and Log Concentration/Fraction axes.")
    if crop_medusa:
        c_mx, c_my = st.columns(2)
        with c_mx:
            medusa_xmin = st.number_input("Min X (pH)", value=4.0, step=0.5)
            medusa_xmax = st.number_input("Max X (pH)", value=5.0, step=0.5)
        with c_my:
            medusa_ymin = st.number_input("Min Y", value=-4.0, step=0.5)
            medusa_ymax = st.number_input("Max Y", value=-1.0, step=0.5)
    else:
        medusa_xmin, medusa_xmax, medusa_ymin, medusa_ymax = None, None, None, None

    st.markdown("---")
    st.header("🎨 Plot Formatting")
    scientific_style = st.toggle("Scientific Paper Style (ACS/Elsevier)", value=True)
    show_sd_shadow = st.toggle("Show SD Shadow on Averages", value=True)
    leg_pos = st.selectbox("Quick Positions", ["Top-Right", "Top-Left", "Bottom-Right", "Bottom-Left", "Outside Right", "Custom..."])
    if leg_pos == "Top-Right": lx, ly, lxa, lya = 0.99, 0.99, "right", "top"
    elif leg_pos == "Top-Left": lx, ly, lxa, lya = 0.01, 0.99, "left", "top"
    elif leg_pos == "Bottom-Right": lx, ly, lxa, lya = 0.99, 0.01, "right", "bottom"
    elif leg_pos == "Bottom-Left": lx, ly, lxa, lya = 0.01, 0.01, "left", "bottom"
    elif leg_pos == "Outside Right": lx, ly, lxa, lya = 1.02, 1.0, "left", "top"
    else:
        lx = st.slider("X Coordinate", min_value=-0.2, max_value=1.5, value=0.99, step=0.01)
        ly = st.slider("Y Coordinate", min_value=-0.2, max_value=1.5, value=0.99, step=0.01)
        lxa, lya = "auto", "auto"

    st.markdown("---")
    st.header("📄 Export Full Report")
    components.html("""<button onclick="window.parent.print();" style="background-color:#FF4B4B; color:white; border:none; border-radius:4px; padding:0.5rem 1rem; font-size:1rem; font-weight:600; cursor:pointer; width:100%;">🖨️ Save Page as PDF</button>""", height=50)
    st.markdown("<div style='text-align: center; margin-top: 50px;'><p style='color: #888888; font-size: 0.85rem; font-family: sans-serif;'>Developed by<br><b>PhD(c) Carlos A. Torres-Ramírez</b><br><br></p></div>", unsafe_allow_html=True)

# Define dynamic labels globally after variables are loaded
cp_axis_label = E_label_base  # CP data are not iR-corrected
j_axis_label = "j (mA cm⁻²)" if scientific_style else "Current Density j (mA/cm²)"
jabs_axis_label = "|j| (mA cm⁻²)" if scientific_style else "|j| (mA/cm²)"

publication_palette = ['#000000', '#E41A1C', '#377EB8', '#4DAF4A', '#984EA3', '#FF7F00', '#A65628', '#F781BF'] + px.colors.qualitative.Alphabet
combined_palette = publication_palette

# Configuraciones de exportación fotográfica
dl_config = {'toImageButtonOptions': {'format': 'png', 'filename': 'electrochem_plot', 'height': 720, 'width': 960, 'scale': 4}}
# Config de exportación HD en proporción cuadrada (el ancho compensa los 20px extra de margen)
dl_config_square = {'toImageButtonOptions': {'format': 'png', 'filename': 'eis_square_plot', 'height': 800, 'width': 820, 'scale': 4}}


# ============================================================
# STREAMLIT COMPATIBILITY HELPERS (old and new Streamlit versions)
# ============================================================
def _param_names(fn):
    try:
        return inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return {}


_PLOTLY_PARAMS = _param_names(st.plotly_chart)
_DF_PARAMS = _param_names(st.dataframe)
_PLOTLY_NEW_WIDTH = "width" in _PLOTLY_PARAMS and isinstance(_PLOTLY_PARAMS["width"].default, str)
_DF_NEW_WIDTH = "width" in _DF_PARAMS and isinstance(_DF_PARAMS["width"].default, str)
_UI_KEY_COUNTER = {"n": 0}


def _next_key(prefix: str) -> str:
    _UI_KEY_COUNTER["n"] += 1
    return f"spark_{prefix}_{_UI_KEY_COUNTER['n']}"


def show_chart(fig, stretch: bool = True, config: Optional[dict] = None):
    """Render a Plotly figure with its own white styling (Streamlit theme disabled)."""
    kw = {"config": config or {}}
    if "theme" in _PLOTLY_PARAMS: kw["theme"] = None
    if "key" in _PLOTLY_PARAMS: kw["key"] = _next_key("chart")
    if _PLOTLY_NEW_WIDTH: kw["width"] = "stretch" if stretch else "content"
    else: kw["use_container_width"] = stretch
    st.plotly_chart(fig, **kw)


def show_table(data):
    kw = {}
    if "key" in _DF_PARAMS: kw["key"] = _next_key("table")
    if _DF_NEW_WIDTH: kw["width"] = "stretch"
    else: kw["use_container_width"] = True
    st.dataframe(data, **kw)


# ============================================================
# UTILITIES & MATH FUNCTIONS
# ============================================================
def to_rgba(color_str: str, alpha: float = 0.2) -> str:
    color_str = color_str.strip().lower()
    if color_str.startswith('#'):
        h = color_str.lstrip('#')
        if len(h) == 6:
            return f"rgba({int(h[0:2], 16)}, {int(h[2:4], 16)}, {int(h[4:6], 16)}, {alpha})"
    elif color_str.startswith('rgb('):
        return color_str.replace('rgb(', 'rgba(').replace(')', f', {alpha})')
    return f"rgba(150, 150, 150, {alpha})"

def _to_float(x):
    if x is None: return None
    try: return float(str(x).replace(",", "."))
    except (TypeError, ValueError): return None

def mad_sigma(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) < 10: return float(np.std(x)) if len(x) else np.nan
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * mad if mad > 0 else float(np.std(x))

# Accepts "50 mV/s", "50mVs", "CV_50mV_s", "20 mV s-1", "5 mV·s⁻¹"... (file names cannot contain "/")
_SR_PATTERN = re.compile(r"(\d+(?:[.,]\d+)?)\s*[-_ ]?\s*mV\s*(?:/|_|-|\.|·|\s)?\s*s(?:ec)?(?:\s*\^?\s*-\s*1|⁻¹)?(?![A-Za-z])", re.IGNORECASE)

def get_sr_from_name(name: str, default_sr: float) -> float:
    m = _SR_PATTERN.search(str(name))
    if m:
        try: return float(m.group(1).replace(",", "."))
        except ValueError: pass
    return default_sr if default_sr is not None else 0.0

def decode_bytes(raw: bytes) -> str:
    """Decode instrument text files (UTF-8, UTF-16 with/without BOM, Windows-1252)."""
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw[3:].decode("utf-8", errors="replace")
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16", errors="replace")
    head = raw[:4000]
    if head and head.count(b"\x00") > 0.2 * len(head):
        try:
            return raw.decode("utf-16-le" if head[1:2] == b"\x00" else "utf-16-be")
        except UnicodeDecodeError:
            pass
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace")

def get_averaged_curve(processed_curves: List[Tuple[str, pd.DataFrame]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not processed_curves: return None, None, None
    max_points = max(len(dd) for _, dd in processed_curves)
    common_idx = np.linspace(0, 1, max_points)

    E_interp, I_interp = [], []
    for _, dd in processed_curves:
        idx = np.linspace(0, 1, len(dd))
        E_interp.append(np.interp(common_idx, idx, dd["x"].values))
        I_interp.append(np.interp(common_idx, idx, dd["y"].values))

    return np.mean(E_interp, axis=0), np.mean(I_interp, axis=0), np.std(I_interp, axis=0)

def get_averaged_eis_curve(processed_curves: List[Tuple[str, pd.DataFrame]]):
    """Average spectra on a common frequency grid (log-f interpolation, overlapping range only)."""
    none = (None, None, None, None, None)
    if not processed_curves: return none
    specs = []
    for _, dd in processed_curves:
        f = np.asarray(dd["f"], float); zr = np.asarray(dd["x"], float); zi = np.asarray(dd["y"], float)
        ok = np.isfinite(f) & (f > 0) & np.isfinite(zr) & np.isfinite(zi)
        f, zr, zi = f[ok], zr[ok], zi[ok]
        if len(f) < 2: continue
        o = np.argsort(f)
        specs.append((f[o], zr[o], zi[o]))
    if not specs: return none
    f_lo = max(s[0][0] for s in specs)
    f_hi = min(s[0][-1] for s in specs)
    if f_hi <= f_lo: return none
    ref = specs[0][0]
    same_grid = all(len(s[0]) == len(ref) and np.allclose(s[0], ref, rtol=5e-3) for s in specs)
    grid = ref if same_grid else np.logspace(np.log10(f_lo), np.log10(f_hi), max(len(s[0]) for s in specs))
    lg = np.log10(grid)
    ZR = np.array([np.interp(lg, np.log10(s[0]), s[1]) for s in specs])
    ZI = np.array([np.interp(lg, np.log10(s[0]), s[2]) for s in specs])
    o = np.argsort(grid)[::-1]  # high -> low frequency
    return grid[o], ZR.mean(axis=0)[o], ZI.mean(axis=0)[o], ZR.std(axis=0)[o], ZI.std(axis=0)[o]


# ============================================================
# EIS EQUIVALENT-CIRCUIT FITTING (CNLS)
# ============================================================
def _rq(w, R, Q, n):
    """Impedance of R || CPE."""
    return R / (1.0 + R * Q * np.power(1j * w, n))

def _ycpe(w, Q, n):
    """Admittance of a CPE."""
    return Q * np.power(1j * w, n)

def _rl_par(w, RL, L):
    """RL || L (RL may be negative)."""
    jwL = 1j * w * L
    return RL * jwL / (RL + jwL)

def _m_randles(w, Rs, Q, n, Rct):
    return Rs + _rq(w, Rct, Q, n)

def _m_randles_w(w, Rs, Q, n, Rct, W):
    return Rs + 1.0 / (_ycpe(w, Q, n) + 1.0 / (Rct + W / np.sqrt(1j * w)))

def _m_2tc(w, Rs, Q1, n1, R1, Q2, n2, R2):
    return Rs + _rq(w, R1, Q1, n1) + _rq(w, R2, Q2, n2)

def _m_par_ads(w, Rs, Q, n, Rct, RL, L):
    return Rs + 1.0 / (_ycpe(w, Q, n) + 1.0 / Rct + 1.0 / (RL + 1j * w * L))

def _m_ser_ads(w, Rs, Q, n, Rct, RL, L):
    return Rs + 1.0 / (_ycpe(w, Q, n) + 1.0 / (Rct + _rl_par(w, RL, L)))

def _m_ads_cap(w, Rs, Q1, n1, Rct, Q2, n2, Rads):
    return Rs + 1.0 / (_ycpe(w, Q1, n1) + 1.0 / (Rct + _rq(w, Rads, Q2, n2)))

def _m_bilayer_ser_ads(w, Rs, Q1, n1, R1, Q2, n2, Rct, RL, L):
    return Rs + _rq(w, R1, Q1, n1) + 1.0 / (_ycpe(w, Q2, n2) + 1.0 / (Rct + _rl_par(w, RL, L)))

def _m_3tc(w, Rs, Q1, n1, R1, Q2, n2, R2, Q3, n3, R3):
    return Rs + _rq(w, R1, Q1, n1) + _rq(w, R2, Q2, n2) + _rq(w, R3, Q3, n3)

# Parameter kinds: R (resistance > 0, log), Rsig (signed resistance), Q (CPE-T, log), n (CPE-P, 0.5–1), L (H, log), W (log)
EIS_MODEL_SPECS = {
    "Randles: Rs-(CPE||Rct) [Ma et al. 2022]": dict(
        func=_m_randles,
        params=[("Rs (Ω)", "R"), ("CPE-T", "Q"), ("CPE-P", "n"), ("Rct (Ω)", "R")],
        rc_groups=None, rct="Rct (Ω)", ceff=[("C_eff (F)", "CPE-T", "CPE-P", "Rct (Ω)")], dc_finite=True),
    "Randles + Warburg: Rs-(CPE||(Rct+W)) [Metrohm/Generic]": dict(
        func=_m_randles_w,
        params=[("Rs (Ω)", "R"), ("CPE-T", "Q"), ("CPE-P", "n"), ("Rct (Ω)", "R"), ("W (Ω·s^-0.5)", "W")],
        rc_groups=None, rct="Rct (Ω)", ceff=[("C_eff (F)", "CPE-T", "CPE-P", "Rct (Ω)")], dc_finite=False),
    "Two Time Constants: Rs-(CPE1||R1)-(CPE2||R2) [Guo et al. 2016]": dict(
        func=_m_2tc,
        params=[("Rs (Ω)", "R"), ("CPE1-T", "Q"), ("CPE1-P", "n"), ("R1 (Ω)", "R"),
                ("CPE2-T", "Q"), ("CPE2-P", "n"), ("R2 (Ω)", "R")],
        rc_groups=[("CPE1-T", "CPE1-P", "R1 (Ω)"), ("CPE2-T", "CPE2-P", "R2 (Ω)")],
        rct="R2 (Ω)",
        ceff=[("C_eff1 (F)", "CPE1-T", "CPE1-P", "R1 (Ω)"), ("C_eff2 (F)", "CPE2-T", "CPE2-P", "R2 (Ω)")],
        dc_finite=True),
    "Parallel Adsorption: Rs-(CPE||(Rct||(RL+L))) [Classic UOR]": dict(
        func=_m_par_ads,
        params=[("Rs (Ω)", "R"), ("CPE-T", "Q"), ("CPE-P", "n"), ("Rct (Ω)", "R"), ("RL (Ω)", "R"), ("L (H)", "L")],
        rc_groups=None, rct="Rct (Ω)", ceff=[("C_eff (F)", "CPE-T", "CPE-P", "Rct (Ω)")], dc_finite=True),
    "Series Adsorption: Rs-(CPE||(Rct+(RL||L))) [Harrington-Conway]": dict(
        func=_m_ser_ads,
        params=[("Rs (Ω)", "R"), ("CPE-T", "Q"), ("CPE-P", "n"), ("Rct (Ω)", "R"), ("RL (Ω)", "Rsig"), ("L (H)", "L")],
        rc_groups=None, rct="Rct (Ω)", ceff=[("C_eff (F)", "CPE-T", "CPE-P", "Rct (Ω)")], dc_finite=True),
    "Adsorption Capacitance: Rs-(CPE1||(Rct+(CPE2||Rads))) [ACS Appl. Mater. 2026]": dict(
        func=_m_ads_cap,
        params=[("Rs (Ω)", "R"), ("CPE1-T", "Q"), ("CPE1-P", "n"), ("Rct (Ω)", "R"),
                ("CPE2-T", "Q"), ("CPE2-P", "n"), ("Rads (Ω)", "R")],
        rc_groups=None, rct="Rct (Ω)",
        ceff=[("C_eff1 (F)", "CPE1-T", "CPE1-P", "Rct (Ω)"), ("C_eff2 (F)", "CPE2-T", "CPE2-P", "Rads (Ω)")],
        dc_finite=True),
    "Bilayer + Series Adsorption: Rs-(CPE1||R1)-(CPE2||(Rct+(RL||L))) [NiOOH/Ni UOR]": dict(
        func=_m_bilayer_ser_ads,
        params=[("Rs (Ω)", "R"), ("CPE1-T", "Q"), ("CPE1-P", "n"), ("R1 (Ω)", "R"),
                ("CPE2-T", "Q"), ("CPE2-P", "n"), ("Rct (Ω)", "R"), ("RL (Ω)", "Rsig"), ("L (H)", "L")],
        rc_groups=None, rct="Rct (Ω)",
        ceff=[("C_eff1 (F)", "CPE1-T", "CPE1-P", "R1 (Ω)"), ("C_eff2 (F)", "CPE2-T", "CPE2-P", "Rct (Ω)")],
        dc_finite=True),
    "Three Time Constants: Rs-(CPE1||R1)-(CPE2||R2)-(CPE3||R3) [Bilayer NiOOH]": dict(
        func=_m_3tc,
        params=[("Rs (Ω)", "R"), ("CPE1-T", "Q"), ("CPE1-P", "n"), ("R1 (Ω)", "R"),
                ("CPE2-T", "Q"), ("CPE2-P", "n"), ("R2 (Ω)", "R"),
                ("CPE3-T", "Q"), ("CPE3-P", "n"), ("R3 (Ω)", "R")],
        rc_groups=[("CPE1-T", "CPE1-P", "R1 (Ω)"), ("CPE2-T", "CPE2-P", "R2 (Ω)"), ("CPE3-T", "CPE3-P", "R3 (Ω)")],
        rct="R3 (Ω)",
        ceff=[("C_eff1 (F)", "CPE1-T", "CPE1-P", "R1 (Ω)"), ("C_eff2 (F)", "CPE2-T", "CPE2-P", "R2 (Ω)"),
              ("C_eff3 (F)", "CPE3-T", "CPE3-P", "R3 (Ω)")],
        dc_finite=True),
}
EIS_MODELS_LIST = list(EIS_MODEL_SPECS.keys())
_BASE_RANDLES = EIS_MODELS_LIST[0]
_BASE_2TC = EIS_MODELS_LIST[2]
_LN10 = np.log(10.0)

def _eis_bounds(kind, zscale):
    if kind == "R": return np.log10(zscale) - 7.0, np.log10(zscale) + 4.0
    if kind == "Rsig": return -1e3, 1e3
    if kind == "Q": return -16.0, 2.0
    if kind == "n": return 0.5, 1.0
    if kind == "L": return -9.0, 10.0
    if kind == "W": return np.log10(zscale) - 9.0, np.log10(zscale) + 6.0
    raise ValueError(kind)

def _eis_to_x(kind, p, zscale):
    if kind == "n": return float(p)
    if kind == "Rsig": return float(p) / zscale
    return float(np.log10(max(p, 1e-300)))

def _eis_to_p(kind, x, zscale):
    if kind == "n": return x
    if kind == "Rsig": return x * zscale
    return 10.0 ** x

def _eis_features(f, zr, zi):
    """Data-driven starting values: HF intercept, total R, characteristic frequencies."""
    order = np.argsort(f)[::-1]  # high -> low frequency
    fs, zrs, zis = f[order], zr[order], zi[order]
    zscale = float(np.max(np.hypot(zrs, zis)))
    hf_phase = np.degrees(np.arctan2(zis[0], zrs[0]))
    zr_min = float(np.min(zrs))
    rs_cands = []
    if hf_phase < 15 and zrs[0] > 0:
        rs_cands.append(max(zrs[0], 1e-6 * zscale))
    rs_cands += [max(0.5 * zr_min, 1e-6 * zscale), max(0.05 * zr_min, 1e-6 * zscale)]
    r_tot = max(float(np.max(zrs)) - min(rs_cands), 1e-3 * zscale)
    peaks_f = []
    if len(zis) >= 7:
        win = min(len(zis) - (1 - len(zis) % 2), 7)
        try: zis_s = savgol_filter(zis, win, 2)
        except Exception: zis_s = zis
        pk, prop = find_peaks(zis_s, prominence=0.02 * max(np.max(np.abs(zis_s)), 1e-30))
        if len(pk):
            pk = pk[np.argsort(prop["prominences"])[::-1]]
            peaks_f = [float(fs[i]) for i in pk[:3]]
    return dict(zscale=zscale, rs_cands=rs_cands, r_tot=r_tot, peaks_f=peaks_f)

def _eis_generic_seeds(model_key, feat):
    rt, pk, rs = feat["r_tot"], feat["peaks_f"], feat["rs_cands"][0]
    q_grid = [1e-12, 1e-9, 1e-6, 1e-4, 1e-2]
    def q_from(fc, R): return 1.0 / (2 * np.pi * fc * max(R, 1e-12))
    seeds = []
    if model_key == _BASE_RANDLES:
        qs = ([q_from(pk[0], rt)] if pk else []) + q_grid
        for rs_ in feat["rs_cands"][:2]:
            for q in qs:
                seeds.append({"Rs (Ω)": rs_, "CPE-T": q, "CPE-P": 0.85, "Rct (Ω)": rt})
    elif model_key == _BASE_2TC:
        pairs = []
        if len(pk) >= 2:
            f_hi, f_lo = max(pk[0], pk[1]), min(pk[0], pk[1])
            pairs.append((q_from(f_hi, 0.5 * rt), q_from(f_lo, 0.5 * rt)))
        pairs += [(a, b) for i, a in enumerate(q_grid[:-1]) for b in q_grid[i + 1:]]
        for q1, q2 in pairs:
            for s in (0.1, 0.5, 0.9):
                seeds.append({"Rs (Ω)": rs, "CPE1-T": q1, "CPE1-P": 0.9, "R1 (Ω)": s * rt,
                              "CPE2-T": q2, "CPE2-P": 0.8, "R2 (Ω)": (1 - s) * rt})
    else:  # three time constants
        trip = []
        if len(pk) >= 3:
            trip.append(tuple(q_from(fc, rt / 3) for fc in sorted(pk[:3], reverse=True)))
        g = [1e-12, 1e-9, 1e-6, 1e-3]
        trip += [(g[a], g[b], g[c]) for a in range(4) for b in range(a + 1, 4) for c in range(b + 1, 4)]
        for q1, q2, q3 in trip:
            for s in ((0.1, 0.3, 0.6), (0.34, 0.33, 0.33), (0.6, 0.3, 0.1)):
                seeds.append({"Rs (Ω)": rs, "CPE1-T": q1, "CPE1-P": 0.9, "R1 (Ω)": s[0] * rt,
                              "CPE2-T": q2, "CPE2-P": 0.85, "R2 (Ω)": s[1] * rt,
                              "CPE3-T": q3, "CPE3-P": 0.8, "R3 (Ω)": s[2] * rt})
    return seeds

def _eis_fit_core(f, Z, model_key, seeds, feat, n_polish=3):
    """Two-stage multi-start CNLS (modulus weighting). Returns (params, result, names, kinds, lb, ub) or None."""
    spec = EIS_MODEL_SPECS[model_key]
    names = [p[0] for p in spec["params"]]
    kinds = [p[1] for p in spec["params"]]
    n_par, zscale = len(names), feat["zscale"]
    w = 2 * np.pi * f
    wts = 1.0 / np.maximum(np.abs(Z), 1e-300)
    lb = np.array([_eis_bounds(k, zscale)[0] for k in kinds])
    ub = np.array([_eis_bounds(k, zscale)[1] for k in kinds])

    def unpack(x): return [_eis_to_p(k, xi, zscale) for k, xi in zip(kinds, x)]

    def resid(x):
        with np.errstate(all="ignore"):
            r = (spec["func"](w, *unpack(x)) - Z) * wts
        out = np.concatenate([r.real, r.imag])
        out[~np.isfinite(out)] = 1e6
        return out

    trial = []
    for s in seeds:
        if not all(k in s for k in names): continue
        x0 = np.clip(np.array([_eis_to_x(k, s[nm], zscale) for nm, k in zip(names, kinds)]), lb + 1e-9, ub - 1e-9)
        try:
            r = least_squares(resid, x0, bounds=(lb, ub), method="trf", x_scale="jac", max_nfev=40 * n_par)
            trial.append((r.cost, r.x))
        except Exception:
            continue
    if not trial: return None
    trial.sort(key=lambda t: t[0])
    best = None
    for _, x in trial[:n_polish]:
        try:
            r = least_squares(resid, x, bounds=(lb, ub), method="trf", x_scale="jac",
                              max_nfev=1500 * n_par, ftol=1e-12, xtol=1e-12, gtol=1e-12)
            if best is None or r.cost < best.cost: best = r
        except Exception:
            continue
    if best is None: return None
    return dict(zip(names, unpack(best.x))), best, names, kinds, lb, ub

def _eis_seeds(model_key, feat, f, Z):
    """Simple models: grid of starting points. Composite models: start from the best fit of the
    parent model (Randles or 2TC) plus a small grid of the extra elements (hierarchical fitting)."""
    rt = feat["r_tot"]
    if model_key in (_BASE_RANDLES, _BASE_2TC):
        return _eis_generic_seeds(model_key, feat)
    if model_key.startswith("Three Time"):
        seeds = _eis_generic_seeds(model_key, feat)
        base = _eis_fit_core(f, Z, _BASE_2TC, _eis_generic_seeds(_BASE_2TC, feat), feat, n_polish=2)
        if base is not None:
            b = base[0]
            seeds.insert(0, {"Rs (Ω)": b["Rs (Ω)"],
                             "CPE1-T": b["CPE1-T"] * 0.1, "CPE1-P": b["CPE1-P"], "R1 (Ω)": 0.5 * b["R1 (Ω)"],
                             "CPE2-T": b["CPE1-T"] * 10, "CPE2-P": b["CPE1-P"], "R2 (Ω)": 0.5 * b["R1 (Ω)"],
                             "CPE3-T": b["CPE2-T"], "CPE3-P": b["CPE2-P"], "R3 (Ω)": b["R2 (Ω)"]})
            seeds.insert(0, {"Rs (Ω)": b["Rs (Ω)"],
                             "CPE1-T": b["CPE1-T"], "CPE1-P": b["CPE1-P"], "R1 (Ω)": b["R1 (Ω)"],
                             "CPE2-T": b["CPE2-T"] * 0.1, "CPE2-P": b["CPE2-P"], "R2 (Ω)": 0.5 * b["R2 (Ω)"],
                             "CPE3-T": b["CPE2-T"] * 10, "CPE3-P": b["CPE2-P"], "R3 (Ω)": 0.5 * b["R2 (Ω)"]})
        return seeds
    if model_key.startswith(("Randles + Warburg", "Parallel Adsorption", "Series Adsorption")):
        base = _eis_fit_core(f, Z, _BASE_RANDLES, _eis_generic_seeds(_BASE_RANDLES, feat), feat, n_polish=2)
        bases = [base[0]] if base is not None else []
        bases += [{"Rs (Ω)": feat["rs_cands"][0], "CPE-T": q, "CPE-P": 0.85, "Rct (Ω)": rt} for q in (1e-9, 1e-6, 1e-3)]
        seeds = []
        for b in bases:
            R0 = max(b["Rct (Ω)"], 1e-9)
            if model_key.startswith("Randles + Warburg"):
                for wf in (0.03, 0.3, 3.0):
                    seeds.append({**b, "Rct (Ω)": 0.7 * R0, "W (Ω·s^-0.5)": wf * R0})
            elif model_key.startswith("Parallel Adsorption"):
                for tau in (0.01, 1.0, 100.0):
                    for frac in (0.5, 2.0):
                        seeds.append({**b, "Rct (Ω)": 1.5 * R0, "RL (Ω)": frac * R0, "L (H)": tau * frac * R0})
            else:
                for tau in (0.01, 1.0, 100.0):
                    for rl in (0.5 * R0, -0.3 * R0):
                        seeds.append({**b, "Rct (Ω)": 0.8 * R0, "RL (Ω)": rl, "L (H)": tau * abs(rl)})
        return seeds
    # Two-arc composite models, seeded from the 2TC fit
    base = _eis_fit_core(f, Z, _BASE_2TC, _eis_generic_seeds(_BASE_2TC, feat), feat, n_polish=2)
    bases = [base[0]] if base is not None else []
    bases += _eis_generic_seeds(_BASE_2TC, feat)[:6 if model_key.startswith("Adsorption Capacitance") else 2]
    seeds = []
    for b in bases:
        if model_key.startswith("Adsorption Capacitance"):
            seeds.append({"Rs (Ω)": b["Rs (Ω)"], "CPE1-T": b["CPE1-T"], "CPE1-P": b["CPE1-P"], "Rct (Ω)": b["R1 (Ω)"],
                          "CPE2-T": b["CPE2-T"], "CPE2-P": b["CPE2-P"], "Rads (Ω)": b["R2 (Ω)"]})
        else:  # Bilayer + Series Adsorption
            R2 = max(b["R2 (Ω)"], 1e-9)
            for tau in (0.01, 1.0, 100.0):
                for rl in (0.5 * R2, -0.3 * R2):
                    seeds.append({"Rs (Ω)": b["Rs (Ω)"], "CPE1-T": b["CPE1-T"], "CPE1-P": b["CPE1-P"],
                                  "R1 (Ω)": b["R1 (Ω)"], "CPE2-T": b["CPE2-T"], "CPE2-P": b["CPE2-P"],
                                  "Rct (Ω)": 0.8 * R2, "RL (Ω)": rl, "L (H)": tau * abs(rl)})
    return seeds

def fit_eis_model(f, zr, zi, model_type, area_cm2, rf_val, n_sim=200):
    """Fit an equivalent circuit (all parameters, Rs included) to one spectrum.

    Returns (results, f_sim, Z'_sim, -Z''_sim, info) or five Nones when the fit is not possible.
    info = {"rel_err": {param: % error}, "at_bound": {param: "lower"/"upper"}}.
    """
    none = (None, None, None, None, None)
    f = np.asarray(f, float); zr = np.asarray(zr, float); zi = np.asarray(zi, float)
    ok = np.isfinite(f) & np.isfinite(zr) & np.isfinite(zi) & (f > 0)
    f, zr, zi = f[ok], zr[ok], zi[ok]
    if model_type not in EIS_MODEL_SPECS: return none
    spec = EIS_MODEL_SPECS[model_type]
    N, n_par = len(f), len(spec["params"])
    if N < n_par + 2: return none
    try:
        Z = zr - 1j * zi
        feat = _eis_features(f, zr, zi)
        out = _eis_fit_core(f, Z, model_type, _eis_seeds(model_type, feat, f, Z), feat, n_polish=3)
        if out is None: return none
        p, best, names, kinds, lb, ub = out
        x = best.x

        # Parameter uncertainties from the Jacobian (in the fitted, transformed space)
        dof = max(2 * N - n_par, 1)
        s2 = float(np.sum(best.fun ** 2)) / dof
        try:
            sx = np.sqrt(np.clip(np.diag(np.linalg.pinv(best.jac.T @ best.jac)) * s2, 0, None))
        except Exception:
            sx = np.full(n_par, np.nan)
        rel_err, at_bound = {}, {}
        for nm, k, xi, si, lo, hi in zip(names, kinds, x, sx, lb, ub):
            rel_err[nm] = 100.0 * _LN10 * si if k in ("R", "Q", "L", "W") else 100.0 * si / max(abs(xi), 1e-30)
            tol = 0.002 if k == "n" else 0.01 * (hi - lo)
            if xi - lo <= tol: at_bound[nm] = "lower"
            elif hi - xi <= tol: at_bound[nm] = "upper"

        # Order RC elements by time constant (1 = highest frequency) so labels are consistent
        if spec["rc_groups"]:
            groups = spec["rc_groups"]
            vals = [(p[q], p[n_], p[r]) for (q, n_, r) in groups]
            errs = [(rel_err[q], rel_err[n_], rel_err[r]) for (q, n_, r) in groups]
            bnds = [(at_bound.get(q), at_bound.get(n_), at_bound.get(r)) for (q, n_, r) in groups]
            taus = [(v[2] * v[0]) ** (1.0 / v[1]) for v in vals]
            new_bound = {k: v for k, v in at_bound.items() if not any(k in g for g in groups)}
            for new_pos, old_pos in enumerate(np.argsort(taus)):
                q, n_, r = groups[new_pos]
                p[q], p[n_], p[r] = vals[old_pos]
                rel_err[q], rel_err[n_], rel_err[r] = errs[old_pos]
                for key_, b_ in zip((q, n_, r), bnds[old_pos]):
                    if b_: new_bound[key_] = b_
            at_bound = new_bound

        pars = [p[nm] for nm in names]
        w = 2 * np.pi * f
        Zfit = spec["func"](w, *pars)
        y = np.concatenate([zr, zi])
        yfit = np.concatenate([Zfit.real, -Zfit.imag])
        ss_tot = np.sum((y - y.mean()) ** 2)

        results = {nm: p[nm] for nm in names}
        results["R²"] = 1 - np.sum((y - yfit) ** 2) / ss_tot if ss_tot > 0 else np.nan
        results["χ²"] = s2

        def calc_ceff(Q, n, R):
            if not (np.isfinite(Q) and np.isfinite(n) and np.isfinite(R)) or Q <= 0 or n <= 0 or R <= 0: return np.nan
            return (Q ** (1.0 / n)) * (R ** ((1.0 - n) / n))

        for cname, qk, nk, rk in spec["ceff"]:
            results[cname] = calc_ceff(p[qk], p[nk], p[rk])

        rct_val = p[spec["rct"]]
        results["_rct"] = rct_val
        results["Rct from"] = spec["rct"].replace(" (Ω)", "") + (" (LF)" if spec["rc_groups"] else "")
        if spec["dc_finite"]:
            Zdc = spec["func"](np.array([2 * np.pi * 1e-9]), *pars)[0]
            results["R_p (Ω)"] = float(Zdc.real - p["Rs (Ω)"])
        results["Rct·A (Ω·cm²)"] = rct_val * area_cm2
        if rf_val and rf_val > 0 and abs(rf_val - 1.0) > 1e-12:
            results["Rct·A·RF (Ω·cm²_ECSA)"] = rct_val * area_cm2 * rf_val

        f_sim = np.logspace(np.log10(np.max(f)), np.log10(np.min(f)), n_sim)
        Z_sim = spec["func"](2 * np.pi * f_sim, *pars)
        return results, f_sim, Z_sim.real, -Z_sim.imag, {"rel_err": rel_err, "at_bound": at_bound}
    except Exception:
        return none

@st.cache_data(show_spinner=False, max_entries=512)
def fit_eis_cached(f, zr, zi, model_type, area_cm2, rf_val):
    return fit_eis_model(f, zr, zi, model_type, area_cm2, rf_val)


# ============================================================
# VOLTAMMETRY ANALYSIS (CV / LSV)
# ============================================================
def sweep_segments(E, rel_hyst: float = 0.01):
    """Split a potential program into monotonic sweeps (vertex detection with hysteresis, robust to noise).
    Returns [(start_idx, end_idx_inclusive, direction)], direction = +1 anodic, -1 cathodic, 0 static."""
    E = np.asarray(E, float)
    n = len(E)
    if n < 2: return [(0, max(n - 1, 0), 0)]
    span = np.nanmax(E) - np.nanmin(E)
    if not np.isfinite(span) or span <= 0: return [(0, n - 1, 0)]
    thr = rel_hyst * span
    turning, direction, ext_i = [], 0, 0
    for i in range(1, n):
        if not np.isfinite(E[i]): continue
        if direction == 0:
            if E[i] - E[0] > thr: direction, ext_i = 1, int(np.nanargmax(E[:i + 1]))
            elif E[0] - E[i] > thr: direction, ext_i = -1, int(np.nanargmin(E[:i + 1]))
            continue
        if direction == 1:
            if E[i] >= E[ext_i]: ext_i = i
            elif E[ext_i] - E[i] > thr: turning.append(ext_i); direction, ext_i = -1, i
        else:
            if E[i] <= E[ext_i]: ext_i = i
            elif E[i] - E[ext_i] > thr: turning.append(ext_i); direction, ext_i = 1, i
    bounds = [0] + turning + [n - 1]
    segs = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b > a: segs.append((a, b, 1 if E[b] > E[a] else (-1 if E[b] < E[a] else 0)))
    return segs or [(0, n - 1, 0)]

def cv_delta_j_half(E, I, e_nf, area_cm2):
    """Δj/2 (mA/cm²) at E_nf, averaging every anodic and every cathodic sweep that crosses E_nf."""
    E, I = np.asarray(E, float), np.asarray(I, float)
    ja, jc = [], []
    for a, b, d in sweep_segments(E):
        if d == 0 or b - a < 2: continue
        Es, Is = E[a:b + 1], I[a:b + 1]
        if not (np.nanmin(Es) <= e_nf <= np.nanmax(Es)): continue
        o = np.argsort(Es, kind="mergesort")
        (ja if d > 0 else jc).append(float(np.interp(e_nf, Es[o], Is[o])))
    if not ja or not jc: return np.nan
    return (np.mean(ja) - np.mean(jc)) * 1000.0 / area_cm2 / 2.0

def cv_anodic_peak(E, I, e_min=None, e_max=None):
    """Highest anodic peak (local maximum on the anodic sweeps); falls back to the anodic maximum."""
    E, I = np.asarray(E, float), np.asarray(I, float)
    segs = [s for s in sweep_segments(E) if s[2] > 0] or [(0, len(E) - 1, 1)]
    use_win = e_min is not None and e_max is not None
    best, pool = None, []
    for a, b, _ in segs:
        idx = np.arange(a, b + 1)
        if use_win: idx = idx[(E[idx] >= e_min) & (E[idx] <= e_max)]
        pool.extend(idx.tolist())
        if len(idx) < 3: continue
        pk, _ = find_peaks(I[idx])
        for c in idx[pk]:
            if best is None or I[c] > I[best]: best = c
    if best is None:
        if not pool:
            pool = [i for a, b, _ in segs for i in range(a, b + 1)]
        pool = np.array(pool)
        best = pool[np.argmax(I[pool])]
    return float(E[best]), float(I[best])

def extract_limits_from_data(df: pd.DataFrame, technique: str) -> Tuple[float, float, float]:
    if len(df) == 0: return None, None, None
    Ecol = "Vf" if "Vf" in df.columns else ("Vu" if "Vu" in df.columns else (df.columns[0] if len(df.columns) > 0 else None))
    if Ecol is None or Ecol not in df.columns: return None, None, None

    col_data = df[Ecol]
    if isinstance(col_data, pd.DataFrame):
        col_data = col_data.iloc[:, 0]

    v_data = pd.to_numeric(col_data, errors='coerce').dropna().values
    if len(v_data) == 0: return None, None, None

    vinit = round(float(v_data[0]), 3)
    if "LSV" in technique:
        return vinit, round(float(v_data[-1]), 3), None
    vertices = [b for a, b, d in sweep_segments(v_data)[:-1]]
    if vertices:
        vlim1 = round(float(v_data[vertices[0]]), 3)
        vlim2 = round(float(v_data[vertices[1]]), 3) if len(vertices) >= 2 else round(float(v_data[-1]), 3)
    else:
        idx_max_dist = np.argmax(np.abs(v_data - v_data[0]))
        vlim1 = round(float(v_data[idx_max_dist]), 3)
        vlim2 = round(float(v_data[-1]), 3)
    return vinit, vlim1, vlim2

def recommend_operating_ranges_for_curve(df_curve, baseline_E_window=0.20, smooth_window=151, smooth_poly=3, local_window=101, threshold_mode="percentile", nr_fixed=1.30, nr_percentile=95, min_run_points=60, I_tol=0.0):
    if "x" not in df_curve.columns or "y" not in df_curve.columns:
        return {"N_points": 0, "noisy_intervals_E": [], "E_cut_cathodic_V": None, "recommended_noise_safe_V": None, "recommended_reduction_only_V": None}

    df = df_curve[["x", "y"]].copy()
    df.columns = ["E", "I"]
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    dfE = df.sort_values("E").reset_index(drop=True)
    E, I = dfE["E"].values, dfE["I"].values
    N = len(dfE)

    if N < 15: return {"N_points": N, "noisy_intervals_E": [], "E_cut_cathodic_V": None, "recommended_noise_safe_V": (float(np.min(E)), float(np.max(E))) if N else None, "recommended_reduction_only_V": None}

    def odd_cap(n):
        n = n if n % 2 == 1 else n + 1
        return max(11, min(n, N if (N % 2 == 1) else N - 1))

    smooth_window = odd_cap(smooth_window)
    local_window = odd_cap(local_window)
    smooth_poly = min(smooth_poly, smooth_window - 2)

    Is = savgol_filter(I, window_length=smooth_window, polyorder=smooth_poly)
    resid = I - Is

    Emax = float(np.max(E))
    base_mask = (E >= (Emax - baseline_E_window)) & (E <= Emax)
    base_resid = resid[base_mask] if base_mask.sum() >= 10 else resid[np.argsort(E)[-max(10, int(0.10 * N)):]]
    sigma_base = mad_sigma(base_resid)
    if not np.isfinite(sigma_base) or sigma_base == 0: sigma_base = float(np.std(resid)) if np.std(resid) > 0 else 1e-12

    half = local_window // 2
    NR = np.empty(N, dtype=float)
    for i in range(N):
        lo, hi = max(0, i - half), min(N, i + half + 1)
        sigma_loc = mad_sigma(resid[lo:hi])
        NR[i] = sigma_loc / sigma_base if np.isfinite(sigma_loc) and sigma_base > 0 else np.nan

    NR_finite = NR[np.isfinite(NR)]
    thr = float(nr_fixed) if threshold_mode == "fixed" else float(np.percentile(NR_finite, nr_percentile))
    bad = np.isfinite(NR) & (NR >= thr)

    min_run_eff, noisy_intervals, i = min(min_run_points, max(10, N // 6)), [], 0
    while i < N:
        if bad[i]:
            j = i
            while j < N and bad[j]: j += 1
            if (j - i) >= min_run_eff: noisy_intervals.append((float(E[i]), float(E[j - 1])))
            i = j
        else: i += 1

    idx_desc = np.argsort(E)[::-1]
    bad_desc, E_desc = bad[idx_desc], E[idx_desc]
    E_cut, k = None, 0
    while k < N:
        if bad_desc[k]:
            m = k
            while m < N and bad_desc[m]: m += 1
            if (m - k) >= min_run_eff:
                E_cut = float(E_desc[k])
                break
            k = m
        else: k += 1

    noise_safe = (float(np.min(E)), Emax) if E_cut is None else (E_cut, Emax)
    df_safe = df[(df["E"] >= noise_safe[0]) & (df["E"] <= noise_safe[1])].dropna()
    red_range = None
    if not df_safe.empty:
        mask_red = df_safe["I"].values <= I_tol
        if np.any(mask_red): red_range = (float(np.min(df_safe["E"].values[mask_red])), float(np.max(df_safe["E"].values[mask_red])))

    return {"N_points": N, "noisy_intervals_E": noisy_intervals, "E_cut_cathodic_V": E_cut, "recommended_noise_safe_V": noise_safe, "recommended_reduction_only_V": red_range}

def _first_sustained_crossing(E_b, J_b, target, sustain=3):
    """First potential (walking along the branch) where J >= target and stays there for `sustain` points."""
    above = J_b >= target
    if not np.any(above): return np.nan
    n = len(J_b)
    for i in np.where(above)[0]:
        if np.all(above[i:min(n, i + sustain)]):
            if i == 0: return float(E_b[0])
            j0, j1, e0, e1 = J_b[i - 1], J_b[i], E_b[i - 1], E_b[i]
            return float(e1 if j1 == j0 else e0 + (target - j0) * (e1 - e0) / (j1 - j0))
    return np.nan

def extract_lsv_catalytic_parameters(df_curve: pd.DataFrame, area_cm2: float, e_rev: float) -> Tuple[dict, dict]:
    """LSV descriptors for cathodic (HER, reductions) and anodic (OER, UOR, oxidations) polarization curves.
    An optional 'std' column (averaged curves) is carried along with the same ordering."""
    if "x" not in df_curve.columns or "y" not in df_curve.columns: return {}, {}
    d = df_curve.replace([np.inf, -np.inf], np.nan).dropna(subset=["x", "y"])
    if len(d) < 20: return {}, {}
    order = np.argsort(d["x"].values, kind="mergesort")
    E, I = d["x"].values[order], d["y"].values[order]
    std = d["std"].values[order] if "std" in d.columns else None
    abs_I = np.abs(I)
    k_max = int(np.argmax(abs_I))
    I_max = float(abs_I[k_max])
    if not np.isfinite(I_max) or I_max <= 0: return {}, {}
    cathodic = I[k_max] < 0
    sign = -1.0 if cathodic else 1.0

    j_dens = abs_I * 1000.0 / area_cm2
    eta_mV = sign * (E - e_rev) * 1000.0     # positive in the direction of the reaction
    j_max = I_max * 1000.0 / area_cm2

    # Reaction branch: from the low-current end of the curve up to the current maximum
    br = np.arange(len(E) - 1, k_max - 1, -1) if cathodic else np.arange(0, k_max + 1)
    E_b, I_b = E[br], abs_I[br]
    E_onset = _first_sustained_crossing(E_b, I_b, 0.05 * I_max)

    # Robust Tafel slope: sliding window over 2–40 % of |I|max on the reaction branch
    search_mask = (I_b >= 0.02 * I_max) & (I_b <= 0.40 * I_max)
    E_search, I_search = E_b[search_mask], I_b[search_mask]
    best_r2, best_slope, best_intercept, best_log_I_fit = -1, np.nan, np.nan, []
    if len(E_search) > 10:
        log_I_search = np.log10(I_search)
        win_size = max(10, len(E_search) // 5)
        for i in range(len(E_search) - win_size + 1):
            x_win, y_win = log_I_search[i:i + win_size], E_search[i:i + win_size]
            if np.ptp(x_win) <= 0: continue
            slope, intercept, r_value, _, _ = linregress(x_win, y_win)
            r2 = r_value ** 2
            if not np.isnan(r2) and r2 > best_r2:
                best_r2, best_slope, best_intercept, best_log_I_fit = r2, slope, intercept, x_win
    tafel_slope = abs(best_slope * 1000) if not np.isnan(best_slope) else np.nan

    etas = {}
    for target in [10, 20, 50, 100]:
        if target <= j_max:
            E_t = _first_sustained_crossing(E_b, I_b * 1000.0 / area_cm2, target)
            etas[f"η_{target} (mV)"] = sign * (E_t - e_rev) * 1000.0 if np.isfinite(E_t) else np.nan
        else:
            etas[f"η_{target} (mV)"] = np.nan
    if j_max < 100: etas[f"η_max@{j_max:.1f} (mV)"] = float(eta_mV[k_max])

    params = {
        "Polarity": "Cathodic (HER/red.)" if cathodic else "Anodic (OER/UOR/ox.)",
        "j_max (mA/cm²)": j_max, "E_onset (V)": E_onset,
        "Tafel Slope (mV/dec)": tafel_slope, "Tafel R²": best_r2 if best_r2 != -1 else np.nan,
        **etas
    }

    try:
        win_len = min(31, len(abs_I) if len(abs_I) % 2 == 1 else len(abs_I) - 1)
        win_len = win_len if win_len % 2 == 1 else win_len - 1
        I_smooth = savgol_filter(abs_I, window_length=max(5, win_len), polyorder=2)
    except Exception:
        I_smooth = abs_I
    I_smooth = np.where(I_smooth <= 0, np.nan, I_smooth)

    fit_data = {
        "E_full": E, "log_I_full": np.log10(I_smooth),
        "log_I_fit": best_log_I_fit, "slope": best_slope, "intercept": best_intercept,
        "log_I_max": np.log10(I_max),
        "j_dens": j_dens, "eta_mV": eta_mV,
        "j_std": (np.abs(std) * 1000.0 / area_cm2) if std is not None else None,
    }
    return params, fit_data

def apply_scientific_style(fig, is_scientific, lx, ly, lxa, lya):
    if is_scientific:
        fig.update_layout(
            title="", plot_bgcolor='white', paper_bgcolor='white',
            font=dict(family="Arial, sans-serif", size=20, color="black"),
            margin=dict(l=80, r=40, t=40, b=60)
        )
        fig.update_xaxes(
            showgrid=False, showline=True, linecolor='black', linewidth=2,
            mirror="all", ticks='inside', tickcolor='black', tickwidth=2, ticklen=8,
            title_font=dict(size=24, family="Arial, sans-serif", color="black"),
            tickfont=dict(size=18, family="Arial, sans-serif", color="black"),
            zeroline=False
        )
        fig.update_yaxes(
            showgrid=False, showline=True, linecolor='black', linewidth=2,
            mirror="all", ticks='inside', tickcolor='black', tickwidth=2, ticklen=8,
            title_font=dict(size=24, family="Arial, sans-serif", color="black"),
            tickfont=dict(size=18, family="Arial, sans-serif", color="black"),
            zeroline=False
        )
    else:
        fig.update_layout(template="plotly_white", plot_bgcolor='white', paper_bgcolor='white', font=dict(color="#1F2328"))
    fig.update_layout(
        legend=dict(
            x=lx, y=ly, xanchor=lxa, yanchor=lya,
            bgcolor='rgba(255, 255, 255, 0.9)', bordercolor='black',
            borderwidth=1 if is_scientific else 0,
            font=dict(size=18, color="black" if is_scientific else "#1F2328")
        )
    )
    return fig

# ============================================================
# PARSERS
# ============================================================
TECH_EIS = "Electrochemical Impedance Spectroscopy (EIS)"
TECH_CV = "Cyclic Voltammetry (CV)"
TECH_LSV = "Linear Sweep Voltammetry (LSV)"
TECH_CP = "Chronopotentiometry (CP)"
TECH_CA = "Chronoamperometry (CA)"

def parse_medusa_plt(raw: bytes) -> Tuple[Dict[str, str], List[Tuple[str, pd.DataFrame]]]:
    text = raw.decode('utf-8', errors='ignore')
    lines = text.splitlines()

    meta = {"TECHNIQUE": "Chemical Speciation (Medusa)"}
    curves = []

    x_low, x_high, y_low, y_high = 0.0, 14.0, -10.0, 0.0

    for line in lines[:50]:
        if "X/Y low and high:" in line:
            parts = line.split(":")[-1].split()
            if len(parts) >= 4:
                x_low, x_high, y_low, y_high = map(float, parts[:4])

    parsing_curves = False
    current_curve = None
    current_data = []

    for line in lines:
        line = line.strip()
        if "-- CURVES --" in line:
            parsing_curves = True
            continue
        if parsing_curves and "-- LABELS ON CURVES --" in line:
            break

        if parsing_curves:
            if line.startswith("5"):
                if current_curve and current_data:
                    df = pd.DataFrame(current_data, columns=["x_px", "y_px"])
                    df["x"] = x_low + (df["x_px"] - 263) / (1763 - 263) * (x_high - x_low)
                    df["y"] = y_low + (df["y_px"] - 140) / (1140 - 140) * (y_high - y_low)
                    curves.append((current_curve, df[["x", "y"]].copy()))

                parts = line.split(maxsplit=2)
                if len(parts) >= 3:
                    current_curve = parts[2].strip()
                    current_data = []
            else:
                m = re.match(r"^([01])\s*(\d+)\s+(\d+)$", line)
                if m:
                    current_data.append([float(m.group(2)), float(m.group(3))])

    if current_curve and current_data:
        df = pd.DataFrame(current_data, columns=["x_px", "y_px"])
        df["x"] = x_low + (df["x_px"] - 263) / (1763 - 263) * (x_high - x_low)
        df["y"] = y_low + (df["y_px"] - 140) / (1140 - 140) * (y_high - y_low)
        curves.append((current_curve, df[["x", "y"]].copy()))

    return meta, curves

def parse_gamry_dta_multi_curve(raw: str) -> Tuple[Dict[str, str], List[Tuple[str, pd.DataFrame]]]:
    lines = raw.splitlines()
    meta: Dict[str, str] = {}
    first_curve_idx = None
    for i, line in enumerate(lines):
        if "CURVE" in line.upper() and "TABLE" in line.upper():
            first_curve_idx = i; break
        if "\t" in line:
            parts = line.split("\t")
            if parts[0].strip():
                val = parts[2].strip() if len(parts) >= 3 else (parts[1].strip() if len(parts) >= 2 else "")
                if val: meta[parts[0].strip()] = val
        else:
            m = re.match(r"^\s*([A-Za-z0-9_]+)\s*:\s*(.+?)\s*$", line)
            if m: meta[m.group(1).strip()] = m.group(2).strip()

    if first_curve_idx is None: return meta, []
    curves: List[Tuple[str, pd.DataFrame]] = []
    i = first_curve_idx
    while i < len(lines):
        line = lines[i]
        if "CURVE" in line.upper() and "TABLE" in line.upper():
            m = re.search(r"(Z?CURVE)(\d*)", line, flags=re.IGNORECASE)
            curve_id = "OCV" if "OCVCURVE" in line.upper() else f"Curve {m.group(2) if m and m.group(2) else '1'}"

            j, col_line_idx = i + 1, None
            while j < len(lines) and j < i + 60:
                s = lines[j].strip()
                parts = s.split("\t")
                if len(parts) >= 3 and any(c.upper() in ["PT", "V", "I", "FREQ", "ZREAL", "T", "TIME", "VF", "VU", "IM"] for c in parts):
                    col_line_idx = j; break
                j += 1

            if col_line_idx is None:
                i += 1; continue

            cols = [c.strip() for c in lines[col_line_idx].split("\t") if c.strip()]
            if len(cols) < 3: cols = [c.strip() for c in re.split(r"\s{2,}", lines[col_line_idx].strip()) if c.strip()]

            data_start = col_line_idx + 1
            if data_start < len(lines) and lines[data_start].lstrip().startswith("#"): data_start += 1
            rows: List[List[str]] = []
            k = data_start
            while k < len(lines):
                s = lines[k].strip()
                if not s:
                    k += 1; continue
                if "CURVE" in s.upper() and "TABLE" in s.upper(): break
                parts = [p.strip() for p in lines[k].split("\t")]
                if len(parts) == 1: parts = [p.strip() for p in re.split(r"\s{2,}", s)]
                if parts and parts[0] == "": parts = parts[1:]
                if len(parts) >= len(cols): rows.append(parts[:len(cols)])
                k += 1

            df = pd.DataFrame(rows, columns=cols)
            for c in df.columns: df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ".", regex=False).str.strip(), errors="coerce")
            df = df.replace([np.inf, -np.inf], np.nan).dropna(how="all").reset_index(drop=True)

            col_map = {}
            for c in df.columns:
                cu = c.upper()
                if cu in ["VF", "VU", "VM", "V"]: col_map[c] = "Vf"
                elif cu in ["IM", "I"]: col_map[c] = "Im"
                elif cu in ["FREQ", "FREQUENCY"]: col_map[c] = "Frequency"
                elif cu in ["ZREAL", "Z'"]: col_map[c] = "Z_real"
                elif cu in ["ZIMAG", "Z''"]: col_map[c] = "neg_Z_imag"
                elif cu in ["T", "TIME"]: col_map[c] = "Time"
            df = df.rename(columns=col_map)
            df = df.loc[:, ~df.columns.duplicated()]

            if "neg_Z_imag" in df.columns and any("Zimag" in c for c in cols):
                df["neg_Z_imag"] = -df["neg_Z_imag"]

            curves.append((curve_id, df))
            i = k
        else:
            i += 1

    # Technique: explicit metadata first, then the columns of ANY curve (an OCV pre-curve must not decide it)
    method_clues = " ".join(meta.get(k, "") for k in ("METHOD", "TAG", "TITLE", "EXPLAIN")).upper()
    has_eis_cols = any("Frequency" in d.columns and "Z_real" in d.columns for _, d in curves)
    has_im = any("Im" in d.columns for _, d in curves)
    has_time_v = any("Time" in d.columns and "Vf" in d.columns for _, d in curves)
    if has_eis_cols or re.search(r"\bEIS|IMPEDANCE", method_clues):
        meta["TECHNIQUE"] = TECH_EIS
    elif "CHRONOP" in method_clues or re.search(r"\bCP\b", method_clues) or (has_time_v and not has_im):
        meta["TECHNIQUE"] = TECH_CP
    elif "CHRONOA" in method_clues or re.search(r"\bCA\b", method_clues):
        meta["TECHNIQUE"] = TECH_CA
    elif re.search(r"(?<![A-Z])CV", method_clues) or "CYCLIC" in method_clues:
        meta["TECHNIQUE"] = TECH_CV
    elif "LSV" in method_clues or "LINEAR" in method_clues or "POTENTIODYNAMIC" in method_clues:
        meta["TECHNIQUE"] = TECH_LSV

    return meta, curves

def _is_numeric_row(line: str) -> bool:
    parts = [p.strip() for p in re.split(r"\t|;", line) if p.strip()]
    if not parts: return False
    n_num = sum(_to_float(p) is not None for p in parts)
    return n_num >= max(1, int(0.6 * len(parts)))

def _unit_scale_after_slash(col_lower: str, base: str) -> float:
    """Biologic style units, e.g. '<I>/mA' -> 1e-3 (to A), 'Ewe/V' -> 1."""
    unit = col_lower.rsplit("/", 1)[-1].strip() if "/" in col_lower else ""
    prefixes = {"": 1.0, "m": 1e-3, "µ": 1e-6, "μ": 1e-6, "u": 1e-6, "n": 1e-9, "p": 1e-12, "k": 1e3}
    if unit.endswith(base):
        return prefixes.get(unit[: len(unit) - len(base)], 1.0)
    return 1.0

def parse_biologic_mpt(raw: str):
    lines = raw.splitlines()
    meta, header_lines = {}, 0
    for line in lines[:10]:
        if "Nb header lines" in line:
            try: header_lines = int(line.split(":")[-1].strip())
            except ValueError: header_lines = 0
            break

    if header_lines == 0:
        # EC-Lab "Export as text" (no header block): the first non-empty row holds the column names
        first = next((i for i, l in enumerate(lines) if l.strip()), None)
        if first is not None and not _is_numeric_row(lines[first]):
            header_lines = first + 1

    for i in range(min(header_lines, len(lines))):
        line = lines[i].strip()
        if not line: continue
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
        else:
            parts = re.split(r'\s{2,}|\t+', line)
            if len(parts) >= 2: meta[parts[0].strip()] = parts[1].strip()

    data_lines = [line for line in lines[header_lines:] if line.strip()]
    if not data_lines: return meta, []

    line_minus_1 = [c for c in [c.strip() for c in lines[header_lines - 1].split('\t')] if c] if header_lines >= 1 else []
    line_minus_2 = [c for c in [c.strip() for c in lines[header_lines - 2].split('\t')] if c] if header_lines >= 2 else []

    num_cols = len([c.strip() for c in data_lines[0].split('\t') if c.strip()])
    if len(line_minus_1) == num_cols: cols = line_minus_1
    elif len(line_minus_2) + len(line_minus_1) == num_cols: cols = line_minus_2 + line_minus_1
    else:
        cols = [f"Col_{i}" for i in range(num_cols)]

    rows = []
    for line in data_lines:
        parts = [p.strip() for p in line.split("\t") if p.strip()]
        if len(parts) >= num_cols: rows.append(parts[:num_cols])
        elif len(parts) > 0: rows.append(parts + [np.nan] * (num_cols - len(parts)))

    unique_cols, seen = [], set()
    for c in cols:
        new_c, counter = c, 1
        while new_c in seen:
            new_c = f"{c}_{counter}"
            counter += 1
        unique_cols.append(new_c)
        seen.add(new_c)

    df = pd.DataFrame(rows, columns=unique_cols)
    for c in df.columns: df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(how="all").reset_index(drop=True)

    col_map, i_scale, v_scale = {}, None, None
    for c in df.columns:
        cl = c.lower()
        if "ewe" in cl or "potential" in cl or "voltage" in cl or "v" == cl:
            col_map[c] = "Vf"
            if v_scale is None: v_scale = _unit_scale_after_slash(cl, "v")
        elif "<i>" in cl or "current" in cl or "i/ma" in cl or "i" == cl:
            col_map[c] = "Im"
            if i_scale is None: i_scale = _unit_scale_after_slash(cl, "a")
        elif cl == "cycle number" or cl == "cycle": col_map[c] = "Cycle"
        elif "freq" in cl: col_map[c] = "Frequency"
        elif "re(z)" in cl or "z'" in cl: col_map[c] = "Z_real"
        elif "-im(z)" in cl or "-z''" in cl: col_map[c] = "neg_Z_imag"
        elif "im(z)" in cl or "z''" in cl: col_map[c] = "z_imag"
        elif "time" in cl or "t/s" in cl: col_map[c] = "Time"
    df = df.rename(columns=col_map)
    df = df.loc[:, ~df.columns.duplicated()]

    if "z_imag" in df.columns and "neg_Z_imag" not in df.columns:
        df["neg_Z_imag"] = -df["z_imag"]

    # Units from the column header (EC-Lab exports current in mA): always convert to A / V
    if "Im" in df.columns and i_scale is not None: df["Im"] = df["Im"] * i_scale
    if "Vf" in df.columns and v_scale is not None and v_scale != 1.0: df["Vf"] = df["Vf"] * v_scale

    curves = []
    if "Cycle" in df.columns:
        for cyc in sorted(df["Cycle"].dropna().unique()):
            df_cyc = df[df["Cycle"] == cyc].copy()
            if len(df_cyc) > 0: curves.append((f"Cycle {int(cyc) if float(cyc).is_integer() else cyc}", df_cyc.reset_index(drop=True)))
    else: curves.append(("Curve 1", df))

    header_text = "\n".join(lines[:header_lines]).lower()
    if "Frequency" in df.columns and "Z_real" in df.columns: meta["TECHNIQUE"] = TECH_EIS
    elif "linear sweep voltammetry" in header_text or "linear polarization" in header_text: meta["TECHNIQUE"] = TECH_LSV
    elif "cyclic voltammetry" in header_text: meta["TECHNIQUE"] = TECH_CV
    elif "chronopotentiometry" in header_text: meta["TECHNIQUE"] = TECH_CP
    elif "chronoamperometry" in header_text: meta["TECHNIQUE"] = TECH_CA
    elif "Time" in df.columns and "Vf" in df.columns and "Im" not in df.columns: meta["TECHNIQUE"] = TECH_CP
    elif "Vf" in df.columns and "Im" in df.columns: meta["TECHNIQUE"] = TECH_CV

    return meta, curves

def parse_pstrace_csv(text: str) -> Tuple[Dict[str, str], List[Tuple[str, pd.DataFrame]]]:
    if not text: return {}, []

    lines = text.splitlines()
    meta = {}
    curves = []
    scan_names = []
    unit_row_idx = -1

    is_eis, is_cp = False, False

    if any("Fraction" in l or "fraction" in l for l in lines[:10]):
        meta["TECHNIQUE"] = "Chemical Speciation (Medusa)"

        header_idx = -1
        for i, line in enumerate(lines[:20]):
            if "pH" in line or "Fraction" in line:
                header_idx = i
                break

        if header_idx != -1:
            delimiter = "\t" if "\t" in lines[header_idx] else ","
            df = pd.read_csv(io.StringIO(text), sep=delimiter, header=header_idx)
            df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how='all')

            x_col = df.columns[0]
            for c in df.columns[1:]:
                df_clean = df[[x_col, c]].dropna().copy()
                df_clean.columns = ["x", "y"]
                curves.append((str(c).strip(), df_clean))
        return meta, curves

    for line in lines[:20]:
        if "Linear Sweep" in line or "LSV" in line: meta["TECHNIQUE"] = TECH_LSV
        elif "Cyclic Voltammetry" in line or "CV" in line: meta["TECHNIQUE"] = TECH_CV
        elif "Impedance" in line or "EIS" in line:
            meta["TECHNIQUE"] = TECH_EIS
            is_eis = True
        elif "Chronopotentiometry" in line or "CP E vs t" in line:
            meta["TECHNIQUE"] = TECH_CP
            is_cp = True

    if is_eis:
        for i, line in enumerate(lines[:50]):
            if "freq / Hz" in line:
                unit_row_idx = i; break
        if unit_row_idx != -1:
            cols = [p.strip() for p in lines[unit_row_idx].split(",") if p.strip()]
            rows = []
            for line in lines[unit_row_idx+1:]:
                if not line.strip(): continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) > len(cols): parts = parts[:len(cols)]
                elif len(parts) < len(cols): parts += [""] * (len(cols) - len(parts))
                rows.append(parts)
            df = pd.DataFrame(rows, columns=cols).replace("", np.nan).dropna(axis=1, how='all')
            for c in df.columns:
                if c: df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "."), errors="coerce")

            df_clean = pd.DataFrame()
            if "Z' / Ohm" in df.columns: df_clean["Z_real"] = df["Z' / Ohm"]
            if "-Z'' / Ohm" in df.columns: df_clean["neg_Z_imag"] = df["-Z'' / Ohm"]
            elif "Z'' / Ohm" in df.columns: df_clean["neg_Z_imag"] = -df["Z'' / Ohm"]
            if "freq / Hz" in df.columns: df_clean["Frequency"] = df["freq / Hz"]

            if "Z_real" in df_clean.columns and "neg_Z_imag" in df_clean.columns and "Frequency" in df_clean.columns:
                df_clean = df_clean.dropna()
                curves.append(("EIS Data", df_clean))
        return meta, curves

    if is_cp:
        for i, line in enumerate(lines[:50]):
            cols = [p.strip() for p in line.split(",")]
            if "s" in cols and "V" in cols:
                unit_row_idx = i; break
        if unit_row_idx != -1:
            cols = [p.strip() for p in lines[unit_row_idx].split(",") if p.strip()]
            rows = []
            for line in lines[unit_row_idx+1:]:
                if not line.strip(): continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) > len(cols): parts = parts[:len(cols)]
                elif len(parts) < len(cols): parts += [""] * (len(cols) - len(parts))
                rows.append(parts)
            df = pd.DataFrame(rows, columns=cols).replace("", np.nan).dropna(axis=1, how='all')
            for c in df.columns:
                if c: df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "."), errors="coerce")

            df_clean = pd.DataFrame()
            if "s" in df.columns: df_clean["Time"] = df["s"]
            if "V" in df.columns: df_clean["Vf"] = df["V"]

            if "Time" in df_clean.columns and "Vf" in df_clean.columns:
                df_clean = df_clean.dropna()
                curves.append(("CP Data", df_clean))
        return meta, curves

    for i, line in enumerate(lines[:50]):
        if ("Scan" in line or "Curve" in line or "vs E" in line) and "Date" not in line and "Voltammetry" in line:
            parts = [p.strip() for p in line.split(",") if p.strip()]
            if (len(parts) > 1 or (len(parts)==1 and "Scan" in parts[0])) and not scan_names: scan_names = parts
        parts = [p.strip() for p in line.split(",") if p.strip()]
        if len(parts) >= 2 and any(v in parts[0] for v in ['V', 'mV', 'E']) and any('A' in u for u in parts):
            unit_row_idx = i; break

    if unit_row_idx == -1: return meta, curves

    unit_parts = [p.strip() for p in lines[unit_row_idx].split(",")]
    rows = [[p.strip() for p in line.split(",")] for line in lines[unit_row_idx+1:] if line.strip()]
    if not rows: return meta, curves

    df_raw = pd.DataFrame(rows).replace("", np.nan)
    num_scans = len(unit_parts) // 2
    if not scan_names: scan_names = [f"Scan {i+1}" for i in range(num_scans)]

    for i in range(num_scans):
        col_v, col_i = i * 2, i * 2 + 1
        if col_i >= df_raw.shape[1]: break
        df_scan = df_raw.iloc[:, [col_v, col_i]].copy()
        df_scan.columns = ["Vf", "Im"]
        df_scan["Vf"] = pd.to_numeric(df_scan["Vf"].astype(str).str.replace(",", "."), errors="coerce")
        df_scan["Im"] = pd.to_numeric(df_scan["Im"].astype(str).str.replace(",", "."), errors="coerce")
        df_scan = df_scan.dropna()
        if len(df_scan) == 0: continue

        if "mV" in unit_parts[col_v]: df_scan["Vf"] = df_scan["Vf"] / 1000.0
        i_unit = unit_parts[col_i]
        if "mA" in i_unit: df_scan["Im"] = df_scan["Im"] * 1e-3
        elif any(u in i_unit for u in ("µA", "μA", "uA", "Âµ")): df_scan["Im"] = df_scan["Im"] * 1e-6
        elif "nA" in i_unit: df_scan["Im"] = df_scan["Im"] * 1e-9
        elif "pA" in i_unit: df_scan["Im"] = df_scan["Im"] * 1e-12

        name = scan_names[i] if i < len(scan_names) else f"Scan {i+1}"
        if "TECHNIQUE" not in meta:
            meta["TECHNIQUE"] = TECH_LSV if "Linear Sweep" in name or "LSV" in name else TECH_CV
        curves.append((name, df_scan))
    return meta, curves

# ---------- Metrohm Autolab NOVA (ASCII export: tab / semicolon / comma separated) ----------
_UNIT_PREFIX = {"": 1.0, "k": 1e3, "M": 1e6, "G": 1e9, "m": 1e-3, "µ": 1e-6, "μ": 1e-6, "u": 1e-6, "n": 1e-9, "p": 1e-12}

def _unit_factor(col_name: str, bases: Tuple[str, ...]) -> float:
    m = re.search(r"\(([^()]*)\)\s*$", str(col_name))
    if not m: return 1.0
    u = m.group(1).strip()
    for b in bases:
        if u.endswith(b):
            return _UNIT_PREFIX.get(u[: len(u) - len(b)].strip(), 1.0)
    return 1.0

def _norm_col(c) -> str:
    s = str(c).strip().lower()
    for a, b in (("′", "'"), ("’", "'"), ("‘", "'"), ("´", "'"), ("″", "''"), ('"', "''"), ("ω", "ohm"), ("ohms", "ohm"), ("º", "°")):
        s = s.replace(a, b)
    return re.sub(r"\s+", "", s)

def _nova_kind(c) -> Optional[str]:
    s = _norm_col(c)
    if s in ("index", "#", "no.", "point", "pt"): return "Index"
    if s.startswith("frequency") or s.startswith("freq"): return "Frequency"
    if re.match(r"^-(z''|zim|zimag|im\(z\))(\(|/|$)", s): return "neg_Z_imag"
    if re.match(r"^(z''|zim|zimag|im\(z\))(\(|/|$)", s): return "Z_imag"
    if re.match(r"^(z'|zre|zreal|re\(z\))(\(|/|$)", s): return "Z_real"
    if re.match(r"^(\|z\||z|zmod|mod\(z\))(\(|/|$)", s): return "Z_mod"
    if s.startswith("-phase"): return "neg_Phase"
    if s.startswith("phase"): return "Phase"
    if s.startswith("potentialapplied"): return "E_applied"
    if s.startswith("currentapplied"): return "I_applied"
    if re.match(r"^we\(\d+\)\.potential", s): return "E_we"
    if re.match(r"^we\(\d+\)\.current", s): return "I_we"
    if re.match(r"^(potential|e)\(dc\)", s) or s.startswith("dcpotential"): return "E_dc"
    if re.match(r"^(current|i)\(dc\)", s) or s.startswith("dccurrent"): return "I_dc"
    if s.startswith("correctedtime"): return "Time_corr"
    if s == "time" or s.startswith("time("): return "Time"
    if s == "scan": return "Scan"
    if re.match(r"^(potential|e|ewe)\((m?v)\)$", s): return "E_generic"
    if re.match(r"^(current|i)\(([mµμun]?a)\)$", s): return "I_generic"
    return None

def _is_nova_text(text: str) -> bool:
    for line in text.splitlines()[:60]:
        s = _norm_col(line)
        if ("frequency(hz)" in s and ("z'(" in s or "-z''(" in s or "phase(" in s)) \
                or re.search(r"we\(\d+\)\.(current|potential)", s) or "potentialapplied(" in s or "currentapplied(" in s:
            return True
    return False

def _split_nova_spectra(df: pd.DataFrame) -> List[pd.DataFrame]:
    """Several spectra exported in one file: split where Index restarts or the frequency jumps back."""
    n = len(df)
    if n < 4: return [df]
    starts = {0}
    if "Index" in df.columns and df["Index"].notna().all():
        idx = df["Index"].values
        starts.update(i for i in range(1, n) if idx[i] <= idx[i - 1])
    lf = np.log10(df["Frequency"].values)
    span, d = lf.max() - lf.min(), np.diff(lf)
    if span > 0 and len(d):
        main = np.sign(np.median(d)) or -1.0
        starts.update(i + 1 for i, di in enumerate(d) if np.sign(di) == -main and abs(di) > 0.5 * span)
    starts = sorted(starts)
    pieces = [df.iloc[a:b] for a, b in zip(starts, starts[1:] + [n])]
    return [p for p in pieces if len(p) >= 3] or [df]

def _derive_scan_rate(E, t) -> Optional[float]:
    """Scan rate (mV/s) from the potential program and time (NOVA exports have no scan-rate header)."""
    rates = []
    for a, b, d in sweep_segments(E):
        if d != 0 and b - a >= 4 and np.ptp(t[a:b + 1]) > 0:
            rates.append(abs(np.polyfit(t[a:b + 1], E[a:b + 1], 1)[0]))
    if not rates: return None
    r = float(np.median(rates)) * 1000.0
    return float(f"{r:.3g}") if r > 0 else None

def parse_nova_ascii(text: str) -> Tuple[Dict[str, str], List[Tuple[str, pd.DataFrame]]]:
    """Metrohm Autolab NOVA ASCII export (e.g. PGSTAT204). EIS: Frequency (Hz), Z' (Ω), -Z'' (Ω), Z (Ω), -Phase (°), Time (s).
    Voltammetry / chrono: Potential applied (V), WE(1).Current (A), WE(1).Potential (V), Time (s), Scan."""
    lines = text.splitlines()
    hdr, delim, cols = None, None, None
    for i, line in enumerate(lines[:60]):
        if not line.strip(): continue
        for d in ("\t", ";", ","):
            if d not in line: continue
            parts = [p.strip() for p in line.split(d)]
            if sum(_nova_kind(p) is not None for p in parts) >= 2:
                hdr, delim, cols = i, d, parts
                break
        if hdr is not None: break
    if hdr is None: return {}, []

    ncol = len(cols)
    rows = []
    for line in lines[hdr + 1:]:
        if not line.strip(): continue
        parts = [p.strip() for p in line.split(delim)]
        if len(parts) < ncol: parts += [""] * (ncol - len(parts))
        rows.append(parts[:ncol])
    if not rows: return {}, []
    raw_df = pd.DataFrame(rows, columns=[f"c{i}" for i in range(ncol)])

    def num(i):
        s = raw_df[f"c{i}"].astype(str).str.strip()
        if delim != ",": s = s.str.replace(",", ".", regex=False)  # decimal comma (Spanish locale)
        return pd.to_numeric(s, errors="coerce")

    kind_idx: Dict[str, int] = {}
    for i, c in enumerate(cols):
        k = _nova_kind(c)
        if k and k not in kind_idx: kind_idx[k] = i

    meta = {"INSTRUMENT": "Metrohm Autolab (NOVA)"}
    ohm = ("Ω", "Ω", "Ohm", "ohm")
    has_z = "Z_real" in kind_idx and ("neg_Z_imag" in kind_idx or "Z_imag" in kind_idx)
    has_polar = "Z_mod" in kind_idx and ("neg_Phase" in kind_idx or "Phase" in kind_idx)

    # ---------------- EIS ----------------
    if "Frequency" in kind_idx and (has_z or has_polar):
        out = pd.DataFrame()
        out["Frequency"] = num(kind_idx["Frequency"]) * _unit_factor(cols[kind_idx["Frequency"]], ("Hz",))
        if has_z:
            out["Z_real"] = num(kind_idx["Z_real"]) * _unit_factor(cols[kind_idx["Z_real"]], ohm)
            if "neg_Z_imag" in kind_idx:
                out["neg_Z_imag"] = num(kind_idx["neg_Z_imag"]) * _unit_factor(cols[kind_idx["neg_Z_imag"]], ohm)
            else:
                out["neg_Z_imag"] = -num(kind_idx["Z_imag"]) * _unit_factor(cols[kind_idx["Z_imag"]], ohm)
        if has_polar:
            k_ph = "neg_Phase" if "neg_Phase" in kind_idx else "Phase"
            ph = num(kind_idx[k_ph]) * (1.0 if k_ph == "neg_Phase" else -1.0)   # -phase
            if "rad" in _norm_col(cols[kind_idx[k_ph]]): ph = np.degrees(ph)
            out["Z_mod"] = num(kind_idx["Z_mod"]) * _unit_factor(cols[kind_idx["Z_mod"]], ohm)
            out["neg_Phase (°)"] = ph
            if not has_z:
                out["Z_real"] = out["Z_mod"] * np.cos(np.radians(ph))
                out["neg_Z_imag"] = out["Z_mod"] * np.sin(np.radians(ph))
        for k, name, base in (("Time", "Time", ("s",)), ("E_dc", "E_dc", ("V",)), ("I_dc", "I_dc", ("A",)), ("Index", "Index", ())):
            if k in kind_idx:
                out[name] = num(kind_idx[k]) * (_unit_factor(cols[kind_idx[k]], base) if base else 1.0)
        out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["Frequency", "Z_real", "neg_Z_imag"])
        out = out[out["Frequency"] > 0].reset_index(drop=True)
        if len(out) == 0: return meta, []
        meta["TECHNIQUE"] = TECH_EIS
        pieces = _split_nova_spectra(out)
        curves = []
        for k, piece in enumerate(pieces, 1):
            name = f"Spectrum {k}"
            if len(pieces) > 1 and "E_dc" in piece.columns and piece["E_dc"].notna().any():
                name += f" (E_dc = {piece['E_dc'].median():.3f} V)"
            curves.append((name, piece.reset_index(drop=True)))
        return meta, curves

    # ---------------- Voltammetry / chronometry ----------------
    E_key = next((k for k in ("E_applied", "E_we", "E_generic") if k in kind_idx), None)
    I_key = next((k for k in ("I_we", "I_generic", "I_applied") if k in kind_idx), None)
    t_key = next((k for k in ("Time", "Time_corr") if k in kind_idx), None)
    if E_key is None or (I_key is None and t_key is None): return meta, []
    out = pd.DataFrame()
    out["Vf"] = num(kind_idx[E_key]) * _unit_factor(cols[kind_idx[E_key]], ("V",))
    if I_key: out["Im"] = num(kind_idx[I_key]) * _unit_factor(cols[kind_idx[I_key]], ("A",))
    if t_key: out["Time"] = num(kind_idx[t_key]) * _unit_factor(cols[kind_idx[t_key]], ("s",))
    if E_key == "E_applied" and "E_we" in kind_idx:
        out["E_we"] = num(kind_idx["E_we"]) * _unit_factor(cols[kind_idx["E_we"]], ("V",))
    if "Scan" in kind_idx: out["Scan"] = num(kind_idx["Scan"])
    if "Index" in kind_idx: out["Index"] = num(kind_idx["Index"])
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["Vf"] + (["Im"] if "Im" in out.columns else [])).reset_index(drop=True)
    if len(out) < 3: return meta, []

    E = out["Vf"].values
    e_span = float(np.nanmax(E) - np.nanmin(E))
    I = out["Im"].values if "Im" in out.columns else None
    i_const = I is not None and np.nanstd(I) <= 0.02 * max(np.nanmax(np.abs(I)), 1e-15)
    if "I_applied" in kind_idx or (E_key != "E_applied" and t_key and (I is None or i_const) and e_span > 0.005):
        tech = TECH_CP
    elif t_key and e_span < 0.01:
        tech = TECH_CA
    else:
        first = out[out["Scan"] == out["Scan"].iloc[0]] if "Scan" in out.columns else out
        tech = TECH_CV if ("Scan" in out.columns or len(sweep_segments(first["Vf"].values)) >= 2) else TECH_LSV
    meta["TECHNIQUE"] = tech
    if tech in (TECH_CV, TECH_LSV) and "Time" in out.columns:
        first = out[out["Scan"] == out["Scan"].iloc[0]] if "Scan" in out.columns else out
        sr = _derive_scan_rate(first["Vf"].values, first["Time"].values)
        if sr: meta["SCANRATE"] = sr

    curves = []
    if "Scan" in out.columns and tech in (TECH_CV, TECH_LSV):
        for sv in pd.unique(out["Scan"].dropna()):
            piece = out[out["Scan"] == sv].reset_index(drop=True)
            if len(piece) >= 3:
                curves.append((f"Scan {int(sv) if float(sv).is_integer() else sv}", piece))
    if not curves:
        curves = [("Curve 1", out)]
    return meta, curves

# ---------- Format detection ----------
def detect_format(text: str) -> Optional[str]:
    head = text[:6000]
    if "EC-Lab ASCII FILE" in head or "Nb header lines" in head: return "biologic"
    if re.search(r"^\s*EXPLAIN\s*$", head, re.M) or (re.search(r"^TAG\t", head, re.M) and re.search(r"CURVE\d*\s*\tTABLE", text)):
        return "gamry"
    first = next((l for l in head.splitlines() if l.strip()), "")
    if "\t" in first and re.search(r"(Ewe/V|<I>/mA|Re\(Z\)/Ohm|freq/Hz|control/V|cycle number)", first):
        return "biologic"
    if _is_nova_text(head): return "nova"
    return None

def parse_any(fname: str, raw: bytes, instrument_hint: str):
    try:
        if fname.lower().endswith(".plt"):
            return parse_medusa_plt(raw)
        text = decode_bytes(raw)
        fmt = detect_format(text)
        if fmt == "biologic": return parse_biologic_mpt(text)
        if fmt == "gamry": return parse_gamry_dta_multi_curve(text)
        if fmt == "nova":
            meta, curves = parse_nova_ascii(text)
            if curves: return meta, curves
        meta, curves = parse_pstrace_csv(text)
        if curves: return meta, curves
        # Fall back on the parser selected in the sidebar
        if instrument_hint.startswith("Gamry"): return parse_gamry_dta_multi_curve(text)
        if instrument_hint.startswith("Metrohm"): return parse_nova_ascii(text)
        return parse_biologic_mpt(text)
    except Exception as e:
        return {"PARSE_ERROR": f"{type(e).__name__}: {e}"}, []

@st.cache_data(show_spinner=False, max_entries=256)
def parse_uploaded(fname: str, raw: bytes, instrument_hint: str):
    return parse_any(fname, raw, instrument_hint)

def convert_df_to_excel(curves_list: List[Tuple[str, pd.DataFrame]]) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        seen_names = set()
        sheets_written = 0
        for cid, df in curves_list:
            if "Z_real" in df.columns and "neg_Z_imag" in df.columns:
                clean_df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["Z_real", "neg_Z_imag"])
            elif "Time" in df.columns and "Vf" in df.columns:
                clean_df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["Time", "Vf"])
            elif "x" in df.columns and "y" in df.columns:
                clean_df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["x", "y"])
            else:
                Ecol = "Vf" if "Vf" in df.columns else ("Vu" if "Vu" in df.columns else None)
                if Ecol is None or "Im" not in df.columns: continue
                clean_df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[Ecol, "Im"])

            if len(clean_df) >= 2:
                safe_name = re.sub(r'[\\*?:/\[\]]', '_', str(cid))[:31].strip() or "Sheet"
                original_safe_name, counter = safe_name, 1
                while safe_name in seen_names:
                    suffix = f"_{counter}"
                    safe_name = f"{original_safe_name[:31-len(suffix)]}{suffix}"
                    counter += 1
                seen_names.add(safe_name)
                clean_df.to_excel(writer, index=False, sheet_name=safe_name)
                sheets_written += 1

        if sheets_written == 0:
            pd.DataFrame({"Message": ["No parseable data found"]}).to_excel(writer, index=False, sheet_name="Empty")

    return output.getvalue()

@st.cache_data(show_spinner=False, max_entries=256)
def excel_for_upload(fname: str, raw: bytes, instrument_hint: str) -> bytes:
    _, curves = parse_uploaded(fname, raw, instrument_hint)
    return convert_df_to_excel(curves)

def technique_kind(tech: str) -> str:
    if "EIS" in tech: return "EIS"
    if "CP" in tech or "Chronopotentiometry" in tech: return "CP"
    if "Medusa" in tech: return "Medusa"
    if "Chronoamperometry" in tech: return "CA"
    return "VOLT"


# ============================================================
# APP MAIN LOGIC & PLOTTING
# ============================================================
uploaded_files = st.file_uploader("Upload CV/LSV/CP/EIS or Medusa (.plt) files", type=["csv", "CSV", "DTA", "dta", "mpt", "MPT", "txt", "TXT", "dat", "DAT", "plt", "PLT"], accept_multiple_files=True)

if uploaded_files:
    file_dict = {f.name: f for f in uploaded_files}
    display_names = set([f"⋮⋮ {name}" for name in file_dict.keys()])

    if 'file_groups' not in st.session_state: st.session_state.file_groups = [{"header": "📥 Unassigned Files", "items": []}, {"header": "📊 Group 1", "items": []}]
    for group in st.session_state.file_groups: group["items"] = [item for item in group["items"] if item in display_names]
    existing_items = set([item for group in st.session_state.file_groups for item in group["items"]])
    new_items = display_names - existing_items
    if new_items: st.session_state.file_groups[0]["items"].extend(sorted(new_items))

    with st.sidebar:
        st.markdown("---")
        st.header("🗂️ Drag & Drop Groups")
        c1, c2 = st.columns(2)
        if c1.button("➕ Add Group"): st.session_state.file_groups.append({"header": f"📊 Group {len(st.session_state.file_groups)}", "items": []}); st.rerun()
        if c2.button("➖ Remove Group") and len(st.session_state.file_groups) > 1:
            st.session_state.file_groups[0]["items"].extend(st.session_state.file_groups[-1]["items"])
            st.session_state.file_groups.pop(); st.rerun()
        unassigned_count = len(st.session_state.file_groups[0]["items"])
        if unassigned_count > 0 and len(st.session_state.file_groups) > 1:
            bc1, bc2 = st.columns([2, 1])
            with bc1: target_g = st.selectbox("Target", [g["header"] for g in st.session_state.file_groups[1:]], label_visibility="collapsed")
            with bc2:
                if st.button("Move All"):
                    for g in st.session_state.file_groups:
                        if g["header"] == target_g:
                            g["items"].extend(st.session_state.file_groups[0]["items"]); st.session_state.file_groups[0]["items"] = []; st.rerun()
        st.session_state.file_groups = sort_items(st.session_state.file_groups, multi_containers=True)

    prepared_group_data = {}
    valid_groups_for_super = []

    for g_idx, group in enumerate(st.session_state.file_groups):
        if g_idx == 0 or not group["items"]: continue
        valid_groups_for_super.append(group["header"])

        group_data_parsed = []
        is_group_lsv, is_group_eis, is_group_cp, is_group_medusa = False, False, False, False

        for item in group["items"]:
            fname = item.replace("⋮⋮ ", "")
            if fname not in file_dict: continue
            meta_sg, curves_comp = parse_uploaded(fname, file_dict[fname].getvalue(), instrument)

            tech_sg = meta_sg.get("TECHNIQUE", "")
            kind_sg = technique_kind(tech_sg)
            sr = manual_scan_rate if manual_scan_rate > 0.0 else _to_float(meta_sg.get("SCANRATE", meta_sg.get("dE/dt")))

            if "LSV" in tech_sg: is_group_lsv = True
            if kind_sg == "EIS": is_group_eis = True
            if kind_sg == "CP": is_group_cp = True
            if kind_sg == "Medusa": is_group_medusa = True

            processed_curves = []
            for cid, df_comp in curves_comp:
                if kind_sg == "EIS":
                    if "Z_real" in df_comp.columns and "neg_Z_imag" in df_comp.columns and "Frequency" in df_comp.columns:
                        _df = df_comp.loc[:, ~df_comp.columns.duplicated()]
                        dd_comp = _df[["Z_real", "neg_Z_imag", "Frequency"]].replace([np.inf, -np.inf], np.nan).dropna().copy()
                        dd_comp.columns = ["x", "y", "f"]
                        if crop_eis:
                            dd_comp = dd_comp[(dd_comp["f"] >= eis_min_f) & (dd_comp["f"] <= eis_max_f)]
                        if len(dd_comp) >= 5: processed_curves.append((cid, dd_comp.reset_index(drop=True)))
                elif kind_sg == "CP":
                    if "Time" in df_comp.columns and "Vf" in df_comp.columns:
                        _df = df_comp.loc[:, ~df_comp.columns.duplicated()]
                        dd_comp = _df[["Time", "Vf"]].replace([np.inf, -np.inf], np.nan).dropna().copy()
                        if convert_to_rhe: dd_comp["Vf"] = dd_comp["Vf"] + e0_ref + (0.0591 * ph_val)
                        dd_comp.columns = ["x", "y"]
                        if len(dd_comp) >= 2: processed_curves.append((cid, dd_comp))
                elif kind_sg == "Medusa":
                    processed_curves.append((cid, df_comp))
                elif kind_sg == "CA":
                    continue
                else:
                    Ecol = "Vf" if "Vf" in df_comp.columns else ("Vu" if "Vu" in df_comp.columns else None)
                    if Ecol and "Im" in df_comp.columns:
                        _df = df_comp.loc[:, ~df_comp.columns.duplicated()]
                        dd_comp = _df[[Ecol, "Im"]].replace([np.inf, -np.inf], np.nan).dropna().copy()
                        if apply_ir: dd_comp[Ecol] = dd_comp[Ecol] - dd_comp["Im"] * ru_ohms * (comp_percent / 100.0)
                        if convert_to_rhe: dd_comp[Ecol] = dd_comp[Ecol] + e0_ref + (0.0591 * ph_val)
                        dd_comp.columns = ["x", "y"]
                        if len(dd_comp) >= 10: processed_curves.append((cid, dd_comp))
            if processed_curves:
                group_data_parsed.append({"fname": fname, "tech": tech_sg, "sr": sr, "curves": processed_curves})

        group_plot_data = []
        if len(group_data_parsed) > 0:
            for dat in group_data_parsed:
                for cid, dd_comp in dat["curves"]:
                    trace_name = f"{cid}" if "Medusa" in dat["tech"] else (f"{dat['fname']}" if len(dat['curves']) == 1 else f"{dat['fname']} ({cid})")
                    sr_val = get_sr_from_name(trace_name, dat['sr'] if dat['sr'] else 0.0)
                    group_plot_data.append({"x": dd_comp["x"].values, "y": dd_comp["y"].values, "f": dd_comp["f"].values if "f" in dd_comp.columns else np.array([]), "std": None, "name": trace_name, "df": dd_comp, "tech": dat["tech"], "sr": sr_val})

        prepared_group_data[group['header']] = {"traces": group_plot_data, "is_lsv": is_group_lsv, "is_eis": is_group_eis, "is_cp": is_group_cp, "is_medusa": is_group_medusa}

    # --- ZONA: SUPER GROUPS ---
    st.markdown("---")
    st.header("🧬 Super Groups (Combine & Fit)")
    st.markdown("Merge groups into a single plot. **Supports Cycle Averaging, Kinetic Analysis ($b$-value), and EIS Equivalent Circuit Fitting.**")

    if 'num_super_groups' not in st.session_state: st.session_state.num_super_groups = 0

    col_sg1, col_sg2, _ = st.columns([1, 1, 6])
    with col_sg1:
        if st.button("➕ Add Super Group"): st.session_state.num_super_groups += 1
    with col_sg2:
        if st.session_state.num_super_groups > 0:
            if st.button("➖ Remove Last"): st.session_state.num_super_groups -= 1

    for sg in range(st.session_state.num_super_groups):
        with st.expander(f"Super Group {sg+1} Configurations", expanded=True):
            selected_groups = st.multiselect("Select groups to merge:", options=valid_groups_for_super, key=f"super_group_select_{sg}")

            if selected_groups:
                is_sg_eis = any(prepared_group_data[g].get("is_eis", False) for g in selected_groups if g in prepared_group_data)
                is_sg_lsv = any(prepared_group_data[g].get("is_lsv", False) for g in selected_groups if g in prepared_group_data)
                is_sg_cp = any(prepared_group_data[g].get("is_cp", False) for g in selected_groups if g in prepared_group_data)
                is_sg_medusa = any(prepared_group_data[g].get("is_medusa", False) for g in selected_groups if g in prepared_group_data)
                sg_kind = "EIS" if is_sg_eis else ("CP" if is_sg_cp else ("Medusa" if is_sg_medusa else "VOLT"))

                avg_mode = st.radio("Super Group Mode:", ["Plot Individual Files", "Average ALL Files inside each Group"], key=f"sg_avg_{sg}", horizontal=True)

                sg_eis_models = {}
                fit_eis_model_toggle = False
                if is_sg_eis:
                    col_fit, col_model = st.columns([1, 2])
                    with col_fit:
                        fit_eis_model_toggle = st.toggle("🔋 Perform EIS Fit", value=False, key=f"fit_eis_{sg}")
                    with col_model:
                        if fit_eis_model_toggle:
                            st.markdown("**Select Equivalent Circuit Model for each Group:**")
                            cols_models = st.columns(min(3, len(selected_groups)))
                            for i, g_name in enumerate(selected_groups):
                                clean_name = g_name.replace("📊 ", "")
                                sg_eis_models[g_name] = cols_models[i%3].selectbox(
                                    f"Model for {clean_name}:",
                                    EIS_MODELS_LIST,
                                    key=f"eis_model_sel_{sg}_{i}"
                                )

                st.markdown("**Customize Legend Labels for this Super Group:**")
                sg_custom_labels = {}
                cols = st.columns(3)
                for i, g_name in enumerate(selected_groups):
                    clean_name = g_name.replace("📊 ", "")
                    sg_custom_labels[g_name] = cols[i%3].text_input(f"Label for {clean_name}", value=clean_name, key=f"sg_lbl_{sg}_{i}")

                fig_super = go.Figure()
                fig_super_tafel, fig_super_jeta = go.Figure(), go.Figure()
                fig_bode_mod, fig_bode_phase = go.Figure(), go.Figure()

                sg_lsv_params, sg_cv_kinetics, sg_eis_params, sg_eis_errors = [], [], [], []
                sg_skipped = []
                sg_max_log_I = -10
                y_axis_label_medusa = "Fraction"

                sg_min_z, sg_max_z = 0.0, 1.0

                for g_idx, g_name in enumerate(selected_groups):
                    if g_name not in prepared_group_data: continue
                    g_data = prepared_group_data[g_name]
                    base_color = combined_palette[g_idx % len(combined_palette)]

                    # Only traces compatible with the super-group type are combined
                    g_traces = [tr for tr in g_data["traces"] if technique_kind(tr["tech"]) == sg_kind]
                    sg_skipped += [tr["name"] for tr in g_data["traces"] if technique_kind(tr["tech"]) != sg_kind]

                    traces_to_plot = []
                    if avg_mode == "Average ALL Files inside each Group" and len(g_traces) > 0:
                        if is_sg_eis:
                            common_f, zr_mean, zi_mean, zr_std, zi_std = get_averaged_eis_curve([(tr["name"], tr["df"]) for tr in g_traces])
                            if common_f is None:
                                st.warning(f"{sg_custom_labels[g_name]}: the spectra have no common frequency range, cannot average.")
                            else:
                                df_mean = pd.DataFrame({"x": zr_mean, "y": zi_mean, "f": common_f})
                                traces_to_plot = [{"x": zr_mean, "y": zi_mean, "f": common_f, "std": None, "name": sg_custom_labels[g_name], "df": df_mean, "tech": "EIS", "group": g_name}]
                        elif is_sg_cp:
                            E_mean, I_mean, I_std = get_averaged_curve([(tr["name"], tr["df"]) for tr in g_traces])
                            df_mean = pd.DataFrame({"x": E_mean, "y": I_mean})
                            traces_to_plot = [{"x": E_mean, "y": I_mean, "f": None, "std": I_std, "name": sg_custom_labels[g_name], "df": df_mean, "tech": "CP", "sr": 0.0, "group": g_name}]
                        elif is_sg_medusa:
                            st.warning("Averaging is not supported for Medusa diagrams. Plotting individually.")
                            for tr in g_traces:
                                tr_copy = tr.copy()
                                tr_copy["group"] = g_name
                                traces_to_plot.append(tr_copy)
                        else:
                            avg_traces = g_traces
                            if is_sg_lsv:  # never average CV scans together with LSV curves
                                avg_traces = [t for t in g_traces if "LSV" in t["tech"]]
                                sg_skipped += [t["name"] for t in g_traces if "LSV" not in t["tech"]]
                            if avg_traces:
                                E_mean, I_mean, I_std = get_averaged_curve([(tr["name"], tr["df"]) for tr in avg_traces])
                                df_mean = pd.DataFrame({"x": E_mean, "y": I_mean, "std": I_std})
                                fallback_sr = np.mean([t["sr"] for t in avg_traces if t["sr"]>0]) if any(t["sr"]>0 for t in avg_traces) else 0.0
                                sr_val = get_sr_from_name(sg_custom_labels[g_name], fallback_sr)
                                traces_to_plot = [{"x": E_mean, "y": I_mean, "f": None, "std": I_std, "name": sg_custom_labels[g_name], "df": df_mean, "tech": "LSV" if is_sg_lsv else "CV", "sr": sr_val, "group": g_name}]
                    else:
                        for idx_tr, tr in enumerate(g_traces):
                            if is_sg_medusa:
                                final_name = f"{tr['name']}"
                                c_color = combined_palette[idx_tr % len(combined_palette)]
                            else:
                                final_name = sg_custom_labels[g_name] if len(g_traces) == 1 else f"{sg_custom_labels[g_name]} - {tr['name']}"
                                c_color = base_color

                            tr_copy = tr.copy()
                            tr_copy["name"] = final_name
                            tr_copy["group"] = g_name
                            tr_copy["color"] = c_color
                            traces_to_plot.append(tr_copy)

                    for tr in traces_to_plot:
                        c_color = tr.get("color", base_color)

                        if is_sg_eis:
                            zr, zi, f_hz = np.asarray(tr["x"], float), np.asarray(tr["y"], float), np.asarray(tr["f"], float)

                            sg_min_z = min(sg_min_z, np.min(zr), np.min(zi))
                            sg_max_z = max(sg_max_z, np.max(zr), np.max(zi))

                            fig_super.add_trace(go.Scatter(x=zr, y=zi, mode='markers', name=tr["name"], marker=dict(color=c_color, size=6)))

                            z_mod = np.sqrt(zr**2 + zi**2)
                            phase = np.degrees(np.arctan2(zi, zr))  # = -phase (zi is -Z'')

                            fig_bode_mod.add_trace(go.Scatter(x=f_hz, y=z_mod, mode='markers', name=tr["name"], marker=dict(color=c_color, size=6)))
                            fig_bode_phase.add_trace(go.Scatter(x=f_hz, y=phase, mode='markers', name=tr["name"], marker=dict(color=c_color, size=6)))

                            if fit_eis_model_toggle and tr["group"] in sg_eis_models:
                                selected_model = sg_eis_models[tr["group"]]
                                with st.spinner(f"Fitting {tr['name']}…"):
                                    results_dict, f_sim, zr_sim, zi_sim, fit_info = fit_eis_cached(f_hz, zr, zi, selected_model, float(electrode_area), float(manual_rf))
                                if results_dict is not None:
                                    row = dict(results_dict)
                                    row["Curve"] = tr["name"]
                                    row["Model"] = selected_model.split(" [")[0]
                                    sg_eis_params.append(row)
                                    err_row = {"Curve": tr["name"], "Model": row["Model"]}
                                    for p_name, err in fit_info["rel_err"].items():
                                        bnd = fit_info["at_bound"].get(p_name)
                                        err_row[p_name] = f"at {bnd} bound" if bnd else (f"±{err:.1f}%" if np.isfinite(err) else "n/a")
                                    sg_eis_errors.append(err_row)

                                    sg_min_z = min(sg_min_z, np.min(zr_sim), np.min(zi_sim))
                                    sg_max_z = max(sg_max_z, np.max(zr_sim), np.max(zi_sim))

                                    fig_super.add_trace(go.Scatter(x=zr_sim, y=zi_sim, mode='lines', name=f"{tr['name']} Model", line=dict(color=c_color, width=2), showlegend=True, hoverinfo='skip'))
                                    z_mod_sim = np.sqrt(zr_sim**2 + zi_sim**2)
                                    phase_sim = np.degrees(np.arctan2(zi_sim, zr_sim))
                                    fig_bode_mod.add_trace(go.Scatter(x=f_sim, y=z_mod_sim, mode='lines', name=f"{tr['name']} Model", line=dict(color=c_color, width=2), showlegend=True, hoverinfo='skip'))
                                    fig_bode_phase.add_trace(go.Scatter(x=f_sim, y=phase_sim, mode='lines', name=f"{tr['name']} Model", line=dict(color=c_color, width=2), showlegend=True, hoverinfo='skip'))
                                else:
                                    st.warning(f"EIS fit failed for {tr['name']} (not enough points for this model or no convergence).")
                        elif is_sg_cp:
                            if tr.get("std") is not None and show_sd_shadow:
                                fig_super.add_trace(go.Scatter(x=tr["x"], y=tr["y"]+tr["std"], mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
                                fig_super.add_trace(go.Scatter(x=tr["x"], y=tr["y"]-tr["std"], mode='lines', line=dict(width=0), fill='tonexty', fillcolor=to_rgba(c_color, 0.2), showlegend=False, hoverinfo='skip'))
                            fig_super.add_trace(go.Scatter(x=tr["x"], y=tr["y"], mode='lines', name=tr["name"], line=dict(color=c_color, width=2.5)))
                        elif is_sg_medusa:
                            fig_super.add_trace(go.Scatter(x=tr["x"], y=tr["y"], mode='lines', name=tr["name"], line=dict(color=c_color, width=2.5)))
                            if np.min(tr["y"]) < -1.0: y_axis_label_medusa = "Log Concentration"
                        else:
                            x_arr = np.asarray(tr["x"], float)
                            y_arr = np.asarray(tr["y"], float)

                            delta_j_half = cv_delta_j_half(x_arr, y_arr, e_non_faradaic, electrode_area)
                            E_pa, i_pa = cv_anodic_peak(x_arr, y_arr, peak_min_v if limit_peak_search else None, peak_max_v if limit_peak_search else None)

                            j_pa = (i_pa * 1000) / electrode_area
                            if np.isfinite(j_pa) and j_pa > 0 and tr.get("sr") and tr["sr"] > 0 and not is_sg_lsv:
                                sg_cv_kinetics.append({
                                    "Curve": tr["name"], "v (mV/s)": tr["sr"], "log_v": np.log10(tr["sr"]),
                                    "E_p (V)": E_pa, "j_p (mA/cm²)": j_pa, "log_jp": np.log10(j_pa),
                                    "delta_j_half": delta_j_half
                                })

                            j_y = y_arr * 1000 / electrode_area
                            if tr.get("std") is not None and show_sd_shadow:
                                j_std = tr["std"] * 1000 / electrode_area
                                fig_super.add_trace(go.Scatter(x=x_arr, y=j_y+j_std, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
                                fig_super.add_trace(go.Scatter(x=x_arr, y=j_y-j_std, mode='lines', line=dict(width=0), fill='tonexty', fillcolor=to_rgba(c_color, 0.2), showlegend=False, hoverinfo='skip'))
                            fig_super.add_trace(go.Scatter(x=x_arr, y=j_y, mode='lines', name=tr["name"], line=dict(color=c_color, width=2.5)))

                            if "LSV" in tr["tech"]:
                                cat_params, fit_data = extract_lsv_catalytic_parameters(tr["df"], electrode_area, e_rev)
                                if cat_params:
                                    sg_lsv_params.append({"Group": g_name, "Curve": tr["name"], **cat_params})
                                    sg_max_log_I = max(sg_max_log_I, fit_data["log_I_max"])
                                    fig_super_tafel.add_trace(go.Scatter(x=fit_data["log_I_full"], y=fit_data["E_full"], mode='lines', name=tr["name"], line=dict(color=c_color, width=2.5)))
                                    if not np.isnan(fit_data["slope"]) and len(fit_data["log_I_fit"]) > 0:
                                        min_x, max_x = np.min(fit_data["log_I_fit"]), np.max(fit_data["log_I_fit"])
                                        span = max_x - min_x
                                        fit_x = np.array([min_x - (span*1.5), max_x + (span*1.5)])
                                        fig_super_tafel.add_trace(go.Scatter(x=fit_x, y=fit_data["slope"]*fit_x + fit_data["intercept"], mode='lines', name=f"Fit: {cat_params['Tafel Slope (mV/dec)']:.1f} mV/dec", line=dict(color=c_color, width=2, dash='dot')))

                                    if fit_data["j_std"] is not None and show_sd_shadow:
                                        j_dens, j_std = fit_data["j_dens"], fit_data["j_std"]
                                        fig_super_jeta.add_trace(go.Scatter(x=fit_data["eta_mV"], y=j_dens+j_std, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
                                        fig_super_jeta.add_trace(go.Scatter(x=fit_data["eta_mV"], y=j_dens-j_std, mode='lines', line=dict(width=0), fill='tonexty', fillcolor=to_rgba(c_color, 0.2), showlegend=False, hoverinfo='skip'))
                                    fig_super_jeta.add_trace(go.Scatter(x=fit_data["eta_mV"], y=fit_data["j_dens"], mode='lines', name=tr["name"], line=dict(color=c_color, width=2.5)))

                if sg_skipped:
                    sg_label = {"EIS": "EIS", "CP": "CP", "Medusa": "speciation"}.get(sg_kind, "LSV" if is_sg_lsv else "CV")
                    st.warning(f"Skipped {len(sg_skipped)} curve(s) that don't match this {sg_label} super group: " + ", ".join(sg_skipped[:6]) + ("…" if len(sg_skipped) > 6 else ""))

                if is_sg_eis:
                    axis_min = sg_min_z * 1.05 if sg_min_z < 0 else 0.0
                    axis_max = sg_max_z * 1.05

                    # Geometría estricta de 620x600 compensando los márgenes de apply_scientific_style (120px ancho, 100px alto)
                    # El área interna quedará exactamente en 500x500 píxeles.
                    fig_super.update_layout(title="Nyquist Plot", xaxis_title="Z' (Ω)", yaxis_title="-Z'' (Ω)", width=620, height=600)
                    fig_super.update_xaxes(range=[axis_min, axis_max], constrain="domain")
                    fig_super.update_yaxes(range=[axis_min, axis_max], scaleanchor="x", scaleratio=1, constrain="domain")
                    fig_super = apply_scientific_style(fig_super, scientific_style, lx, ly, lxa, lya)
                    show_chart(fig_super, stretch=False, config=dl_config_square)

                    c1, c2 = st.columns(2)
                    with c1:
                        fig_bode_mod.update_layout(title="Bode Plot (|Z|)", xaxis_title="Frequency f (Hz)", yaxis_title="|Z| (Ω)", xaxis_type="log", yaxis_type="log", width=620, height=600)
                        fig_bode_mod = apply_scientific_style(fig_bode_mod, scientific_style, lx, ly, lxa, lya)
                        show_chart(fig_bode_mod, stretch=False, config=dl_config_square)
                    with c2:
                        fig_bode_phase.update_layout(title="Bode Plot (Phase)", xaxis_title="Frequency f (Hz)", yaxis_title="-Phase (°)", xaxis_type="log", width=620, height=600)
                        fig_bode_phase = apply_scientific_style(fig_bode_phase, scientific_style, lx, ly, lxa, lya)
                        show_chart(fig_bode_phase, stretch=False, config=dl_config_square)

                    if fit_eis_model_toggle and sg_eis_params:
                        st.markdown("#### ⚡ Equivalent Circuit Fit Results")
                        df_eis = pd.DataFrame(sg_eis_params)

                        if "_rct" in df_eis.columns:
                            df_eis["Rct Ratio (Max/Current)"] = df_eis["_rct"].max() / df_eis["_rct"]

                        all_possible_cols = ["Curve", "Model", "Rs (Ω)", "CPE-T", "CPE-P", "Rct (Ω)", "W (Ω·s^-0.5)", "RL (Ω)", "L (H)", "CPE1-T", "CPE1-P", "R1 (Ω)", "CPE2-T", "CPE2-P", "R2 (Ω)", "CPE3-T", "CPE3-P", "R3 (Ω)", "Rads (Ω)", "R_p (Ω)", "Rct from", "Rct·A (Ω·cm²)", "Rct·A·RF (Ω·cm²_ECSA)", "C_eff (F)", "C_eff1 (F)", "C_eff2 (F)", "C_eff3 (F)", "Rct Ratio (Max/Current)", "R²", "χ²"]
                        cols = [c for c in all_possible_cols if c in df_eis.columns]
                        df_eis = df_eis[cols]

                        format_dict = {
                            "Rs (Ω)": "{:.2f}", "CPE-T": "{:.2e}", "CPE-P": "{:.3f}",
                            "Rct (Ω)": "{:.2f}", "W (Ω·s^-0.5)": "{:.2e}", "RL (Ω)": "{:.2f}", "L (H)": "{:.2e}",
                            "CPE1-T": "{:.2e}", "CPE1-P": "{:.3f}", "R1 (Ω)": "{:.2f}",
                            "CPE2-T": "{:.2e}", "CPE2-P": "{:.3f}", "R2 (Ω)": "{:.2f}",
                            "CPE3-T": "{:.2e}", "CPE3-P": "{:.3f}", "R3 (Ω)": "{:.2f}",
                            "Rads (Ω)": "{:.2f}", "R_p (Ω)": "{:.2f}", "Rct·A (Ω·cm²)": "{:.2f}", "Rct·A·RF (Ω·cm²_ECSA)": "{:.2f}",
                            "C_eff (F)": "{:.2e}", "C_eff1 (F)": "{:.2e}", "C_eff2 (F)": "{:.2e}", "C_eff3 (F)": "{:.2e}",
                            "Rct Ratio (Max/Current)": "{:.2f}", "R²": "{:.4f}", "χ²": "{:.2e}"
                        }
                        show_table(df_eis.style.format({k: v for k, v in format_dict.items() if k in df_eis.columns}, na_rep="—"))

                        with st.expander("📐 Parameter uncertainties (±%, from the fit Jacobian)"):
                            df_err = pd.DataFrame(sg_eis_errors)
                            show_table(df_err[[c for c in all_possible_cols if c in df_err.columns]].fillna("—"))
                        bound_msgs = [f"{e['Curve']}: " + ", ".join(k for k, v in e.items() if isinstance(v, str) and "bound" in v) for e in sg_eis_errors if any(isinstance(v, str) and "bound" in v for v in e.values())]
                        if bound_msgs:
                            st.warning("⚠️ Some parameters reached a fitting bound (not determined by the data — e.g. an arc that does not close in the measured range, or a negligible Rs). Consider a simpler model or a wider frequency range:\n\n" + "\n".join(f"- {m}" for m in bound_msgs))
                elif is_sg_cp:
                    fig_super.update_layout(title="Chronopotentiometry" if not scientific_style else "", xaxis_title="Time t (s)", yaxis_title=cp_axis_label, height=500)
                    fig_super = apply_scientific_style(fig_super, scientific_style, lx, ly, lxa, lya)
                    show_chart(fig_super, config=dl_config)
                elif is_sg_medusa:
                    fig_super.update_layout(title="Chemical Speciation" if not scientific_style else "", xaxis_title="pH", yaxis_title=y_axis_label_medusa, height=500)
                    if crop_medusa:
                        fig_super.update_xaxes(range=[medusa_xmin, medusa_xmax])
                        fig_super.update_yaxes(range=[medusa_ymin, medusa_ymax])
                    fig_super = apply_scientific_style(fig_super, scientific_style, lx, ly, lxa, lya)
                    show_chart(fig_super, config=dl_config)
                else:
                    fig_super.update_layout(title="", xaxis_title=x_axis_label, yaxis_title=j_axis_label, height=500)
                    fig_super = apply_scientific_style(fig_super, scientific_style, lx, ly, lxa, lya)
                    show_chart(fig_super, config=dl_config)

                    if is_sg_lsv:
                        c1, c2 = st.columns(2)
                        with c1:
                            fig_super_jeta.update_layout(title="Catalytic Performance" if not scientific_style else "", xaxis_title="Overpotential η (mV)", yaxis_title=jabs_axis_label, height=500)
                            fig_super_jeta = apply_scientific_style(fig_super_jeta, scientific_style, lx, ly, lxa, lya)
                            show_chart(fig_super_jeta, config=dl_config)
                        with c2:
                            fig_super_tafel.update_layout(title="Tafel Plot" if not scientific_style else "", xaxis_title="log₁₀|I| (A)", yaxis_title=x_axis_label, xaxis=dict(range=[sg_max_log_I - 4.5, sg_max_log_I + 0.2]), height=500)
                            fig_super_tafel = apply_scientific_style(fig_super_tafel, scientific_style, lx, ly, lxa, lya)
                            show_chart(fig_super_tafel, config=dl_config)

                        if sg_lsv_params:
                            st.markdown("#### 🧪 Catalytic Parameters (LSV)")
                            show_table(pd.DataFrame(sg_lsv_params))

                    elif len(sg_cv_kinetics) > 1:
                        st.markdown("#### 🔋 CV Kinetics & ECSA ($C_{dl}$ y $b$-value)")
                        df_cv = pd.DataFrame(sg_cv_kinetics).sort_values("v (mV/s)")

                        df_cv["v (V/s)"] = df_cv["v (mV/s)"] / 1000.0
                        df_cv_clean = df_cv.dropna(subset=["delta_j_half", "v (V/s)"])

                        c_dl, rf, r2_cdl = np.nan, np.nan, np.nan
                        fig_cdl = go.Figure()

                        if df_cv_clean["v (V/s)"].nunique() >= 2:
                            v_arr = df_cv_clean["v (V/s)"].values
                            dj_arr = df_cv_clean["delta_j_half"].values
                            c_dl = float(np.sum(v_arr * dj_arr) / np.sum(v_arr * v_arr))  # least squares through the origin

                            ss_res = np.sum((dj_arr - c_dl * v_arr)**2)
                            ss_tot = np.sum((dj_arr - np.mean(dj_arr))**2)
                            r2_cdl = max(0, 1 - (ss_res / ss_tot)) if ss_tot > 0 else 0

                            rf = c_dl / c_dl_ref if c_dl_ref > 0 else np.nan
                            df_cv["j_p,ECSA (mA/cm²_ECSA)"] = df_cv["j_p (mA/cm²)"] / rf

                            fig_cdl.add_trace(go.Scatter(x=v_arr, y=dj_arr, mode='markers', marker=dict(size=10, color='blue'), name="Data"))
                            fit_v = np.array([0, v_arr.max() * 1.1])
                            fig_cdl.add_trace(go.Scatter(x=fit_v, y=c_dl * fit_v, mode='lines', line=dict(color='red', dash='dash'), name=f"Fit (C_dl={c_dl:.3f} mF/cm²)"))

                            fig_cdl.update_layout(xaxis_title="Scan Rate v (V/s)", yaxis_title="Δj/2 (mA/cm²)", height=450)
                            fig_cdl = apply_scientific_style(fig_cdl, scientific_style, lx, ly, lxa, lya)

                        c1, c2 = st.columns([1, 1])
                        with c1:
                            if df_cv["v (mV/s)"].nunique() >= 2:
                                slope, intercept, r_value, p_value, std_err = linregress(df_cv["log_v"], df_cv["log_jp"])

                                fig_b = go.Figure()
                                fig_b.add_trace(go.Scatter(x=df_cv["log_v"], y=df_cv["log_jp"], mode='markers', marker=dict(size=10, color='black'), name="Data points"))
                                fit_x = np.array([df_cv["log_v"].min() - 0.1, df_cv["log_v"].max() + 0.1])
                                fig_b.add_trace(go.Scatter(x=fit_x, y=slope * fit_x + intercept, mode='lines', line=dict(color='red', dash='dash'), name=f"Fit: b = {slope:.2f}"))

                                fig_b.update_layout(xaxis_title="log₁₀(v) [mV/s]", yaxis_title="log₁₀(j_p) [mA cm⁻²]" if scientific_style else "log₁₀(j_p) [mA/cm²]", height=450)
                                fig_b = apply_scientific_style(fig_b, scientific_style, lx, ly, lxa, lya)
                                show_chart(fig_b, config=dl_config)
                                st.metric("b-value (Slope ± SE)", f"{slope:.4f} ± {std_err:.4f}", f"R² = {r_value**2:.4f}", delta_color="off")
                                st.info("💡 **b = 0.5**: Diffusion-controlled. **b = 1.0**: Surface-controlled.")
                            else:
                                st.info("The b-value and C_dl need curves recorded at **at least 2 different scan rates** (read from the file, the file name e.g. '50mVs', the legend label, or the manual scan rate).")
                        with c2:
                            if np.isfinite(c_dl):
                                show_chart(fig_cdl, config=dl_config)
                                st.metric("Double Layer Capacitance (C_dl)", f"{c_dl:.3f} mF/cm²", f"R² = {r2_cdl:.4f} (at {e_non_faradaic} V)", delta_color="off")
                                st.metric("Roughness Factor (RF)", f"{rf:.1f}", f"Ref = {c_dl_ref} mF/cm²", delta_color="off")
                            elif df_cv["v (mV/s)"].nunique() >= 2:
                                st.info(f"C_dl not computed: E = {e_non_faradaic} V (non-faradaic potential) is not crossed by both the anodic and cathodic sweeps of at least 2 scan rates.")

                        cols_to_show = ["Curve", "v (mV/s)", "E_p (V)", "j_p (mA/cm²)"]
                        if np.isfinite(rf): cols_to_show.append("j_p,ECSA (mA/cm²_ECSA)")
                        show_table(df_cv[cols_to_show].style.format({c: ("{:.4g}" if c.startswith("j_p") else "{:.3f}") for c in cols_to_show if c != "Curve"}))

    # --- BOTTOM AREA: INDIVIDUAL ANALYSIS ---
    st.markdown("---")
    st.header("📄 Individual Analysis")

    actual_file_order = [item.replace("⋮⋮ ", "") for group in st.session_state.file_groups for item in group["items"]]

    for file_name in actual_file_order:
        if file_name not in file_dict: continue

        file = file_dict[file_name]
        raw_bytes = file.getvalue()
        meta, curves = parse_uploaded(file_name, raw_bytes, instrument)

        technique = meta.get("TECHNIQUE", "Unknown Technique")
        sr = manual_scan_rate if manual_scan_rate > 0.0 else _to_float(meta.get("SCANRATE", meta.get("dE/dt")))

        if not curves:
            st.markdown(f"### {file.name}")
            st.warning("⚠️ Could not recognize this file format." + (f" ({meta['PARSE_ERROR']})" if meta.get("PARSE_ERROR") else "") + " Check the export format or pick the matching parser in the sidebar.")
            continue

        kind = technique_kind(technique)
        vinit, vlim1, vlim2 = None, None, None
        is_eis = kind == "EIS"
        is_cp = kind == "CP"
        is_medusa = kind == "Medusa"
        if kind == "VOLT":
            ref_curve = next((d for _, d in curves if "Im" in d.columns and ("Vf" in d.columns or "Vu" in d.columns)), curves[0][1])
            vinit, vlim1, vlim2 = extract_limits_from_data(ref_curve, technique)

        st.markdown(f"### {file.name}")
        col_title, col_btn = st.columns([4, 1])
        with col_title: st.markdown(f"🔬 **Technique Detected:** `{technique}`" + (f" · {meta['INSTRUMENT']}" if meta.get("INSTRUMENT") else ""))
        with col_btn: st.download_button("📥 Export to Excel", excel_for_upload(file_name, raw_bytes, instrument), f"{os.path.splitext(file.name)[0]}_Data.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key=f"dl_{file.name}")

        if kind == "CA":
            st.info("ℹ️ Chronoamperometry (constant potential) data are not analyzed yet; use **Export to Excel** to get the data.")
            continue
        if is_eis:
            st.info("💡 **Electrochemical Impedance Spectroscopy (EIS) Data:** Generating Nyquist Plot (-Z'' vs Z')")
        elif is_cp:
            st.info("💡 **Chronopotentiometry (CP) Data:** Generating E vs t Plot")
        elif is_medusa:
            st.info("💡 **Chemical Speciation Data:** Generating Distribution Diagram")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Initial Potential", f"{vinit} V" if vinit is not None else "N/A")
            if "LSV" in technique:
                c2.metric("Final Potential", f"{vlim1} V" if vlim1 is not None else "N/A")
                c3.metric("Scan Limit 2", "N/A")
            else:
                c2.metric("Scan Limit 1", f"{vlim1} V" if vlim1 is not None else "N/A")
                c3.metric("Scan Limit 2", f"{vlim2} V" if vlim2 is not None else "N/A")
            c4.metric("Scan Rate", f"{sr:g} mV/s" if sr is not None else "N/A")

        processed_curves = []
        y_axis_label_medusa_ind = "Fraction"

        for i, (cid, dfi) in enumerate(curves):
            if is_eis:
                if "Z_real" in dfi.columns and "neg_Z_imag" in dfi.columns and "Frequency" in dfi.columns:
                    _dfi = dfi.loc[:, ~dfi.columns.duplicated()]
                    dd = _dfi[["Z_real", "neg_Z_imag", "Frequency"]].replace([np.inf, -np.inf], np.nan).dropna().copy()
                    dd.columns = ["x", "y", "f"]
                    if crop_eis:
                        dd = dd[(dd["f"] >= eis_min_f) & (dd["f"] <= eis_max_f)]
                    if len(dd) >= 5: processed_curves.append((cid, dd.reset_index(drop=True)))
            elif is_cp:
                if "Time" in dfi.columns and "Vf" in dfi.columns:
                    _dfi = dfi.loc[:, ~dfi.columns.duplicated()]
                    dd = _dfi[["Time", "Vf"]].replace([np.inf, -np.inf], np.nan).dropna().copy()
                    if convert_to_rhe: dd["Vf"] = dd["Vf"] + e0_ref + (0.0591 * ph_val)
                    dd.columns = ["x", "y"]
                    if len(dd) >= 2: processed_curves.append((cid, dd))
            elif is_medusa:
                if dfi["y"].min() < -1.0: y_axis_label_medusa_ind = "Log Concentration"
                processed_curves.append((cid, dfi))
            else:
                Ecol = "Vf" if "Vf" in dfi.columns else ("Vu" if "Vu" in dfi.columns else None)
                if Ecol is None or "Im" not in dfi.columns: continue
                _dfi = dfi.loc[:, ~dfi.columns.duplicated()]
                dd = _dfi[[Ecol, "Im"]].replace([np.inf, -np.inf], np.nan).dropna().copy()
                if apply_ir: dd[Ecol] = dd[Ecol] - dd["Im"] * ru_ohms * (comp_percent / 100.0)
                if convert_to_rhe: dd[Ecol] = dd[Ecol] + e0_ref + (0.0591 * ph_val)
                dd.columns = ["x", "y"]
                if len(dd) >= 10: processed_curves.append((cid, dd))

        if not processed_curves: continue

        fig, fig_tafel, fig_jeta = go.Figure(), go.Figure(), go.Figure()
        results_list, lsv_cat_list, max_log_I_ind = [], [], -10

        ind_min_z, ind_max_z = 0.0, 1.0

        avg_cycles = st.toggle(f"🌟 Average {len(processed_curves)} Cycles/Scans", key=f"avg_{file.name}") if len(processed_curves) > 1 and not is_medusa else False

        if avg_cycles:
            if is_eis:
                common_f, zr_mean, zi_mean, zr_std, zi_std = get_averaged_eis_curve(processed_curves)
                if common_f is None:
                    st.warning("These spectra have no common frequency range; cannot average.")
                else:
                    ind_min_z = min(ind_min_z, np.min(zr_mean), np.min(zi_mean))
                    ind_max_z = max(ind_max_z, np.max(zr_mean), np.max(zi_mean))
                    mean_color = combined_palette[0]
                    fig.add_trace(go.Scatter(x=zr_mean, y=zi_mean, mode='markers', name='Average', marker=dict(color=mean_color, size=6)))
            elif is_cp:
                E_mean, I_mean, I_std = get_averaged_curve(processed_curves)
                mean_color = combined_palette[0]
                if show_sd_shadow:
                    fig.add_trace(go.Scatter(x=E_mean, y=I_mean + I_std, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
                    fig.add_trace(go.Scatter(x=E_mean, y=I_mean - I_std, mode='lines', line=dict(width=0), fill='tonexty', fillcolor=to_rgba(mean_color, 0.2), showlegend=False, hoverinfo='skip'))
                fig.add_trace(go.Scatter(x=E_mean, y=I_mean, mode='lines', name='Average', line=dict(color=mean_color, width=2.5)))
            else:
                E_mean, I_mean, I_std = get_averaged_curve(processed_curves)
                mean_color = combined_palette[0]

                j_mean = I_mean * 1000 / electrode_area
                j_std = I_std * 1000 / electrode_area

                if show_sd_shadow:
                    fig.add_trace(go.Scatter(x=E_mean, y=j_mean + j_std, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
                    fig.add_trace(go.Scatter(x=E_mean, y=j_mean - j_std, mode='lines', line=dict(width=0), fill='tonexty', fillcolor=to_rgba(mean_color, 0.2), showlegend=False, hoverinfo='skip'))
                fig.add_trace(go.Scatter(x=E_mean, y=j_mean, mode='lines', name='Average', line=dict(color=mean_color, width=2.5)))

                df_mean = pd.DataFrame({"x": E_mean, "y": I_mean, "std": I_std})
                out = recommend_operating_ranges_for_curve(df_mean)
                ns, ro = out["recommended_noise_safe_V"], out["recommended_reduction_only_V"]
                results_list.append({"Curve": "Average Curve", "Points": out["N_points"], "Noise-Safe Min (V)": round(ns[0], 4) if ns else None, "Noise-Safe Max (V)": round(ns[1], 4) if ns else None, "Reduction Min (V)": round(ro[0], 4) if ro else None, "Reduction Max (V)": round(ro[1], 4) if ro else None})

                if "LSV" in technique:
                    cat_params, fit_data = extract_lsv_catalytic_parameters(df_mean, electrode_area, e_rev)
                    if cat_params:
                        cat_params = {"Curve": "Average Curve", **cat_params}
                        lsv_cat_list.append(cat_params)
                        max_log_I_ind = max(max_log_I_ind, fit_data["log_I_max"])
                        fig_tafel.add_trace(go.Scatter(x=fit_data["log_I_full"], y=fit_data["E_full"], mode='lines', name="Average (Log Curve)", line=dict(color=mean_color, width=2.5)))
                        if not np.isnan(fit_data["slope"]) and len(fit_data["log_I_fit"]) > 0:
                            min_x, max_x = np.min(fit_data["log_I_fit"]), np.max(fit_data["log_I_fit"])
                            span = max_x - min_x
                            fit_x = np.array([min_x - (span*1.5), max_x + (span*1.5)])
                            fig_tafel.add_trace(go.Scatter(x=fit_x, y=fit_data["slope"]*fit_x + fit_data["intercept"], mode='lines', name=f"Fit: {cat_params['Tafel Slope (mV/dec)']:.1f} mV/dec", line=dict(color=mean_color, width=2, dash='dot')))

                        if show_sd_shadow and fit_data["j_std"] is not None:
                            j_dens, j_std_s = fit_data["j_dens"], fit_data["j_std"]
                            fig_jeta.add_trace(go.Scatter(x=fit_data["eta_mV"], y=j_dens+j_std_s, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
                            fig_jeta.add_trace(go.Scatter(x=fit_data["eta_mV"], y=j_dens-j_std_s, mode='lines', line=dict(width=0), fill='tonexty', fillcolor=to_rgba(mean_color, 0.2), showlegend=False, hoverinfo='skip'))
                        fig_jeta.add_trace(go.Scatter(x=fit_data["eta_mV"], y=fit_data["j_dens"], mode='lines', name="Average", line=dict(color=mean_color, width=2.5)))
        else:
            for i, (cid, dd) in enumerate(processed_curves):
                line_color = combined_palette[i % len(combined_palette)]
                if is_eis:
                    ind_min_z = min(ind_min_z, np.min(dd["x"]), np.min(dd["y"]))
                    ind_max_z = max(ind_max_z, np.max(dd["x"]), np.max(dd["y"]))

                    fig.add_trace(go.Scatter(x=dd["x"], y=dd["y"], mode='markers', name=cid, marker=dict(color=line_color, size=6)))
                elif is_cp:
                    fig.add_trace(go.Scatter(x=dd["x"], y=dd["y"], mode='lines', name=cid, line=dict(color=line_color, width=2)))
                elif is_medusa:
                    fig.add_trace(go.Scatter(x=dd["x"], y=dd["y"], mode='lines', name=cid, line=dict(color=line_color, width=2.5)))
                else:
                    j_y = dd["y"] * 1000 / electrode_area
                    fig.add_trace(go.Scatter(x=dd["x"], y=j_y, mode='lines', name=cid, line=dict(color=line_color, width=2)))

                    out = recommend_operating_ranges_for_curve(dd)
                    ns, ro = out["recommended_noise_safe_V"], out["recommended_reduction_only_V"]
                    results_list.append({"Curve": cid, "Points": out["N_points"], "Noise-Safe Min (V)": round(ns[0], 4) if ns else None, "Noise-Safe Max (V)": round(ns[1], 4) if ns else None, "Reduction Min (V)": round(ro[0], 4) if ro else None, "Reduction Max (V)": round(ro[1], 4) if ro else None})

                    if "LSV" in technique:
                        cat_params, fit_data = extract_lsv_catalytic_parameters(dd, electrode_area, e_rev)
                        if cat_params:
                            cat_params = {"Curve": cid, **cat_params}
                            lsv_cat_list.append(cat_params)
                            max_log_I_ind = max(max_log_I_ind, fit_data["log_I_max"])
                            fig_tafel.add_trace(go.Scatter(x=fit_data["log_I_full"], y=fit_data["E_full"], mode='lines', name=f"{cid}", line=dict(color=line_color, width=2)))
                            if not np.isnan(fit_data["slope"]) and len(fit_data["log_I_fit"]) > 0:
                                min_x, max_x = np.min(fit_data["log_I_fit"]), np.max(fit_data["log_I_fit"])
                                span = max_x - min_x
                                fit_x = np.array([min_x - (span*1.5), max_x + (span*1.5)])
                                fig_tafel.add_trace(go.Scatter(x=fit_x, y=fit_data["slope"]*fit_x + fit_data["intercept"], mode='lines', name=f"Fit: {cat_params['Tafel Slope (mV/dec)']:.1f} mV/dec", line=dict(color=line_color, width=2, dash='dot')))
                            fig_jeta.add_trace(go.Scatter(x=fit_data["eta_mV"], y=fit_data["j_dens"], mode='lines', name=cid, line=dict(color=line_color, width=2)))

        if is_eis:
            x_title = "Z' (Ω)"
            y_title = "-Z'' (Ω)"
            axis_min = ind_min_z * 1.05 if ind_min_z < 0 else 0.0
            axis_max = ind_max_z * 1.05
        elif is_cp:
            x_title = "Time t (s)"
            y_title = cp_axis_label
        elif is_medusa:
            x_title = "pH"
            y_title = y_axis_label_medusa_ind
        else:
            x_title = x_axis_label
            y_title = j_axis_label

        title_txt = "Nyquist Plot" if is_eis else "Chronopotentiometry" if is_cp else "Chemical Speciation" if is_medusa else "Raw Data" if not scientific_style else ""

        # Geometría estricta también en Individual Analysis
        if is_eis:
            fig.update_layout(title=title_txt, xaxis_title=x_title, yaxis_title=y_title, width=620, height=600)
            fig.update_xaxes(range=[axis_min, axis_max], constrain="domain")
            fig.update_yaxes(range=[axis_min, axis_max], scaleanchor="x", scaleratio=1, constrain="domain")
        else:
            fig.update_layout(title=title_txt, xaxis_title=x_title, yaxis_title=y_title, height=500)

        if is_medusa and crop_medusa:
            fig.update_xaxes(range=[medusa_xmin, medusa_xmax])
            fig.update_yaxes(range=[medusa_ymin, medusa_ymax])

        fig = apply_scientific_style(fig, scientific_style, lx, ly, lxa, lya)

        if is_eis:
            show_chart(fig, stretch=False, config=dl_config_square)
        else:
            show_chart(fig, config=dl_config)

        if "LSV" in technique and lsv_cat_list and not is_eis and not is_cp and not is_medusa:
            c1, c2 = st.columns(2)
            with c1:
                fig_jeta.update_layout(title="Catalytic Performance" if not scientific_style else "", xaxis_title="Overpotential η (mV)", yaxis_title=jabs_axis_label, height=500)
                fig_jeta = apply_scientific_style(fig_jeta, scientific_style, lx, ly, lxa, lya)
                show_chart(fig_jeta, config=dl_config)
            with c2:
                fig_tafel.update_layout(title="Tafel Plot" if not scientific_style else "", xaxis_title="log₁₀|I| (A)", yaxis_title=x_axis_label, xaxis=dict(range=[max_log_I_ind - 4.5, max_log_I_ind + 0.2]), height=500)
                fig_tafel = apply_scientific_style(fig_tafel, scientific_style, lx, ly, lxa, lya)
                show_chart(fig_tafel, config=dl_config)

        if results_list and not is_eis and not is_cp and not is_medusa: st.write("**Recommended Operating Ranges:**"); show_table(pd.DataFrame(results_list))
        if lsv_cat_list and not is_eis and not is_cp and not is_medusa: st.write("**🧪 Catalytic Parameters:**"); show_table(pd.DataFrame(lsv_cat_list))
        st.markdown("<br><br>", unsafe_allow_html=True)
