# app.py — streamlined; E. coli–like rods (shorter & thicker); English-only UI
import json
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from utils import (
    load_dyes_yaml,
    load_probe_fluor_map,
    build_emission_only_matrix,
    build_effective_with_lasers,
    derive_powers_simultaneous,
    derive_powers_separate,
    solve_lexicographic_k,
    cosine_similarity_matrix,
    top_k_pairwise,
)

st.set_page_config(page_title="Fluorophore Selector", layout="wide")

# -------------------- Load data --------------------
DYES_YAML = "data/dyes.yaml"
PROBE_MAP_YAML = "data/probe_fluor_map.yaml"
READOUT_POOL_YAML = "data/readout_fluorophores.yaml"

wl, dye_db = load_dyes_yaml(DYES_YAML)
probe_map = load_probe_fluor_map(PROBE_MAP_YAML)

def _load_readout_pool(path):
    try:
        import yaml, os
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        items = data.get("fluorophores", []) or []
        pool = sorted({s.strip() for s in items if isinstance(s, str) and s.strip()})
        return [f for f in pool if f in dye_db]
    except Exception:
        return []

readout_pool = _load_readout_pool(READOUT_POOL_YAML)

# -------------------- Sidebar --------------------
st.sidebar.header("Configuration")
mode = st.sidebar.radio(
    "Mode",
    options=("Emission spectra", "Predicted spectra"),
    help=(
        "Emission spectra: emission-only, normalized peaks.\n"
        "Predicted spectra: excitation·QY·EC with lasers."
    ),
)
source_mode = st.sidebar.radio("Selection source", ("By probes", "From readout pool"))
k_show = st.sidebar.slider("Show top-K similarities", 5, 50, 10, 1)

laser_list = []
laser_strategy = None
if mode == "Predicted spectra":
    laser_strategy = st.sidebar.radio("Laser usage", ("Simultaneous", "Separate"))
    n = st.sidebar.number_input("Number of lasers", 1, 8, 4, 1)
    cols = st.sidebar.columns(2)
    defaults = [405, 488, 561, 639]
    for i in range(n):
        lam = cols[i % 2].number_input(
            f"Laser {i+1} (nm)", int(wl.min()), int(max(700, wl.max())),
            defaults[i] if i < len(defaults) else int(wl.min()), 1
        )
        laser_list.append(int(lam))

# -------------------- Helpers: colors, tables --------------------
DEFAULT_COLORS = np.array([
    [0.95, 0.25, 0.25], [0.25, 0.65, 0.95],
    [0.25, 0.85, 0.35], [0.90, 0.70, 0.20],
    [0.80, 0.40, 0.80], [0.25, 0.80, 0.80],
    [0.85, 0.50, 0.35], [0.60, 0.60, 0.60],
], dtype=float)

def _ensure_colors(R):
    if R <= len(DEFAULT_COLORS): 
        return DEFAULT_COLORS[:R]
    hs = np.linspace(0, 1, R, endpoint=False)
    extra = np.stack([
        np.abs(np.sin(2*np.pi*hs))*0.7+0.3,
        np.abs(np.sin(2*np.pi*(hs+0.33)))*0.7+0.3,
        np.abs(np.sin(2*np.pi*(hs+0.66)))*0.7+0.3
    ], axis=1)
    return extra[:R]

def _rgb01_to_plotly(col):
    r, g, b = (int(255*x) for x in col)
    return f"rgb({r},{g},{b})"

def _pair_only_fluor(a, b):
    fa = a.split(" – ", 1)[1] if " – " in a else a
    fb = b.split(" – ", 1)[1] if " – " in b else b
    return f"{fa} vs {fb}"

def _html_two_row_table(row0_label, row1_label, row0_vals, row1_vals,
                        color_second_row=False, color_thresh=0.9, fmt2=False):
    def esc(x):
        return (str(x).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;"))
    def fmtv(v):
        if fmt2:
            try: return f"{float(v):.3f}"
            except: return esc(v)
        return esc(v)
    cells0 = "".join(f"<td style='padding:6px 10px;border:1px solid #ddd;'>{esc(v)}</td>" for v in row0_vals)
    tds0 = f"<td style='padding:6px 10px;border:1px solid #ddd;white-space:nowrap;'>{esc(row0_label)}</td>{cells0}"
    tds1_list = []
    for v in row1_vals:
        style = "padding:6px 10px;border:1px solid #ddd;"
        if color_second_row:
            try:
                vv=float(v); style+=f"color:{'red' if vv>color_thresh else 'green'};"
            except: pass
        tds1_list.append(f"<td style='{style}'>{fmtv(v)}</td>")
    tds1 = f"<td style='padding:6px 10px;border:1px solid #ddd;white-space:nowrap;'>{esc(row1_label)}</td>{''.join(tds1_list)}"
    st.markdown(f"""
    <div style="overflow-x:auto;">
      <table style="border-collapse:collapse;width:100%;table-layout:auto;">
        <tbody><tr>{tds0}</tr><tr>{tds1}</tr></tbody>
      </table>
    </div>
    """, unsafe_allow_html=True)

# -------------------- Cache wrappers --------------------
@st.cache_data(show_spinner=False)
def cached_build_effective_with_lasers(wl, dye_db, groups, laser_list, laser_strategy, powers):
    groups_key = json.dumps({k:sorted(v) for k,v in sorted(groups.items())}, ensure_ascii=False)
    _ = (tuple(sorted(laser_list)), laser_strategy, tuple(np.asarray(powers,float)) if powers is not None else None, groups_key)
    return build_effective_with_lasers(wl, dye_db, groups, laser_list, laser_strategy, powers)

@st.cache_data(show_spinner=False)
def cached_interpolate_E_on_channels(wl, spectra_cols, chan_centers_nm):
    spectra_cols = np.asarray(spectra_cols, dtype=float)
    if spectra_cols.ndim == 1: spectra_cols = spectra_cols[:, None]
    W, N = spectra_cols.shape
    E = np.zeros((len(chan_centers_nm), N), dtype=float)
    for j in range(N):
        y = spectra_cols[:, j]
        E[:, j] = np.interp(chan_centers_nm, wl, y, left=float(y[0]), right=float(y[-1]))
    return np.nan_to_num(E, nan=0.0, posinf=0.0, neginf=0.0)

# -------------------- NLS + colors --------------------
def nls_unmix(Timg, E, iters=2000, tol=1e-6):
    """Fast MU with pixelwise normalization; expects Timg(H,W,C), E(C,R)."""
    H, W, C = Timg.shape
    E = np.asarray(E, dtype=np.float32)
    if E.ndim != 2 or E.shape[0] != C:
        raise ValueError(f"E shape {E.shape} mismatch with Timg channels {C}")
    M = Timg.reshape(-1, C).astype(np.float32, copy=False)
    scale = np.sqrt(np.mean(M**2, axis=1, keepdims=True)); scale[scale<=0]=1.0
    Mn = M / scale
    EtE = E.T @ E
    # LS init
    A = Mn @ E @ np.linalg.pinv(EtE)
    A[A < 0] = 0
    # MU updates
    for _ in range(iters):
        numer = Mn @ E
       
