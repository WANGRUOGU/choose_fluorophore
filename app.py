# app.py
import os
import yaml
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
    solve_lexicographic_k,   # strict lexicographic optimizer
    cosine_similarity_matrix,
    top_k_pairwise,
)

st.set_page_config(page_title="Choose Fluorophore", layout="wide")

# --- Optional logo in sidebar (robust) ---
from pathlib import Path
from PIL import Image, UnidentifiedImageError

APP_DIR = Path(__file__).parent
LOGO_CANDIDATES = [
    APP_DIR / "assets" / "lab logo.jpg",
    APP_DIR / "assets" / "lab_logo.jpg",
    APP_DIR / "assets" / "lab_logo.png",
    APP_DIR / "assets" / "logo.png",
]

def try_show_sidebar_logo(paths):
    for p in paths:
        if p.exists() and p.is_file():
            try:
                img = Image.open(p)
                st.sidebar.image(img, use_container_width=True)
                return
            except UnidentifiedImageError:
                # file exists but not a valid image — try next candidate
                continue
    # If none worked, just skip (no error)
    # st.sidebar.caption(" ")  # (optional) keep some spacing

try_show_sidebar_logo(LOGO_CANDIDATES)

# --- Paths ---
DYES_YAML = "data/dyes.yaml"
PROBE_MAP_YAML = "data/probe_fluor_map.yaml"
READOUT_POOL_YAML = "data/readout_fluorophores.yaml"

# ---------- Load data ----------
wl, dye_db = load_dyes_yaml(DYES_YAML)
probe_map = load_probe_fluor_map(PROBE_MAP_YAML)

def load_readout_pool(path):
    """Read 'fluorophores: [...]' and keep only those present in dyes.yaml."""
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    items = data.get("fluorophores", []) or []
    pool = sorted({s.strip() for s in items if isinstance(s, str) and s.strip()})
    pool = [f for f in pool if f in dye_db]
    return pool

readout_pool = load_readout_pool(READOUT_POOL_YAML)

# ---------- Sidebar ----------
st.sidebar.header("Configuration")

mode = st.sidebar.radio(
    "Mode",
    options=("Emission spectra", "Predicted spectra"),
    help=(
        "Emission spectra: peak-normalized emission, optimize by cosine.\n\n"
        "Predicted spectra: build effective spectra with lasers "
        "using excitation · QY · EC, then optimize by cosine on those effective spectra."
    ),
)

laser_strategy = None
laser_list = []
if mode == "Predicted spectra":
    laser_strategy = st.sidebar.radio(
        "Laser usage", options=("Simultaneous", "Separate"),
        help="Simultaneous: cumulative within wavelength segments (B-leveling). "
             "Separate: per-laser scaled to the same B, spectra concatenated horizontally."
    )
    n = st.sidebar.number_input("Number of lasers", 1, 8, 4, 1)
    cols = st.sidebar.columns(2)
    default_seeds = [405, 488, 561, 639]
    lasers = []
    for i in range(n):
        lam = cols[i % 2].number_input(
            f"Laser {i+1} (nm)", int(wl.min()), int(max(700, wl.max())),
            default_seeds[i] if i < len(default_seeds) else int(wl.min()), 1
        )
        lasers.append(int(lam))
    laser_list = lasers

k_show = st.sidebar.slider(
    "Show top-K largest pairwise similarities",
    min_value=5, max_value=50, value=10, step=1,
)

source_mode = st.sidebar.radio(
    "Selection source",
    options=("By probes", "From readout pool"),
    help="Pick per-probe, or directly select N fluorophores from the readout pool."
)

# ---------- Small HTML 2-row table ----------
def html_two_row_table(row0_label, row1_label, row0_vals, row1_vals,
                       color_second_row=False, color_thresh=0.9,
                       format_second_row=False):
    """Render a compact 2-row table with a label column on the left."""
    def esc(x):
        return (str(x)
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;"))

    cells0 = "".join(
        f"<td style='padding:6px 10px;border:1px solid #ddd;'>{esc(v)}</td>"
        for v in row0_vals
    )
    tds0 = (
        f"<td style='padding:6px 10px;border:1px solid #ddd;white-space:nowrap;'>{esc(row0_label)}</td>"
        f"{cells0}"
    )

    def fmt(v):
        if format_second_row:
            try:
                return f"{float(v):.3f}"
            except Exception:
                return esc(v)
        return esc(v)

    tds1_list = []
    for v in row1_vals:
        style = "padding:6px 10px;border:1px solid #ddd;"
        if color_second_row:
            try:
                vv = float(v)
                style += "color:{};".format("red" if vv > color_thresh else "green")
            except Exception:
                pass
        tds1_list.append(f"<td style='{style}'>{fmt(v)}</td>")

    tds1 = (
        f"<td style='padding:6px 10px;border:1px solid #ddd;white-space:nowrap;'>{esc(row1_label)}</td>"
        f"{''.join(tds1_list)}"
    )

    html = f"""
    <div style="overflow-x:auto;">
      <table style="border-collapse:collapse;width:100%;table-layout:auto;">
        <tbody>
          <tr>{tds0}</tr>
          <tr>{tds1}</tr>
        </tbody>
      </table>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def only_fluor_pair(a: str, b: str) -> str:
    """Drop probe names in 'Probe – Fluor' labels and keep 'Fluor vs Fluor'."""
    fa = a.split(" – ", 1)[1]
    fb = b.split(" – ", 1)[1]
    return f"{fa} vs {fb}"

# ---------- Spectra -> 8.9nm channel grid ----------
def interpolate_E_on_channels(wl, spectra_cols, chan_centers_nm):
    """
    wl: (W,) nm grid
    spectra_cols: (W,N)
    chan_centers_nm: (C,)
    Return E: (C,N)
    """
    W, N = spectra_cols.shape
    E = np.zeros((len(chan_centers_nm), N), dtype=float)
    for j in range(N):
        y = spectra_cols[:, j]
        E[:, j] = np.interp(chan_centers_nm, wl, y, left=y[0], right=y[-1])
    return E

# ---------- Rod synthesis + Poisson noise + NLS ----------
DEFAULT_COLORS = np.array([
    [0.95, 0.25, 0.25],
    [0.25, 0.65, 0.95],
    [0.25, 0.85, 0.35],
    [0.90, 0.70, 0.20],
    [0.80, 0.40, 0.80],
    [0.25, 0.80, 0.80],
    [0.85, 0.50, 0.35],
    [0.60, 0.60, 0.60],
], dtype=float)

def _ensure_colors(R):
    if R <= len(DEFAULT_COLORS):
        return DEFAULT_COLORS[:R]
    hs = np.linspace(0, 1, R, endpoint=False)
    extra = np.stack([np.abs(np.sin(2*np.pi*hs))*0.7+0.3,
                      np.abs(np.sin(2*np.pi*(hs+0.33)))*0.7+0.3,
                      np.abs(np.sin(2*np.pi*(hs+0.66)))*0.7+0.3], axis=1)
    return extra[:R]

def _capsule_profile(H, W, cx, cy, length, width, theta_rad):
    """
    Capsule (rounded rectangle) with intensity decaying from centerline to boundary:
      - rectangle: 1 - |yp|/r
      - endcaps:   1 - rho/r
    """
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    X = xx - cx
    Y = yy - cy
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    xp =  c * X + s * Y
    yp = -s * X + c * Y

    half_L = 0.5 * length
    r = 0.5 * width

    rect_mask = (np.abs(xp) <= half_L) & (np.abs(yp) <= r)
    rect_val = np.clip(1.0 - np.abs(yp) / (r + 1e-12), 0.0, 1.0)

    rhoL = np.sqrt((xp + half_L) ** 2 + yp ** 2)
    rhoR = np.sqrt((xp - half_L) ** 2 + yp ** 2)
    capL_mask = rhoL <= r
    capR_mask = rhoR <= r
    capL_val = np.clip(1.0 - rhoL / (r + 1e-12), 0.0, 1.0)
    capR_val = np.clip(1.0 - rhoR / (r + 1e-12), 0.0, 1.0)

    val = np.zeros((H, W), dtype=float)
    val[rect_mask] = rect_val[rect_mask]
    val[capL_mask] = np.maximum(val[capL_mask], capL_val[capL_mask])
    val[capR_mask] = np.maximum(val[capR_mask], capR_val[capR_mask])
    return val, (rect_mask | capL_mask | capR_mask)

def _place_rods_scene(H, W, R, rods_per=3, max_tries=200, rng=None):
    """
    Place non-overlapping rods (shorter, E. coli-like), each per class.
    """
    rng = np.random.default_rng() if rng is None else rng
    Atrue = np.zeros((H, W, R), dtype=float)
    occ = np.zeros((H, W), dtype=bool)

    # shorter rods (more E. coli-like)
    L_min, L_max = 28, 56
    W_min, W_max = 9, 15

    for r in range(R):
        placed, tries = 0, 0
        while placed < rods_per and tries < max_tries:
            tries += 1
            length = int(rng.integers(L_min, L_max + 1))
            width  = int(rng.integers(W_min, W_max + 1))
            theta  = float(rng.uniform(0, np.pi))
            margin = 6 + int(max(length, width) / 2)
            if W - 2*margin <= 2 or H - 2*margin <= 2:
                break
            cx = int(rng.integers(margin, W - margin))
            cy = int(rng.integers(margin, H - margin))

            prof, mask = _capsule_profile(H, W, cx, cy, length, width, theta)
            if not np.any(mask):
                continue
            if np.any(occ & mask):
                continue

            if prof.max() > 0:
                prof = prof / prof.max()
            Atrue[:, :, r] = np.maximum(Atrue[:, :, r], prof)
            occ |= mask
            placed += 1

    return np.clip(Atrue, 0.0, 1.0)

def add_poisson_noise_per_channel(Tclean, peak=255, rng=None):
    """
    For each channel:
      - scale so max -> peak
      - sample Y ~ Poisson(X_scaled)
      - scale back by /peak
    """
    rng = np.random.default_rng() if rng is None else rng
    H, W, C = Tclean.shape
    Tnoisy = np.empty_like(Tclean, dtype=float)
    for c in range(C):
        img = Tclean[:, :, c]
        m = float(np.max(img))
        if m <= 0:
            Tnoisy[:, :, c] = 0.0
            continue
        scale = peak / m
        lam = img * scale
        counts = rng.poisson(lam).astype(float)
        Tnoisy[:, :, c] = counts / peak
    return Tnoisy

def nls_unmix(Timg, E, iters=10_000, tol=1e-8, verbose=False):
    """
    MATLAB NLS (your code) faithful port:
      - if T is 3D, reshape to (Npix, C)
      - pixel-wise normalization by scale = sqrt(mean(M.^2, 2))
      - A = pinv(E)*M, A<0 -> 0
      - multiplicative updates: A = A .* (M*E) ./ (A*(E.'*E))
      - stop if relative fit change < tol
      - rescale A by 'scale' and then max-normalize to [0,1]
    Inputs:
      Timg: (H,W,C) or (Npix,C)
      E:    (C,R)
    Returns:
      Aout: (H,W,R) or (Npix,R) matching input shape
    """
    # reshape
    if Timg.ndim == 3:
        H, W, C = Timg.shape
        M = Timg.reshape(-1, C).astype(np.float64, copy=False)
        reshape_back = (H, W)
    else:
        M = np.array(Timg, dtype=np.float64, copy=False)
        reshape_back = None
        H = W = None
    Npix, C = M.shape
    R = E.shape[1]

    # pixel-wise scale
    scale = np.sqrt(np.mean(M**2, axis=1, keepdims=True))
    scale[scale <= 0] = 1.0
    Mn = M / scale

    # init A using pseudoinverse
    EtE = E.T @ E
    pinv = np.linalg.pinv(EtE) @ E.T
    A = (Mn @ pinv.T)  # (Npix,R)
    A[A < 0] = 0.0

    # iterative multiplicative updates
    fit = np.linalg.norm(Mn - A @ E.T, 'fro')
    for i in range(iters):
        fit_old = fit
        # A = A .* (M*E) ./ (A*(E.'*E))
        numer = Mn @ E
        denom = (A @ (E.T @ E)) + 1e-12
        A *= numer / denom
        # evaluate fit
        fit = np.linalg.norm(Mn - A @ E.T, 'fro')
        rel = abs(fit_old - fit) / (fit_old + 1e-12)
        if verbose and (i % 200 == 0 or i < 5):
            print(f"Iter {i:4d} fit={fit:.6e} Δrel={rel:.3e}")
        if (rel < tol) or np.isnan(fit):
            break

    # rescale back & normalize
    A *= scale  # undo pixel-wise normalization
    maxA = np.max(A)
    if maxA > 0:
        A = A / maxA

    if reshape_back is not None:
        H, W = reshape_back
        return A.reshape(H, W, R).astype(np.float32)
    else:
        return A.astype(np.float32)


def _to_uint8_gray(img2d):
    p = float(np.percentile(img2d, 99.5))
    if p <= 1e-8:
        return np.zeros_like(img2d, dtype=np.uint8)
    z = np.clip(img2d / p, 0.0, 1.0)
    z = np.power(z, 0.8)
    return (z * 255.0).astype(np.uint8)

def colorize_composite(A, colors):
    """线性渲染：每张图按自身最大值缩放到 [0,1]，无分位数 / 无 gamma。"""
    H, W, R = A.shape
    rgb = np.zeros((H, W, 3), dtype=float)
    for r in range(R):
        ar = np.clip(A[:, :, r], 0.0, 1.0)
        rgb += ar[:, :, None] * colors[r][None, None, :]
    m = float(rgb.max())
    if m > 0:
        rgb = rgb / m
    return rgb

def colorize_single_channel(A_r, color):
    """线性渲染：通道按自身最大值缩放到 [0,1]，无分位数 / 无 gamma。"""
    z = np.clip(A_r, 0.0, 1.0)
    m = float(z.max())
    if m > 0:
        z = z / m
    rgb = z[:, :, None] * np.asarray(color)[None, None, :]
    return rgb


def render_color_legend(names, colors):
    cells = []
    for name, col in zip(names, colors):
        r, g, b = (int(255*x) for x in col)
        dot = f"<span style='display:inline-block;width:12px;height:12px;background:rgb({r},{g},{b});border-radius:3px;margin-right:8px;'></span>"
        cells.append(f"<div style='margin:4px 12px 4px 0;white-space:nowrap;'>{dot}{name}</div>")
    html = "<div style='display:flex;flex-wrap:wrap;align-items:center;'>" + "".join(cells) + "</div>"
    st.markdown(html, unsafe_allow_html=True)

def simulate_rods_and_unmix(E, H=256, W=256, rods_per=3, rng=None):
    """
    Build one mixed image with R*rods_per rods (non-overlapping),
    add Poisson noise, run NLS, and return maps & overall RMSE.
    E: (C,R) spectra on 8.9nm grid
    Returns:
      Atrue (H,W,R), Ahat (H,W,R), rmse_all (float)
    """
    rng = np.random.default_rng() if rng is None else rng
    C, R = E.shape[0], E.shape[1]

    # 1) rods scene
    Atrue = _place_rods_scene(H, W, R, rods_per=rods_per, rng=rng)

    # 2) forward render (C channels)
    Tclean = np.zeros((H, W, C), dtype=float)
    for c in range(C):
        Tclean[:, :, c] = np.tensordot(Atrue, E[c, :], axes=([2], [0]))

    # 3) Poisson noise per channel
    Tnoisy = add_poisson_noise_per_channel(Tclean, peak=255, rng=rng)

    # 4) NLS unmix
    M = Tnoisy.reshape(-1, C)
    Ahat = nls_unmix(Tnoisy, E, iters=10000, tol=1e-8)  # 直接传 (H,W,C)

    # 5) overall RMSE
    rmse_all = float(np.sqrt(np.mean((Ahat - Atrue) ** 2)))

    return Atrue, Ahat, rmse_all

# ---------- Main ----------
st.title("Fluorophore Selection for Multiplexed Imaging")

use_pool = (source_mode == "From readout pool")
groups = {}

if use_pool:
    st.subheader("Pick from readout pool")
    if len(readout_pool) == 0:
        st.info("Readout pool not found or empty. Please add data/readout_fluorophores.yaml.")
        st.stop()
    max_n = len(readout_pool)
    N_pick = st.number_input("How many fluorophores to pick", min_value=1, max_value=max_n, value=min(4, max_n), step=1)
    groups = {"Pool": readout_pool[:]}
else:
    all_probes = sorted(probe_map.keys())
    st.subheader("Pick probes to optimize")
    picked = st.multiselect("Probes", options=all_probes)
    if not picked:
        st.info("Select at least one probe to proceed.")
        st.stop()
    for p in picked:
        cands = [f for f in probe_map.get(p, []) if f in dye_db]
        if cands:
            groups[p] = cands
    if not groups:
        st.error("No valid candidates with spectra in dyes.yaml for the selected probes.")
        st.stop()

# ---------- Core runner ----------
def run_selection_and_display(groups, mode, laser_strategy, laser_list):
    required_count = (N_pick if use_pool else None)

    if mode == "Emission spectra":
        # Build emission-only matrix
        E_norm, labels_pair, idx_groups = build_emission_only_matrix(wl, dye_db, groups)
        if E_norm.shape[1] == 0:
            st.error("No spectra available for optimization.")
            st.stop()

        # Strict lexicographic (K=min(10,#pairs))
        if required_count is None:
            G = len(idx_groups); K = min(10, (G * (G - 1)) // 2)
        else:
            N = E_norm.shape[1]; K = min(10, N * (N - 1) // 2)

        sel_idx, _ = solve_lexicographic_k(
            E_norm, idx_groups, labels_pair,
            levels=K, enforce_unique=True,
            required_count=required_count
        )

        # Selected
        if use_pool:
            chosen = sorted([labels_pair[j].split(" – ", 1)[1] for j in sel_idx])
            st.subheader("Selected Fluorophores")
            html_two_row_table("Slot", "Fluorophore",
                               [f"Slot {i+1}" for i in range(len(chosen))],
                               chosen)
        else:
            sel_pairs = [labels_pair[j] for j in sel_idx]
            probes = [s.split(" – ", 1)[0] for s in sel_pairs]
            fluors = [s.split(" – ", 1)[1] for s in sel_pairs]
            st.subheader("Selected Fluorophores")
            html_two_row_table("Probe", "Fluorophore", probes, fluors)

        # Pairwise similarities
        S = cosine_similarity_matrix(E_norm[:, sel_idx])
        sub_labels = [labels_pair[j] for j in sel_idx]
        tops = top_k_pairwise(S, sub_labels, k=k_show)
        pairs = [only_fluor_pair(a, b) for _, a, b in tops]
        sims = [val for val, _, _ in tops]
        st.subheader("Top pairwise similarities")
        html_two_row_table("Pair", "Similarity", pairs, sims,
                           color_second_row=True, color_thresh=0.9, format_second_row=True)

        # Spectra viewer (normalize traces to 0–1)
        st.subheader("Spectra viewer")
        fig = go.Figure()
        for j in sel_idx:
            y = E_norm[:, j]
            y = y / (np.max(y) + 1e-12)
            fig.add_trace(go.Scatter(x=wl, y=y, mode="lines", name=labels_pair[j]))
        fig.update_layout(
            title_text="",
            xaxis_title="Wavelength (nm)",
            yaxis_title="Normalized intensity",
            yaxis=dict(range=[0, 1.05],
                       tickmode="array",
                       tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                       ticktext=["0", "0.2", "0.4", "0.6", "0.8", "1"])
        )
        st.plotly_chart(fig, use_container_width=True)

                # --- Rod simulation & unmix with Poisson noise ---
        C = 23
        start_nm = 494.0
        step_nm = 8.9
        chan_centers = start_nm + step_nm * np.arange(C)
        E_sel = E_norm[:, sel_idx]
        E = interpolate_E_on_channels(wl, E_sel, chan_centers)  # (C, R)

        st.subheader("Rod-shaped cells: ground-truth vs. NLS unmixing (Poisson noise)")
        Atrue, Ahat, rmse_all = simulate_rods_and_unmix(
            E=E, H=256, W=256, rods_per=3, rng=np.random.default_rng(2025)
        )
        R = E.shape[1]
        colors = _ensure_colors(R)

        # 取当前选择顺序对应的 fluor 名字（不要排序，以免与颜色/索引错位）
        fluor_names = [labels_pair[j].split(" – ", 1)[1] for j in sel_idx]

        # 1（真值合成）+ R（每个 fluor 的 NLS 彩色图），标题用真实名字
        imgs = []
        true_rgb = colorize_composite(Atrue, colors)
        imgs.append(("True (composite)", (true_rgb * 255).astype(np.uint8)))
        for r, name in enumerate(fluor_names):
            rgb_r = colorize_single_channel(Ahat[:, :, r], colors[r])
            imgs.append((f"NLS ({name})", (rgb_r * 255).astype(np.uint8)))

        cols = st.columns(len(imgs))
        for i, (title, im) in enumerate(imgs):
            with cols[i]:
                st.image(im, use_container_width=True)
                st.caption(title)

        # 图例（名字 ↔ 颜色）
        render_color_legend(fluor_names, colors)

        st.caption(f"Overall RMSE (Ahat vs Atrue): {rmse_all:.4f}")


    else:
        if not laser_list:
            st.error("Please specify laser wavelengths.")
            st.stop()

        # Round A: emission-only provisional selection
        E0_norm, labels0, idx0 = build_emission_only_matrix(wl, dye_db, groups)
        if required_count is None:
            G0 = len(idx0); K0 = min(10, (G0 * (G0 - 1)) // 2)
        else:
            N0 = E0_norm.shape[1]; K0 = min(10, N0 * (N0 - 1) // 2)
        sel0, _ = solve_lexicographic_k(
            E0_norm, idx0, labels0,
            levels=K0, enforce_unique=True,
            required_count=required_count
        )
        A_labels = [labels0[j] for j in sel0]

        # (1) Calibrate powers on A
        if laser_strategy == "Simultaneous":
            powers_A, _ = derive_powers_simultaneous(wl, dye_db, A_labels, laser_list)
        else:
            powers_A, _ = derive_powers_separate(wl, dye_db, A_labels, laser_list)

        # Build effective spectra with powers_A, then select
        E_raw_all, E_norm_all, labels_all, idx_groups_all = build_effective_with_lasers(
            wl, dye_db, groups, laser_list, laser_strategy, powers_A
        )
        if required_count is None:
            Gf = len(idx_groups_all); Kf = min(10, (Gf * (Gf - 1)) // 2)
        else:
            Nf = E_norm_all.shape[1]; Kf = min(10, Nf * (Nf - 1) // 2)
        sel_idx, _ = solve_lexicographic_k(
            E_norm_all, idx_groups_all, labels_all,
            levels=Kf, enforce_unique=True,
            required_count=required_count
        )

        # (2) Recalibrate on FINAL selection for display
        final_labels = [labels_all[j] for j in sel_idx]
        if laser_strategy == "Simultaneous":
            powers, B = derive_powers_simultaneous(wl, dye_db, final_labels, laser_list)
        else:
            powers, B = derive_powers_separate(wl, dye_db, final_labels, laser_list)

        # Rebuild effective spectra with FINAL powers for display
        E_raw_all, E_norm_all, labels_all, idx_groups_all = build_effective_with_lasers(
            wl, dye_db, groups, laser_list, laser_strategy, powers
        )

        # Selected
        if use_pool:
            chosen = sorted([labels_all[j].split(" – ", 1)[1] for j in sel_idx])
            st.subheader("Selected Fluorophores (with lasers)")
            html_two_row_table("Slot", "Fluorophore",
                               [f"Slot {i+1}" for i in range(len(chosen))],
                               chosen)
        else:
            sel_pairs = [labels_all[j] for j in sel_idx]
            probes = [s.split(" – ", 1)[0] for s in sel_pairs]
            fluors = [s.split(" – ", 1)[1] for s in sel_pairs]
            st.subheader("Selected Fluorophores (with lasers)")
            html_two_row_table("Probe", "Fluorophore", probes, fluors)

        # Laser powers (relative)
        lam_sorted = list(sorted(laser_list))
        p = np.array(powers, dtype=float)
        maxp = float(np.max(p)) if p.size > 0 else 1.0
        prel = (p / (maxp + 1e-12)).tolist()
        st.subheader("Laser powers (relative)")
        html_two_row_table("Laser (nm)", "Relative power",
                           lam_sorted, [float(f"{v:.6g}") for v in prel])

        # Pairwise similarities
        S = cosine_similarity_matrix(E_norm_all[:, sel_idx])
        sub_labels = [labels_all[j] for j in sel_idx]
        tops = top_k_pairwise(S, sub_labels, k=k_show)
        pairs = [only_fluor_pair(a, b) for _, a, b in tops]
        sims = [val for val, _, _ in tops]
        st.subheader("Top pairwise similarities")
        html_two_row_table("Pair", "Similarity", pairs, sims,
                           color_second_row=True, color_thresh=0.9, format_second_row=True)

        # Spectra viewer
        st.subheader("Spectra viewer")
        fig = go.Figure()
        if laser_strategy == "Separate":
            lam_sorted = list(sorted(laser_list))
            L = len(lam_sorted)
            Wn = len(wl)
            gap = 12.0
            wl_max_vis = float(min(1000.0, wl[-1]))
            seg_widths = [max(0.0, wl_max_vis - float(l)) for l in lam_sorted]
            offsets, acc = [], 0.0
            for wseg in seg_widths:
                offsets.append(acc); acc += wseg + gap
            for j in sel_idx:
                xs_cat, ys_cat = [], []
                for i, l in enumerate(lam_sorted):
                    if seg_widths[i] <= 0: continue
                    off = offsets[i]
                    mask = (wl >= l) & (wl <= wl_max_vis)
                    wl_seg = wl[mask]
                    block = E_raw_all[i * Wn:(i + 1) * Wn, j] / (B + 1e-12)
                    y_seg = block[mask]
                    xs_cat.append(wl_seg + off); ys_cat.append(y_seg)
                if xs_cat:
                    fig.add_trace(go.Scatter(
                        x=np.concatenate(xs_cat),
                        y=np.concatenate(ys_cat),
                        mode="lines",
                        name=labels_all[j]
                    ))
            rights = [offsets[i] + wl_max_vis for i in range(L)]
            mids   = [offsets[i] + (float(lam_sorted[i]) + wl_max_vis) / 2.0 for i in range(L)]
            for i in range(L - 1):
                if seg_widths[i] <= 0: continue
                sep_x = rights[i]
                fig.add_shape(type="line", x0=sep_x, x1=sep_x,
                              y0=0, y1=1, yref="paper", xref="x",
                              line=dict(color="white", width=2, dash="dash"), layer="above")
            for i in range(L):
                if seg_widths[i] <= 0: continue
                fig.add_annotation(x=mids[i], xref="x",
                                   y=1.06, yref="paper",
                                   text=f"{int(lam_sorted[i])} nm",
                                   showarrow=False, font=dict(size=12),
                                   align="center", yanchor="bottom")
            tick_positions = [mids[i] for i in range(L) if seg_widths[i] > 0]
            tick_texts     = [f"{int(lam_sorted[i])}–{int(wl_max_vis)} nm" for i in range(L) if seg_widths[i] > 0]
            fig.update_layout(
                title_text="",
                xaxis_title="Wavelength (nm)",
                yaxis_title="Normalized intensity",
                xaxis=dict(tickmode="array", tickvals=tick_positions, ticktext=tick_texts, ticks="outside", automargin=True),
                yaxis=dict(range=[0, 1.05],
                           tickmode="array",
                           tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                           ticktext=["0", "0.2", "0.4", "0.6", "0.8", "1"]),
                margin=dict(t=80)
            )
        else:
            for j in sel_idx:
                y = E_raw_all[:, j] / (B + 1e-12)
                fig.add_trace(go.Scatter(x=wl, y=y, mode="lines", name=labels_all[j]))
            fig.update_layout(
                title_text="",
                xaxis_title="Wavelength (nm)",
                yaxis_title="Normalized intensity",
                yaxis=dict(range=[0, 1.05],
                           tickmode="array",
                           tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                           ticktext=["0", "0.2", "0.4", "0.6", "0.8", "1"])
            )
        st.plotly_chart(fig, use_container_width=True)

                # --- Rod simulation & unmix (Poisson) ---
        C = 23
        start_nm = 494.0
        step_nm = 8.9
        chan_centers = start_nm + step_nm * np.arange(C)
        E_sel = E_norm_all[:, sel_idx]
        E = interpolate_E_on_channels(wl, E_sel, chan_centers)  # (C, R)

        st.subheader("Rod-shaped cells: ground-truth vs. NLS unmixing (Poisson noise)")
        Atrue, Ahat, rmse_all = simulate_rods_and_unmix(
            E=E, H=256, W=256, rods_per=3, rng=np.random.default_rng(2025)
        )
        R = E.shape[1]
        colors = _ensure_colors(R)

        # 取当前选择顺序对应的 fluor 名字（不要排序）
        fluor_names = [labels_all[j].split(" – ", 1)[1] for j in sel_idx]

        # 1（真值合成）+ R（每个 fluor 的 NLS 彩色图），标题用真实名字
        imgs = []
        true_rgb = colorize_composite(Atrue, colors)
        imgs.append(("True (composite)", (true_rgb * 255).astype(np.uint8)))
        for r, name in enumerate(fluor_names):
            rgb_r = colorize_single_channel(Ahat[:, :, r], colors[r])
            imgs.append((f"NLS ({name})", (rgb_r * 255).astype(np.uint8)))

        cols = st.columns(len(imgs))
        for i, (title, im) in enumerate(imgs):
            with cols[i]:
                st.image(im, use_container_width=True)
                st.caption(title)

        # 图例（名字 ↔ 颜色）
        render_color_legend(fluor_names, colors)

        st.caption(f"Overall RMSE (Ahat vs Atrue): {rmse_all:.4f}")


# Execute
run_selection_and_display(groups, mode, laser_strategy, laser_list)
