# app.py
import os
import yaml
import h5py
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

# --- Optional logo ---
LOGO_PATH = "assets/lab logo.jpg"
if os.path.exists(LOGO_PATH):
    st.sidebar.image(LOGO_PATH, use_container_width=True)

# --- Paths ---
DYES_YAML = "data/dyes.yaml"
PROBE_MAP_YAML = "data/probe_fluor_map.yaml"
READOUT_POOL_YAML = "data/readout_fluorophores.yaml"
NOISE_H5 = "assets/noise_quantiles.h5"

# ---------- Load core data ----------
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

# ---------- Noise DB (optional) ----------
def try_load_noise_db(path):
    """Load edges, quantiles, Q from H5; return (edges, qs, Q) or None."""
    if not os.path.exists(path):
        return None
    try:
        with h5py.File(path, "r") as f:
            edges = np.array(f["/edges"][:], dtype=float)          # (B+1,)
            qs    = np.array(f["/quantiles"][:], dtype=float)      # (K,)
            Q     = np.array(f["/Q"][:], dtype=float)              # (B,C,K)
        # sanity
        if edges.ndim != 1 or qs.ndim != 1 or Q.ndim != 3:
            return None
        if edges.size < 2 or qs.size < 2 or Q.shape[2] != qs.size:
            return None
        return (edges, qs, Q)
    except Exception:
        return None

noise_db = try_load_noise_db(NOISE_H5)

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

# ---------- Tiny 2-row HTML table ----------
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

# ---------- Build E on 8.9nm grid ----------
def interpolate_E_on_channels(wl, spectra_cols, chan_centers_nm):
    """
    wl: (W,) nm grid of dyes.yaml
    spectra_cols: (W,N) columns of spectra (emission-normalized or effective)
    chan_centers_nm: (C,) channel center wavelengths (nm)
    Return E: (C,N)
    """
    W, N = spectra_cols.shape
    E = np.zeros((len(chan_centers_nm), N), dtype=float)
    for j in range(N):
        y = spectra_cols[:, j]
        E[:, j] = np.interp(chan_centers_nm, wl, y, left=y[0], right=y[-1])
    return E

# ---------- Rod synthesis & NLS ----------
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

def _capsule_mask(H, W, cx, cy, length, width, theta_rad):
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    X = xx - cx
    Y = yy - cy
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    xp =  c * X + s * Y
    yp = -s * X + c * Y
    half_L = 0.5 * length
    r = 0.5 * width
    rect = (np.abs(xp) <= half_L) & (np.abs(yp) <= r)
    dl = (xp + half_L) ** 2 + yp ** 2 <= r ** 2
    dr = (xp - half_L) ** 2 + yp ** 2 <= r ** 2
    return (rect | dl | dr)

def _place_rods_scene(H, W, R, rods_per=3, max_tries=200, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    Atrue = np.zeros((H, W, R), dtype=float)
    rod_masks = [[] for _ in range(R)]
    occ = np.zeros((H, W), dtype=bool)
    L_min, L_max = 40, 80
    W_min, W_max = 10, 18
    for r in range(R):
        placed, tries = 0, 0
        while placed < rods_per and tries < max_tries:
            tries += 1
            length = rng.integers(L_min, L_max + 1)
            width  = rng.integers(W_min, W_max + 1)
            theta  = rng.uniform(0, np.pi)
            margin = 5 + int(max(length, width) / 2)
            if W - 2*margin <= 2 or H - 2*margin <= 2:
                break
            cx = rng.integers(margin, W - margin)
            cy = rng.integers(margin, H - margin)
            m = _capsule_mask(H, W, cx, cy, length, width, theta)
            if not np.any(m):      continue
            if np.any(occ & m):    continue
            yy, xx = np.mgrid[0:H, 0:W]
            dist = ((xx - cx) ** 2 + (yy - cy) ** 2) ** 0.5
            prof = np.clip(1.0 - (dist / (0.6 * length)), 0.2, 1.0)
            a = np.zeros((H, W), dtype=float); a[m] = prof[m]
            if a.max() > 0: a = a / a.max()
            Atrue[:, :, r] += a
            rod_masks[r].append(m.copy())
            occ |= m
            placed += 1
        while len(rod_masks[r]) < rods_per:
            rod_masks[r].append(np.zeros((H, W), dtype=bool))
    return Atrue, rod_masks

def sample_noise_per_channel(mu, edges, qs, Qc, rng):
    H, W = mu.shape
    B = len(edges) - 1
    old_q = np.linspace(0, 1, Qc.shape[1])
    if not np.allclose(qs, old_q):
        Qc = np.interp(qs[None, :], old_q[None, :], Qc, left=Qc[:, :1], right=Qc[:, -1])
    bins = np.clip(np.digitize(mu.ravel(), edges[1:-1], right=False), 0, B - 1)
    u = rng.random(mu.size)
    k = np.clip((u * (len(qs) - 1)).astype(int), 0, len(qs) - 1)
    noise = Qc[bins, k]
    return noise.reshape(H, W)

def colorize_dominant(A, colors):
    H, W, R = A.shape
    idx = np.argmax(A, axis=2)
    val = A[np.arange(H)[:, None], np.arange(W)[None, :], idx]
    rgb = np.zeros((H, W, 3), dtype=float)
    for r in range(R):
        mask = (idx == r)
        for c in range(3):
            rgb[..., c][mask] = colors[r, c] * val[mask]
    return rgb

def nls_unmix(T, E, iters=2000, tol=1e-7):
    M = T  # (Npix, C)
    Npix, C = M.shape
    R = E.shape[1]
    EtE = E.T @ E
    pinv = np.linalg.pinv(EtE) @ E.T
    A = (pinv @ M.T).T
    A[A < 0] = 0
    fit = np.linalg.norm(M - A @ E.T, 'fro')
    for _ in range(iters):
        fit_old = fit
        denom = (A @ (E.T @ E)) + 1e-12
        numer = M @ E
        A *= numer / denom
        fit = np.linalg.norm(M - A @ E.T, 'fro')
        if abs(fit - fit_old) / (fit_old + 1e-12) < tol:
            break
    return A

def simulate_rods_and_unmix(E, edges, qs, Q, H=256, W=256, rods_per=3, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    C, R = E.shape[0], E.shape[1]
    colors = _ensure_colors(R)
    # Scene
    Atrue, _ = _place_rods_scene(H, W, R, rods_per=rods_per, rng=rng)
    # Forward (C channels)
    Tclean = np.zeros((H, W, C), dtype=float)
    for c in range(C):
        Tclean[:, :, c] = np.tensordot(Atrue, E[c, :], axes=([2], [0]))
    # Noise
    Tnoisy = np.empty_like(Tclean)
    for c in range(C):
        mu = Tclean[:, :, c]
        res = sample_noise_per_channel(mu, edges, qs, Q[:, c, :], rng)
        Tnoisy[:, :, c] = np.clip(mu + res, 0.0, None)
    # NLS
    M = Tnoisy.reshape(-1, C)
    Ahat = nls_unmix(M, E).reshape(H, W, R)
    # Display per fluor
    imgs_rgb, rmses = [], []
    for r in range(R):
        err = Ahat[:, :, r] - Atrue[:, :, r]
        rmses.append(float(np.sqrt(np.mean(err**2))))
        rgb = colorize_dominant(Ahat, colors)
        p = float(np.percentile(rgb, 99.0))
        if p > 1e-8:
            rgb = np.clip(rgb / p, 0.0, 1.0)
        rgb = np.power(np.clip(rgb + 0.03, 0.0, 1.0), 0.7)
        imgs_rgb.append(rgb)
    return imgs_rgb, rmses, Atrue, Ahat

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
        E_norm, labels_pair, idx_groups = build_emission_only_matrix(wl, dye_db, groups)
        if E_norm.shape[1] == 0:
            st.error("No spectra available for optimization.")
            st.stop()

        # Strict lexicographic: K = min(10, #pairs)
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

        # Similarities
        S = cosine_similarity_matrix(E_norm[:, sel_idx])
        sub_labels = [labels_pair[j] for j in sel_idx]
        tops = top_k_pairwise(S, sub_labels, k=k_show)
        pairs = [only_fluor_pair(a, b) for _, a, b in tops]
        sims = [val for val, _, _ in tops]
        st.subheader("Top pairwise similarities")
        html_two_row_table("Pair", "Similarity", pairs, sims,
                           color_second_row=True, color_thresh=0.9, format_second_row=True)

        # Spectra viewer (normalized 0–1)
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

        # --- Rod simulation & unmix (if noise DB available) ---
        if noise_db is not None:
            edges, qs, Q = noise_db
            C = Q.shape[1]
            start_nm = 494.0
            step_nm = 8.9
            chan_centers = start_nm + step_nm * np.arange(C)
            # use E_norm on channels as spectra (columns are selected)
            E_sel = E_norm[:, sel_idx]
            E = interpolate_E_on_channels(wl, E_sel, chan_centers)  # (C, R)
            st.subheader("Simulate & unmix with rod-shaped cells")
            imgs_rgb, rmses, _, _ = simulate_rods_and_unmix(
                E=E, edges=edges, qs=qs, Q=Q,
                H=256, W=256, rods_per=3, rng=np.random.default_rng(2025)
            )
            cols = st.columns(E.shape[1])
            for r in range(E.shape[1]):
                with cols[r]:
                    st.image((imgs_rgb[r]*255).astype(np.uint8), use_container_width=True)
                    st.caption(f"Fluor #{r+1} • RMSE = {rmses[r]:.4f}")
        else:
            st.info("Noise DB not found. Skipping rod simulation.")

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

        # Similarities
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

        # --- Rod simulation & unmix (if noise DB available) ---
        if noise_db is not None:
            edges, qs, Q = noise_db
            C = Q.shape[1]
            start_nm = 494.0
            step_nm = 8.9
            chan_centers = start_nm + step_nm * np.arange(C)
            # build spectra columns for selected (use normalized effective E_norm_all)
            E_sel = E_norm_all[:, sel_idx]
            E = interpolate_E_on_channels(wl, E_sel, chan_centers)  # (C, R)
            st.subheader("Simulate & unmix with rod-shaped cells")
            imgs_rgb, rmses, _, _ = simulate_rods_and_unmix(
                E=E, edges=edges, qs=qs, Q=Q,
                H=256, W=256, rods_per=3, rng=np.random.default_rng(2025)
            )
            cols = st.columns(E.shape[1])
            for r in range(E.shape[1]):
                with cols[r]:
                    st.image((imgs_rgb[r]*255).astype(np.uint8), use_container_width=True)
                    st.caption(f"Fluor #{r+1} • RMSE = {rmses[r]:.4f}")
        else:
            st.info("Noise DB not found. Skipping rod simulation.")


# Execute
run_selection_and_display(groups, mode, laser_strategy, laser_list)
