import streamlit as st
import numpy as np
import plotly.graph_objects as go
import yaml
import os
import h5py  # NEW: for H5 datasets (cells & noise)

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

# --- Logo (optional) ---
LOGO_PATH = "assets/lab logo.jpg"
if os.path.exists(LOGO_PATH):
    st.sidebar.image(LOGO_PATH, use_container_width=True)

DYES_YAML = "data/dyes.yaml"
PROBE_MAP_YAML = "data/probe_fluor_map.yaml"
READOUT_POOL_YAML = "data/readout_fluorophores.yaml"  # optional pool file

# ===== Channel wavelengths (23 bands, ~8.9 nm step) =====
CHANNEL_WL = np.array([
    494, 503, 512, 521, 530, 539, 548, 557, 566, 575, 583, 592,
    601, 610, 619, 628, 637, 646, 655, 664, 673, 681, 690
], dtype=float)

# ===== H5 datasets for synthetic generation =====
# use your actual filenames:
CELL_H5_PATH  = "assets/cell_abundance.h5"
NOISE_H5_PATH = "assets/noise_quantiles.h5"

# optional import guard
try:
    import h5py
    HAS_H5PY = True
except Exception:
    HAS_H5PY = False

def _describe_file_issue(path):
    import os
    if not os.path.exists(path):
        return f"file not found: {path}"
    size = os.path.getsize(path)
    if size < 32:
        return f"file too small ({size} bytes): {path}"
    # LFS pointer check
    with open(path, "rb") as f:
        head = f.read(200)
    head_txt = head.decode("utf-8", errors="ignore")
    if "git-lfs.github.com/spec/v1" in head_txt:
        return ("Looks like a Git LFS pointer file (not the real binary). "
                "Use git lfs pull or download the real .h5 and place it under assets/.")
    # HDF5 signature
    if head[:8] != b"\x89HDF\r\n\x1a\n":
        return f"invalid HDF5 signature (first 8 bytes={head[:8]!r})"
    return None

@st.cache_resource(show_spinner=False)
def load_cell_db(path=CELL_H5_PATH):
    if not HAS_H5PY:
        raise RuntimeError("h5py is not installed.")
    issue = _describe_file_issue(path)
    if issue:
        raise RuntimeError(f"cell DB invalid: {issue}")
    f = h5py.File(path, "r")
    cells  = f["/cells"]            # H5 dataset view
    labels = f["/labels"][:].astype(str)
    meta   = {k: f["/meta/"+k][()] for k in f["/meta"].keys()}
    return f, cells, labels, meta

@st.cache_resource(show_spinner=False)
def load_noise_db(path=NOISE_H5_PATH):
    if not HAS_H5PY:
        raise RuntimeError("h5py is not installed.")
    issue = _describe_file_issue(path)
    if issue:
        raise RuntimeError(f"noise DB invalid: {issue}")
    f = h5py.File(path, "r")
    edges = f["/edges"][:]
    qs    = f["/qs"][:]
    Q     = f["/Q"][:]   # (B, C, nq)
    return f, edges, qs, Q

# ---------- Data loading ----------
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
    preset = st.sidebar.radio(
        "Lasers", options=("405/488/561/639", "Custom"),
        help="Use preset or define your wavelengths."
    )
    if preset == "405/488/561/639":
        laser_list = [405, 488, 561, 639]
        st.sidebar.caption("Using lasers: 405, 488, 561, 639 nm")
    else:
        n = st.sidebar.number_input("Number of lasers", 1, 8, 4, 1)
        cols = st.sidebar.columns(2)
        default_seeds = [405, 488, 561, 639]
        lasers = []
        for i in range(n):
            lam = cols[i % 2].number_input(
                f"Laser {i+1} (nm)", int(wl.min()), int(wl.max()),
                default_seeds[i] if i < len(default_seeds) else int(wl.min()), 1
            )
            lasers.append(int(lam))
        laser_list = lasers

k_show = st.sidebar.slider(
    "Show top-K largest pairwise similarities",
    min_value=5, max_value=50, value=10, step=1,
)

# Choose selection source
source_mode = st.sidebar.radio(
    "Selection source",
    options=("By probes", "From readout pool"),
    help="Pick per-probe, or directly select N fluorophores from the readout pool."
)

# ---------- Tiny 2-row HTML table (no header/index) ----------
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

# ---------- Simulation helpers (NEW) ----------
def resample_spectrum_to_channels(wl_grid, y, channel_wl=CHANNEL_WL):
    """Linear interpolate y(wl_grid) to channel centers."""
    y = np.asarray(y, dtype=float)
    wl = np.asarray(wl_grid, dtype=float)
    return np.interp(channel_wl, wl, y, left=y[0], right=y[-1])

def build_E_matrix_from_selected(wl, labels_all, sel_idx, E_source, mode, B=None):
    """
    Build CxR mixing matrix by sampling per-dye curves to 23 channels.
    E_source: emission (Emission spectra) or effective spectra (Predicted) columns.
    """
    R = len(sel_idx)
    C = len(CHANNEL_WL)
    E = np.zeros((C, R), dtype=float)
    for k, j in enumerate(sel_idx):
        y = E_source[:, j]
        if mode == "Emission spectra":
            y = y / (np.max(y) + 1e-12)
        else:
            y = (y / (float(B) + 1e-12)) if (B is not None) else (y / (np.max(y)+1e-12))
        E[:, k] = resample_spectrum_to_channels(wl, y, CHANNEL_WL)
        m = np.max(E[:, k])
        if m > 0:
            E[:, k] /= m
    return E  # C x R

def sample_noise_per_channel(Tc, edges, qs, Qc, rng):
    """
    Tc: (H,W) target mean image of one channel
    edges: (B+1,) bin edges (should be ascending)
    qs: (nq,) quantile levels in [0,1]
    Qc: (B, nq) quantile table for this channel
    rng: numpy Generator
    """
    H, W = Tc.shape
    mu = np.asarray(Tc, dtype=float).ravel()

    # ---- sanitize edges ----
    e = np.asarray(edges, dtype=float).ravel()
    e = e[np.isfinite(e)]
    if e.size < 2:
        e = np.array([0.0, np.inf], dtype=float)
    e = np.unique(e)
    if e.size < 2:
        e = np.array([0.0, np.inf], dtype=float)

    # ---- sanitize qs & align with Qc ----
    q = np.asarray(qs, dtype=float).ravel()
    q = q[np.isfinite(q)]
    q = np.clip(q, 0.0, 1.0)
    if q.size < 2:
        q = np.linspace(0, 1, 51, dtype=float)

    Qc = np.asarray(Qc, dtype=float)
    Bm1 = e.size - 1  # number of bins

    # fix Qc dims
    if Qc.ndim != 2:
        Qc = np.zeros((Bm1, q.size), dtype=float)
    else:
        B_qc, nq_qc = Qc.shape

        # (A) 修正分位数轴长度：逐行 1D 插值到新的 q
        if nq_qc != q.size:
            old_q = np.linspace(0, 1, max(2, nq_qc))
            new_Qc = np.zeros((B_qc, q.size), dtype=float)
            for b in range(B_qc):
                row = Qc[b, :]
                # np.interp 的 left/right 需要标量，这里用端点值
                new_Qc[b, :] = np.interp(q, old_q, row, left=row[0], right=row[-1])
            Qc = new_Qc  # (B_qc, len(q))

        # (B) 修正 bin 数：裁剪或用最后一行 pad
        if B_qc != Bm1:
            if Bm1 < B_qc:
                Qc = Qc[:Bm1, :]
            else:
                pad_rows = Bm1 - B_qc
                last_row = Qc[-1:, :] if B_qc > 0 else np.zeros((1, Qc.shape[1]), dtype=float)
                Qc = np.vstack([Qc, np.repeat(last_row, pad_rows, axis=0)])

    # ---- 分箱 ----
    if e.size <= 2:
        bins = np.zeros_like(mu, dtype=int)
    else:
        bins = np.digitize(mu, e[1:-1], right=False)
        bins = np.clip(bins, 0, Bm1 - 1)

    # ---- 逐 bin 逆 CDF 采样 ----
    # 确保 q 递增，并对 Qc 行做同样的重排（以防 q 原始不是单调的）
    sort_q_idx = np.argsort(q)
    q_sorted = q[sort_q_idx]
    Qc_sorted = Qc[:, sort_q_idx]

    u = rng.random(mu.shape)
    noise = np.zeros_like(mu)

    for b in np.unique(bins):
        sel = (bins == b)
        if not np.any(sel):
            continue
        row = Qc_sorted[b, :]             # (nq,)
        noise[sel] = np.interp(u[sel], q_sorted, row, left=row[0], right=row[-1])

    return noise.reshape(H, W)



def generate_synthetic_images(E, cells_ds, n_images=3, rng=None, noise_db=None):
    """
    E: CxR, cells_ds: (N,Ht,Wt)
    Return: lists of length n_images: T_list (HxWxC), Atrue_list (HxWxR)
    """
    if rng is None:
        rng = np.random.default_rng(123)
    N, Ht, Wt = cells_ds.shape
    C, R = E.shape

    T_list, A_list = [], []
    for _ in range(n_images):
        idx = rng.integers(0, N, size=R)
        A = np.stack([cells_ds[i, :, :] for i in idx], axis=-1)  # HxWxR
        A = np.maximum(A, 0.0)

        T = np.zeros((Ht, Wt, C), dtype=float)
        for c in range(C):
            T[:, :, c] = (A * E[c, :]).sum(axis=-1)

        if noise_db is not None:
            edges, qs, Q = noise_db
            for c in range(C):
                res = sample_noise_per_channel(T[:, :, c], edges, qs, Q[:, c, :], rng)
                T[:, :, c] = T[:, :, c] + res

                # clamp -> contrast stretch to [0,1] per image
        T = np.maximum(T, 0.0)
        T = T - T.min()
        T = T / (T.max() + 1e-12)

        T_list.append(T.astype(float))
        A_list.append(A.astype(float))

    return T_list, A_list

def nls_unmix(T, E, iters=400, tol=1e-8, verbose=False):
    """Pixelwise NLS like your MATLAB code (init by normal eq., mult. updates, re-scale, global max-norm)."""
    H, W, C = T.shape
    C2, R = E.shape
    assert C2 == C
    M = T.reshape(-1, C).astype(float)
    N = M.shape[0]

    scale = np.sqrt(np.mean(M**2, axis=1, keepdims=True)) + 1e-12
    M_n = M / scale

    EtE = E.T @ E + 1e-12*np.eye(R)
    A = (np.linalg.solve(EtE, E.T @ M_n.T)).T
    A[A < 0] = 0

    num = M_n @ E
    G = E.T @ E
    for _ in range(iters):
        den = (A @ G) + 1e-12
        A_new = A * (num / den)
        if np.linalg.norm(A_new - A) / (np.linalg.norm(A) + 1e-12) < tol:
            A = A_new; break
        A = A_new

    A = A * scale
    m = np.max(A) + 1e-12
    A = A / m
    return A.reshape(H, W, R)

def colorize_dominant(A, colors):
    """Colorize by dominant fluor (argmax over R), brightness = max abundance per pixel."""
    H, W, R = A.shape
    idx = np.argmax(A, axis=-1)
    val = np.max(A, axis=-1)
    rgb = np.zeros((H, W, 3), dtype=float)
    for r in range(R):
        mask = (idx == r)
        if not np.any(mask): continue
        rgb[mask, 0] = colors[r][0] * val[mask]
        rgb[mask, 1] = colors[r][1] * val[mask]
        rgb[mask, 2] = colors[r][2] * val[mask]
    return np.clip(rgb, 0, 1)

def compute_rmse(A_hat, A_true):
    return float(np.sqrt(np.mean((A_hat - A_true)**2)))

DEFAULT_COLORS = [
    (0.121, 0.466, 0.705),
    (1.000, 0.498, 0.054),
    (0.172, 0.627, 0.172),
    (0.839, 0.152, 0.156),
    (0.580, 0.404, 0.741),
    (0.549, 0.337, 0.294),
    (0.890, 0.467, 0.761),
    (0.498, 0.498, 0.498),
    (0.737, 0.741, 0.133),
    (0.090, 0.745, 0.811),
]

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

# ---------- Core run ----------
def run_selection_and_display(groups, mode, laser_strategy, laser_list):
    required_count = (N_pick if use_pool else None)

    if mode == "Emission spectra":
        # Build emission-only matrix
        E_norm, labels_pair, idx_groups = build_emission_only_matrix(wl, dye_db, groups)
        if E_norm.shape[1] == 0:
            st.error("No spectra available for optimization.")
            st.stop()

        # Strict lexicographic optimization (K = min(10, #pairs))
        if required_count is None:
            G = len(idx_groups)
            K = min(10, (G * (G - 1)) // 2)
        else:
            N = E_norm.shape[1]
            K = min(10, N * (N - 1) // 2)

        sel_idx, _ = solve_lexicographic_k(
            E_norm, idx_groups, labels_pair,
            levels=K, enforce_unique=True,
            required_count=required_count
        )

        # Selected Fluorophores
        if use_pool:
            chosen_fluors = [labels_pair[j].split(" – ", 1)[1] for j in sel_idx]
            chosen_fluors = sorted(chosen_fluors)
            st.subheader("Selected Fluorophores")
            html_two_row_table("Slot", "Fluorophore",
                               [f"Slot {i+1}" for i in range(len(chosen_fluors))],
                               chosen_fluors)
        else:
            sel_pairs = [labels_pair[j] for j in sel_idx]
            probes = [s.split(" – ", 1)[0] for s in sel_pairs]
            fluors = [s.split(" – ", 1)[1] for s in sel_pairs]
            st.subheader("Selected Fluorophores")
            html_two_row_table("Probe", "Fluorophore", probes, fluors)

        # Top pairwise similarities
        S = cosine_similarity_matrix(E_norm[:, sel_idx])
        sub_labels = [labels_pair[j] for j in sel_idx]
        tops = top_k_pairwise(S, sub_labels, k=k_show)
        pairs = [only_fluor_pair(a, b) for _, a, b in tops]
        sims = [val for val, _, _ in tops]
        st.subheader("Top pairwise similarities")
        html_two_row_table("Pair", "Similarity", pairs, sims,
                           color_second_row=True, color_thresh=0.9, format_second_row=True)

        # Spectra viewer (normalize to 0–1 per trace)
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

        # ===== Simulate & Unmix (8.9 nm channels) =====
        st.subheader("Simulate & unmix (8.9 nm channels)")
        try:
            f_cells, cells_ds, cells_labels, cell_meta = load_cell_db()
            f_noise, edges, qs, Q = load_noise_db()
            have_data = True
        except Exception as e:
            have_data = False
            st.info(f"Cell/noise DB not available: {e}")

        if have_data:
            # Build E (C×R) from emission curves used above
            E = build_E_matrix_from_selected(wl, labels_pair, sel_idx, E_source=E_norm, mode="Emission spectra", B=None)

            # Generate three images and unmix
            T_list, Atrue_list = generate_synthetic_images(
                E=E, cells_ds=cells_ds, n_images=3,
                rng=np.random.default_rng(123), noise_db=(edges, qs, Q)
            )

            R = len(sel_idx)
            COLORS = DEFAULT_COLORS[:R]
            cols = st.columns(R)

            for row in range(3):
                T = T_list[row]
                A_true = Atrue_list[row]
                A_hat = nls_unmix(T, E, iters=400, tol=1e-8, verbose=False)
                rmse = compute_rmse(A_hat, A_true)
                rgb = colorize_dominant(A_hat, COLORS)
                rgb = rgb / (rgb.max() + 1e-12)
                for c in range(R):
                    with cols[c]:
                        st.image((rgb*255).astype(np.uint8), caption=f"RMSE={rmse:.4f}", use_container_width=True)

    else:
        if not laser_list:
            st.error("Please specify laser wavelengths.")
            st.stop()

        # --- Round A: emission-only for a provisional selection (initial power guess) ---
        E0_norm, labels0, idx0 = build_emission_only_matrix(wl, dye_db, groups)
        if required_count is None:
            G0 = len(idx0)
            K0 = min(10, (G0 * (G0 - 1)) // 2)
        else:
            N0 = E0_norm.shape[1]
            K0 = min(10, N0 * (N0 - 1) // 2)

        sel0, _ = solve_lexicographic_k(
            E0_norm, idx0, labels0,
            levels=K0, enforce_unique=True,
            required_count=required_count
        )
        A_labels = [labels0[j] for j in sel0]

        # (1) Calibrate powers on A
        if laser_strategy == "Simultaneous":
            powers_A, B_A = derive_powers_simultaneous(wl, dye_db, A_labels, laser_list)
        else:
            powers_A, B_A = derive_powers_separate(wl, dye_db, A_labels, laser_list)

        # Build effective spectra with powers_A, then select
        E_raw_all, E_norm_all, labels_all, idx_groups_all = build_effective_with_lasers(
            wl, dye_db, groups, laser_list, laser_strategy, powers_A
        )
        if required_count is None:
            Gf = len(idx_groups_all)
            Kf = min(10, (Gf * (Gf - 1)) // 2)
        else:
            Nf = E_norm_all.shape[1]
            Kf = min(10, Nf * (Nf - 1) // 2)

        sel_idx, _ = solve_lexicographic_k(
            E_norm_all, idx_groups_all, labels_all,
            levels=Kf, enforce_unique=True,
            required_count=required_count
        )

        # (2) Recalibrate powers on the FINAL selection
        final_labels = [labels_all[j] for j in sel_idx]
        if laser_strategy == "Simultaneous":
            powers, B = derive_powers_simultaneous(wl, dye_db, final_labels, laser_list)
        else:
            powers, B = derive_powers_separate(wl, dye_db, final_labels, laser_list)

        # Rebuild effective spectra with FINAL powers for display/metrics
        E_raw_all, E_norm_all, labels_all, idx_groups_all = build_effective_with_lasers(
            wl, dye_db, groups, laser_list, laser_strategy, powers
        )

        # ----- Display -----
        # Selected Fluorophores
        if use_pool:
            chosen_fluors = [labels_all[j].split(" – ", 1)[1] for j in sel_idx]
            chosen_fluors = sorted(chosen_fluors)
            st.subheader("Selected Fluorophores (with lasers)")
            html_two_row_table("Slot", "Fluorophore",
                               [f"Slot {i+1}" for i in range(len(chosen_fluors))],
                               chosen_fluors)
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

        # Top pairwise similarities
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
                fig.add_shape(
                    type="line",
                    x0=sep_x, x1=sep_x,
                    y0=0, y1=1, yref="paper", xref="x",
                    line=dict(color="white", width=2, dash="dash"),
                    layer="above"
                )
            for i in range(L):
                if seg_widths[i] <= 0: continue
                fig.add_annotation(
                    x=mids[i], xref="x",
                    y=1.12 if (i % 2 == 0) else 1.06, yref="paper",
                    text=f"{int(lam_sorted[i])} nm",
                    showarrow=False,
                    font=dict(size=12),
                    align="center",
                    yanchor="bottom",
                    xshift=(-12 if (i % 2 == 0) else 12)
                )

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
                           ticktext=["0", "0.2", "0.4", "0.6", "1"]),
                margin=dict(t=90)
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

        # ===== Simulate & Unmix (8.9 nm channels) =====
        st.subheader("Simulate & unmix (8.9 nm channels)")
        try:
            f_cells, cells_ds, cells_labels, cell_meta = load_cell_db()
            f_noise, edges, qs, Q = load_noise_db()
            have_data = True
        except Exception as e:
            have_data = False
            st.info(f"Cell/noise DB not available: {e}")

        if have_data:
            # Build E (C×R) from final effective spectra displayed above
            E = build_E_matrix_from_selected(
                wl, labels_all, sel_idx, E_source=E_raw_all, mode="Predicted spectra", B=B
            )

            T_list, Atrue_list = generate_synthetic_images(
                E=E, cells_ds=cells_ds, n_images=3,
                rng=np.random.default_rng(123), noise_db=(edges, qs, Q)
            )

            R = len(sel_idx)
            COLORS = DEFAULT_COLORS[:R]
            cols = st.columns(R)

            for row in range(3):
                T = T_list[row]
                A_true = Atrue_list[row]
                A_hat = nls_unmix(T, E, iters=400, tol=1e-8, verbose=False)
                rmse = compute_rmse(A_hat, A_true)
                rgb = colorize_dominant(A_hat, COLORS)
                rgb = rgb / (rgb.max() + 1e-12)
                for c in range(R):
                    with cols[c]:
                        st.image((rgb*255).astype(np.uint8), caption=f"RMSE={rmse:.4f}", use_container_width=True)

# Execute
run_selection_and_display(groups, mode, laser_strategy, laser_list)
