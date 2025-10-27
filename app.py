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
    solve_lexicographic_k,   # strict lexicographic optimizer (levels=K)
    cosine_similarity_matrix,
    top_k_pairwise,
)

# ==============================
# Basic page & logo
# ==============================
st.set_page_config(page_title="Choose Fluorophore", layout="wide")

LOGO_PATH = "assets/lab logo.jpg"
if os.path.exists(LOGO_PATH):
    st.sidebar.image(LOGO_PATH, use_container_width=True)

# ==============================
# Data locations
# ==============================
DYES_YAML = "data/dyes.yaml"
PROBE_MAP_YAML = "data/probe_fluor_map.yaml"
READOUT_POOL_YAML = "data/readout_fluorophores.yaml"  # optional

# H5 datasets (cell shapes & noise quantiles)
CELL_H5_PATH  = "assets/cell_abundance.h5"
NOISE_H5_PATH = "assets/noise_quantiles.h5"

# ==============================
# Try h5py (optional)
# ==============================
try:
    import h5py
    HAS_H5PY = True
except Exception:
    HAS_H5PY = False

# ==============================
# Robust file checks & dummy builders
# ==============================
def _describe_file_issue(path):
    if not os.path.exists(path):
        return f"file not found: {path}"
    size = os.path.getsize(path)
    if size < 32:
        return f"file too small ({size} bytes): {path}"
    with open(path, "rb") as f:
        head = f.read(200)
    head_txt = head.decode("utf-8", errors="ignore")
    if "git-lfs.github.com/spec/v1" in head_txt:
        return ("Looks like a Git LFS pointer file (not the real binary). "
                "Use git lfs pull or place the real .h5 under assets/.")
    if head[:8] != b"\x89HDF\r\n\x1a\n":
        return f"invalid HDF5 signature (first 8 bytes={head[:8]!r})"
    return None

def _normal_ppf(u):
    """
    Fast approximation to standard normal inverse CDF (Acklam's method).
    u in [0,1]; returns z with ~1e-6 relative accuracy for typical use.
    """
    u = np.asarray(u, dtype=float)
    # Coefficients
    a = [ -3.969683028665376e+01,  2.209460984245205e+02, -2.759285104469687e+02,
           1.383577518672690e+02, -3.066479806614716e+01,  2.506628277459239e+00 ]
    b = [ -5.447609879822406e+01,  1.615858368580409e+02, -1.556989798598866e+02,
           6.680131188771972e+01, -1.328068155288572e+01 ]
    c = [ -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
          -2.549732539343734e+00,  4.374664141464968e+00,  2.938163982698783e+00 ]
    d = [  7.784695709041462e-03,  3.224671290700398e-01,  2.445134137142996e+00,
           3.754408661907416e+00 ]
    plow, phigh = 0.02425, 1 - 0.02425

    z = np.empty_like(u)

    # Lower region
    mask = (u < plow)
    if np.any(mask):
        q = np.sqrt(-2*np.log(u[mask]))
        z[mask] = ((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]
        z[mask] /= (((d[0]*q + d[1])*q + d[2])*q + d[3])
        z[mask] *= -1

    # Central region
    mask = (u >= plow) & (u <= phigh)
    if np.any(mask):
        q = u[mask] - 0.5
        r = q*q
        z[mask] = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q
        z[mask] /= (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0)

    # Upper region
    mask = (u > phigh)
    if np.any(mask):
        q = np.sqrt(-2*np.log(1.0 - u[mask]))
        z[mask] = ((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]
        z[mask] /= (((d[0]*q + d[1])*q + d[2])*q + d[3])

    return z

def create_dummy_cell_db(path, N=2000, Ht=64, Wt=64):
    """
    Build a small H5 with synthetic 'cells' (2D blobs), labels and meta.
    """
    if not HAS_H5PY:
        raise RuntimeError("h5py not installed; cannot create dummy H5.")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rng = np.random.default_rng(123)
    cells = np.zeros((N, Ht, Wt), dtype=np.float32)
    labels = np.array([f"dummy{i%8}" for i in range(N)], dtype=h5py.string_dtype())

    yy, xx = np.mgrid[0:Ht, 0:Wt].astype(np.float32)
    for i in range(N):
        img = np.zeros((Ht, Wt), dtype=np.float32)
        k = rng.integers(1, 4)  # number of blobs per cell
        for _ in range(k):
            cy = rng.uniform(10, Ht-10)
            cx = rng.uniform(10, Wt-10)
            sy = rng.uniform(3, 9)
            sx = rng.uniform(3, 9)
            amp = rng.uniform(0.4, 1.0)
            img += amp * np.exp(-(((yy-cy)/sy)**2 + ((xx-cx)/sx)**2)/2.0)
        img /= (img.max() + 1e-6)
        cells[i] = img

    with h5py.File(path, "w") as f:
        f.create_dataset("/cells", data=cells, compression="gzip", compression_opts=4)
        f.create_dataset("/labels", data=labels)
        f.create_dataset("/meta/target_size", data=np.array([Ht, Wt], dtype=np.float64))
        f.create_dataset("/meta/source_count", data=np.array([N], dtype=np.float64))
        f.create_dataset("/meta/source_hw", data=np.array([Ht, Wt], dtype=np.float64))
        f.create_dataset("/meta/n_cells", data=np.array([N], dtype=np.float64))

def create_dummy_noise_db(path, C=23):
    """
    Build a small H5 with noise quantiles:
    edges:(B+1,), qs:(nq,), Q:(B,C,nq).
    """
    if not HAS_H5PY:
        raise RuntimeError("h5py not installed; cannot create dummy H5.")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    B = 12
    nq = 51
    edges = np.linspace(0, 1, B+1, dtype=np.float32)
    qs = np.linspace(0, 1, nq, dtype=np.float32)
    Q = np.zeros((B, C, nq), dtype=np.float32)
    # Zero-mean Gaussian; sigma increases with intensity bin
    for b in range(B):
        sigma = 0.02 + 0.10 * (b/(B-1) if B > 1 else 0.0)
        Q[b, :, :] = (sigma * _normal_ppf(qs))[None, :]
    with h5py.File(path, "w") as f:
        f.create_dataset("/edges", data=edges)
        f.create_dataset("/qs", data=qs)
        f.create_dataset("/Q", data=Q, compression="gzip", compression_opts=4)

@st.cache_resource(show_spinner=False)
def load_cell_db(path=CELL_H5_PATH):
    if not HAS_H5PY:
        raise RuntimeError("h5py is not installed.")
    issue = _describe_file_issue(path)
    if issue:
        # Try build a dummy so the app can run end-to-end
        create_dummy_cell_db(path, N=2000, Ht=64, Wt=64)
    f = h5py.File(path, "r")
    cells  = f["/cells"]                 # (N,H,W) H5 dataset view
    labels = f["/labels"][:].astype(str)
    meta   = {k: f["/meta/"+k][()] for k in f["/meta"].keys()}
    return f, cells, labels, meta

@st.cache_resource(show_spinner=False)
def load_noise_db(path=NOISE_H5_PATH):
    if not HAS_H5PY:
        raise RuntimeError("h5py is not installed.")
    issue = _describe_file_issue(path)
    if issue:
        create_dummy_noise_db(path, C=23)
    f = h5py.File(path, "r")
    edges = f["/edges"][:]
    qs    = f["/qs"][:]
    Q     = f["/Q"][:]   # (B, C, nq)
    return f, edges, qs, Q

# ==============================
# UI helpers
# ==============================
def html_two_row_table(row0_label, row1_label, row0_vals, row1_vals,
                       color_second_row=False, color_thresh=0.9,
                       format_second_row=False):
    """Compact 2-row table (no header/index)."""
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
    """Drop probe names from 'Probe – Fluor' -> 'Fluor vs Fluor'."""
    fa = a.split(" – ", 1)[1]
    fb = b.split(" – ", 1)[1]
    return f"{fa} vs {fb}"

# ==============================
# Load YAMLs
# ==============================
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

# ==============================
# Sidebar controls
# ==============================
st.sidebar.header("Configuration")

mode = st.sidebar.radio(
    "Mode",
    options=("Emission spectra", "Predicted spectra"),
    help=("Emission spectra: peak-normalized emission, optimize by cosine.\n\n"
          "Predicted spectra: build effective spectra with lasers using excitation·QY·EC, "
          "then optimize by cosine on those effective spectra.")
)

laser_strategy = None
laser_list = []
if mode == "Predicted spectra":
    laser_strategy = st.sidebar.radio(
        "Laser usage", options=("Simultaneous", "Separate"),
        help=("Simultaneous: cumulative within wavelength segments (B-leveling). "
              "Separate: per-laser scaled to the same B, spectra concatenated horizontally (for optimization).")
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

source_mode = st.sidebar.radio(
    "Selection source",
    options=("By probes", "From readout pool"),
    help="Pick per-probe, or directly select N fluorophores from the readout pool."
)

# ==============================
# Main title
# ==============================
st.title("Fluorophore Selection for Multiplexed Imaging")

# ==============================
# Build selection groups
# ==============================
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

# ==============================
# 8.9 nm channel grid (23 channels)
# ==============================
CHANNEL_WL = np.array([
    494, 503, 512, 521, 530, 539, 548, 557, 566, 575, 583, 592,
    601, 610, 619, 628, 637, 646, 655, 664, 673, 681, 690
], dtype=float)

# ==============================
# Simulation & unmix helpers
# ==============================
DEFAULT_COLORS = np.array([
    [0.90, 0.20, 0.20],
    [0.20, 0.70, 0.20],
    [0.20, 0.40, 0.90],
    [0.85, 0.60, 0.15],
    [0.55, 0.25, 0.70],
    [0.10, 0.70, 0.70],
    [0.80, 0.35, 0.15],
    [0.35, 0.75, 0.45],
    [0.40, 0.40, 0.40],
    [0.80, 0.10, 0.50],
    [0.15, 0.55, 0.85],
    [0.65, 0.70, 0.20],
], dtype=float)

def _nearest_idx_from_grid(wavelengths, lam):
    idx = int(np.clip(np.round(lam - wavelengths[0]), 0, len(wavelengths)-1))
    return idx

def build_E_matrix_from_selected(wl, labels_all, sel_idx, E_source, mode, B=None, laser_list_for_separate=None):
    """
    Build E (C x R) where C=23 channel centers, R=#selected.
    - mode == "Emission spectra": E_source is W x N peak-normalized emissions.
    - mode == "Predicted spectra": E_source is:
        * Simultaneous -> W x N effective raw spectra (scale by 1/B before sampling).
        * Separate     -> (W*L) x N concatenated; we sum L blocks to a single W-length spectrum, then sample.
    """
    C = len(CHANNEL_WL)
    R = len(sel_idx)
    E = np.zeros((C, R), dtype=float)

    # Detect source width(s)
    if mode == "Predicted spectra":
        if E_source.shape[0] != len(wl):  # maybe concatenated (W*L)
            # Try infer L
            L = int(round(E_source.shape[0] / len(wl)))
        else:
            L = 1
    else:
        L = 1

    for r, j in enumerate(sel_idx):
        if mode == "Emission spectra":
            spec = E_source[:, j]  # already peak-normalized, then L2 for cosine
        else:
            if L == 1:
                spec = E_source[:, j] / (B + 1e-12)
            else:
                # concatenate blocks -> sum blocks into one spectrum for channel sampling
                Wn = len(wl)
                acc = np.zeros(Wn, dtype=float)
                for b in range(L):
                    acc += E_source[b*Wn:(b+1)*Wn, j]
                spec = acc / (B + 1e-12)

        # sample at 23 channel centers by nearest index on wl grid
        vals = []
        for cw in CHANNEL_WL:
            idx = _nearest_idx_from_grid(wl, cw)
            vals.append(spec[idx])
        E[:, r] = np.asarray(vals, dtype=float)

    # normalize each column to unit 2-norm so NLS behaves well
    norms = np.linalg.norm(E, axis=0, keepdims=True) + 1e-12
    E = E / norms
    return E

def nls_unmix(T, E, iters=400, tol=1e-8, verbose=False):
    """
    Nonnegative least squares via multiplicative updates (like your Matlab code).
    T: (H,W,C), E: (C,R) -> returns A_hat (H*W, R) in [0,1]
    """
    if T.ndim == 3:
        H, W, C = T.shape
        M = T.reshape(-1, C).astype(float)
    else:
        M = T.astype(float)
        H = 1; W = M.shape[0]; C = M.shape[1]
    R = E.shape[1]
    idx = np.any(M > 0, axis=1)
    N = M.shape[0]
    M2 = M[idx, :]
    scale = np.sqrt(np.mean(M2**2, axis=1, keepdims=True))  # (n,1)
    scale[scale == 0] = 1.0
    M2n = M2 / scale

    # LS init
    EtE = E.T @ E
    EtM = E.T @ M2n.T      # (R,n)
    try:
        A = (np.linalg.pinv(EtE) @ EtM).T
    except Exception:
        A = (np.eye(R) @ EtM).T
    A[A < 0] = 0

    fit = np.linalg.norm(M2n - A @ E.T, 'fro')
    for i in range(iters):
        fit_old = fit
        numer = (M2n @ E)
        denom = (A @ (E.T @ E)) + 1e-12
        A *= numer / denom
        fit = np.linalg.norm(M2n - A @ E.T, 'fro')
        change = abs(fit_old - fit) / (fit_old + 1e-12)
        if verbose:
            st.write(f"Iter {i+1}: fit={fit:.4e} delta={change:.2e}")
        if (change < tol) or np.isnan(fit):
            break

    A *= scale  # undo pixel normalization
    if A.size > 0:
        A /= (A.max() + 1e-12)
    A_out = np.zeros((N, R), dtype=float)
    A_out[idx, :] = A
    return A_out

def compute_rmse(A_hat, A_true):
    diff = (A_hat - A_true).ravel()
    return float(np.sqrt(np.mean(diff*diff)))

def colorize_dominant(A_hat, colors):
    """
    A_hat: (N,R) in [0,1]; colors: (R,3) in [0,1].
    Returns (H,W,3) after reshaping by caller.
    """
    if A_hat.size == 0:
        return np.zeros((1,1,3), dtype=float)
    N, R = A_hat.shape
    idx = np.argmax(A_hat, axis=1)  # dominant
    mag = np.max(A_hat, axis=1)
    rgb = colors[idx] * mag[:, None]
    return rgb

def sample_noise_per_channel(Tc, edges, qs, Qc, rng):
    """
    Robust bin-quantile noise sampler for one channel.
    Tc: (H,W) target mean image of one channel
    edges: (B+1,)
    qs: (nq,)
    Qc: (B, nq)  [this channel's quantile table]
    """
    H, W = Tc.shape
    mu = np.asarray(Tc, dtype=float).ravel()

    # sanitize edges
    e = np.asarray(edges, dtype=float).ravel()
    e = e[np.isfinite(e)]
    if e.size < 2:
        e = np.array([0.0, np.inf], dtype=float)
    e = np.unique(e)
    if e.size < 2:
        e = np.array([0.0, np.inf], dtype=float)

    # sanitize qs
    q = np.asarray(qs, dtype=float).ravel()
    q = q[np.isfinite(q)]
    q = np.clip(q, 0.0, 1.0)
    if q.size < 2:
        q = np.linspace(0, 1, 51, dtype=float)

    Qc = np.asarray(Qc, dtype=float)
    Bm1 = e.size - 1

    # fix Qc dims
    if Qc.ndim != 2:
        Qc = np.zeros((Bm1, q.size), dtype=float)
    else:
        B_qc, nq_qc = Qc.shape
        if nq_qc != q.size:
            old_q = np.linspace(0, 1, max(2, nq_qc))
            new_Qc = np.zeros((B_qc, q.size), dtype=float)
            for b in range(B_qc):
                row = Qc[b, :]
                new_Qc[b, :] = np.interp(q, old_q, row, left=row[0], right=row[-1])
            Qc = new_Qc
        if B_qc != Bm1:
            if Bm1 < B_qc:
                Qc = Qc[:Bm1, :]
            else:
                if B_qc == 0:
                    last = np.zeros((1, Qc.shape[1]), dtype=float)
                else:
                    last = Qc[-1:, :]
                Qc = np.vstack([Qc, np.repeat(last, Bm1 - B_qc, axis=0)])

    # digitize
    if e.size <= 2:
        bins = np.zeros_like(mu, dtype=int)
    else:
        bins = np.digitize(mu, e[1:-1], right=False)
        bins = np.clip(bins, 0, Bm1 - 1)

    # inverse-CDF sampling
    sort_q_idx = np.argsort(q)
    q_sorted = q[sort_q_idx]
    Qc_sorted = Qc[:, sort_q_idx]
    u = rng.random(mu.shape)
    noise = np.zeros_like(mu)
    for b in np.unique(bins):
        sel = (bins == b)
        if not np.any(sel):
            continue
        row = Qc_sorted[b, :]
        noise[sel] = np.interp(u[sel], q_sorted, row, left=row[0], right=row[-1])
    return noise.reshape(H, W)

def generate_synthetic_images(E, cells_ds, n_images, rng, noise_db):
    """
    Synthesize a small set of images and their ground-truth abundances:
    - pick R random cell exemplars, build A_true by stacking per-channel abundances
    - form T = A_true @ E^T (then reshape to HxWxC)
    - add noise sampled via the (edges, qs, Q) quantile table
    - final per-image contrast stretch to [0,1] for visualization
    Returns: list of T (H,W,C), list of A_true (N,R)
    """
    edges, qs, Q = noise_db
    Ncells, H, W = cells_ds.shape
    C, R = E.shape
    T_list, A_list = [], []
    for k in range(n_images):
        # pick R cells (with replacement) and random amplitudes
        take = rng.integers(0, Ncells, size=R)
        amps = rng.uniform(0.5, 1.0, size=R)
        A_true = np.zeros((H*W, R), dtype=float)
        for r, idx in enumerate(take):
            cell = cells_ds[idx, :, :][()]  # (H,W)
            vec = (cell.ravel() * amps[r]).astype(float)
            A_true[:, r] = vec

        # forward model
        M = A_true @ E.T                      # (N, C)
        T = M.reshape(H, W, C)

        # add noise per channel
        for c in range(C):
            res = sample_noise_per_channel(T[:, :, c], edges, qs, Q[:, c, :], rng)
            T[:, :, c] = np.maximum(T[:, :, c] + res, 0.0)

        # per-image contrast stretch to [0,1]
        T = T - T.min()
        mx = float(T.max())
        if mx > 0:
            T = T / mx

        T_list.append(T.astype(float))
        A_list.append(A_true.astype(float))
    return T_list, A_list
def _ensure_colors(R):
    """Return an (R,3) color table in [0,1]. Extend DEFAULT_COLORS if needed."""
    base = DEFAULT_COLORS
    if R <= base.shape[0]:
        return base[:R]
    # extend by HSV evenly
    extra = R - base.shape[0]
    hs = np.linspace(0, 1, extra, endpoint=False)
    ext = np.stack([np.abs(np.sin(2*np.pi*hs))*0.7+0.3,
                    np.abs(np.sin(2*np.pi*(hs+0.33)))*0.7+0.3,
                    np.abs(np.sin(2*np.pi*(hs+0.66)))*0.7+0.3], axis=1)
    out = np.vstack([base, ext])
    return out[:R]

def _pad_to_square(img_hw3, value=0.0):
    """Pad an HxWx3 image to square (largest side) without distorting content."""
    H, W, _ = img_hw3.shape
    S = max(H, W)
    pad_top = (S - H) // 2
    pad_bottom = S - H - pad_top
    pad_left = (S - W) // 2
    pad_right = S - W - pad_left
    return np.pad(img_hw3,
                  ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                  mode="constant", constant_values=value)

def simulate_and_unmix_block(
    wl, sel_idx, labels, E_source, spectra_mode, B=None,
    title="Simulate & unmix (8.9 nm channels)"
):
    """Shared UI block for simulation + NLS unmixing."""
    st.subheader(title)
    try:
        f_cells, cells_ds, cells_labels, cell_meta = load_cell_db()
        f_noise, edges, qs, Q = load_noise_db()
        have_data = True
    except Exception as e:
        have_data = False
        st.info(f"Cell/noise DB not available: {e}")

    if not have_data:
        return

    E = build_E_matrix_from_selected(
        wl=wl, labels_all=labels, sel_idx=sel_idx,
        E_source=E_source, mode=spectra_mode, B=B
    )

    T_list, Atrue_list = generate_synthetic_images(
        E=E, cells_ds=cells_ds, n_images=3,
        rng=np.random.default_rng(123), noise_db=(edges, qs, Q)
    )

    R = len(sel_idx)
    COLORS = DEFAULT_COLORS[:max(R, 1)]

    for row in range(3):
        T = T_list[row]
        A_true = Atrue_list[row]
        A_hat = nls_unmix(T, E, iters=400, tol=1e-8, verbose=False)
        rmse = compute_rmse(A_hat, A_true)

        # 颜色：严格按 R 个来（你选了 4 个就用 4 种）
        COLORS = _ensure_colors(R)
        
        rgb = colorize_dominant(A_hat, COLORS).reshape(T.shape[0], T.shape[1], 3)
        
        # ---- 可视化增强：更激进的亮度提升 ----
        # 1) 高分位拉伸（99%）
        p = float(np.percentile(rgb, 99.0))
        if p > 1e-8:
            rgb = np.clip(rgb / p, 0.0, 1.0)
        
        # 2) 伽马校正（更亮一点）
        gamma = 0.7
        rgb = np.power(np.clip(rgb, 0.0, 1.0), gamma)
        
        # 3) 微弱抬底，避免纯黑背景吞细节
        rgb = np.clip(rgb + 0.03, 0.0, 1.0)
        
        # 4) 显示前“填充为正方形”
        rgb_sq = _pad_to_square(rgb, value=0.0)
        
        st.image((rgb_sq * 255).astype(np.uint8), use_container_width=True)
        st.caption(f"RMSE = {rmse:.4f} | R = {R}")


# ==============================
# Core run
# ==============================
def run_selection_and_display(groups, mode, laser_strategy, laser_list):
    required_count = (N_pick if use_pool else None)

    if mode == "Emission spectra":
        E_norm, labels_pair, idx_groups = build_emission_only_matrix(wl, dye_db, groups)
        if E_norm.shape[1] == 0:
            st.error("No spectra available for optimization.")
            st.stop()

        # strict lexicographic levels = min(10, number of pairs)
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

        # Selected
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

        # Spectra viewer
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

        # Simulate & unmix
        simulate_and_unmix_block(
            wl=wl,
            sel_idx=sel_idx,
            labels=labels_pair,
            E_source=E_norm,
            spectra_mode="Emission spectra",
            B=None
        )

    else:
        if not laser_list:
            st.error("Please specify laser wavelengths.")
            st.stop()

        # Round A: emission-only selection (for initial powers)
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

        # (1) powers on A
        if laser_strategy == "Simultaneous":
            powers_A, B_A = derive_powers_simultaneous(wl, dye_db, A_labels, laser_list)
        else:
            powers_A, B_A = derive_powers_separate(wl, dye_db, A_labels, laser_list)

        # Build effective (all candidates) with powers_A, then select
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

        # (2) powers on FINAL selection
        final_labels = [labels_all[j] for j in sel_idx]
        if laser_strategy == "Simultaneous":
            powers, B = derive_powers_simultaneous(wl, dye_db, final_labels, laser_list)
        else:
            powers, B = derive_powers_separate(wl, dye_db, final_labels, laser_list)

        # Rebuild effective with FINAL powers for display/metrics
        E_raw_all, E_norm_all, labels_all, idx_groups_all = build_effective_with_lasers(
            wl, dye_db, groups, laser_list, laser_strategy, powers
        )

        # Display
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

        lam_sorted = list(sorted(laser_list))
        p = np.array(powers, dtype=float)
        maxp = float(np.max(p)) if p.size > 0 else 1.0
        prel = (p / (maxp + 1e-12)).tolist()
        st.subheader("Laser powers (relative)")
        html_two_row_table("Laser (nm)", "Relative power",
                           lam_sorted, [float(f"{v:.6g}") for v in prel])

        S = cosine_similarity_matrix(E_norm_all[:, sel_idx])
        sub_labels = [labels_all[j] for j in sel_idx]
        tops = top_k_pairwise(S, sub_labels, k=k_show)
        pairs = [only_fluor_pair(a, b) for _, a, b in tops]
        sims = [val for val, _, _ in tops]
        st.subheader("Top pairwise similarities")
        html_two_row_table("Pair", "Similarity", pairs, sims,
                           color_second_row=True, color_thresh=0.9, format_second_row=True)

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
                    y=1.06, yref="paper",
                    text=f"{int(lam_sorted[i])} nm",
                    showarrow=False,
                    font=dict(size=12),
                    align="center",
                    yanchor="bottom"
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
                           ticktext=["0", "0.2", "0.4", "0.6", "0.8", "1"]),
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

        # Simulate & unmix
        simulate_and_unmix_block(
            wl=wl,
            sel_idx=sel_idx,
            labels=labels_all,
            E_source=E_raw_all,
            spectra_mode="Predicted spectra",
            B=B
        )

# Execute
run_selection_and_display(groups, mode, laser_strategy, laser_list)
