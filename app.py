# app.py — Fluorophore Selection with sampler-based realism
# - Abundance scaled to sampler.yaml max intensity
# - Spectra subset to fixed 33 wavelengths and globally normalized
# - Predicted intensity replaced by empirical per-channel samples from sampler.yaml

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

st.set_page_config(page_title="Fluorophore Selection", layout="wide")

# -------------------- Fixed channels (nm) & sampler path --------------------
CHANNEL_WAVELENGTHS = np.array([
    414, 423, 432, 441, 450, 459, 468, 477, 486, 494, 503, 512, 521, 530, 539,
    548, 557, 566, 575, 583, 592, 601, 610, 619, 628, 637, 646, 655, 664, 673,
    681, 690, 717
], dtype=float)  # 33 channels

SAMPLER_YAML = "data/sampler.yaml"  # your uploaded empirical database

# -------------------- Data --------------------
DYES_YAML = "data/dyes.yaml"
PROBE_MAP_YAML = "data/probe_fluor_map.yaml"
READOUT_POOL_YAML = "data/readout_fluorophores.yaml"

# -------------------- Loaders --------------------
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

def _get_inventory_from_probe_map():
    inv = set()
    for _, vals in probe_map.items():
        if not isinstance(vals, (list, tuple)):
            continue
        for f in vals:
            if isinstance(f, str):
                fs = f.strip()
                if fs and fs in dye_db:
                    inv.add(fs)
    return sorted(inv)

inventory_pool = _get_inventory_from_probe_map()

def _get_eub338_pool():
    targets = {"eub338", "eub 338", "eub-338"}
    def norm(s): return "".join(s.lower().split())
    for k in probe_map.keys():
        if norm(k) in targets:
            cands = [f for f in probe_map.get(k, []) if f in dye_db]
            return sorted({c.strip() for c in cands})
    import re
    def norm2(s): return re.sub(r"[^a-z0-9]+", "", s.lower())
    for k in probe_map.keys():
        if norm2(k) == "eub338":
            cands = [f for f in probe_map.get(k, []) if f in dye_db]
            return sorted({c.strip() for c in cands})
    return []

# -------------------- Sampler loader --------------------
def _open_text(path):
    import os, gzip
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")

def _load_sampler_yaml(path):
    import os, yaml, numpy as np

    # try .yaml then .yaml.gz
    if not os.path.exists(path):
        if os.path.exists(path + ".gz"):
            path = path + ".gz"
        else:
            return None

    with _open_text(path) as f:
        yml = yaml.safe_load(f) or {}

    if (yml.get("type") or "").lower() != "per_channel_bins":
        return None

    storage = (yml.get("meta", {}) or {}).get("storage") or yml.get("storage") or "samples"
    storage = str(storage).lower()
    bin_edges = yml.get("bin_edges") or []
    C = len(bin_edges)
    if C == 0:
        return None

    out = {"C": C, "bin_edges": bin_edges, "storage": storage}

    if storage == "samples":
        out["samples"] = yml.get("samples") or [[] for _ in range(C)]
    elif storage == "hist":
        out["hist"] = yml.get("hist") or [[] for _ in range(C)]
    elif storage == "quantiles":
        out["q_levels"] = yml.get("q_levels") or yml.get("quantile_levels")
        out["q_values"] = yml.get("q_values") or [[] for _ in range(C)]
    else:
        if "samples" in yml:
            storage = "samples"
            out["storage"] = "samples"
            out["samples"] = yml.get("samples") or [[] for _ in range(C)]
        else:
            return None

    # ---- robust numeric max (skip garbage & nested) ----
    def _safe_max_from_iter(x):
        m = None
        stack = [x]
        while stack:
            cur = stack.pop()
            if isinstance(cur, (list, tuple)):
                stack.extend(cur)
            else:
                try:
                    v = float(cur)
                    if np.isfinite(v):
                        m = v if (m is None or v > m) else m
                except Exception:
                    pass
        return m

    max_val = 0.0
    if storage == "samples":
        for c in range(C):
            chan = out["samples"][c] if c < len(out["samples"]) else []
            for b in chan:
                mv = _safe_max_from_iter(b)
                if mv is not None and mv > max_val:
                    max_val = mv

    elif storage == "hist":
        for c in range(C):
            chan = out["hist"][c] if c < len(out["hist"]) else []
            for hb in chan:
                centers = (hb or {}).get("centers", [])
                mv = _safe_max_from_iter(centers)
                if mv is not None and mv > max_val:
                    max_val = mv

    elif storage == "quantiles":
        qvals = out["q_values"]
        for c in range(C):
            chan = qvals[c] if c < len(qvals) else []
            for qb in chan:
                mv = _safe_max_from_iter(qb)
                if mv is not None and mv > max_val:
                    max_val = mv

    out["max_intensity"] = float(max_val)
    return out

# -------------------- Sidebar --------------------
st.sidebar.header("Configuration")
mode = st.sidebar.radio(
    "Mode",
    options=("Emission spectra", "Predicted spectra"),
    help=("Emission: emission-only, then subset to 33 channels & global normalize.\n"
          "Predicted: include lasers (excitation · QY · EC), then same subset & normalize."),
    key="mode_radio"
)
source_mode = st.sidebar.radio(
    "Selection source",
    ("By probes", "From readout pool", "All fluorophores", "EUB338 only"),
    key="source_radio"
)
k_show = st.sidebar.slider("Show top-K similarities", 5, 50, 10, 1, key="k_show_slider")

laser_list = []
laser_strategy = None
if mode == "Predicted spectra":
    laser_strategy = st.sidebar.radio("Laser usage", ("Simultaneous", "Separate"), key="laser_strategy_radio")
    n = st.sidebar.number_input("Number of lasers", 1, 8, 4, 1, key="num_lasers_input")
    cols = st.sidebar.columns(2)
    defaults = [405, 488, 561, 639]
    for i in range(n):
        lam = cols[i % 2].number_input(
            f"Laser {i+1} (nm)", int(wl.min()), int(max(700, wl.max())),
            defaults[i] if i < len(defaults) else int(wl.min()), 1, key=f"laser_{i+1}"
        )
        laser_list.append(int(lam))

# -------------------- UI helpers --------------------
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
    def esc(x): return (str(x).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
    def fmtv(v):
        if fmt2:
            try:
                return f"{float(v):.3f}"
            except:
                return esc(v)
        return esc(v)
    cells0 = "".join(f"<td style='padding:6px 10px;border:1px solid #ddd;'>{esc(v)}</td>" for v in row0_vals)
    tds0 = f"<td style='padding:6px 10px;border:1px solid #ddd;white-space:nowrap;'>{esc(row0_label)}</td>{cells0}"
    tds1_list = []
    for v in row1_vals:
        style = "padding:6px 10px;border:1px solid #ddd;"
        if color_second_row:
            try:
                vv = float(v)
                style += f"color:{'red' if vv > color_thresh else 'green'};"
            except:
                pass
        tds1_list.append(f"<td style='{style}'>{fmtv(v)}</td>")
    tds1 = f"<td style='padding:6px 10px;border:1px solid #ddd;white-space:nowrap;'>{esc(row1_label)}</td>{''.join(tds1_list)}"
    st.markdown(f"""
    <div style="overflow-x:auto;">
      <table style="border-collapse:collapse;width:100%;table-layout:auto;">
        <tbody><tr>{tds0}</tr><tr>{tds1}</tr></tbody>
      </table>
    </div>
    """, unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def cached_build_effective_with_lasers(wl, dye_db, groups, laser_list, laser_strategy, powers):
    groups_key = json.dumps({k: sorted(v) for k, v in sorted(groups.items())}, ensure_ascii=False)
    _ = (tuple(sorted(laser_list)), laser_strategy, tuple(np.asarray(powers, float)) if powers is not None else None, groups_key)
    return build_effective_with_lasers(wl, dye_db, groups, laser_list, laser_strategy, powers)

@st.cache_data(show_spinner=False)
def cached_interpolate_E_on_channels(wl, spectra_cols, chan_centers_nm):
    spectra_cols = np.asarray(spectra_cols, dtype=float)
    if spectra_cols.ndim == 1:
        spectra_cols = spectra_cols[:, None]
    W, N = spectra_cols.shape
    E = np.zeros((len(chan_centers_nm), N), dtype=float)
    for j in range(N):
        y = spectra_cols[:, j]
        E[:, j] = np.interp(chan_centers_nm, wl, y, left=float(y[0]), right=float(y[-1]))
    return np.nan_to_num(E, nan=0.0, posinf=0.0, neginf=0.0)

def _to_uint8_gray(img2d):
    z = np.asarray(img2d, dtype=float)
    m = float(z.max())
    if m > 0:
        z = z / m
    return (np.clip(z, 0, 1) * 255).astype(np.uint8)

def _argmax_labelmap(Ahat, colors, rescale_global=False):
    H, W, R = Ahat.shape
    idx = np.argmax(Ahat, axis=2)
    mx  = np.max(Ahat, axis=2)
    if rescale_global:
        m = float(mx.max())
        if m > 0:
            mx = mx / m
    cols = np.asarray(colors, dtype=float)
    rgb = cols[idx] * mx[:, :, None]
    rgb = np.clip(rgb, 0, 1)
    return (rgb * 255).astype(np.uint8)

def _chunk(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def _show_bw_grid(title, imgs_uint8, labels, cols_per_row=6):
    st.markdown(f"**{title}**")
    n = len(imgs_uint8)
    for i in range(0, n, cols_per_row):
        chunk_imgs = imgs_uint8[i:i+cols_per_row]
        chunk_labels = labels[i:i+cols_per_row]
        cols = st.columns(cols_per_row)
        for j in range(cols_per_row):
            if j < len(chunk_imgs):
                cols[j].image(chunk_imgs[j], use_container_width=True, clamp=True)
                cols[j].caption(chunk_labels[j])
            else:
                cols[j].markdown("&nbsp;")

def _html_table(headers, rows, num_cols=None):
    num_cols = num_cols or set()
    def esc(x):
        return str(x).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
    thead = "".join(f"<th style='padding:6px 10px;border:1px solid #ddd;text-align:left'>{esc(h)}</th>" for h in headers)
    trs = []
    for r in rows:
        tds=[]
        for j, v in enumerate(r):
            text = f"{float(v):.4f}" if j in num_cols else esc(v)
            align = "right" if j in num_cols else "left"
            tds.append(f"<td style='padding:6px 10px;border:1px solid #ddd;text-align:{align}'>{text}</td>")
        trs.append(f"<tr>{''.join(tds)}</tr>")
    st.markdown(
        f"""
        <div style="overflow-x:auto;">
          <table style="border-collapse:collapse;width:100%;table-layout:auto;">
            <thead><tr>{thead}</tr></thead>
            <tbody>{''.join(trs)}</tbody>
          </table>
        </div>
        """,
        unsafe_allow_html=True
    )

# -------------------- NLS & color --------------------
def nls_unmix(Timg, E, iters=2000, tol=1e-6):
    """Fast MU-style NLS with per-pixel normalization. Timg(H,W,C), E(C,R) -> A(H,W,R)."""
    H, W, C = Timg.shape
    E = np.asarray(E, dtype=np.float32)
    if E.ndim != 2 or E.shape[0] != C:
        raise ValueError(f"E shape {E.shape} mismatch with Timg channels {C}")
    M = Timg.reshape(-1, C).astype(np.float32, copy=False)
    scale = np.sqrt(np.mean(M**2, axis=1, keepdims=True)); scale[scale<=0]=1.0
    Mn = M/scale
    EtE = E.T @ E
    A = Mn @ E @ np.linalg.pinv(EtE)
    A[A<0]=0
    for _ in range(iters):
        numer = Mn @ E
        denom = (A @ EtE) + 1e-12
        A *= numer / denom
        if np.max(numer / (denom + 1e-12)) < 1 + tol:
            break
    A *= scale
    mA = float(np.max(A))
    if mA>0: A /= mA
    return A.reshape(H, W, E.shape[1])

def colorize_single(A_r, color):
    z = np.clip(A_r, 0, 1); m = float(z.max())
    if m > 0: z /= m
    return z[:, :, None] * np.asarray(color)[None, None, :]

def colorize_composite(A, colors):
    rgb = np.zeros((A.shape[0], A.shape[1], 3), dtype=float)
    for r in range(A.shape[2]):
        rgb += colorize_single(A[:, :, r], colors[r])
    m = float(rgb.max())
    if m > 0: rgb /= m
    return rgb

# -------------------- Rod (capsule) scene --------------------
def _capsule_profile(H, W, cx, cy, length, width, theta):
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    X = xx - cx; Y = yy - cy
    c, s = np.cos(theta), np.sin(theta)
    xp =  c*X + s*Y
    yp = -s*X + c*Y
    half_L = 0.5*length
    r = 0.5*width
    rect = (np.abs(xp) <= half_L) & (np.abs(yp) <= r)
    val = np.zeros((H, W))
    if np.any(rect): val[rect] = 1 - np.abs(yp[rect])/(r+1e-12)
    for side in (-1, 1):
        rho = np.sqrt((xp + side*half_L)**2 + yp**2)
        cap = rho <= r
        if np.any(cap): val[cap] = np.maximum(val[cap], 1 - rho[cap]/(r+1e-12))
    return np.clip(val, 0, 1), val > 0

def _place_rods_scene(H, W, R, rods_per=3, rng=None, max_trials_per_class=1200):
    rng = np.random.default_rng() if rng is None else rng
    Atrue = np.zeros((H, W, R), dtype=np.float32)
    occ = np.zeros((H, W), dtype=bool)
    placed_counts = np.zeros(R, dtype=int)

    Lmin, Lmax = 18, 30
    Wmin, Wmax = 10, 16

    for r in range(R):
        placed = 0; tries = 0
        while placed < rods_per and tries < max_trials_per_class:
            tries += 1
            length = int(rng.integers(Lmin, Lmax + 1))
            width  = int(rng.integers(Wmin, Wmax + 1))
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
            m = float(prof.max())
            if m > 0:
                prof /= m
            Atrue[:, :, r] = np.maximum(Atrue[:, :, r], prof.astype(np.float32))
            occ |= mask
            placed += 1
        placed_counts[r] = placed

    return np.clip(Atrue, 0, 1), placed_counts

def _capsule_expected_area(Lmin=18, Lmax=30, Wmin=10, Wmax=16):
    L = 0.5 * (Lmin + Lmax)
    W = 0.5 * (Wmin + Wmax)
    r = 0.5 * W
    return 2 * r * L + np.pi * r * r

def _suggest_canvas_size(R, rods_per, target_density=0.22, min_side=160):
    area_one = _capsule_expected_area()
    total_obj_area = R * rods_per * area_one
    canvas_area = total_obj_area / max(1e-6, target_density)
    side = int(np.ceil(np.sqrt(canvas_area)))
    side = max(min_side, side)
    return side, side

# -------------------- Subset & normalize spectra matrix --------------------
def subset_and_normalize_matrix(wl_full, M_full, target_nm):
    """
    Interpolate column-wise spectra matrix M_full (W x N) to target_nm (L,)
    and divide by the GLOBAL max of the resulting matrix (so max=1).
    Returns (E_sub[L x N], global_max).
    """
    wl_full = np.asarray(wl_full, float)
    M_full  = np.asarray(M_full, float)
    target  = np.asarray(target_nm, float)
    if M_full.ndim == 1:
        M_full = M_full[:, None]
    W, N = M_full.shape
    E = np.zeros((target.size, N), dtype=float)
    for j in range(N):
        y = M_full[:, j]
        E[:, j] = np.interp(target, wl_full, y, left=float(y[0]), right=float(y[-1]))
    gmax = float(np.max(E)) if E.size else 1.0
    if gmax <= 0: gmax = 1.0
    return E / gmax, gmax

# -------------------- Sampler-based simulation --------------------
def _sample_channel_from_bins(I_flat, edges, pools, rng):
    """
    I_flat: (N,) predicted intensities for a single channel
    edges:  (B+1,) bin edges (ascending)
    pools:  list[B] of arrays/lists with real intensity samples (可能含空或脏数据)
    Returns y_flat: (N,) sampled real intensities.
    """
    import numpy as np

    edges = np.asarray(edges, float)
    N = I_flat.size
    y = np.zeros(N, dtype=float)

    # locate bins
    b = np.searchsorted(edges, I_flat, side='right') - 1
    b = np.clip(b, 0, len(edges) - 2)

    def _pool_empty(p):
        try:
            return (p is None) or (len(p) == 0)
        except Exception:
            return True

    def _pool_to_array(p):
        # 尽量转为 1D float 数组；失败则返回空
        try:
            arr = np.asarray(p, dtype=float).ravel()
            arr = arr[np.isfinite(arr)]
            return arr
        except Exception:
            return np.array([], dtype=float)

    uniq = np.unique(b)
    for ub in uniq:
        mask = (b == ub)
        pool = pools[ub] if ub < len(pools) else None

        arr = np.array([], dtype=float) if _pool_empty(pool) else _pool_to_array(pool)
        if arr.size == 0:
            # fallback: 最近的非空 bin；再不行就用该 bin 区间内均匀采样
            found = None
            L, R = ub - 1, ub + 1
            while L >= 0 or R < len(pools):
                cand = None
                if L >= 0 and not _pool_empty(pools[L]):
                    cand = _pool_to_array(pools[L])
                    if cand.size: found = cand; break
                if R < len(pools) and not _pool_empty(pools[R]):
                    cand = _pool_to_array(pools[R])
                    if cand.size: found = cand; break
                L -= 1; R += 1
            if found is not None:
                idx = rng.integers(0, found.size, size=mask.sum())
                y[mask] = found[idx]
            else:
                left, right = edges[ub], edges[ub+1]
                y[mask] = rng.uniform(left, right, size=mask.sum())
        else:
            idx = rng.integers(0, arr.size, size=mask.sum())
            y[mask] = arr[idx]

    return y


def simulate_with_sampler(E, sampler, rods_per=3, rng=None):
    """
    Build rods abundance Atrue (scaled so max == sampler.max_intensity),
    compute predicted intensities Tpred = Atrue ⊗ E, then replace each
    predicted channel value by a random real intensity drawn from sampler bins.
    Finally unmix with E.

       E: (C, R) matrix already subset to CHANNEL_WAVELENGTHS and normalized by global max
          C = number of channels (33), R = number of selected fluorophores

    Returns: Atrue (H,W,R), Ahat (H,W,R)
    """
    rng = np.random.default_rng() if rng is None else rng
    E = np.asarray(E, float)
    if E.ndim != 2:
        raise ValueError(f"E must be 2D, got {E.shape}")
    C, R = E.shape

    # auto-size canvas and place rods
    H, W = _suggest_canvas_size(R, rods_per, target_density=0.22, min_side=160)
    Atrue01, _ = _place_rods_scene(H, W, R, rods_per, rng)

    # scale abundance to sampler max intensity
    Atrue = Atrue01 * float(sampler["max_intensity"])

    # predicted intensities Tpred(H,W,C)
    Tpred = np.zeros((H, W, C), dtype=float)
    for c in range(C):
        # sum_r Atrue * E[c, r]
        Tpred[:, :, c] = np.tensordot(Atrue, E[c, :], axes=([2], [0]))

    # empirical resampling per channel
    Tflat = Tpred.reshape(-1, C)
    Yflat = np.zeros_like(Tflat)
    for c in range(C):
        edges = np.asarray(sampler["bin_edges"][c], float)
        pools = sampler["samples"][c]
        Yflat[:, c] = _sample_channel_from_bins(Tflat[:, c], edges, pools, rng)
    Tobs = Yflat.reshape(H, W, C)

    # unmix back with E
    Tobs_n = Tobs / (float(np.max(Tobs)) + 1e-12)
    Ahat = nls_unmix(Tobs_n, E, iters=1200, tol=1e-6)
    return Atrue, Ahat

# -------------------- Main panels --------------------
st.title("Fluorophore Selection for Multiplexed Imaging")

# Build candidate groups
use_pool = False
if source_mode == "From readout pool":
    pool = readout_pool[:]
    if not pool:
        st.info("Readout pool not found (data/readout_fluorophores.yaml)."); st.stop()
    max_n = len(pool)
    N_pick = st.number_input("How many fluorophores", 1, max_n, min(4, max_n), 1, key="n_pick_pool")
    groups = {"Pool": pool}
    use_pool = True

elif source_mode == "All fluorophores":
    pool = inventory_pool[:]
    if not pool:
        st.error("No fluorophores found in probe_fluor_map.yaml that also exist in dyes.yaml."); st.stop()
    max_n = len(pool)
    N_pick = st.number_input("How many fluorophores", 1, max_n, min(4, max_n), 1, key="n_pick_inv")
    groups = {"Pool": pool}
    use_pool = True

elif source_mode == "EUB338 only":
    pool = _get_eub338_pool()
    if not pool:
        st.error("No candidates found for EUB 338 in probe_fluor_map.yaml."); st.stop()
    max_n = len(pool)
    N_pick = st.number_input("How many fluorophores", 1, max_n, min(4, max_n), 1, key="n_pick_eub338")
    groups = {"Pool": pool}
    use_pool = True

else:  # "By probes"
    all_probes = sorted(probe_map.keys())
    picked = st.multiselect("Probes", options=all_probes, key="picked_probes")
    if not picked:
        st.info("Select at least one probe to proceed."); st.stop()
    groups = {}
    for p in picked:
        cands = [f for f in probe_map.get(p, []) if f in dye_db]
        if cands:
            groups[p] = cands
    if not groups:
        st.error("No valid candidates with spectra in dyes.yaml."); st.stop()
    N_pick = None

def _prettify_name(label: str) -> str:
    name = label.split(" – ", 1)[1] if " – " in label else label
    up = name.upper()
    if up.startswith("AF") and name[2:].isdigit():
        return f"AF {name[2:]}"
    return name

def run(groups, mode, laser_strategy, laser_list):
    required_count = (N_pick if use_pool else None)

    # -------------------- EMISSION branch --------------------
    if mode == "Emission spectra":
        if SAMPLER is None:
            st.error("sampler.yaml not found or invalid. Please place it at data/sampler.yaml.")
            st.stop()

        # Build emission-only on full wl, then subset to 33 channels and global normalize
        E_full, labels, idx_groups = build_emission_only_matrix(wl, dye_db, groups)
        if E_full.shape[1] == 0:
            st.error("No spectra."); st.stop()

        E_sub, gmax = subset_and_normalize_matrix(wl, E_full, CHANNEL_WAVELENGTHS)  # (33 x N)

        # Selection on the subset-normalized matrix
        sel_idx, _ = solve_lexicographic_k(
            E_sub, idx_groups, labels,
            levels=10, enforce_unique=True, required_count=required_count
        )
        colors = _ensure_colors(len(sel_idx))

        # Selected list
        if use_pool:
            fluors = [labels[j].split(" – ", 1)[1] for j in sel_idx]
            st.subheader("Selected Fluorophores")
            _html_two_row_table("Slot", "Fluorophore",
                                [f"Slot {i+1}" for i in range(len(fluors))],
                                fluors)
        else:
            sel_pairs = [labels[j] for j in sel_idx]
            st.subheader("Selected Fluorophores")
            _html_two_row_table("Probe", "Fluorophore",
                                [s.split(" – ", 1)[0] for s in sel_pairs],
                                [s.split(" – ", 1)[1] for s in sel_pairs])

        # Pairwise similarities (same E_sub)
        S = cosine_similarity_matrix(E_sub[:, sel_idx])
        tops = top_k_pairwise(S, [labels[j] for j in sel_idx], k=k_show)
        st.subheader("Top pairwise similarities")
        _html_two_row_table("Pair", "Similarity",
                            [_pair_only_fluor(a, b) for _, a, b in tops],
                            [val for val, _, _ in tops],
                            color_second_row=True, color_thresh=0.9, fmt2=True)

        # Spectra viewer on 33 channels
        st.subheader("Spectra viewer (subset & globally normalized)")
        fig = go.Figure()
        xnm = CHANNEL_WAVELENGTHS
        for t, j in enumerate(sel_idx):
            y = E_sub[:, j]
            fig.add_trace(go.Scatter(
                x=xnm, y=y, mode="lines+markers", name=labels[j],
                line=dict(color=_rgb01_to_plotly(colors[t]), width=2)
            ))
        fig.update_layout(xaxis_title="Wavelength (nm)",
                          yaxis_title="Normalized intensity",
                          yaxis=dict(range=[0, 1.05]))
        st.plotly_chart(fig, use_container_width=True)

        # Sampler-based simulation on selected set
        # E for simulation must be (C, R) with C=33, R=#selected dyes
        E_sel = E_sub[:, sel_idx]  # (33 x R)
        Atrue, Ahat = simulate_with_sampler(E_sel, SAMPLER, rods_per=3)

        colL, colR = st.columns(2)
        true_rgb = (colorize_composite(Atrue / (Atrue.max()+1e-12), colors) * 255).astype(np.uint8)
        labelmap_rgb = _argmax_labelmap(Ahat, colors, rescale_global=True)
        with colL:
            st.image(true_rgb, use_container_width=True, clamp=True)
            st.caption("True (abundance scaled to sampler max)")
        with colR:
            st.image(labelmap_rgb, use_container_width=True, clamp=True)
            st.caption("Unmixing results")

        names = [_prettify_name(labels[j]) for j in sel_idx]
        unmix_bw = [_to_uint8_gray(Ahat[:, :, r]) for r in range(Ahat.shape[2])]
        st.divider()
        _show_bw_grid("Per-fluorophore (Unmixing, grayscale)", unmix_bw, names, cols_per_row=6)

        rmse_vals = []
        for r in range(len(names)):
            rmse_vals.append(np.sqrt(np.mean((Ahat[:, :, r] - Atrue[:, :, r])**2)))
        st.subheader("Per-fluorophore RMSE")
        _html_two_row_table("Fluorophore", "RMSE", names, rmse_vals, fmt2=True)
        return

    # -------------------- PREDICTED branch --------------------
    else:
        if SAMPLER is None:
            st.error("sampler.yaml not found or invalid. Please place it at data/sampler.yaml.")
            st.stop()
        if not laser_list:
            st.error("Please specify laser wavelengths."); st.stop()

        # Provisional selection on emission-only
        E0, labels0, idx0 = build_emission_only_matrix(wl, dye_db, groups)
        sel0, _ = solve_lexicographic_k(E0, idx0, labels0, levels=10, enforce_unique=True, required_count=required_count)
        A_labels = [labels0[j] for j in sel0]

        # Powers on provisional set
        if laser_strategy == "Simultaneous":
            powers_A, _ = derive_powers_simultaneous(wl, dye_db, A_labels, laser_list)
        else:
            powers_A, _ = derive_powers_separate(wl, dye_db, A_labels, laser_list)

        # Build full with lasers
        E_raw_all, E_norm_all, labels_all, idx_all = cached_build_effective_with_lasers(
            wl, dye_db, groups, laser_list, laser_strategy, powers_A
        )

        # Final selection on E_norm_all
        sel_idx, _ = solve_lexicographic_k(
            E_norm_all, idx_all, labels_all, levels=10, enforce_unique=True, required_count=required_count
        )
        final_labels = [labels_all[j] for j in sel_idx]

        # Recalibrate powers on final set
        if laser_strategy == "Simultaneous":
            powers, B = derive_powers_simultaneous(wl, dye_db, final_labels, laser_list)
        else:
            powers, B = derive_powers_separate(wl, dye_db, final_labels, laser_list)

        # Build only selected subset at full wavelengths
        if use_pool:
            small_groups = {"Pool": [s.split(" – ", 1)[1] for s in final_labels]}
        else:
            small_groups = {}
            for s in final_labels:
                p, f = s.split(" – ", 1)
                small_groups.setdefault(p, []).append(f)

        E_raw_sel, _, labels_sel, _ = cached_build_effective_with_lasers(
            wl, dye_db, small_groups, laser_list, laser_strategy, powers
        )  # (W x R)

        # Subset to 33 channels & global normalize (matrix-wise)
        E_sub, gmax = subset_and_normalize_matrix(wl, E_raw_sel, CHANNEL_WAVELENGTHS)  # (33 x R)
        colors = _ensure_colors(len(labels_sel))

        # Selected list & pairwise similarities on E_sub
        st.subheader("Selected Fluorophores (with lasers)")
        fluors = [s.split(" – ", 1)[1] for s in labels_sel]
        _html_two_row_table("Slot", "Fluorophore",
                            [f"Slot {i+1}" for i in range(len(fluors))],
                            fluors)

        S = cosine_similarity_matrix(E_sub)
        tops = top_k_pairwise(S, labels_sel, k=k_show)
        st.subheader("Top pairwise similarities")
        _html_two_row_table("Pair", "Similarity",
                            [_pair_only_fluor(a, b) for _, a, b in tops],
                            [val for val, _, _ in tops],
                            color_second_row=True, color_thresh=0.9, fmt2=True)

        st.subheader("Spectra viewer (subset & globally normalized)")
        fig = go.Figure()
        for t in range(len(labels_sel)):
            y = E_sub[:, t]
            fig.add_trace(go.Scatter(
                x=CHANNEL_WAVELENGTHS, y=y, mode="lines+markers", name=labels_sel[t],
                line=dict(color=_rgb01_to_plotly(colors[t]), width=2)
            ))
        fig.update_layout(
            xaxis_title="Wavelength (nm)",
            yaxis_title="Normalized intensity",
            yaxis=dict(range=[0, 1.05])
        )
        st.plotly_chart(fig, use_container_width=True)

        # Sampler-based simulation on selected set
        E_sel = E_sub  # (33 x R) -> channels x fluorophores
        Atrue, Ahat = simulate_with_sampler(E_sel, SAMPLER, rods_per=3)

        colL, colR = st.columns(2)
        true_rgb = (colorize_composite(Atrue / (Atrue.max()+1e-12), colors) * 255).astype(np.uint8)
        labelmap_rgb = _argmax_labelmap(Ahat, colors, rescale_global=True)
        with colL:
            st.image(true_rgb, use_container_width=True, clamp=True)
            st.caption("True (abundance scaled to sampler max)")
        with colR:
            st.image(labelmap_rgb, use_container_width=True, clamp=True)
            st.caption("Unmixing results")

        names = [_prettify_name(s) for s in labels_sel]
        unmix_bw = [_to_uint8_gray(Ahat[:, :, r]) for r in range(Ahat.shape[2])]
        st.divider()
        _show_bw_grid("Per-fluorophore (Unmixing, grayscale)", unmix_bw, names, cols_per_row=6)

        rmse_vals = []
        for r in range(len(names)):
            rmse_vals.append(np.sqrt(np.mean((Ahat[:, :, r] - Atrue[:, :, r])**2)))
        st.subheader("Per-fluorophore RMSE")
        _html_two_row_table("Fluorophore", "RMSE", names, rmse_vals, fmt2=True)
        return

# -------------------- Execute --------------------
if __name__ == "__main__":
    run_groups = None
    # groups are built earlier depending on sidebar selection
    # Recompute here to ensure latest state:
    if source_mode == "From readout pool":
        pool = readout_pool[:]
        run_groups = {"Pool": pool}
    elif source_mode == "All fluorophores":
        pool = inventory_pool[:]
        run_groups = {"Pool": pool}
    elif source_mode == "EUB338 only":
        pool = _get_eub338_pool()
        run_groups = {"Pool": pool}
    else:
        # By probes, pull selected again (Streamlit keeps state)
        picked = st.session_state.get("picked_probes", [])
        run_groups = {}
        for p in picked or []:
            cands = [f for f in probe_map.get(p, []) if f in dye_db]
            if cands:
                run_groups[p] = cands
        if not run_groups:
            st.stop()
    run(run_groups, mode, laser_strategy, laser_list)
