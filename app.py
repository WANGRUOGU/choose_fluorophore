# app.py — streamlined + small rod-shaped cells + brightness preserved in Predicted
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

st.set_page_config(page_title="Choose Fluorophore", layout="wide")

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
    help=("Emission: 仅发射谱、峰值归一；Predicted: 含激发·QY·EC与激光，体现亮度差。")
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

# -------------------- Helper colors/tables --------------------
DEFAULT_COLORS = np.array([
    [0.95, 0.25, 0.25], [0.25, 0.65, 0.95],
    [0.25, 0.85, 0.35], [0.90, 0.70, 0.20],
    [0.80, 0.40, 0.80], [0.25, 0.80, 0.80],
    [0.85, 0.50, 0.35], [0.60, 0.60, 0.60],
], dtype=float)

def _ensure_colors(R):
    if R <= len(DEFAULT_COLORS): return DEFAULT_COLORS[:R]
    hs = np.linspace(0, 1, R, endpoint=False)
    extra = np.stack([
        np.abs(np.sin(2*np.pi*hs))*0.7+0.3,
        np.abs(np.sin(2*np.pi*(hs+0.33)))*0.7+0.3,
        np.abs(np.sin(2*np.pi*(hs+0.66)))*0.7+0.3
    ], axis=1)
    return extra[:R]

def _rgb01_to_plotly(col):
    r,g,b = (int(255*x) for x in col)
    return f"rgb({r},{g},{b})"

def _pair_only_fluor(a,b):
    fa = a.split(" – ",1)[1] if " – " in a else a
    fb = b.split(" – ",1)[1] if " – " in b else b
    return f"{fa} vs {fb}"

def _html_two_row_table(row0_label,row1_label,row0_vals,row1_vals,
                        color_second_row=False,color_thresh=0.9,fmt2=False):
    def esc(x): return (str(x).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;"))
    def fmtv(v):
        if fmt2:
            try: return f"{float(v):.3f}"
            except: return esc(v)
        return esc(v)
    cells0 = "".join(f"<td style='padding:6px 10px;border:1px solid #ddd;'>{esc(v)}</td>" for v in row0_vals)
    tds0 = f"<td style='padding:6px 10px;border:1px solid #ddd;white-space:nowrap;'>{esc(row0_label)}</td>{cells0}"
    tds1_list=[]
    for v in row1_vals:
        style="padding:6px 10px;border:1px solid #ddd;"
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
def cached_build_effective_with_lasers(wl,dye_db,groups,laser_list,laser_strategy,powers):
    groups_key = json.dumps({k:sorted(v) for k,v in sorted(groups.items())}, ensure_ascii=False)
    _ = (tuple(sorted(laser_list)), laser_strategy, tuple(np.asarray(powers,float)) if powers is not None else None, groups_key)
    return build_effective_with_lasers(wl,dye_db,groups,laser_list,laser_strategy,powers)

@st.cache_data(show_spinner=False)
def cached_interpolate_E_on_channels(wl,spectra_cols,chan_centers_nm):
    spectra_cols = np.asarray(spectra_cols, dtype=float)
    if spectra_cols.ndim == 1: spectra_cols = spectra_cols[:,None]
    W,N = spectra_cols.shape
    E = np.zeros((len(chan_centers_nm),N), dtype=float)
    for j in range(N):
        y = spectra_cols[:,j]
        E[:,j] = np.interp(chan_centers_nm, wl, y, left=float(y[0]), right=float(y[-1]))
    return np.nan_to_num(E, nan=0.0, posinf=0.0, neginf=0.0)

# -------------------- NLS + color --------------------
def nls_unmix(Timg,E,iters=2000,tol=1e-6):
    """Fast NMF-like MU with pixelwise normalization; expects Timg(H,W,C), E(C,R)."""
    H,W,C = Timg.shape
    E = np.asarray(E, dtype=np.float32)
    if E.ndim != 2 or E.shape[0] != C:
        raise ValueError(f"E shape {E.shape} mismatch with Timg channels {C}")
    M = Timg.reshape(-1, C).astype(np.float32, copy=False)
    scale = np.sqrt(np.mean(M**2, axis=1, keepdims=True)); scale[scale<=0]=1.0
    Mn = M/scale
    EtE = E.T @ E
    A = Mn @ np.linalg.pinv(EtE) @ E.T
    A[A<0]=0
    for _ in range(iters):
        numer = Mn @ E
        denom = (A @ EtE) + 1e-12
        A *= numer / denom
        # 早停：相对改进很小则退出（简单近似）
        if np.max(numer/ (denom+1e-12)) < 1 + tol:
            break
    A *= scale
    mA = float(np.max(A))
    if mA>0: A /= mA
    return A.reshape(H,W,E.shape[1])

def colorize_single(A_r,color):
    z=np.clip(A_r,0,1); m=float(z.max())
    if m>0: z/=m
    return z[:,:,None]*np.asarray(color)[None,None,:]

def colorize_composite(A,colors):
    rgb=np.zeros((A.shape[0],A.shape[1],3),dtype=float)
    for r in range(A.shape[2]): rgb+=colorize_single(A[:,:,r],colors[r])
    m=float(rgb.max()); 
    if m>0: rgb/=m
    return rgb

# -------------------- Rod (capsule) scene --------------------
def _capsule_profile(H,W,cx,cy,length,width,theta):
    yy,xx = np.mgrid[0:H,0:W].astype(float)
    X=xx-cx; Y=yy-cy
    c,s = np.cos(theta), np.sin(theta)
    xp =  c*X + s*Y
    yp = -s*X + c*Y
    half_L = 0.5*length
    r = 0.5*width
    rect = (np.abs(xp)<=half_L) & (np.abs(yp)<=r)
    val = np.zeros((H,W))
    if np.any(rect): val[rect] = 1 - np.abs(yp[rect])/(r+1e-12)
    for side in (-1,1):
        rho = np.sqrt((xp+side*half_L)**2 + yp**2)
        cap = rho <= r
        if np.any(cap): val[cap] = np.maximum(val[cap], 1 - rho[cap]/(r+1e-12))
    return np.clip(val,0,1), val>0

def _place_rods_scene(H,W,R,rods_per=3,rng=None):
    rng = np.random.default_rng() if rng is None else rng
    Atrue = np.zeros((H,W,R), dtype=np.float32)
    occ = np.zeros((H,W), dtype=bool)
    # smaller rods → faster & clearer
    Lmin,Lmax = 18, 36
    Wmin,Wmax = 6, 10
    for r in range(R):
        placed = 0; tries = 0
        while placed < rods_per and tries < 200:
            tries += 1
            length = int(rng.integers(Lmin, Lmax+1))
            width  = int(rng.integers(Wmin, Wmax+1))
            theta  = float(rng.uniform(0, np.pi))
            margin = 6 + int(max(length,width)/2)
            if W - 2*margin <= 2 or H - 2*margin <= 2: break
            cx = int(rng.integers(margin, W-margin))
            cy = int(rng.integers(margin, H-margin))
            prof, mask = _capsule_profile(H,W,cx,cy,length,width,theta)
            if not np.any(mask): continue
            if np.any(occ & mask): continue
            m=float(prof.max())
            if m>0: prof/=m
            Atrue[:,:,r] = np.maximum(Atrue[:,:,r], prof.astype(np.float32))
            occ |= mask
            placed += 1
    return np.clip(Atrue,0,1)

# -------------------- Simulation --------------------
def simulate_rods_and_unmix(E,H=160,W=160,rods_per=3,rng=None):
    """Use global scaling Poisson noise → preserves inter-dye brightness."""
    rng = np.random.default_rng() if rng is None else rng
    E = np.asarray(E, dtype=float)
    if E.ndim != 2: raise ValueError(f"E must be 2D, got {E.shape}")
    C,R = E.shape

    # scene
    Atrue = _place_rods_scene(H,W,R,rods_per,rng)

    # forward render
    Tclean = np.zeros((H,W,C), dtype=float)
    for c in range(C):
        Tclean[:,:,c] = np.tensordot(Atrue, E[c,:], axes=([2],[0]))

    # global Poisson scaling to peak=255 (preserve brightness ratios)
    peak = 255.0
    Tmax = float(np.max(Tclean))
    if Tmax <= 0: 
        Tnoisy = np.zeros_like(Tclean)
    else:
        lam = Tclean * (peak / Tmax)
        lam = np.nan_to_num(lam, nan=0.0, posinf=1e6, neginf=0.0)
        lam = np.clip(lam, 0.0, 1e6)
        Tnoisy = rng.poisson(lam).astype(float) / peak

    # unmix
    Ahat = nls_unmix(Tnoisy, E, iters=1500, tol=1e-6)
    rmse = float(np.sqrt(np.mean((Ahat - Atrue)**2)))
    return Atrue, Ahat, rmse

# -------------------- Main --------------------
st.title("Fluorophore Selection for Multiplexed Imaging")

use_pool = (source_mode == "From readout pool")
if use_pool:
    if not readout_pool:
        st.info("Readout pool not found (data/readout_fluorophores.yaml)."); st.stop()
    max_n = len(readout_pool)
    N_pick = st.number_input("How many fluorophores", 1, max_n, min(4, max_n), 1)
    groups = {"Pool": readout_pool[:]}
else:
    all_probes = sorted(probe_map.keys())
    picked = st.multiselect("Probes", options=all_probes)
    if not picked:
        st.info("Select at least one probe to proceed."); st.stop()
    groups = {}
    for p in picked:
        cands = [f for f in probe_map.get(p, []) if f in dye_db]
        if cands: groups[p] = cands
    if not groups:
        st.error("No valid candidates with spectra in dyes.yaml."); st.stop()
    N_pick = None

def run(groups, mode, laser_strategy, laser_list):
    required_count = (N_pick if use_pool else None)

    if mode == "Emission spectra":
        # selection on emission-only (peak-normalized in utils)
        E_norm, labels, idx_groups = build_emission_only_matrix(wl, dye_db, groups)
        if E_norm.shape[1] == 0: st.error("No spectra."); st.stop()

        sel_idx, _ = solve_lexicographic_k(
            E_norm, idx_groups, labels,
            levels=10, enforce_unique=True, required_count=required_count
        )
        colors = _ensure_colors(len(sel_idx))

        # Selected table
        if use_pool:
            fluors = [labels[j].split(" – ",1)[1] for j in sel_idx]
            st.subheader("Selected Fluorophores")
            _html_two_row_table("Slot","Fluorophore",[f"Slot {i+1}" for i in range(len(fluors))],fluors)
        else:
            sel_pairs = [labels[j] for j in sel_idx]
            st.subheader("Selected Fluorophores")
            _html_two_row_table("Probe","Fluorophore",
                                [s.split(" – ",1)[0] for s in sel_pairs],
                                [s.split(" – ",1)[1] for s in sel_pairs])

        # Pairwise similarities
        S = cosine_similarity_matrix(E_norm[:, sel_idx])
        tops = top_k_pairwise(S, [labels[j] for j in sel_idx], k=k_show)
        st.subheader("Top pairwise similarities")
        _html_two_row_table("Pair","Similarity",
                            [_pair_only_fluor(a,b) for _,a,b in tops],
                            [val for val,_,_ in tops],
                            color_second_row=True, color_thresh=0.9, fmt2=True)

        # Spectra viewer (per-trace normalized)
        st.subheader("Spectra viewer")
        fig = go.Figure()
        for t,j in enumerate(sel_idx):
            y = E_norm[:,j]; y = y/(np.max(y)+1e-12)
            fig.add_trace(go.Scatter(
                x=wl, y=y, mode="lines", name=labels[j],
                line=dict(color=_rgb01_to_plotly(colors[t]), width=2)
            ))
        fig.update_layout(xaxis_title="Wavelength (nm)",
                          yaxis_title="Normalized intensity",
                          yaxis=dict(range=[0,1.05]))
        st.plotly_chart(fig, use_container_width=True)

        # Optional simulation (emission-only -> per-channel max scaling并不体现亮度差，这里仅用于结构观感)
        if st.checkbox("Run rod simulation + NLS (heavier)", value=False):
            C = 23; chan = 494.0 + 8.9*np.arange(C)
            E = cached_interpolate_E_on_channels(wl, E_norm[:, sel_idx], chan)
            Atrue, Ahat, rmse = simulate_rods_and_unmix(E, H=160, W=160, rods_per=3)
            imgs = [("True (composite)", (colorize_composite(Atrue, colors)*255).astype(np.uint8))]
            names = [labels[j].split(" – ",1)[1] for j in sel_idx]
            for r, name in enumerate(names):
                rgb = (colorize_single(Ahat[:,:,r], colors[r])*255).astype(np.uint8)
                imgs.append((f"NLS ({name})", rgb))
            cs = st.columns(len(imgs))
            for c,(title,im) in zip(cs, imgs):
                c.image(im, use_container_width=True); c.caption(title)
            st.caption(f"Overall RMSE: {rmse:.4f}")

    else:
        if not laser_list:
            st.error("Please specify laser wavelengths."); st.stop()

        # Round A: emission-only provisional selection
        E0, labels0, idx0 = build_emission_only_matrix(wl, dye_db, groups)
        sel0, _ = solve_lexicographic_k(E0, idx0, labels0, levels=10, enforce_unique=True, required_count=required_count)
        A_labels = [labels0[j] for j in sel0]

        # (1) powers on A
        if laser_strategy == "Simultaneous":
            powers_A, _ = derive_powers_simultaneous(wl, dye_db, A_labels, laser_list)
        else:
            powers_A, _ = derive_powers_separate(wl, dye_db, A_labels, laser_list)

        # first build (cached)
        E_raw_all, E_norm_all, labels_all, idx_all = cached_build_effective_with_lasers(
            wl, dye_db, groups, laser_list, laser_strategy, powers_A
        )

        # final selection
        sel_idx, _ = solve_lexicographic_k(
            E_norm_all, idx_all, labels_all, levels=10, enforce_unique=True, required_count=required_count
        )
        final_labels = [labels_all[j] for j in sel_idx]

        # (2) recalibrate on final only
        if laser_strategy == "Simultaneous":
            powers, B = derive_powers_simultaneous(wl, dye_db, final_labels, laser_list)
        else:
            powers, B = derive_powers_separate(wl, dye_db, final_labels, laser_list)

        # second build: only selected subset (cached)
        if use_pool:
            small_groups = {"Pool":[s.split(" – ",1)[1] for s in final_labels]}
        else:
            small_groups = {}
            for s in final_labels:
                p,f = s.split(" – ",1)
                small_groups.setdefault(p,[]).append(f)

        E_raw_sel, E_norm_sel, labels_sel, _ = cached_build_effective_with_lasers(
            wl, dye_db, small_groups, laser_list, laser_strategy, powers
        )

        # Selected table
        st.subheader("Selected Fluorophores (with lasers)")
        fluors = [s.split(" – ",1)[1] for s in labels_sel]
        _html_two_row_table("Slot","Fluorophore",
                            [f"Slot {i+1}" for i in range(len(fluors))],
                            fluors)

        # Pairwise on normalized effective spectra
        S = cosine_similarity_matrix(E_norm_sel)
        tops = top_k_pairwise(S, labels_sel, k=k_show)
        st.subheader("Top pairwise similarities")
        _html_two_row_table("Pair","Similarity",
                            [_pair_only_fluor(a,b) for _,a,b in tops],
                            [val for val,_,_ in tops],
                            color_second_row=True, color_thresh=0.9, fmt2=True)

        # Spectra viewer (use absolute effective spectra /B → preserves relative brightness)
        st.subheader("Spectra viewer")
        colors = _ensure_colors(len(labels_sel))
        fig = go.Figure()
        for t in range(len(labels_sel)):
            y = E_raw_sel[:, t] / (B + 1e-12)
            fig.add_trace(go.Scatter(
                x=wl, y=y, mode="lines", name=labels_sel[t],
                line=dict(color=_rgb01_to_plotly(colors[t]), width=2)
            ))
        fig.update_layout(
            xaxis_title="Wavelength (nm)",
            yaxis_title="Normalized intensity (relative to B)",
            yaxis=dict(range=[0,1.05])
        )
        st.plotly_chart(fig, use_container_width=True)

        # Optional simulation: Predicted uses absolute spectra + global scaling → show brightness differences
        if st.checkbox("Run rod simulation + NLS (heavier)", value=False):
            C = 23
            chan = 494.0 + 8.9*np.arange(C)
            E = cached_interpolate_E_on_channels(wl, E_raw_sel/(B+1e-12), chan)
            Atrue, Ahat, rmse = simulate_rods_and_unmix(E, H=160, W=160, rods_per=3)
            imgs = [("True (composite)", (colorize_composite(Atrue, colors)*255).astype(np.uint8))]
            names = [s.split(" – ",1)[1] for s in labels_sel]
            for r, name in enumerate(names):
                rgb = (colorize_single(Ahat[:,:,r], colors[r])*255).astype(np.uint8)
                imgs.append((f"NLS ({name})", rgb))
            cs = st.columns(len(imgs))
            for c,(title,im) in zip(cs, imgs):
                c.image(im, use_container_width=True); c.caption(title)
            st.caption(f"Overall RMSE: {rmse:.4f}")

# -------------------- Execute --------------------
if __name__ == "__main__":
    run(groups, mode, laser_strategy, laser_list)

