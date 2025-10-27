# app.py — minimal pipeline per your spec
# Flow: A (max=1) -> E@8.9nm -> scale to 255 -> Poisson -> NLS with same E -> global-normalize Ahat -> display

import numpy as np
import streamlit as st
import plotly.graph_objects as go

from utils import load_dyes_yaml  # 只用这个：读取 data/dyes.yaml -> (wl, dye_db)

st.set_page_config(page_title="Spectral Unmix Demo", layout="wide")

# -------------------- Load data --------------------
DYES_YAML = "data/dyes.yaml"
wl, dye_db = load_dyes_yaml(DYES_YAML)  # wl: (W,), dye_db: {name: {"emission": (W,)...}}

ALL_DYES = sorted([k for k,v in dye_db.items() if isinstance(v, dict) and "emission" in v])

# -------------------- Sidebar Config --------------------
st.sidebar.header("Setup")

picked = st.sidebar.multiselect("Fluorophores", ALL_DYES, default=ALL_DYES[:4])
if not picked:
    st.info("请选择至少一个染料（左侧）。")
    st.stop()

# Imaging channels (8.9 nm sampling)
C = st.sidebar.number_input("Number of channels (C)", 4, 64, 23, 1)
start_nm = st.sidebar.number_input("Channel start (nm)", int(max(350, wl.min())), int(min(800, wl.max())), 494, 1)
step_nm = st.sidebar.number_input("Channel step (nm)", 1.0, 20.0, 8.9, 0.1)
chan_centers = start_nm + step_nm * np.arange(int(C))

# Image shape & rods
H = st.sidebar.number_input("Image height", 64, 512, 160, 8)
W = st.sidebar.number_input("Image width", 64, 512, 160, 8)
rods_per = st.sidebar.slider("Rods per dye", 1, 6, 3, 1)
Lmin = st.sidebar.slider("Rod length min (px)", 8, 80, 24, 1)
Lmax = st.sidebar.slider("Rod length max (px)", 10, 120, 48, 1)
Wmin_px = st.sidebar.slider("Rod width min (px)", 4, 60, 12, 1)
Wmax_px = st.sidebar.slider("Rod width max (px)", 6, 80, 20, 1)

# -------------------- Helper: colors, interp, rods --------------------
DEFAULT_COLORS = np.array([
    [0.95, 0.25, 0.25], [0.25, 0.65, 0.95],
    [0.25, 0.85, 0.35], [0.90, 0.70, 0.20],
    [0.80, 0.40, 0.80], [0.25, 0.80, 0.80],
    [0.85, 0.50, 0.35], [0.60, 0.60, 0.60]
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

def interpolate_E_on_channels(wl, spectra_cols, chan_centers_nm):
    """ wl:(W,), spectra_cols:(W,R) -> E:(C,R) @ chan_centers """
    spectra_cols = np.asarray(spectra_cols, dtype=float)
    if spectra_cols.ndim == 1:
        spectra_cols = spectra_cols[:, None]
    Cn = len(chan_centers_nm); W, R = spectra_cols.shape
    E = np.zeros((Cn, R), dtype=float)
    for j in range(R):
        y = spectra_cols[:, j]
        E[:, j] = np.interp(chan_centers_nm, wl, y, left=float(y[0]), right=float(y[-1]))
    return np.nan_to_num(E, nan=0.0, posinf=0.0, neginf=0.0)

def _capsule_profile(H,W,cx,cy,length,width,theta):
    yy,xx = np.mgrid[0:H,0:W].astype(float)
    X=xx-cx; Y=yy-cy
    c,s = np.cos(theta), np.sin(theta)
    xp =  c*X + s*Y
    yp = -s*X + c*Y
    half_L = 0.5*length; r = 0.5*width
    rect = (np.abs(xp)<=half_L) & (np.abs(yp)<=r)
    val = np.zeros((H,W))
    if np.any(rect): val[rect] = 1 - np.abs(yp[rect])/(r+1e-12)
    for side in (-1,1):
        rho = np.sqrt((xp+side*half_L)**2 + yp**2)
        cap = rho <= r
        if np.any(cap): val[cap] = np.maximum(val[cap], 1 - rho[cap]/(r+1e-12))
    return np.clip(val,0,1), val>0

def place_rods_scene(H,W,R,rods_per=3,rng=None,Lmin=24,Lmax=48,Wmin=12,Wmax=20):
    rng = np.random.default_rng() if rng is None else rng
    Atrue = np.zeros((H,W,R), dtype=np.float32)
    occ = np.zeros((H,W), dtype=bool)
    for r in range(R):
        placed = 0; tries = 0
        while placed < rods_per and tries < 300:
            tries += 1
            length = int(rng.integers(Lmin, Lmax+1))
            width  = int(rng.integers(Wmin, Wmax+1))
            theta  = float(rng.uniform(0, np.pi))
            margin = 6 + int(max(length,width)//2)
            if W - 2*margin <= 2 or H - 2*margin <= 2: break
            cx = int(rng.integers(margin, W-margin))
            cy = int(rng.integers(margin, H-margin))
            prof, mask = _capsule_profile(H,W,cx,cy,length,width,theta)
            if not np.any(mask): continue
            if np.any(occ & mask): continue
            m = float(prof.max())
            if m > 0: prof /= m
            Atrue[:,:,r] = np.maximum(Atrue[:,:,r], prof.astype(np.float32))
            occ |= mask
            placed += 1
    return np.clip(Atrue, 0, 1)

# -------------------- NLS (no internal max-normalization) --------------------
def nls_unmix(Timg, E, iters=2000, tol=1e-6):
    """Timg:(H,W,C), E:(C,R) -> Ahat:(H,W,R)"""
    H, W, C = Timg.shape
    E = np.asarray(E, dtype=np.float32)
    if E.ndim != 2 or E.shape[0] != C:
        raise ValueError(f"E shape {E.shape} mismatch with Timg channels {C}")
    M = Timg.reshape(-1, C).astype(np.float32, copy=False)

    # pixel-wise normalization
    scale = np.sqrt(np.mean(M**2, axis=1, keepdims=True)); scale[scale<=0]=1.0
    Mn = M / scale

    # LS init: A0 = Mn @ E @ (E^T E)^-1
    EtE = E.T @ E
    A = Mn @ E @ np.linalg.pinv(EtE)
    A[A < 0] = 0

    # MU iterations
    for _ in range(iters):
        numer = Mn @ E
        denom = (A @ EtE) + 1e-12
        A *= numer / denom
        if np.max(numer / (denom + 1e-12)) < 1 + tol:
            break

    A *= scale  # restore per-pixel scale
    A[A < 0] = 0
    return A.reshape(H, W, E.shape[1])

# -------------------- UI: spectra viewer --------------------
R = len(picked)
colors = _ensure_colors(R)

st.title("Spectra → Forward (255) → Poisson → NLS → Global-normalize(Â)")

# gather emission spectra (as stored)
W = len(wl)
S = np.zeros((W, R), dtype=float)
for j, name in enumerate(picked):
    S[:, j] = np.asarray(dye_db[name]["emission"], dtype=float)

# Spectra viewer (normalize each trace to [0,1] for可视化，但 unmix/forward 用按 8.9nm 采样后的数值)
st.subheader("Spectra viewer")
fig = go.Figure()
for j, name in enumerate(picked):
    y = S[:, j]
    y_plot = y / (np.max(y) + 1e-12)
    fig.add_trace(go.Scatter(x=wl, y=y_plot, mode="lines", name=name,
                             line=dict(color=_rgb01_to_plotly(colors[j]), width=2)))
fig.update_layout(xaxis_title="Wavelength (nm)", yaxis_title="Normalized intensity", yaxis=dict(range=[0,1.05]))
st.plotly_chart(fig, use_container_width=True)

# 8.9 nm sampling for both forward and unmix
E = interpolate_E_on_channels(wl, S, chan_centers)  # (C,R)

# -------------------- Run button --------------------
# -------------------- Run button --------------------
if st.button("Run simulation + unmix"):
    rng = np.random.default_rng(2025)

    # 1) Atrue (per-dye abundance maps), each in [0,1]
    Atrue = place_rods_scene(
        H, W, R,
        rods_per=rods_per,             # ← 修正这里
        rng=rng,
        Lmin=Lmin, Lmax=Lmax,
        Wmin=Wmin_px, Wmax=Wmax_px
    )

    # 2) Forward: Tclean = Atrue ⊗ E
    Tclean = np.zeros((H, W, C), dtype=float)
    for c in range(C):
        # sum_r Atrue[:,:,r] * E[c,r]
        Tclean[:, :, c] = np.tensordot(Atrue, E[c, :], axes=([2], [0]))

    # 3) Scale so max=255
    Tmax = float(np.max(Tclean))
    if Tmax <= 0:
        st.error("Forward image is all zeros; try adjusting rods or spectra.")
        st.stop()
    s = 255.0 / Tmax
    lam = Tclean * s

    # 4) Poisson noise (scalar-wise => per-pixel-per-channel)
    lam = np.nan_to_num(lam, nan=0.0, posinf=1e6, neginf=0.0)
    lam = np.clip(lam, 0.0, 1e6)
    Y = rng.poisson(lam).astype(float) / 255.0  # back to ~[0,1] scale

    # 5) Unmix with the same E (8.9 nm sampled)
    Ahat = nls_unmix(Y, E, iters=1500, tol=1e-6)

    # 6) Global normalize ALL Ahat maps together (single global max)
    Amax = float(np.max(Ahat))
    Ahat_vis = (Ahat / (Amax + 1e-12)) if Amax > 0 else np.zeros_like(Ahat)

    # -------------------- Display --------------------
    st.subheader("Abundance maps (global-normalized across all dyes)")

    # True composite（把 Atrue 也做一次全局归一后合成）
    Atrue_max = float(np.max(Atrue))
    Atrue_vis = Atrue / (Atrue_max + 1e-12)

    def colorize_stack(A, cols):
        H_, W_, R_ = A.shape
        rgb = np.zeros((H_, W_, 3), dtype=float)
        for r in range(R_):
            z = np.clip(A[:, :, r], 0.0, 1.0)
            rgb += z[:, :, None] * cols[r][None, None, :]
        m = float(rgb.max())
        if m > 0:
            rgb /= m
        return (rgb * 255).astype(np.uint8)

    tiles = []
    tiles.append(("True", colorize_stack(Atrue_vis, colors)))

    # 单通道展示：直接用 Ahat_vis（已是“整体 normalize”后的值），不要再对每通道各自拉满
    for r, name in enumerate(picked):
        z = np.clip(Ahat_vis[:, :, r], 0.0, 1.0)
        rgb = (z[:, :, None] * colors[r][None, None, :])  # 不再二次归一
        tiles.append((name, (rgb * 255).astype(np.uint8)))

    cols_ui = st.columns(len(tiles))
    for c_ui, (title, im) in zip(cols_ui, tiles):
        c_ui.image(im, use_container_width=True)
        c_ui.caption(title)

else:
    st.info("点击上面的 **Run simulation + unmix** 按钮执行流程。")

