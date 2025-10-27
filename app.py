# app.py (minimal + optimized)
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
    help=("Emission: 仅发射谱，峰值归一。\n"
          "Predicted: 激发·QY·EC + 激光方案生成有效光谱。")
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

# -------------------- Helpers --------------------
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
    r, g, b = (int(255*col[0]), int(255*col[1]), int(255*col[2]))
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
                vv = float(v); style += f"color:{'red' if vv>color_thresh else 'green'};"
            except: pass
        tds1_list.append(f"<td style='{style}'>{fmtv(v)}</td>")
    tds1 = f"<td style='padding:6px 10px;border:1px solid #ddd;white-space:nowrap;'>{esc(row1_label)}</td>{''.join(tds1_list)}"
    st.markdown(f"""
    <div style="overflow-x:auto;">
      <table style="border-collapse:collapse;width:100%;table-layout:auto;">
        <tbody>
          <tr>{tds0}</tr>
          <tr>{tds1}</tr>
        </tbody>
      </table>
    </div>
    """, unsafe_allow_html=True)

def _groups_from_labels(labels, use_pool):
    g = {}
    if use_pool:
        g["Pool"] = [s.split(" – ",1)[1] if " – " in s else s for s in labels]
    else:
        for s in labels:
            p, f = s.split(" – ", 1)
            g.setdefault(p, []).append(f)
    return g

# -------------------- Cache wrappers --------------------
@st.cache_data(show_spinner=False)
def cached_build_effective_with_lasers(wl, dye_db, groups, laser_list, laser_strategy, powers):
    groups_key = json.dumps({k: sorted(v) for k, v in sorted(groups.items())}, ensure_ascii=False)
    _ = (tuple(sorted(laser_list)), laser_strategy, tuple(np.asarray(powers,float)) if powers is not None else None, groups_key)
    return build_effective_with_lasers(wl, dye_db, groups, laser_list, laser_strategy, powers)

@st.cache_data(show_spinner=False)
@st.cache_data(show_spinner=False)
def cached_interpolate_E_on_channels(wl, spectra_cols, chan_centers_nm):
    # 强制把输入转成 float 的 2D 数组（防止 object/ragged）
    spectra_cols = np.asarray(spectra_cols, dtype=float)
    if spectra_cols.ndim == 1:
        spectra_cols = spectra_cols[:, None]  # (W,) -> (W,1)
    W, N = spectra_cols.shape
    E = np.zeros((len(chan_centers_nm), N), dtype=float)
    for j in range(N):
        y = spectra_cols[:, j]
        E[:, j] = np.interp(chan_centers_nm, wl, y, left=float(y[0]), right=float(y[-1]))
    # 去 NaN/Inf
    if not np.isfinite(E).all():
        E = np.nan_to_num(E, nan=0.0, posinf=0.0, neginf=0.0)
    return E


# -------------------- Imaging & NLS --------------------
def add_poisson_noise(Tclean, peak=255, mode="per_channel", rng=None):
    rng = np.random.default_rng() if rng is None else rng
    Tclean = np.clip(Tclean, 0.0, None)
    H, W, C = Tclean.shape
    if mode == "global":
        m = float(np.max(Tclean))
        if not np.isfinite(m) or m <= 0.0:
            return np.zeros_like(Tclean)
        lam = Tclean * (peak / m)
        lam = np.nan_to_num(lam, nan=0.0, posinf=float(peak), neginf=0.0)
        lam = np.clip(lam, 0.0, 1e6)
        return rng.poisson(lam).astype(float) / peak
    # per-channel
    out = np.empty_like(Tclean, dtype=float)
    for c in range(C):
        img = Tclean[:,:,c]; m = float(np.max(img))
        if not np.isfinite(m) or m <= 0.0:
            out[:,:,c] = 0.0; continue
        lam = img * (peak / m)
        lam = np.nan_to_num(lam, nan=0.0, posinf=float(peak), neginf=0.0)
        lam = np.clip(lam, 0.0, 1e6)
        out[:,:,c] = rng.poisson(lam).astype(float) / peak
    return out
def _ensure_2d_float_matrix(E, target_cols=None, dtype=np.float32):
    """
    将任意 E 转成规则 2D 浮点矩阵，去除 NaN/Inf，并在必要时尝试转置以匹配列数=target_cols。
    参数
      E: 任意可能是 list/tuple/ndarray（甚至 ragged）的对象
      target_cols: 目标列数（通常=像素向量的通道数 C = M.shape[1]）
    返回
      E2: np.ndarray, dtype=dtype, shape=(*, *)
    """
    # 先尝试直接转成 float 数组
    try:
        A = np.asarray(E, dtype=dtype)
    except Exception:
        # 如果失败，先以 object 读入，再逐列堆叠
        obj = np.asarray(E, dtype=object)
        # 处理 1 维 ragged （每个元素是一个列向量）
        if obj.ndim == 1 and obj.size > 0:
            cols = [np.asarray(v, dtype=dtype).ravel() for v in obj]
            # 对齐长度（截短到最短）
            minlen = min(len(c) for c in cols)
            if minlen == 0:
                raise ValueError("Empty columns in E.")
            A = np.stack([c[:minlen] for c in cols], axis=1)  # (W, N)
        else:
            # 尝试逐行堆叠
            rows = []
            for v in obj.reshape(-1):
                vv = np.asarray(v, dtype=dtype).ravel()
                if vv.size == 0:
                    continue
                rows.append(vv)
            if not rows:
                raise ValueError("E cannot be coerced to non-empty 2D matrix.")
            minlen = min(len(r) for r in rows)
            A = np.stack([r[:minlen] for r in rows], axis=0)

    if A.ndim == 1:
        A = A[:, None]

    # 去 NaN/Inf
    if not np.isfinite(A).all():
        A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)

    # 如果目标列数提供了，但不匹配，尝试一次转置修正
    if target_cols is not None:
        if A.shape[1] != target_cols and A.shape[0] == target_cols:
            A = A.T  # 只转置一次
    return A.astype(dtype, copy=False)

def nls_unmix(Timg, E, EtE=None, iters=3000, tol=1e-6, verbose=False, dtype=np.float32):
    # 统一处理 Timg -> M
    if Timg.ndim == 3:
        H, W, C = Timg.shape
        M = Timg.reshape(-1, C).astype(dtype, copy=False)
        back = (H, W)
    else:
        M = np.asarray(Timg, dtype=dtype, copy=False)
        back = None
        if M.ndim != 2:
            raise ValueError(f"nls_unmix: Timg must be (H,W,C) or (Npix,C), got {M.shape}")

    C = M.shape[1]  # 通道数

    # 强力整形 E，确保是 (C, R)
    E = _ensure_2d_float_matrix(E, target_cols=C, dtype=dtype)
    if E.ndim != 2 or E.shape[0] != C:
        raise ValueError(f"nls_unmix: E bad shape {E.shape}; expected (C={C}, R).")

    # 像素归一
    scale = np.sqrt(np.mean(M**2, axis=1, keepdims=True))
    scale[scale <= 0] = 1.0
    Mn = M / scale

    # 预备 (E^T E)
    if EtE is None:
        EtE = (E.T @ E).astype(dtype, copy=False)
    else:
        EtE = np.asarray(EtE, dtype=dtype, copy=False)

    # 伪逆初始化 A0 = Mn pinv(E)
    pinv_EtE = np.linalg.pinv(EtE).astype(dtype, copy=False)
    A = (Mn @ (pinv_EtE @ E.T).T)
    A[A < 0] = 0

    # 乘法更新
    fit = np.linalg.norm(Mn - A @ E.T, 'fro')
    for _ in range(iters):
        fit_old = fit
        numer = Mn @ E
        denom = (A @ EtE) + 1e-12
        A *= numer / denom
        fit = np.linalg.norm(Mn - A @ E.T, 'fro')
        if abs(fit_old - fit) / (fit_old + 1e-12) < tol or np.isnan(fit):
            break

    # 复原 & 全局标准化到 [0,1]
    A *= scale
    mA = float(np.max(A))
    if mA > 0:
        A /= mA

    if back is not None:
        H, W = back
        return A.reshape(H, W, E.shape[1])
    return A



def simulate_rods_and_unmix(E, H=192, W=192, rods_per=3, rng=None, noise_mode="per_channel"):
    rng = np.random.default_rng() if rng is None else rng

    # 兜底：把 E 强制成 2D float；若列数=R，后续按 (C,R) 使用
    E = _ensure_2d_float_matrix(E, dtype=np.float32)
    if E.ndim != 2:
        raise ValueError(f"simulate_rods_and_unmix: E must be 2D, got {E.shape}")
    C, R = E.shape


    # rods GT (简单快速：随机不重叠小圆片 + 线性渐变核)
    Atrue = np.zeros((H, W, R), dtype=np.float32)
    occ = np.zeros((H, W), dtype=bool)
    for r in range(R):
        placed = 0; tries = 0
        while placed < rods_per and tries < 200:
            tries += 1
            cy = int(rng.integers(12, H-12)); cx = int(rng.integers(12, W-12))
            rad = int(rng.integers(6, 10))
            yy, xx = np.mgrid[0:H, 0:W]
            mask = (xx-cx)**2 + (yy-cy)**2 <= rad**2
            if np.any(occ & mask): continue
            occ |= mask
            dist = np.sqrt((xx-cx)**2 + (yy-cy)**2) / (rad+1e-6)
            prof = np.clip(1.0 - dist, 0.0, 1.0)
            Atrue[:,:,r] = np.maximum(Atrue[:,:,r], prof.astype(np.float32))
            placed += 1

    # forward render
    Tclean = np.zeros((H, W, C), dtype=np.float32)
    for c in range(C):
        Tclean[:,:,c] = np.tensordot(Atrue, E[c,:], axes=([2],[0]))

    # noise
    Tnoisy = add_poisson_noise(Tclean, peak=255, mode=noise_mode, rng=rng)

    # unmix
    EtE = E.T @ E
    Ahat = nls_unmix(Tnoisy, E, EtE=EtE, iters=2500, tol=5e-6)

    rmse_all = float(np.sqrt(np.mean((Ahat - Atrue) ** 2)))
    return Atrue, Ahat, rmse_all

def colorize_single(A_r, color):
    z = np.clip(A_r, 0.0, 1.0)
    m = float(z.max()); 
    if m > 0: z = z / m
    return (z[:,:,None] * np.asarray(color)[None,None,:])

def colorize_composite(A, colors):
    H, W, R = A.shape
    rgb = np.zeros((H, W, 3), dtype=float)
    for r in range(R):
        rgb += colorize_single(A[:,:,r], colors[r])
    m = float(rgb.max()); 
    if m > 0: rgb /= m
    return rgb

# -------------------- Main --------------------
st.title("Fluorophore Selection for Multiplexed Imaging")

use_pool = (source_mode == "From readout pool")
if use_pool:
    if not readout_pool:
        st.info("Readout pool not found or empty (data/readout_fluorophores.yaml).")
        st.stop()
    max_n = len(readout_pool)
    N_pick = st.number_input("How many fluorophores to pick", 1, max_n, min(4, max_n), 1)
    groups = {"Pool": readout_pool[:]}
else:
    all_probes = sorted(probe_map.keys())
    picked = st.multiselect("Probes", options=all_probes)
    if not picked:
        st.info("Select at least one probe to proceed.")
        st.stop()
    groups = {}
    for p in picked:
        cands = [f for f in probe_map.get(p, []) if f in dye_db]
        if cands: groups[p] = cands
    if not groups:
        st.error("No valid candidates with spectra in dyes.yaml.")
        st.stop()
    N_pick = None

# -------------------- Runner --------------------
def run(groups, mode, laser_strategy, laser_list):
    required_count = (N_pick if use_pool else None)

    if mode == "Emission spectra":
        # 1) emission-only selection
        E_norm, labels_pair, idx_groups = build_emission_only_matrix(wl, dye_db, groups)
        if E_norm.shape[1] == 0:
            st.error("No spectra available.")
            st.stop()

        K = min(10, (E_norm.shape[1]*(E_norm.shape[1]-1))//2)
        sel_idx, _ = solve_lexicographic_k(E_norm, idx_groups, labels_pair,
                                           levels=K, enforce_unique=True,
                                           required_count=required_count)

        # Selected
        if use_pool:
            chosen = [labels_pair[j].split(" – ",1)[1] for j in sel_idx]
            st.subheader("Selected Fluorophores")
            _html_two_row_table("Slot", "Fluorophore",
                                [f"Slot {i+1}" for i in range(len(chosen))],
                                chosen)
        else:
            sel_pairs = [labels_pair[j] for j in sel_idx]
            st.subheader("Selected Fluorophores")
            _html_two_row_table("Probe", "Fluorophore",
                                [s.split(" – ",1)[0] for s in sel_pairs],
                                [s.split(" – ",1)[1] for s in sel_pairs])

        # Pairwise
        S = cosine_similarity_matrix(E_norm[:, sel_idx])
        sub_labels = [labels_pair[j] for j in sel_idx]
        tops = top_k_pairwise(S, sub_labels, k=k_show)
        st.subheader("Top pairwise similarities")
        _html_two_row_table("Pair", "Similarity",
                            [_pair_only_fluor(a,b) for _,a,b in tops],
                            [val for val,_,_ in tops],
                            color_second_row=True, color_thresh=0.9, fmt2=True)

        # Spectra viewer (each trace normalized to 1)
        st.subheader("Spectra viewer")
        colors = _ensure_colors(len(sel_idx))
        fig = go.Figure()
        for t, j in enumerate(sel_idx):
            y = E_norm[:, j]; y = y / (np.max(y)+1e-12)
            fig.add_trace(go.Scatter(
                x=wl, y=y, mode="lines", name=labels_pair[j],
                line=dict(color=_rgb01_to_plotly(colors[t]), width=2)
            ))
        fig.update_layout(xaxis_title="Wavelength (nm)",
                          yaxis_title="Normalized intensity",
                          yaxis=dict(range=[0,1.05]))
        st.plotly_chart(fig, use_container_width=True)

        # Optional simulation (per-channel scaling,保持原逻辑)
        run_sim = st.checkbox("Run rod simulation + NLS (heavier)", value=False)
        if run_sim:
            # channel grid
            C = 23; chan_centers = 494.0 + 8.9*np.arange(C)
            E = cached_interpolate_E_on_channels(wl, E_norm[:, sel_idx], chan_centers)
            Atrue, Ahat, rmse = simulate_rods_and_unmix(
                E=E, H=192, W=192, rods_per=3, rng=np.random.default_rng(2025),
                noise_mode="per_channel"
            )
            cols = _ensure_colors(E.shape[1])
            names = [labels_pair[j].split(" – ",1)[1] for j in sel_idx]
            imgs = [("True (composite)", (colorize_composite(Atrue, cols)*255).astype(np.uint8))]
            for r, name in enumerate(names):
                rgb = (colorize_single(Ahat[:,:,r], cols[r])*255).astype(np.uint8)
                imgs.append((f"NLS ({name})", rgb))
            cs = st.columns(len(imgs))
            for c,(title,im) in zip(cs, imgs):
                c.image(im, use_container_width=True); c.caption(title)
            st.caption(f"Overall RMSE: {rmse:.4f}")

    else:
        if not laser_list:
            st.error("Please specify laser wavelengths.")
            st.stop()

        # Round A: emission-only provisional selection
        E0_norm, labels0, idx0 = build_emission_only_matrix(wl, dye_db, groups)
        K0 = min(10, (E0_norm.shape[1]*(E0_norm.shape[1]-1))//2)
        sel0, _ = solve_lexicographic_k(E0_norm, idx0, labels0,
                                        levels=K0, enforce_unique=True,
                                        required_count=required_count)
        A_labels = [labels0[j] for j in sel0]

        # (1) calibrate on A
        if laser_strategy == "Simultaneous":
            powers_A, _ = derive_powers_simultaneous(wl, dye_db, A_labels, laser_list)
        else:
            powers_A, _ = derive_powers_separate(wl, dye_db, A_labels, laser_list)

        # first build (cached)
        E_raw_all, E_norm_all, labels_all, idx_groups_all = cached_build_effective_with_lasers(
            wl, dye_db, groups, laser_list, laser_strategy, powers_A
        )

        # final selection
        Kf = min(10, (E_norm_all.shape[1]*(E_norm_all.shape[1]-1))//2)
        sel_idx, _ = solve_lexicographic_k(E_norm_all, idx_groups_all, labels_all,
                                           levels=Kf, enforce_unique=True,
                                           required_count=required_count)
        final_labels = [labels_all[j] for j in sel_idx]

        # (2) recalibrate on final only
        if laser_strategy == "Simultaneous":
            powers, B = derive_powers_simultaneous(wl, dye_db, final_labels, laser_list)
        else:
            powers, B = derive_powers_separate(wl, dye_db, final_labels, laser_list)

        # second build (only selected subset; cached)
        small_groups = _groups_from_labels(final_labels, use_pool=use_pool)
        E_raw_sel, E_norm_sel, labels_sel, _ = cached_build_effective_with_lasers(
            wl, dye_db, small_groups, laser_list, laser_strategy, powers
        )

        # Selected table
        if use_pool:
            chosen = [s.split(" – ",1)[1] for s in labels_sel]
            st.subheader("Selected Fluorophores (with lasers)")
            _html_two_row_table("Slot", "Fluorophore",
                                [f"Slot {i+1}" for i in range(len(chosen))],
                                chosen)
        else:
            st.subheader("Selected Fluorophores (with lasers)")
            _html_two_row_table("Probe", "Fluorophore",
                                [s.split(" – ",1)[0] for s in labels_sel],
                                [s.split(" – ",1)[1] for s in labels_sel])

        # Pairwise (on normalized effective spectra)
        S = cosine_similarity_matrix(E_norm_sel)
        tops = top_k_pairwise(S, labels_sel, k=k_show)
        st.subheader("Top pairwise similarities")
        _html_two_row_table("Pair", "Similarity",
                            [_pair_only_fluor(a,b) for _,a,b in tops],
                            [val for val,_,_ in tops],
                            color_second_row=True, color_thresh=0.9, fmt2=True)

        # Spectra viewer（颜色一致；Predicted 显示绝对有效谱 E_raw/B，不对每条单独归一）
        st.subheader("Spectra viewer")
        colors = _ensure_colors(len(labels_sel))
        fig = go.Figure()
        if laser_strategy == "Separate":
            lam_sorted = list(sorted(laser_list))
            L = len(lam_sorted); Wn = len(wl); gap = 12.0
            wl_max = float(min(1000.0, wl[-1]))
            seg_w = [max(0.0, wl_max - float(l)) for l in lam_sorted]
            offs, acc = [], 0.0
            for wseg in seg_w: offs.append(acc); acc += wseg + gap
            for t in range(len(labels_sel)):
                xs, ys = [], []
                for i,l in enumerate(lam_sorted):
                    if seg_w[i] <= 0: continue
                    mask = (wl >= l) & (wl <= wl_max)
                    wl_seg = wl[mask]
                    block = E_raw_sel[i*Wn:(i+1)*Wn, t] / (B + 1e-12)
                    y_seg = block[mask]
                    xs.append(wl_seg + offs[i]); ys.append(y_seg)
                if xs:
                    fig.add_trace(go.Scatter(
                        x=np.concatenate(xs), y=np.concatenate(ys),
                        mode="lines", name=labels_sel[t],
                        line=dict(color=_rgb01_to_plotly(colors[t]), width=2)
                    ))
            fig.update_layout(xaxis_title="Wavelength (nm, concatenated)",
                              yaxis_title="Normalized intensity (relative to B)",
                              yaxis=dict(range=[0,1.05]))
        else:
            for t in range(len(labels_sel)):
                y = E_raw_sel[:, t] / (B + 1e-12)
                fig.add_trace(go.Scatter(
                    x=wl, y=y, mode="lines", name=labels_sel[t],
                    line=dict(color=_rgb01_to_plotly(colors[t]), width=2)
                ))
            fig.update_layout(xaxis_title="Wavelength (nm)",
                              yaxis_title="Normalized intensity (relative to B)",
                              yaxis=dict(range=[0,1.05]))
        st.plotly_chart(fig, use_container_width=True)

        # Optional simulation：Predicted 用绝对谱 + 全局缩放（体现亮度差）
        run_sim = st.checkbox("Run rod simulation + NLS (heavier)", value=False)
        if run_sim:
            C = 23; chan_centers = 494.0 + 8.9*np.arange(C)
            E_abs = (E_raw_sel / (B + 1e-12))  # (W or concat, R)
            E = cached_interpolate_E_on_channels(wl, E_abs, chan_centers)  # (C,R)
            Atrue, Ahat, rmse = simulate_rods_and_unmix(
                E=E, H=192, W=192, rods_per=3, rng=np.random.default_rng(2025),
                noise_mode="global"
            )
            imgs = [("True (composite)", (colorize_composite(Atrue, colors)*255).astype(np.uint8))]
            names = [s.split(" – ",1)[1] for s in labels_sel]
            for r, name in enumerate(names):
                rgb = (colorize_single(Ahat[:,:,r], colors[r])*255).astype(np.uint8)
                imgs.append((f"NLS ({name})", rgb))
            cs = st.columns(len(imgs))
            for c,(title,im) in zip(cs, imgs):
                c.image(im, use_container_width=True); c.caption(title)
            st.caption(f"Overall RMSE: {rmse:.4f}")

# run
run(groups, mode, laser_strategy, laser_list)
