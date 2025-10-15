# app.py
import streamlit as st
import numpy as np
import plotly.graph_objects as go

from utils import (
    load_dyes_yaml,
    load_probe_fluor_map,
    build_emission_only_matrix,
    build_effective_with_lasers,
    derive_powers_simultaneous,
    derive_powers_separate,
    solve_lexicographic,
    cosine_similarity_matrix,
    top_k_pairwise,
)

st.set_page_config(page_title="Choose Fluorophore", layout="wide")

DYES_YAML = "data/dyes.yaml"
PROBE_MAP_YAML = "data/probe_fluor_map.yaml"

# ---------- Load data ----------
wl, dye_db = load_dyes_yaml(DYES_YAML)
probe_map = load_probe_fluor_map(PROBE_MAP_YAML)  # {probe: [fluor,...]}

# ---------- Sidebar ----------
st.sidebar.header("Configuration")

mode = st.sidebar.radio(
    "Mode",
    options=("Emission only", "Emission + Excitation + Brightness"),
    help=(
        "Emission only: use emission spectra only (each emission first peak-normalized, "
        "then cosine-based optimization).\n\n"
        "Emission + Excitation + Brightness: build effective spectra with lasers using "
        "excitation · QY · EC, and optimize on cosine of those effective spectra."
    ),
)

laser_strategy = None
laser_list = []
if mode == "Emission + Excitation + Brightness":
    laser_strategy = st.sidebar.radio(
        "Laser usage", options=("Simultaneous", "Separate"),
        help="Simultaneous: cumulative-by-segment leveling to a common B. Separate: each laser scaled to a common peak."
    )
    preset = st.sidebar.radio(
        "Lasers", options=("488/561/639 (preset)", "Custom"),
        help="Use preset or define your wavelengths."
    )
    if preset == "488/561/639 (preset)":
        laser_list = [488, 561, 639]
        st.sidebar.caption("Using lasers: 488, 561, 639 nm")
    else:
        n = st.sidebar.number_input("Number of lasers", 1, 8, 3, 1)
        cols = st.sidebar.columns(2)
        lasers = []
        for i in range(n):
            lam = cols[i % 2].number_input(
                f"Laser {i+1} (nm)", int(wl.min()), int(wl.max()),
                [488, 561, 639][i] if i < 3 else int(wl.min()), 1
            )
            lasers.append(int(lam))
        laser_list = lasers

k_show = st.sidebar.slider(
    "Show top-K largest pairwise similarities",
    min_value=5, max_value=50, value=10, step=1,
)

# ---------- Helpers ----------
def desired_lexi_levels(idx_groups, cap=10):
    """Use min(10, number of pairwise terms C(G,2))."""
    G = len(idx_groups)
    return min(cap, (G * (G - 1)) // 2)

def effective_levels(levels_desired):
    """utils.solve_lexicographic 目前实现到第2层，这里保护一下。"""
    return min(levels_desired, 2)

def html_two_row_table(row0_label, row1_label, row0_vals, row1_vals,
                       color_second_row=False, color_thresh=0.9,
                       format_second_row=False):
    """
    Render a 2-row horizontal table (no headers/index).
    Leftmost cell is the label text (not bold).
    """
    def esc(x):
        return (str(x)
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;"))

    # Row 0
    cells0 = "".join(
        f"<td style='padding:6px 10px;border:1px solid #ddd;'>{esc(v)}</td>"
        for v in row0_vals
    )
    tds0 = (
        f"<td style='padding:6px 10px;border:1px solid #ddd;white-space:nowrap;'>{esc(row0_label)}</td>"
        f"{cells0}"
    )

    # Row 1
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
    fa = a.split(" – ", 1)[1]
    fb = b.split(" – ", 1)[1]
    return f"{fa} vs {fb}"

# ---------- Main ----------
st.title("Fluorophore Selection for Multiplexed Imaging")

# Probe picker（不折叠）
all_probes = sorted(probe_map.keys())
st.subheader("Pick probes to optimize")
picked = st.multiselect("Probes", options=all_probes)

if not picked:
    st.info("Select at least one probe to proceed.")
    st.stop()

# Build groups dict in the chosen order (过滤无效候选)
groups = {}
for p in picked:
    cands = [f for f in probe_map.get(p, []) if f in dye_db]
    if cands:
        groups[p] = cands
if not groups:
    st.error("No valid candidates with spectra in dyes.yaml for the selected probes.")
    st.stop()

# ---------- Optimization & Display ----------
if mode == "Emission only":
    E_norm, labels_pair, idx_groups = build_emission_only_matrix(wl, dye_db, groups)
    if E_norm.shape[1] == 0:
        st.error("No spectra available for optimization.")
        st.stop()

    levels_eff = effective_levels(desired_lexi_levels(idx_groups))
    sel_idx, layer_vals = solve_lexicographic(
        E_norm, idx_groups, labels_pair, levels=levels_eff, enforce_unique=True
    )

    # Selected Fluorophores
    sel_pairs = [labels_pair[j] for j in sel_idx]  # "Probe – Fluor"
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

    st.subheader("Spectra viewer")
    # Spectra viewer (0–1 ticks)
    fig = go.Figure()
    for j in sel_idx:
        y = E_norm[:, j]
        y = y / (np.max(y) + 1e-12)
        fig.add_trace(go.Scatter(x=wl, y=y, mode="lines", name=labels_pair[j]))
    fig.update_layout(
        title=None,
        xaxis_title="Wavelength (nm)",
        yaxis_title="Normalized intensity",
        yaxis=dict(range=[0, 1.05],
                   tickmode="array",
                   tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                   ticktext=["0", "0.2", "0.4", "0.6", "0.8", "1"])
    )
    st.plotly_chart(fig, use_container_width=True)

else:
    if not laser_list:
        st.error("Please specify laser wavelengths.")
        st.stop()

    # First pass (A) on emission-only for power calibration
    E0_norm, labels0, idx0 = build_emission_only_matrix(wl, dye_db, groups)
    levels_eff0 = effective_levels(desired_lexi_levels(idx0))
    sel0, _ = solve_lexicographic(E0_norm, idx0, labels0, levels=levels_eff0, enforce_unique=True)
    A_labels = [labels0[j] for j in sel0]

    # Powers and B
    if laser_strategy == "Simultaneous":
        powers, B = derive_powers_simultaneous(wl, dye_db, A_labels, laser_list)
    else:
        powers, B = derive_powers_separate(wl, dye_db, A_labels, laser_list)

    # Effective spectra (all candidates)
    E_raw_all, E_norm_all, labels_all, idx_groups_all = build_effective_with_lasers(
        wl, dye_db, groups, laser_list, laser_strategy, powers
    )

    # Optimization on effective spectra
    levels_eff = effective_levels(desired_lexi_levels(idx_groups_all))
    sel_idx, layer_vals = solve_lexicographic(
        E_norm_all, idx_groups_all, labels_all, levels=levels_eff, enforce_unique=True
    )

    # Selected Fluorophores
    sel_pairs = [labels_all[j] for j in sel_idx]  # "Probe – Fluor"
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
                       lam_sorted, [float(f"{v:.6g}") for v in prel],
                       color_second_row=False, format_second_row=False)

    # Top pairwise similarities
    S = cosine_similarity_matrix(E_norm_all[:, sel_idx])
    sub_labels = [labels_all[j] for j in sel_idx]
    tops = top_k_pairwise(S, sub_labels, k=k_show)
    pairs = [only_fluor_pair(a, b) for _, a, b in tops]
    sims = [val for val, _, _ in tops]
    st.subheader("Top pairwise similarities")
    html_two_row_table("Pair", "Similarity", pairs, sims,
                       color_second_row=True, color_thresh=0.9, format_second_row=True)

# ======== Spectra viewer ========
st.subheader("Spectra viewer")
fig = go.Figure()

if laser_strategy == "Separate":
    lam_sorted = list(sorted(laser_list))
    L = len(lam_sorted)
    W = len(wl)

    gap = 12.0  # 段间视觉间隙（坐标单位）
    wl_max_vis = float(min(1000.0, wl[-1]))  # 可视范围上限

    # 每段“可视宽度”= min(1000, wl[-1]) - laser_nm（若为负则置 0）
    seg_widths = [max(0.0, wl_max_vis - float(l)) for l in lam_sorted]

    # offsets：用 seg_widths 累加（与曲线长度完全一致）
    offsets = []
    acc = 0.0
    for wseg in seg_widths:
        offsets.append(acc)
        acc += wseg + gap

    # 画曲线：逐段切 wl，再做位移；y 用拼接块同样切 mask，二者长度一致
    for j in sel_idx:
        xs_cat, ys_cat = [], []
        for i, l in enumerate(lam_sorted):
            if seg_widths[i] <= 0:
                continue
            off = offsets[i]
            # 该段可视 wl 片段：laser_nm .. min(1000, wl[-1])
            mask = (wl >= l) & (wl <= wl_max_vis)
            wl_seg = wl[mask]
            # 对应的 y 从第 i 个拼接块取再按相同 mask 切
            block = E_raw_all[i * W:(i + 1) * W, j] / (B + 1e-12)
            y_seg = block[mask]
            # 平移到自定义拼接轴
            xs_cat.append(wl_seg + off)
            ys_cat.append(y_seg)
        if xs_cat:
            x_concat = np.concatenate(xs_cat)
            y_concat = np.concatenate(ys_cat)
            fig.add_trace(go.Scatter(x=x_concat, y=y_concat, mode="lines", name=labels_all[j]))

    # 白色虚线分隔：画在每段右边界（= offset + seg_width）
    for i in range(L - 1):
        if seg_widths[i] <= 0:
            continue
        sep_x = offsets[i] + seg_widths[i]
        fig.add_shape(
            type="line",
            x0=sep_x, x1=sep_x,
            y0=0, y1=1, yref="paper", xref="x",
            line=dict(color="white", width=2, dash="dash"),
            layer="above"
        )

    # 段标题：放在段中点；交错高度 + 轻微 xshift，减少重叠
    for i, l in enumerate(lam_sorted):
        if seg_widths[i] <= 0:
            continue
        midx = offsets[i] + seg_widths[i] / 2.0
        fig.add_annotation(
            x=midx, xref="x",
            y=1.12 if (i % 2 == 0) else 1.06, yref="paper",
            text=f"{int(l)} nm",
            showarrow=False,
            font=dict(size=12),
            align="center",
            yanchor="bottom",
            xshift=(-12 if (i % 2 == 0) else 12)
        )

    # x 轴刻度：每段仅 1 个（中点），文本“laser–upper”
    tick_positions = [offsets[i] + seg_widths[i] / 2.0 for i in range(L) if seg_widths[i] > 0]
    tick_texts = [f"{int(l)}–{int(wl_max_vis)} nm" for i, l in enumerate(lam_sorted) if seg_widths[i] > 0]

    fig.update_layout(
        title_text="",  # 防止 'undefined'
        xaxis_title="Wavelength (nm)",
        yaxis_title="Normalized intensity",
        xaxis=dict(
            tickmode="array",
            tickvals=tick_positions,
            ticktext=tick_texts,
            ticks="outside",
            automargin=True
        ),
        yaxis=dict(
            range=[0, 1.05],
            tickmode="array",
            tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
            ticktext=["0", "0.2", "0.4", "0.6", "0.8", "1"]
        ),
        margin=dict(t=90)
    )

else:
    # ------- Simultaneous：保持原样 -------
    for j in sel_idx:
        y = E_raw_all[:, j] / (B + 1e-12)
        fig.add_trace(go.Scatter(x=wl, y=y, mode="lines", name=labels_all[j]))
    fig.update_layout(
        title_text="",
        xaxis_title="Wavelength (nm)",
        yaxis_title="Normalized intensity",
        yaxis=dict(
            range=[0, 1.05],
            tickmode="array",
            tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
            ticktext=["0", "0.2", "0.4", "0.6", "0.8", "1"]
        )
    )

st.plotly_chart(fig, use_container_width=True)

st.plotly_chart(fig, use_container_width=True)




