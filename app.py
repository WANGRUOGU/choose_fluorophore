# app.py
import streamlit as st
import numpy as np
import pandas as pd
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

levels = st.sidebar.selectbox(
    "Lexicographic levels",
    options=[1, 2],
    index=1,
    help="Number of layers to minimize lexicographically (1=minimax only; 2=also shrink the second-largest)."
)

# ---------- Helpers ----------
def _blank_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Make visually blank (but unique) column headers using zero-width spaces."""
    df2 = df.copy()
    df2.columns = [("\u200B" * (i + 1)) for i in range(len(df2.columns))]
    return df2

def _render_two_row_table(row0_label: str, row1_label: str, row0_vals, row1_vals,
                          color_second_row: bool = False, color_thresh: float = 0.9,
                          format_second_row: bool = False):
    """
    Build a 2-row horizontal table with a labeled first column.
    - First column shows row labels (bold).
    - Optionally color the second row (values) by threshold.
    - Optionally format the second row to 3 decimals.
    """
    # Capitalize labels and bold via Styler
    row0_label = row0_label.capitalize()
    row1_label = row1_label.capitalize()

    # 第一列是行标签，其余列是值
    data = [
        [f"**{row0_label}**"] + list(row0_vals),
        [f"**{row1_label}**"] + list(row1_vals),
    ]
    df = pd.DataFrame(data)

    # 去掉列头数字（但要保持唯一）
    df = _blank_cols(df)

    sty = df.style

    from pandas import IndexSlice as idx

    # 只对第二行的“非首列”做数值格式化/着色
    cols_nonfirst = df.columns[1:]

    if format_second_row:
        sty = sty.format("{:.3f}", subset=idx[[1], cols_nonfirst])

    if color_second_row:
        def _color_row(v):
            # v 是整张表；我们只对第二行非首列着色
            s = pd.DataFrame("", index=v.index, columns=v.columns)
            for c in cols_nonfirst:
                try:
                    val = float(v.loc[1, c])
                except Exception:
                    val = 0.0
                s.loc[1, c] = "color: red" if val > color_thresh else "color: green"
            return s
        sty = sty.apply(_color_row, axis=None)

    return sty

def _pairs_only(a: str, b: str) -> str:
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

    sel_idx, layer_vals = solve_lexicographic(
        E_norm, idx_groups, labels_pair, levels=levels, enforce_unique=True
    )

    # ======== Selected Fluorophores（横向两行，首格粗体标签） ========
    sel_pairs = [labels_pair[j] for j in sel_idx]  # "Probe – Fluor"
    probes = [s.split(" – ", 1)[0] for s in sel_pairs]
    fluors = [s.split(" – ", 1)[1] for s in sel_pairs]
    st.subheader("Selected Fluorophores")
    st.dataframe(
        _render_two_row_table("Probe", "Fluorophore", probes, fluors),
        use_container_width=True
    )

    # ======== Top pairwise similarities（横向两行，>0.9 红，否则绿；去掉 “(largest first)”） ========
    S = cosine_similarity_matrix(E_norm[:, sel_idx])
    sub_labels = [labels_pair[j] for j in sel_idx]
    tops = top_k_pairwise(S, sub_labels, k=k_show)
    pairs = [_pairs_only(a, b) for _, a, b in tops]
    sims = [val for val, _, _ in tops]

    st.subheader("Top pairwise similarities")
    st.dataframe(
        _render_two_row_table("Pair", "Similarity", pairs, sims,
                              color_second_row=True, format_second_row=True),
        use_container_width=True
    )

    # ======== 光谱图：Spectra viewer；每条曲线各自 0–1 归一，显示 0–1 刻度 ========
    fig = go.Figure()
    for j in sel_idx:
        y = E_norm[:, j]
        y = y / (np.max(y) + 1e-12)  # 0–1
        lbl = labels_pair[j]
        fig.add_trace(go.Scatter(x=wl, y=y, mode="lines", name=lbl))
    fig.update_layout(
        title="Spectra viewer",
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

    # 先用 emission-only 做一轮，得到 A（用于功率标定）
    E0_norm, labels0, idx0 = build_emission_only_matrix(wl, dye_db, groups)
    sel0, _ = solve_lexicographic(E0_norm, idx0, labels0, levels=levels, enforce_unique=True)
    A_labels = [labels0[j] for j in sel0]

    # 由 A 计算功率（需返回 powers, B）
    if laser_strategy == "Simultaneous":
        powers, B = derive_powers_simultaneous(wl, dye_db, A_labels, laser_list)
    else:
        powers, B = derive_powers_separate(wl, dye_db, A_labels, laser_list)

    # 用功率构建全候选的“有效光谱”（E_raw 展示用；E_norm 用于相似度/优化）
    E_raw_all, E_norm_all, labels_all, idx_groups_all = build_effective_with_lasers(
        wl, dye_db, groups, laser_list, laser_strategy, powers
    )

    # 基于有效光谱再做层级优化
    sel_idx, layer_vals = solve_lexicographic(
        E_norm_all, idx_groups_all, labels_all, levels=levels, enforce_unique=True
    )

    # ======== Selected Fluorophores（横向两行） ========
    sel_pairs = [labels_all[j] for j in sel_idx]  # "Probe – Fluor"
    probes = [s.split(" – ", 1)[0] for s in sel_pairs]
    fluors = [s.split(" – ", 1)[1] for s in sel_pairs]
    st.subheader("Selected Fluorophores (with lasers)")
    st.dataframe(
        _render_two_row_table("Probe", "Fluorophore", probes, fluors),
        use_container_width=True
    )

    # ======== Laser powers（相对值，横向两行；首格标签加粗；去掉表头数字） ========
    lam_sorted = list(sorted(laser_list))
    p = np.array(powers, dtype=float)
    maxp = float(np.max(p)) if p.size > 0 else 1.0
    prel = (p / (maxp + 1e-12)).tolist()

    st.subheader("Laser powers (relative)")
    st.dataframe(
        _render_two_row_table("Laser (nm)", "Relative power", lam_sorted, [float(f"{v:.6g}") for v in prel]),
        use_container_width=True
    )

    # ======== 相似度 Top-K（横向两行，>0.9 红，否则绿；pair 去掉 probe；去掉 “(largest first)”) ========
    S = cosine_similarity_matrix(E_norm_all[:, sel_idx])
    sub_labels = [labels_all[j] for j in sel_idx]
    tops = top_k_pairwise(S, sub_labels, k=k_show)
    pairs = [_pairs_only(a, b) for _, a, b in tops]
    sims = [val for val, _, _ in tops]

    st.subheader("Top pairwise similarities")
    st.dataframe(
        _render_two_row_table("Pair", "Similarity", pairs, sims,
                              color_second_row=True, format_second_row=True),
        use_container_width=True
    )

    # ======== 光谱图：Spectra viewer；所有谱统一 ÷ B 显示，但 y 轴标题去掉“(÷B)” ========
    fig = go.Figure()
    for j in sel_idx:
        lbl = labels_all[j]
        y = E_raw_all[:, j] / (B + 1e-12)  # 仍按你的 B 做统一归一
        fig.add_trace(go.Scatter(x=wl, y=y, mode="lines", name=lbl))
    fig.update_layout(
        title="Spectra viewer",
        xaxis_title="Wavelength (nm)",
        yaxis_title="Normalized intensity",
        yaxis=dict(range=[0, 1.05],
                   tickmode="array",
                   tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                   ticktext=["0", "0.2", "0.4", "0.6", "0.8", "1"])
    )
    st.plotly_chart(fig, use_container_width=True)
