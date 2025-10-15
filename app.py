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

# ---------- Tiny HTML table renderer (no headers, no index) ----------
def html_two_row_table(row0_label, row1_label, row0_vals, row1_vals,
                       color_second_row=False, color_thresh=0.9,
                       format_second_row=False):
    """
    Render a 2-row horizontal table with a labeled first column.
    - No column headers, no index.
    - First column contains row labels (plain text).
    - Optional color for second row cells by threshold.
    - Optional numeric formatting for second row.
    """
    def esc(x):
        # basic escape
        return (str(x)
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;"))

    cells0 = "".join(f"<td style='padding:6px 10px;border:1px solid #ddd;'>{esc(v)}</td>" for v in row0_vals)
    tds0 = f"<td style='padding:6px 10px;border:1px solid #ddd;white-space:nowrap;'>{esc(row0_label)}</td>{cells0}"

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
    tds1 = "<td style='padding:6px 10px;border:1px solid #ddd;white-space:nowrap;'>{}</td>{}".format(
        esc(row1_label), "".join(tds1_list)
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

    # ======== Selected Fluorophores（两行 HTML 表，无列头/索引） ========
    sel_pairs = [labels_pair[j] for j in sel_idx]  # "Probe – Fluor"
    probes = [s.split(" – ", 1)[0] for s in sel_pairs]
    fluors = [s.split(" – ", 1)[1] for s in sel_pairs]
    st.subheader("Selected Fluorophores")
    html_two_row_table("Probe", "Fluorophore", probes, fluors)

    # ======== Top pairwise similarities（两行 HTML 表，第二行阈值着色；去掉 “(largest first)”） ========
    S = cosine_similarity_matrix(E_norm[:, sel_idx])
    sub_labels = [labels_pair[j] for j in sel_idx]
    tops = top_k_pairwise(S, sub_labels, k=k_show)
    pairs = [f"{a.split(' – ',1)[1]} vs {b.split(' – ',1)[1]}" for _, a, b in tops]
    sims = [val for val, _, _ in tops]

    st.subheader("Top pairwise similarities")
    html_two_row_table("Pair", "Similarity", pairs, sims,
                       color_second_row=True, color_thresh=0.9, format_second_row=True)

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

    # ======== Selected Fluorophores（两行 HTML 表） ========
    sel_pairs = [labels_all[j] for j in sel_idx]  # "Probe – Fluor"
    probes = [s.split(" – ", 1)[0] for s in sel_pairs]
    fluors = [s.split(" – ", 1)[1] for s in sel_pairs]
    st.subheader("Selected Fluorophores (with lasers)")
    html_two_row_table("Probe", "Fluorophore", probes, fluors)

    # ======== Laser powers（相对值，两行 HTML 表；无表头/索引） ========
    lam_sorted = list(sorted(laser_list))
    p = np.array(powers, dtype=float)
    maxp = float(np.max(p)) if p.size > 0 else 1.0
    prel = (p / (maxp + 1e-12)).tolist()

    st.subheader("Laser powers (relative)")
    html_two_row_table("Laser (nm)", "Relative power", lam_sorted, [float(f"{v:.6g}") for v in prel],
                       color_second_row=False, format_second_row=False)

    # ======== 相似度 Top-K（两行 HTML 表；>0.9 红，否则绿） ========
    S = cosine_similarity_matrix(E_norm_all[:, sel_idx])
    sub_labels = [labels_all[j] for j in sel_idx]
    tops = top_k_pairwise(S, sub_labels, k=k_show)
    pairs = [f"{a.split(' – ',1)[1]} vs {b.split(' – ',1)[1]}" for _, a, b in tops]
    sims = [val for val, _, _ in tops]

    st.subheader("Top pairwise similarities")
    html_two_row_table("Pair", "Similarity", pairs, sims,
                       color_second_row=True, color_thresh=0.9, format_second_row=True)

    # ======== 光谱图：Spectra viewer；所有谱统一 ÷ B 显示（标题不写 ÷B），y 轴 0–1 刻度 ========
    fig = go.Figure()
    for j in sel_idx:
        lbl = labels_all[j]
        y = E_raw_all[:, j] / (B + 1e-12)  # 仍按 B 统一归一（只是标题不显示）
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
