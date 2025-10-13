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
        "excitation · QY · EC, and optimize on cosine of those effective spectra. "
        "Plots show raw effective spectra (not normalized)."
    ),
)

laser_strategy = None
laser_list = []
if mode == "Emission + Excitation + Brightness":
    laser_strategy = st.sidebar.radio(
        "Laser usage", options=("Simultaneous", "Separate"),
        help="Simultaneous: segment leveling to a common B. Separate: each laser scaled to a common peak."
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
                [488,561,639][i] if i < 3 else int(wl.min()), 1
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

# ---------- Main ----------
st.title("Fluorophore Selection for Multiplexed Imaging")

# Probe picker
all_probes = sorted(probe_map.keys())
with st.expander("Pick probes to optimize", expanded=True):
    picked = st.multiselect("Probes", options=all_probes)

if not picked:
    st.info("Select at least one probe to proceed.")
    st.stop()

# Build groups dict in the chosen order
groups = {}
for p in picked:
    # 只保留在 dyes.yaml 里存在的候选
    cands = [f for f in probe_map.get(p, []) if f in dye_db]
    if cands:
        groups[p] = cands
if not groups:
    st.error("No valid candidates with spectra in dyes.yaml for the selected probes.")
    st.stop()

# ---------- Optimization ----------
if mode == "Emission only":
    # E_norm 用于优化；labels_pair 形如 "Probe – Fluor"
    E_norm, labels_pair, idx_groups = build_emission_only_matrix(wl, dye_db, groups)
    if E_norm.shape[1] == 0:
        st.error("No spectra available for optimization.")
        st.stop()

    sel_idx, layer_vals = solve_lexicographic(
        E_norm, idx_groups, labels_pair, levels=levels, enforce_unique=True
    )
    selected = [labels_pair[j] for j in sel_idx]  # "Probe – Fluor"

    st.subheader("Selected Fluorophores")
    for s in selected:
        st.write(f"- **{s.split(' – ',1)[0]}** → {s.split(' – ',1)[1]}")

    # 相似度（在选中集合上），并标注哪两对
    S = cosine_similarity_matrix(E_norm[:, sel_idx])
    sub_labels = [labels_pair[j] for j in sel_idx]
    tops = top_k_pairwise(S, sub_labels, k=k_show)
    st.markdown("**Top pairwise similarities (largest first)**  \n"
                "_Cosine similarity in [0, 1]. Closer to 1 ⇒ more similar; "
                "values > 0.9 often indicate poor separability._")
    if not tops:
        st.write("None.")
    else:
        for val, a, b in tops:
            st.write(f"- {a}  **vs**  {b}  →  {val:.3f}")

    # 画 emission-only（已做峰值归一）谱（展示时也可用 L2 归一后的以保持幅度可比性）
    # 绘制 emission-only 谱线（标准化到自身最大值后绘制）
    fig = go.Figure()
    for j in sel_idx:
        y = E_norm[:, j]
        y = y / (np.max(y) + 1e-12)        # ← 这里标准化到 [0,1]
        fig.add_trace(go.Scatter(
            x=wl, y=y, mode="lines",
            name=sub_labels[sel_idx.index(j)]
        ))
    fig.update_layout(
        title="Normalized spectra of selected fluorophores",
        xaxis_title="Wavelength (nm)",
        yaxis_title="Normalized intensity",
        yaxis_range=[0, 1.05]
    )
    st.plotly_chart(fig, use_container_width=True)


else:
    if not laser_list:
        st.error("Please specify laser wavelengths.")
        st.stop()

    # 先用 emission-only 建 A
    E0_norm, labels0, idx0 = build_emission_only_matrix(wl, dye_db, groups)
    sel0, _ = solve_lexicographic(E0_norm, idx0, labels0, levels=levels, enforce_unique=True)
    A_labels = [labels0[j] for j in sel0]

    # 由 A 计算功率
    if laser_strategy == "Simultaneous":
        powers = derive_powers_simultaneous(wl, dye_db, A_labels, laser_list)
    else:
        powers = derive_powers_separate(wl, dye_db, A_labels, laser_list)

    # 用功率重建 “全候选” 的有效光谱（原始 + 归一）
    E_raw_all, E_norm_all, labels_all, idx_groups_all = build_effective_with_lasers(
        wl, dye_db, groups, laser_list, laser_strategy, powers
    )

    # 基于有效光谱（L2 归一）再做层级优化
    sel_idx, layer_vals = solve_lexicographic(
        E_norm_all, idx_groups_all, labels_all, levels=levels, enforce_unique=True
    )
    selected = [labels_all[j] for j in sel_idx]

    st.subheader("Selected Fluorophores (with lasers)")
    for s in selected:
        st.write(f"- **{s.split(' – ',1)[0]}** → {s.split(' – ',1)[1]}")
    st.caption("Laser powers derived via your B-leveling rule; plots show raw effective spectra (not normalized).")
    st.write(", ".join([f"{lam} nm: {p:.3g}" for lam, p in zip(sorted(laser_list), powers)]))

    # 相似度（基于 E_norm_all 的选中列）
    S = cosine_similarity_matrix(E_norm_all[:, sel_idx])
    sub_labels = [labels_all[j] for j in sel_idx]
    tops = top_k_pairwise(S, sub_labels, k=k_show)
    st.markdown("**Top pairwise similarities (largest first)**  \n"
                "_Cosine similarity in [0, 1]. Closer to 1 ⇒ more similar; "
                "values > 0.9 often indicate poor separability._")
    if not tops:
        st.write("None.")
    else:
        for val, a, b in tops:
            st.write(f"- {a}  **vs**  {b}  →  {val:.3f}")

    # 画“原始有效光谱”（不做归一化）
    fig = go.Figure()
    for j in sel_idx:
        lbl = labels_all[j]
        fig.add_trace(go.Scatter(x=wl, y=E_raw_all[:, j], mode="lines", name=lbl))
    fig.update_layout(title="Effective spectra under lasers (raw, not normalized)",
                      xaxis_title="Wavelength (nm)", yaxis_title="Intensity (a.u.)")
    st.plotly_chart(fig, use_container_width=True)
