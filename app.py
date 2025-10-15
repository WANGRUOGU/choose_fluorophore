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
    solve_min_top_m,            # <— 关键：前10对相似度目标
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
        help="Simultaneous: cumulative-by-segment leveling to a common B. Separate: each laser scaled to match the min-wavelength anchor; spectra concatenated by lasers."
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

# Helper: compute Top-M (=min(10, C(G,2)))
def topM_from_idx_groups(idx_groups):
    G = len(idx_groups)
    return min(10, (G * (G - 1)) // 2)

# ---------- Optimization & Display ----------
if mode == "Emission only":
    E_norm, labels_pair, idx_groups = build_emission_only_matrix(wl, dye_db, groups)
    if E_norm.shape[1] == 0:
        st.error("No spectra available for optimization.")
        st.stop()

    topM = topM_from_idx_groups(idx_groups)
    sel_idx, _ = solve_min_top_m(
        E_norm, idx_groups, labels_pair, top_m=topM, enforce_unique=True
    )

    # ======== Selected Fluorophores（两行 HTML 表） ========
    sel_pairs = [labels_pair[j] for j in sel_idx]  # "Probe – Fluor"
    probes = [s.split(" – ", 1)[0] for s in sel_pairs]
    fluors = [s.split(" – ", 1)[1] for s in sel_pairs]
    st.subheader("Selected Fluorophores")
    html_two_row_table("Probe", "Fluorophore", probes, fluors)

    # ======== Top pairwise similarities（两行 HTML 表；阈值着色） ========
    S = cosine_similarity_matrix(E_norm[:, sel_idx])
    sub_labels = [labels_pair[j] for j in sel_idx]
    tops = top_k_pairwise(S, sub_labels, k=k_show)
    pairs = [only_fluor_pair(a, b) for _, a, b in tops]
    sims = [val for val, _, _ in tops]

    st.subheader("Top pairwise similarities")
    html_two_row_table("Pair", "Similarity", pairs, sims,
                       color_second_row=True, color_thresh=0.9, format_second_row=True)

    # ======== Spectra viewer（0–1 ticks；每条曲线各自 0–1 归一） ========
    st.subheader("Spectra viewer")
    fig = go.Figure()
    for j in sel_idx:
        y = E_norm[:, j]
        y = y / (np.max(y) + 1e-12)
        fig.add_trace(go.Scatter(x=wl, y=y, mode="lines", name=labels_pair[j]))
    fig.update_layout(
        title_text="",
