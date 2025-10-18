import streamlit as st
import numpy as np
import plotly.graph_objects as go
import yaml
import os

from utils import (
    load_dyes_yaml,
    load_probe_fluor_map,
    build_emission_only_matrix,
    build_effective_with_lasers,
    derive_powers_simultaneous,
    derive_powers_separate,
    solve_lexicographic_k,   # ← 关键：导入渐进式字典序优化器
    cosine_similarity_matrix,
    top_k_pairwise,
)

st.set_page_config(page_title="Choose Fluorophore", layout="wide")

DYES_YAML = "data/dyes.yaml"
PROBE_MAP_YAML = "data/probe_fluor_map.yaml"
READOUT_POOL_YAML = "data/readout_fluorophores.yaml"

# ---------- Load data ----------
wl, dye_db = load_dyes_yaml(DYES_YAML)
probe_map = load_probe_fluor_map(PROBE_MAP_YAML)

def load_readout_pool(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    items = data.get("fluorophores", []) or []
    pool = sorted({s.strip() for s in items if isinstance(s, str) and s.strip()})
    pool = [f for f in pool if f in dye_db]
    return pool

readout_pool = load_readout_pool(READOUT_POOL_YAML)

# ---------- Sidebar ----------
st.sidebar.header("Configuration")

mode = st.sidebar.radio(
    "Mode",
    options=("Emission only", "Emission + Excitation + Brightness"),
    help=(
        "Emission only: use emission spectra only (each emission first peak-normalized, "
        "then optimize by cosine).\n\n"
        "Emission + Excitation + Brightness: build effective spectra with lasers using "
        "excitation · QY · EC, and optimize on cosine of those effective spectra."
    ),
)

laser_strategy = None
laser_list = []
if mode == "Emission + Excitation + Brightness":
    laser_strategy = st.sidebar.radio(
        "Laser usage", options=("Simultaneous", "Separate"),
        help="Simultaneous: cumulative-by-segment leveling to a common B. Separate: per-laser power aligned, spectra concatenated horizontally."
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

source_mode = st.sidebar.radio(
    "Selection source",
    options=("By probes", "From readout pool"),
    help="Choose dyes by probe mapping, or pick the most separable subset directly from the readout pool."
)

# ---------- Tiny 2-row HTML table ----------
def html_two_row_table(row0_label, row1_label, row0_vals, row1_vals,
                       color_second_row=False, color_thresh=0.9,
                       format_second_row=False):
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
    fa = a.split(" – ", 1)[1]
    fb = b.split(" – ", 1)[1]
    return f"{fa} vs {fb}"

# ---------- Main ----------
st.title("Fluorophore Selection for Multiplexed Imaging")

use_pool = (source_mode == "From readout pool")
groups = {}

if use_pool:
    st.subheader("Pick from readout pool")
    if len(readout_pool) == 0:
        st.info("Readout pool not found or empty. Please add data/readout_fluorophores.yaml.")
        st.stop()
    max_n = len(readout_pool)
    N_pick = st.number_input("How many fluorophores to pick", min_value=1, max_value=max_n, value=min(4, max_n), step=1)
    for i in range(1, N_pick + 1):
        groups[f"Slot {i}"] = readout_pool[:]
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

def run_selection_and_display(groups, mode, laser_strategy, laser_list):
    if mode == "Emission only":
        E_norm, labels_pair, idx_groups = build_emission_only_matrix(wl, dye_db, groups)
        if E_norm.shape[1] == 0:
            st.error("No spectra available for optimization.")
            st.stop()

        # 渐进式字典序（K= min(10, 对数)）
        G = len(idx_groups)
        K = min(10, (G * (G - 1)) // 2)
        sel_idx, _ = solve_lexicographic_k(E_norm, idx_groups, labels_pair, levels=K, enforce_unique=True)

        # Selected
        sel_pairs = [labels_pair[j] for j in sel_idx]
        first_row = [s.split(" – ", 1)[0] for s in sel_pairs]
        second_row = [s.split(" – ", 1)[1] for s in sel_pairs]
        st.subheader("Selected Fluorophores")
        html_two_row_table("Probe" if not use_pool else "Slot", "Fluorophore", first_row, second_row)

        # Top-K similarities
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

    else:
        if not laser_list:
            st.error("Please specify laser wavelengths.")
            st.stop()

        # 第一轮（A 集）用于功率标定
        E0_norm, labels0, idx0 = build_emission_only_matrix(wl, dye_db, groups)
        G0 = len(idx0)
        K0 = min(10, (G0 * (G0 - 1)) // 2)
        sel0, _ = solve_lexicographic_k(E0_norm, idx0, labels0, levels=K0, enforce_unique=True)
        A_labels = [labels0[j] for j in sel0]

        # 功率与 B
        if laser_strategy == "Simultaneous":
            powers, B = derive_powers_simultaneous(wl, dye_db, A_labels, laser_list)
        else:
            powers, B = derive_powers_separate(wl, dye_db, A_labels, laser_list)

        # 有效光谱
        E_raw_all, E_norm_all, labels_all, idx_groups_all = build_effective_with_lasers(
            wl, dye_db, groups, laser_list, laser_strategy, powers
        )

        # 第二轮：基于有效光谱的字典序
        Gf = len(idx_groups_all)
        Kf = min(10, (Gf * (Gf - 1)) // 2)
        sel_idx, _ = solve_lexicographic_k(E_norm_all, idx_groups_all, labels_all, levels=Kf, enforce_unique=True)

        # Selected
        sel_pairs = [labels_all[j] for j in sel_idx]
        first_row = [s.split(" – ", 1)[0] for s in sel_pairs]
        second_row = [s.split(" – ", 1)[1] for s in sel_pairs]
        st.subheader("Selected Fluorophores (with lasers)")
        html_two_row_table("Probe" if not use_pool else "Slot", "Fluorophore", first_row, second_row)

        # Laser powers (relative)
        lam_sorted = list(sorted(laser_list))
        p = np.array(powers, dtype=float)
        maxp = float(np.max(p)) if p.size > 0 else 1.0
        prel = (p / (maxp + 1e-12)).tolist()
        st.subheader("Laser powers (relative)")
        html_two_row_table("Laser (nm)", "Relative power",
                           lam_sorted, [float(f"{v:.6g}") for v in prel],
                           color_second_row=False, format_second_row=False)

        # Top-K similarities
        S = cosine_similarity_matrix(E_norm_all[:, sel_idx])
        sub_labels = [labels_all[j] for j in sel_idx]
        tops = top_k_pairwise(S, sub_labels, k=k_show)
        pairs = [only_fluor_pair(a, b) for _, a, b in tops]
        sims = [val for val, _, _ in tops]
        st.subheader("Top pairwise similarities")
        html_two_row_table("Pair", "Similarity", pairs, sims,
                           color_second_row=True, color_thresh=0.9, format_second_row=True)

        # Spectra viewer
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
                    y=1.12 if (i % 2 == 0) else 1.06, yref="paper",
                    text=f"{int(lam_sorted[i])} nm",
                    showarrow=False,
                    font=dict(size=12),
                    align="center",
                    yanchor="bottom",
                    xshift=(-12 if (i % 2 == 0) else 12)
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

run_selection_and_display(groups, mode, laser_strategy, laser_list)
