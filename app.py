import os
import yaml
import streamlit as st
import numpy as np
import plotly.graph_objects as go

from utils import (
    load_dyes_yaml,
    read_probes_and_mapping,
    build_effective_emission_emission_only,
    iterate_selection_with_lasers,
    cosine_similarity_matrix,
    top_k_pairwise,
    build_E_and_groups_from_mapping,
    lexicographic_select_grouped_milp,
)

st.set_page_config(page_title="Choose Fluorophore", layout="wide")

APP_VERSION = "v3.0-lexiMILP"

# -----------------------------
# Paths
# -----------------------------
DYES_YAML = "data/dyes.yaml"
PROBE_MAP_YAML = "data/probe_fluor_map.yaml"

# -----------------------------
# Load data
# -----------------------------
wl, dye_db = load_dyes_yaml(DYES_YAML)
all_probes, probe_to_fluors_raw = read_probes_and_mapping(PROBE_MAP_YAML)

# -----------------------------
# Sidebar: pipeline configuration (简洁版，无debug/clear-cache)
# -----------------------------
st.sidebar.header("Configuration")
st.sidebar.caption(APP_VERSION)

mode = st.sidebar.radio(
    "Mode",
    options=("Emission only", "Emission + Excitation + Brightness"),
    help="Optimize using emission spectra only, or include excitation, quantum yield, and extinction coefficient."
)

laser_strategy = None
laser_list = []
custom_lasers = []
if mode == "Emission + Excitation + Brightness":
    laser_strategy = st.sidebar.radio(
        "Laser usage",
        options=("Simultaneous", "Separate"),
        help=(
            "Simultaneous: all lasers on together; powers chosen so segment peaks align.\n"
            "Separate: lasers fired separately; each power scaled to a common peak."
        ),
    )
    preset = st.sidebar.radio(
        "Lasers",
        options=("488/561/639 (preset)", "Custom"),
        help="Use 488, 561, 639 nm, or specify your own wavelengths."
    )
    if preset == "488/561/639 (preset)":
        laser_list = [488, 561, 639]
        st.sidebar.caption("Using lasers: 488, 561, 639 nm")
    else:
        n_lasers = st.sidebar.number_input(
            "Number of lasers", min_value=1, max_value=8, value=3, step=1
        )
        cols = st.sidebar.columns(2)
        for i in range(n_lasers):
            lam = cols[i % 2].number_input(
                f"Laser {i+1} (nm)",
                min_value=int(wl.min()), max_value=int(wl.max()),
                value=[488, 561, 639][i] if i < 3 else int(wl.min()),
                step=1,
            )
            custom_lasers.append(int(lam))
        laser_list = custom_lasers

k_top_show = st.sidebar.slider(
    "Show top-K largest pairwise similarities",
    min_value=5, max_value=50, value=10, step=1,
    help="Only the largest K similarities will be displayed (sorted)."
)

# -----------------------------
# Main UI
# -----------------------------
st.title("Fluorophore Selection for Multiplexed Imaging")

st.markdown(
    "Select one fluorophore per probe to minimize spectral similarity. "
    "This version uses the **layer-by-layer (lexicographic) MILP** solver—no greedy."
)

with st.expander("Pick probes to optimize", expanded=True):
    picked = st.multiselect(
        "Probes",
        options=all_probes,
        help="Choices are read from the `probes:` list in data/probe_fluor_map.yaml."
    )

if not picked:
    st.info("Select at least one probe to proceed.")
    st.stop()

# -----------------------------
# Build candidate groups from mapping, and filter by spectra availability
# -----------------------------
E_emission, labels_emission, groups_idx, group_names = build_E_and_groups_from_mapping(
    wl=wl,
    dye_db=dye_db,
    picked_probes=picked,
    probe_to_fluors=probe_to_fluors_raw,
    mode="emission_only"
)

if E_emission.shape[1] == 0 or len(groups_idx) == 0:
    st.error("No valid candidates after filtering by available spectra.")
    st.stop()

# -----------------------------
# Optimization
# -----------------------------
if mode == "Emission only":
    # Columns already correspond to emission-only spectra (not normalized here)
    # We will normalize inside the MILP builder via cosine constants.
    sel_indices = lexicographic_select_grouped_milp(E_emission, groups_idx)

    selected = [labels_emission[i] for i in sel_indices]
    st.subheader("Selected Fluorophores")
    for probe, fluor in zip(group_names, selected):
        st.write(f"- **{probe}** → {fluor}")

    # Show top-K similarities among the selected set
    # Normalize for cosine matrix here just for reporting
    En = E_emission / (np.linalg.norm(E_emission, axis=0, keepdims=True) + 1e-12)
    S = cosine_similarity_matrix(En[:, sel_indices])
    vals, pairs = top_k_pairwise(S, k=k_top_show)
    st.markdown("**Top pairwise similarities (largest first):**")
    st.write("None." if len(vals) == 0 else ", ".join([f"{v:.3f}" for v in vals]))

    # Spectra viewer-like plot
    fig = go.Figure()
    for i, fluor in zip(sel_indices, selected):
        y = En[:, i]  # normalized for visual
        fig.add_trace(go.Scatter(x=wl, y=y, mode="lines", name=fluor))
    fig.update_layout(
        title="Emission (normalized) spectra of selected fluorophores",
        xaxis_title="Wavelength (nm)", yaxis_title="Intensity (normalized)"
    )
    st.plotly_chart(fig, use_container_width=True)

else:
    # Full pipeline with lasers:
    if len(laser_list) == 0:
        st.error("Please specify laser wavelengths.")
        st.stop()

    # 迭代：由 emission-only 的首次选择出发，每次重建有效谱并用“分层MILP”重新选择
    selected, powers, iters, converged, E_final, labels_final, groups_idx_final = iterate_selection_with_lasers(
        wl=wl,
        dye_db=dye_db,
        picked_probes=picked,
        probe_to_fluors=probe_to_fluors_raw,
        laser_wavelengths=laser_list,
        mode=laser_strategy,
        solver="lexi_milp",
        max_iter=8
    )

    st.subheader("Selected Fluorophores (with lasers)")
    for probe, fluor in zip(picked, selected):
        st.write(f"- **{probe}** → {fluor}")
    st.caption(f"Iterations: {iters}  |  Converged: {converged}")

    st.subheader("Derived Laser Powers")
    st.write(", ".join([f"{lam} nm: {p:.3g}" for lam, p in zip(sorted(laser_list), powers)]))

    # Report top-K on final selection (normalize for cosine)
    En_final = E_final / (np.linalg.norm(E_final, axis=0, keepdims=True) + 1e-12)
    idx_final = [labels_final.index(f) for f in selected]
    S = cosine_similarity_matrix(En_final[:, idx_final])
    vals, _ = top_k_pairwise(S, k=k_top_show)
    st.markdown("**Top pairwise similarities (largest first):**")
    st.write("None." if len(vals) == 0 else ", ".join([f"{v:.3f}" for v in vals]))

    # Spectra viewer-like plot
    fig = go.Figure()
    for fluor in selected:
        j = labels_final.index(fluor)
        y = En_final[:, j]  # normalized for display
        fig.add_trace(go.Scatter(x=wl, y=y, mode="lines", name=fluor))
    fig.update_layout(
        title="Effective spectra under lasers",
        xaxis_title="Wavelength (nm)", yaxis_title="Intensity (a.u.)"
    )
    st.plotly_chart(fig, use_container_width=True)
