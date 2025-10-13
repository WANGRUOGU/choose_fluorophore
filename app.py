import streamlit as st
import numpy as np
import plotly.graph_objects as go

from utils import (
    load_dyes_yaml,
    load_probe_fluor_map,
    cosine_similarity_matrix,
    select_minimax_subset_grouped,
    select_lexicographic_grouped,  # kept if you still want it from before
    build_effective_emission_emission_only,
    build_effective_emission_with_lasers,
    iterate_selection_with_lasers,
    top_k_pairwise,
    normalize_probe_mapping,  # <- we will correctly unpack its return
)

st.set_page_config(page_title="Choose Fluorophore", layout="wide")

# -----------------------------
# Load data
# -----------------------------
yaml_path = "data/dyes.yaml"
probe_map_path = "data/probe_fluor_map.yaml"

wl, dye_db = load_dyes_yaml(yaml_path)

# Load raw mapping (supports multiple shapes)
mapping_raw = load_probe_fluor_map(probe_map_path)

# Normalize & filter (returning (clean_mapping, dropped_report))
probe_to_fluors, _dropped = normalize_probe_mapping(mapping_raw, dye_db, alias=None)

# -----------------------------
# Sidebar: Pipeline configuration
# -----------------------------
st.sidebar.header("Configuration")

# 1) Objective flavor
mode = st.sidebar.radio(
    "Mode",
    options=("Emission only", "Emission + Excitation + Brightness"),
    help="Choose whether to optimize using emission spectra only, or to include excitation, quantum yield, and extinction coefficient.",
)

# 2) If including lasers: strategy + lasers
laser_strategy = None
laser_list = []
custom_lasers = []

if mode == "Emission + Excitation + Brightness":
    laser_strategy = st.sidebar.radio(
        "Laser usage",
        options=("Simultaneous", "Separate"),
        help=(
            "Simultaneous: all lasers on together. We assign powers so that the per-segment peak across fluorophores "
            "is leveled to the same maximum.\n"
            "Separate: lasers fired separately; each laser’s power is set so its global peak matches a common target."
        ),
    )
    preset = st.sidebar.radio(
        "Lasers",
        options=("488/561/639 (preset)", "Custom"),
        help="Either use a fixed set (488, 561, 639 nm) or specify your own laser wavelengths."
    )
    if preset == "488/561/639 (preset)":
        laser_list = [488, 561, 639]
        st.sidebar.caption("Using lasers: 488, 561, 639 nm")
    else:
        n_lasers = st.sidebar.number_input(
            "Number of lasers",
            min_value=1, max_value=8, value=3, step=1,
            help="How many lasers you will specify."
        )
        cols = st.sidebar.columns(2)
        custom_lasers = []
        for i in range(n_lasers):
            default_val = [488, 561, 639][i] if i < 3 else int(wl.min())
            lam = cols[i % 2].number_input(
                f"Laser {i+1} (nm)",
                min_value=int(wl.min()), max_value=int(wl.max()),
                value=int(default_val),
                step=1, help="Laser wavelength in nm on the same grid as spectra."
            )
            custom_lasers.append(int(lam))
        laser_list = custom_lasers

# 3) How many “largest similarities” to display (not the heatmap)
k_top_show = st.sidebar.slider(
    "Show top-K largest pairwise similarities",
    min_value=5, max_value=50, value=10, step=1,
    help="Only the largest K similarities will be displayed (sorted)."
)

# 4) Optimization flavor: single-level or lexicographic
lexi = st.sidebar.checkbox(
    "Lexicographic (layer-by-layer) minimization",
    value=False,
    help="If checked, apply the multi-level (lexicographic) optimization. Otherwise, do single-level minimax."
)

# -----------------------------
# Main UI
# -----------------------------
st.title("Fluorophore Selection for Multiplexed Imaging")

st.markdown(
    "Use this app to select one fluorophore per probe to minimize spectral similarity. "
    "If you include excitation and brightness, laser powers are derived automatically from your selection and the specified lasers."
)

# Probe selection UI (no preset number; user decides)
all_probes = sorted(probe_to_fluors.keys())
with st.expander("Pick probes to optimize", expanded=True):
    picked = st.multiselect(
        "Probes",
        options=all_probes,
        help="Choose the probes you will include in the optimization. Each probe will contribute one fluorophore."
    )

if not picked:
    st.info("Select at least one probe to proceed.")
    st.stop()

# Build candidate groups (one group per probe)
groups = []
group_names = []
for probe in picked:
    # keep only candidates that exist in dye_db (others have no spectra)
    cands = [f for f in probe_to_fluors.get(probe, []) if f in dye_db]
    if not cands:
        st.warning(f"No spectra found in dyes.yaml for probe '{probe}' candidates; this probe will be skipped.")
        continue
    groups.append(cands)
    group_names.append(probe)

if not groups:
    st.error("No valid groups after filtering by available spectra.")
    st.stop()

# -----------------------------
# Compute effective spectra and run optimization
# -----------------------------
if mode == "Emission only":
    # Build a matrix [W x |candidates_total|] of emission-only spectra (normalized to unit L2 for cosine)
    E, labels = build_effective_emission_emission_only(wl, dye_db, groups)
    # Run the combinatorial selection (single-level or lexicographic)
    if not lexi:
        sel_indices, t_star = select_minimax_subset_grouped(E, groups)
    else:
        sel_indices, t_star = select_lexicographic_grouped(E, groups)

    selected = [labels[i] for i in sel_indices]

    # Display results
    st.subheader("Selected Fluorophores")
    for probe, fluor in zip(group_names, selected):
        st.write(f"- **{probe}** → {fluor}")

    # Similarity (top-K only)
    if len(sel_indices) >= 2:
        S = cosine_similarity_matrix(E[:, sel_indices])
        vals, pairs = top_k_pairwise(S, k=k_top_show)
        st.markdown("**Top pairwise similarities (largest first):**")
        if len(vals) == 0:
            st.write("None.")
        else:
            st.write(", ".join([f"{v:.3f}" for v in vals]))
    else:
        st.markdown("**Top pairwise similarities:** need ≥2 selections.")

    # Spectra viewer-like plot (emission only, normalized per fluor)
    fig = go.Figure()
    for i, fluor in zip(sel_indices, selected):
        norm = np.linalg.norm(E[:, i]) + 1e-12
        fig.add_trace(go.Scatter(
            x=wl, y=E[:, i] / norm,
            mode="lines", name=fluor
        ))
    fig.update_layout(
        title="Emission-only (normalized) spectra of selected fluorophores",
        xaxis_title="Wavelength (nm)", yaxis_title="Intensity (normalized)"
    )
    st.plotly_chart(fig, use_container_width=True)

else:
    # ========== Full model with lasers ==========
    if len(laser_list) == 0:
        st.error("Please specify laser wavelengths.")
        st.stop()

    # 1) Initial selection A under emission-only
    E0, labels0 = build_effective_emission_emission_only(wl, dye_db, groups)
    if not lexi:
        sel0, t0 = select_minimax_subset_grouped(E0, groups)
    else:
        sel0, t0 = select_lexicographic_grouped(E0, groups)
    A = [labels0[i] for i in sel0]

    # 2) Iterate: compute laser powers from A, rebuild effective spectra for all candidates, re-select, until fixed point or max iters
    selected, powers, iters, converged, E_final, labels_final = iterate_selection_with_lasers(
        wl=wl,
        dye_db=dye_db,
        groups=groups,
        laser_wavelengths=laser_list,
        mode=laser_strategy,
        init_selection=A,
        max_iter=8
    )

    # Display chosen fluorophores & laser powers
    st.subheader("Selected Fluorophores (with lasers)")
    for probe, fluor in zip(group_names, selected):
        st.write(f"- **{probe}** → {fluor}")
    st.caption(f"Iterations: {iters}  |  Converged: {converged}")

    st.subheader("Derived Laser Powers")
    st.write(", ".join([f"{lam} nm: {p:.3g}" for lam, p in zip(sorted(laser_list), powers)]))
    st.caption("Powers computed so that per-segment or per-laser peaks are leveled to a common maximum B, following your rules.")

    # Similarity (top-K only)
    if len(selected) >= 2 and E_final is not None:
        idx_final = [labels_final.index(f) for f in selected]
        S = cosine_similarity_matrix(E_final[:, idx_final])
        vals, pairs = top_k_pairwise(S, k=k_top_show)
        st.markdown("**Top pairwise similarities (largest first):**")
        if len(vals) == 0:
            st.write("None.")
        else:
            st.write(", ".join([f"{v:.3f}" for v in vals]))
    else:
        st.markdown("**Top pairwise similarities:** need ≥2 selections.")

    # Spectra viewer-like plot (effective emission with lasers)
    if E_final is not None:
        fig = go.Figure()
        for fluor in selected:
            j = labels_final.index(fluor)
            y = E_final[:, j]
            # normalize for visualization only
            fig.add_trace(go.Scatter(
                x=wl, y=y / (np.linalg.norm(y) + 1e-12),
                mode="lines", name=fluor
            ))
        fig.update_layout(
            title="Effective spectra under lasers (normalized for display)",
            xaxis_title="Wavelength (nm)", yaxis_title="Intensity (normalized)"
        )
        st.plotly_chart(fig, use_container_width=True)
