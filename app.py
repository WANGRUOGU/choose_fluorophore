import streamlit as st
import numpy as np
import plotly.graph_objects as go
import yaml  # <-- used to strictly read probe names

from utils import (
    load_dyes_yaml,
    cosine_similarity_matrix,
    select_minimax_subset_grouped,
    select_lexicographic_grouped,  # keep if you want to toggle
    build_effective_emission_emission_only,
    build_effective_emission_with_lasers,
    iterate_selection_with_lasers,
    top_k_pairwise,
)

st.set_page_config(page_title="Choose Fluorophore", layout="wide")

# -----------------------------
# Load data
# -----------------------------
yaml_path = "data/dyes.yaml"
probe_map_path = "data/probe_fluor_map.yaml"

wl, dye_db = load_dyes_yaml(yaml_path)

# --- STRICT: read probe names & mapping ONLY from the `probes:` array ----
@st.cache_data(show_spinner=False)
def read_probe_names_and_map(path):
    """
    Returns:
      probe_names: list[str] from data['probes'][i]['name']
      probe_to_fluors: dict[str, list[str]] from data['probes'][i]['fluors']
    Ignores top-level keys like schema/updated/notes.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    plist = data.get("probes", [])
    probe_names = []
    probe_to_fluors = {}
    if isinstance(plist, list):
        for item in plist:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            fls = item.get("fluors", []) or []
            if not name:
                continue
            if isinstance(fls, (list, tuple)):
                fls = [str(x).strip() for x in fls if str(x).strip()]
            else:
                fls = [str(fls).strip()] if str(fls).strip() else []
            probe_names.append(name)
            probe_to_fluors[name] = fls
    # unique & sorted names for UI
    names_sorted = sorted(dict.fromkeys(probe_names))
    return names_sorted, probe_to_fluors

all_probes, probe_to_fluors_raw = read_probe_names_and_map(probe_map_path)

# -----------------------------
# Sidebar: Pipeline configuration
# -----------------------------
st.sidebar.header("Configuration")

mode = st.sidebar.radio(
    "Mode",
    options=("Emission only", "Emission + Excitation + Brightness"),
    help="Choose whether to optimize using emission spectra only, or to include excitation, quantum yield, and extinction coefficient.",
)

laser_strategy = None
laser_list = []
custom_lasers = []

if mode == "Emission + Excitation + Brightness":
    laser_strategy = st.sidebar.radio(
        "Laser usage",
        options=("Simultaneous", "Separate"),
        help=(
            "Simultaneous: all lasers on together. Powers are chosen so segment peaks align.\n"
            "Separate: lasers fired separately; each laser power set so its peak aligns to a common target."
        ),
    )
    preset = st.sidebar.radio(
        "Lasers",
        options=("488/561/639 (preset)", "Custom"),
        help="Use fixed (488, 561, 639 nm) or specify your own wavelengths."
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
            lam = cols[i % 2].number_input(
                f"Laser {i+1} (nm)",
                min_value=int(wl.min()), max_value=int(wl.max()),
                value=[488, 561, 639][i] if i < 3 else int(wl.min()),
                step=1, help="Laser wavelength in nm."
            )
            custom_lasers.append(int(lam))
        laser_list = custom_lasers

k_top_show = st.sidebar.slider(
    "Show top-K largest pairwise similarities",
    min_value=5, max_value=50, value=10, step=1,
    help="Only the largest K similarities will be displayed (sorted)."
)

lexi = st.sidebar.checkbox(
    "Lexicographic (layer-by-layer) minimization",
    value=False,
    help="If checked, apply multi-level (lexicographic) minimization; otherwise single-level minimax."
)

# -----------------------------
# Main UI
# -----------------------------
st.title("Fluorophore Selection for Multiplexed Imaging")

st.markdown(
    "Use this app to select one fluorophore per probe to minimize spectral similarity. "
    "If you include excitation and brightness, laser powers are derived automatically from your rules."
)

# === Probe selection: options come ONLY from YAML's `probes:` names ===
with st.expander("Pick probes to optimize", expanded=True):
    picked = st.multiselect(
        "Probes",
        options=all_probes,
        help="Choose the probes to include. Each probe contributes one fluorophore."
    )

if not picked:
    st.info("Select at least one probe to proceed.")
    st.stop()

# Build groups using ONLY the raw mapping from YAML; later we will filter by spectra availability
groups_raw = []
group_names = []
for probe in picked:
    cands = list(probe_to_fluors_raw.get(probe, []))
    if not cands:
        st.warning(f"No candidates listed for probe '{probe}' in probe_fluor_map.yaml.")
        continue
    groups_raw.append(cands)
    group_names.append(probe)

if not groups_raw:
    st.error("No valid groups from the selected probes.")
    st.stop()

# Filter groups by dyes that actually exist in dyes.yaml (have spectra)
groups = []
for cands in groups_raw:
    valid = [f for f in cands if f in dye_db]
    if not valid:
        groups.append([])  # keep structure; will be handled below
    else:
        groups.append(valid)

# If any group ends empty, skip that probe now (and warn)
kept_groups, kept_names = [], []
for g, (cands, pname) in enumerate(zip(groups, group_names)):
    if not cands:
        st.warning(f"Probe '{pname}' has no candidates with spectra in dyes.yaml and will be skipped.")
        continue
    kept_groups.append(cands)
    kept_names.append(pname)

groups = kept_groups
group_names = kept_names

if not groups:
    st.error("No groups remain after filtering by available spectra.")
    st.stop()

# -----------------------------
# Compute & optimize
# -----------------------------
if mode == "Emission only":
    E, labels = build_effective_emission_emission_only(wl, dye_db, groups)
    if not lexi:
        sel_indices, t_star = select_minimax_subset_grouped(E, groups)
    else:
        sel_indices, t_star = select_lexicographic_grouped(E, groups)

    selected = [labels[i] for i in sel_indices]

    st.subheader("Selected Fluorophores")
    for probe, fluor in zip(group_names, selected):
        st.write(f"- **{probe}** → {fluor}")

    S = cosine_similarity_matrix(E[:, sel_indices])
    vals, pairs = top_k_pairwise(S, k=k_top_show)
    st.markdown("**Top pairwise similarities (largest first):**")
    st.write("None." if len(vals) == 0 else ", ".join([f"{v:.3f}" for v in vals]))

    fig = go.Figure()
    for i, fluor in zip(sel_indices, selected):
        y = E[:, i]
        fig.add_trace(go.Scatter(
            x=wl, y=y / (np.linalg.norm(y) + 1e-12),
            mode="lines", name=fluor
        ))
    fig.update_layout(
        title="Emission-only (normalized) spectra of selected fluorophores",
        xaxis_title="Wavelength (nm)", yaxis_title="Intensity (normalized)"
    )
    st.plotly_chart(fig, use_container_width=True)

else:
    if len(laser_list) == 0:
        st.error("Please specify laser wavelengths.")
        st.stop()

    E0, labels0 = build_effective_emission_emission_only(wl, dye_db, groups)
    if not lexi:
        sel0, t0 = select_minimax_subset_grouped(E0, groups)
    else:
        sel0, t0 = select_lexicographic_grouped(E0, groups)
    A = [labels0[i] for i in sel0]

    selected, powers, iters, converged, E_final, labels_final = iterate_selection_with_lasers(
        wl=wl,
        dye_db=dye_db,
        groups=groups,
        laser_wavelengths=laser_list,
        mode=laser_strategy,
        init_selection=A,
        max_iter=8
    )

    st.subheader("Selected Fluorophores (with lasers)")
    for probe, fluor in zip(group_names, selected):
        st.write(f"- **{probe}** → {fluor}")
    st.caption(f"Iterations: {iters}  |  Converged: {converged}")

    st.subheader("Derived Laser Powers")
    st.write(", ".join([f"{lam} nm: {p:.3g}" for lam, p in zip(sorted(laser_list), powers)]))

    idx_final = [labels_final.index(f) for f in selected]
    S = cosine_similarity_matrix(E_final[:, idx_final])
    vals, pairs = top_k_pairwise(S, k=k_top_show)
    st.markdown("**Top pairwise similarities (largest first):**")
    st.write("None." if len(vals) == 0 else ", ".join([f"{v:.3f}" for v in vals]))

    fig = go.Figure()
    for fluor in selected:
        j = labels_final.index(fluor)
        y = E_final[:, j]
        fig.add_trace(go.Scatter(
            x=wl, y=y / (np.linalg.norm(y) + 1e-12),
            mode="lines", name=fluor
        ))
    fig.update_layout(
        title="Effective spectra under lasers (normalized for display)",
        xaxis_title="Wavelength (nm)", yaxis_title="Intensity (normalized)"
    )
    st.plotly_chart(fig, use_container_width=True)
