import streamlit as st
import numpy as np
import plotly.graph_objects as go
import yaml  # <- robustly read probe_fluor_map.yaml here

from utils import (
    load_dyes_yaml,
    load_probe_fluor_map,            # kept as fallback
    normalize_probe_mapping,         # we will correctly unpack (clean, dropped)
    cosine_similarity_matrix,
    select_minimax_subset_grouped,
    select_lexicographic_grouped,
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

# Robustly read probe_fluor_map.yaml: prefer the {probes: [ {name, fluors}, ... ]} shape
def _load_probe_mapping_only_names(path):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    # Preferred shape: dict with key "probes" as a list of dicts
    if isinstance(data, dict) and isinstance(data.get("probes"), list):
        mapping = {}
        for item in data["probes"]:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            fls = item.get("fluors", []) or []
            if not name:
                continue
            if isinstance(fls, (list, tuple)):
                mapping[name] = [str(x).strip() for x in fls if str(x).strip()]
            else:
                mapping[name] = [str(fls).strip()] if str(fls).strip() else []
        return mapping
    # Fallback to legacy util for other shapes
    return load_probe_fluor_map(path)

# --- Strictly read probes list from YAML (ignore schema/updated/notes) ---
def read_probe_names_and_map(path):
    """
    Returns:
      probe_names: list[str]  (from data['probes'][i]['name'])
      probe_to_fluors: dict[str, list[str]]  (from data['probes'][i]['fluors'])
    This ignores top-level keys like schema/updated/notes.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    probe_names = []
    probe_to_fluors_raw = {}
    plist = data.get("probes", [])
    if isinstance(plist, list):
        for item in plist:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            fls = item.get("fluors", []) or []
            if not name:
                continue
            # coerce to clean list of strings
            if isinstance(fls, (list, tuple)):
                fls = [str(x).strip() for x in fls if str(x).strip()]
            else:
                fls = [str(fls).strip()] if str(fls).strip() else []
            probe_names.append(name)
            probe_to_fluors_raw[name] = fls
    return probe_names, probe_to_fluors_raw

# 1) 严格拿到 probe 名称列表 + 原始映射
probe_names_raw, probe_to_fluors_raw = read_probe_names_and_map(probe_map_path)

# 2) 只用于下拉菜单：直接使用 YAML 里的 probe 名称
all_probes = sorted(probe_names_raw)

# 3) 再把原始映射过一遍 normalize（做别名/去重/过滤成 dyes.yaml 中存在的光谱）
from utils import normalize_probe_mapping
probe_to_fluors, _dropped = normalize_probe_mapping(probe_to_fluors_raw, dye_db, alias=None)


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
    E, labels = build_effective_emission_emission_only(wl, dye_db, groups)
    if not lexi:
        sel_indices, t_star = select_minimax_subset_grouped(E, groups)
    else:
        sel_indices, t_star = select_lexicographic_grouped(E, groups)

    selected = [labels[i] for i in sel_indices]

    st.subheader("Selected Fluorophores")
    for probe, fluor in zip(group_names, selected):
        st.write(f"- **{probe}** → {fluor}")

    if len(sel_indices) >= 2:
        S = cosine_similarity_matrix(E[:, sel_indices])
        vals, pairs = top_k_pairwise(S, k=k_top_show)
        st.markdown("**Top pairwise similarities (largest first):**")
        st.write(", ".join([f"{v:.3f}" for v in vals]) if vals else "None.")
    else:
        st.markdown("**Top pairwise similarities:** need ≥2 selections.")

    fig = go.Figure()
    for i, fluor in zip(sel_indices, selected):
        norm = np.linalg.norm(E[:, i]) + 1e-12
        fig.add_trace(go.Scatter(x=wl, y=E[:, i] / norm, mode="lines", name=fluor))
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
    st.caption("Powers computed so that per-segment or per-laser peaks are leveled to a common maximum B, following your rules.")

    if len(selected) >= 2 and E_final is not None:
        idx_final = [labels_final.index(f) for f in selected]
        S = cosine_similarity_matrix(E_final[:, idx_final])
        vals, pairs = top_k_pairwise(S, k=k_top_show)
        st.markdown("**Top pairwise similarities (largest first):**")
        st.write(", ".join([f"{v:.3f}" for v in vals]) if vals else "None.")
    else:
        st.markdown("**Top pairwise similarities:** need ≥2 selections.")

    if E_final is not None:
        fig = go.Figure()
        for fluor in selected:
            j = labels_final.index(fluor)
            y = E_final[:, j]
            fig.add_trace(go.Scatter(x=wl, y=y / (np.linalg.norm(y) + 1e-12), mode="lines", name=fluor))
        fig.update_layout(
            title="Effective spectra under lasers (normalized for display)",
            xaxis_title="Wavelength (nm)", yaxis_title="Intensity (normalized)"
        )
        st.plotly_chart(fig, use_container_width=True)
