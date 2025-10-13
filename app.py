import os
import io
import yaml
import streamlit as st
import numpy as np
import plotly.graph_objects as go

from utils import (
    load_dyes_yaml,
    cosine_similarity_matrix,
    select_minimax_subset_grouped,
    select_lexicographic_grouped,
    build_effective_emission_emission_only,
    build_effective_emission_with_lasers,
    iterate_selection_with_lasers,
    top_k_pairwise,
)

st.set_page_config(page_title="Choose Fluorophore", layout="wide")

APP_VERSION = "v2.3-debug-probes-only"

# -----------------------------
# Paths
# -----------------------------
DYES_YAML = "data/dyes.yaml"
PROBE_MAP_YAML = "data/probe_fluor_map.yaml"

# -----------------------------
# Data loaders (with robust cache invalidation)
# -----------------------------
def _file_mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except Exception:
        return -1.0

@st.cache_data(show_spinner=False)
def load_text_preview(path: str, mtime: float, head_bytes: int = 1200) -> str:
    """Read first ~head_bytes bytes to preview raw YAML (for debug)."""
    try:
        with open(path, "rb") as f:
            raw = f.read(head_bytes)
        return raw.decode("utf-8", errors="replace")
    except Exception as e:
        return f"[ERROR reading {path}: {e}]"

@st.cache_data(show_spinner=False)
def read_probes_and_mapping_no_cache(path: str):
    """
    Read probes from YAML where the TOP-LEVEL is a LIST:
    - name: PROBE
      fluors: [ ... ]
    Returns:
      names_sorted: list[str]
      mapping: dict[str, list[str]]
    """
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Strictly require top-level list (that is your current file)
    if not isinstance(data, list):
        # Try fallback if it's a dict with 'probes'
        if isinstance(data, dict) and isinstance(data.get("probes"), list):
            items = data["probes"]
        else:
            # Show what structure we actually saw
            st.error("probe_fluor_map.yaml is not a top-level list. Parsed type: "
                     f"{type(data).__name__}. Keys: {list(data.keys()) if isinstance(data, dict) else 'N/A'}")
            return [], {}
    else:
        items = data

    names = []
    mapping = {}
    for item in items:
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
        names.append(name)
        mapping[name] = fls

    names_sorted = sorted(dict.fromkeys(names))
    return names_sorted, mapping


# -----------------------------
# Load dyes
# -----------------------------
wl, dye_db = load_dyes_yaml(DYES_YAML)

# -----------------------------
# Sidebar: controls
# -----------------------------
st.sidebar.header("Configuration")
st.sidebar.caption(f"App {APP_VERSION}")

# Force clear cache button
if st.sidebar.button("Force clear cache", help="Clear Streamlit caches and re-read YAML files"):
    st.cache_data.clear()
    st.rerun()

mode = st.sidebar.radio(
    "Mode",
    options=("Emission only", "Emission + Excitation + Brightness"),
    help="Choose whether to optimize using emission-only, or include excitation + brightness.",
)

laser_strategy = None
laser_list = []
custom_lasers = []

if mode == "Emission + Excitation + Brightness":
    laser_strategy = st.sidebar.radio(
        "Laser usage",
        options=("Simultaneous", "Separate"),
        help=(
            "Simultaneous: all lasers on together, powers chosen so segment peaks align.\n"
            "Separate: lasers fired separately; each power scaled to a common peak."
        ),
    )
    preset = st.sidebar.radio(
        "Lasers",
        options=("488/561/639 (preset)", "Custom"),
        help="Use a fixed set (488, 561, 639) or specify your own wavelengths."
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
)

lexi = st.sidebar.checkbox(
    "Lexicographic (layer-by-layer) minimization",
    value=False,
)

# -----------------------------
# Debug panel (so我们能看到读到的到底是什么)
# -----------------------------
with st.expander("🔧 Debug: YAML & Parsing", expanded=False):
    st.markdown("**Resolved paths**")
    st.code(
        f"DYES_YAML:  {os.path.abspath(DYES_YAML)}\n"
        f"PROBE_MAP:  {os.path.abspath(PROBE_MAP_YAML)}",
        language="text",
    )

    pm_mtime = _file_mtime(PROBE_MAP_YAML)
    st.markdown(f"**probe_fluor_map.yaml mtime**: `{pm_mtime}`")

    st.markdown("**Raw YAML preview (first ~1200 bytes)**")
    st.code(load_text_preview(PROBE_MAP_YAML, pm_mtime), language="yaml")

all_probes, probe_to_fluors_raw = read_probes_and_mapping_no_cache(PROBE_MAP_YAML)

with st.expander("🔧 Debug: Parsed probes (first 20)", expanded=False):
    st.write(all_probes[:20])

# -----------------------------
# Main UI
# -----------------------------
st.title("Fluorophore Selection for Multiplexed Imaging")
st.caption(APP_VERSION)

st.markdown(
    "Select one fluorophore per probe to minimize spectral similarity. "
    "With excitation+brightness enabled, laser powers are derived automatically."
)

# Probe selection strictly from parsed names
with st.expander("Pick probes to optimize", expanded=True):
    picked = st.multiselect(
        "Probes (from probes: list in YAML)",
        options=all_probes,
        help="Choices come strictly from the `probes:` list in data/probe_fluor_map.yaml.",
    )

if not picked:
    st.info("Select at least one probe to proceed.")
    st.stop()

# Build groups: use mapping from YAML, then filter to dyes present in dye_db
groups = []
group_names = []
for probe in picked:
    cands = list(probe_to_fluors_raw.get(probe, []))
    # filter by spectra availability
    cands = [f for f in cands if f in dye_db]
    if not cands:
        st.warning(f"Probe '{probe}' has no candidates with spectra in dyes.yaml and will be skipped.")
        continue
    groups.append(cands)
    group_names.append(probe)

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
        fig.add_trace(go.Scatter(x=wl, y=y / (np.linalg.norm(y) + 1e-12), mode="lines", name=fluor))
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
        fig.add_trace(go.Scatter(x=wl, y=y / (np.linalg.norm(y) + 1e-12), mode="lines", name=fluor))
    fig.update_layout(
        title="Effective spectra under lasers (normalized for display)",
        xaxis_title="Wavelength (nm)", yaxis_title="Intensity (normalized)"
    )
    st.plotly_chart(fig, use_container_width=True)
