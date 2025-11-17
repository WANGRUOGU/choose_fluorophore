# app.py — main Streamlit entry for fluorophore selection

import json
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from utils import (
    load_dyes_yaml,
    load_probe_fluor_map,
    build_emission_only_matrix,
    build_effective_with_lasers,
    derive_powers_simultaneous,
    derive_powers_separate,
    solve_lexicographic_k,
    cosine_similarity_matrix,
    top_k_pairwise,
)

from ui_helpers import (
    DEFAULT_COLORS,
    _ensure_colors,
    _rgb01_to_plotly,
    _pair_only_fluor,
    _html_two_row_table,
    _html_table,
    _show_bw_grid,
    _prettify_name,
)

from simulation import (
    _to_uint8_gray,
    _argmax_labelmap,
    colorize_composite,
    simulate_rods_and_unmix,
)

# -------------------- Streamlit config --------------------
st.set_page_config(page_title="Fluorophore Selection", layout="wide")

# -------------------- Data loading --------------------
DYES_YAML = "data/dyes.yaml"
PROBE_MAP_YAML = "data/probe_fluor_map.yaml"
READOUT_POOL_YAML = "data/readout_fluorophores.yaml"

wl, dye_db = load_dyes_yaml(DYES_YAML)
probe_map = load_probe_fluor_map(PROBE_MAP_YAML)


def _load_readout_pool(path):
    """Read fluorphore list from readout_fluorophores.yaml, filtered by dyes.yaml."""
    try:
        import yaml, os
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        items = data.get("fluorophores", []) or []
        pool = sorted({s.strip() for s in items if isinstance(s, str) and s.strip()})
        return [f for f in pool if f in dye_db]
    except Exception:
        return []


readout_pool = _load_readout_pool(READOUT_POOL_YAML)


def _get_inventory_from_probe_map():
    """Union of all fluorophores that appear anywhere in probe_fluor_map.yaml and exist in dyes.yaml."""
    inv = set()
    for _, vals in probe_map.items():
        if not isinstance(vals, (list, tuple)):
            continue
        for f in vals:
            if isinstance(f, str):
                fs = f.strip()
                if fs and fs in dye_db:
                    inv.add(fs)
    return sorted(inv)


inventory_pool = _get_inventory_from_probe_map()


def _get_eub338_pool():
    """Candidates under the EUB 338 probe key (various spellings), filtered to dyes.yaml presence."""
    targets = {"eub338", "eub 338", "eub-338"}

    def norm(s):
        return "".join(s.lower().split())

    for k in probe_map.keys():
        if norm(k) in targets:
            cands = [f for f in probe_map.get(k, []) if f in dye_db]
            return sorted({c.strip() for c in cands})

    # relaxed fallback
    import re

    def norm2(s):
        return re.sub(r"[^a-z0-9]+", "", s.lower())

    for k in probe_map.keys():
        if norm2(k) == "eub338":
            cands = [f for f in probe_map.get(k, []) if f in dye_db]
            return sorted({c.strip() for c in cands})

    return []


# -------------------- Sorting helper --------------------
def _sort_by_emission_peak(wavelengths, labels):
    """
    Sort a list of labels by emission peak wavelength (ascending).

    labels: list of "Probe – Fluor" or just fluor names.
    Returns: list of indices giving the sort order.
    """
    peaks = []
    for lab in labels:
        fluor = lab.split(" – ", 1)[1] if " – " in lab else lab
        rec = dye_db.get(fluor)
        em = rec.get("emission") if rec is not None else None
        if em is None or len(em) != len(wavelengths):
            # unknown or malformed emission -> push to the end
            peaks.append(float("inf"))
        else:
            jmax = int(np.argmax(em))
            peaks.append(float(wavelengths[jmax]))
    return list(np.argsort(peaks))


# -------------------- Cached builders --------------------
@st.cache_data(show_spinner=False)
def cached_build_effective_with_lasers(wavelengths, dye_db_local, groups, laser_list, laser_strategy, powers):
    groups_key = json.dumps(
        {k: sorted(v) for k, v in sorted(groups.items())},
        ensure_ascii=False
    )
    _ = (
        tuple(sorted(laser_list)),
        laser_strategy,
        tuple(np.asarray(powers, float)) if powers is not None else None,
        groups_key,
    )
    return build_effective_with_lasers(
        wavelengths, dye_db_local, groups, laser_list, laser_strategy, powers
    )


@st.cache_data(show_spinner=False)
def cached_interpolate_E_on_channels(wavelengths, spectra_cols, chan_centers_nm):
    spectra_cols = np.asarray(spectra_cols, dtype=float)
    if spectra_cols.ndim == 1:
        spectra_cols = spectra_cols[:, None]
    W, N = spectra_cols.shape
    E = np.zeros((len(chan_centers_nm), N), dtype=float)
    for j in range(N):
        y = spectra_cols[:, j]
        E[:, j] = np.interp(
            chan_centers_nm,
            wavelengths,
            y,
            left=float(y[0]),
            right=float(y[-1]),
        )
    return np.nan_to_num(E, nan=0.0, posinf=0.0, neginf=0.0)


# -------------------- Sidebar UI --------------------
st.sidebar.header("Configuration")
mode = st.sidebar.radio(
    "Mode",
    options=("Emission spectra", "Predicted spectra"),
    help=("Emission: emission-only, peak-normalized.\n"
          "Predicted: effective spectra with lasers (excitation · QY · EC)."),
    key="mode_radio"
)

source_mode = st.sidebar.radio(
    "Selection source",
    ("By probes", "From readout pool", "All fluorophores", "EUB338 only"),
    key="source_radio"
)

k_show = st.sidebar.slider(
    "Show top-K similarities",
    min_value=5,
    max_value=50,
    value=10,
    step=1,
    key="k_show_slider"
)

laser_list = []
laser_strategy = None
if mode == "Predicted spectra":
    laser_strategy = st.sidebar.radio(
        "Laser usage",
        ("Simultaneous", "Separate"),
        key="laser_strategy_radio"
    )
    n = st.sidebar.number_input(
        "Number of lasers", 1, 8, 4, 1, key="num_lasers_input"
    )
    cols = st.sidebar.columns(2)
    defaults = [405, 488, 561, 639]
    for i in range(n):
        lam = cols[i % 2].number_input(
            f"Laser {i+1} (nm)",
            int(wl.min()),
            int(max(700, wl.max())),
            defaults[i] if i < len(defaults) else int(wl.min()),
            1,
            key=f"laser_{i+1}",
        )
        laser_list.append(int(lam))


# -------------------- Source selection -> groups --------------------
st.title("Fluorophore Selection for Multiplexed Imaging")

use_pool = False
N_pick = None

if source_mode == "From readout pool":
    pool = readout_pool[:]
    if not pool:
        st.info("Readout pool not found (data/readout_fluorophores.yaml).")
        st.stop()
    max_n = len(pool)
    N_pick = st.number_input(
        "How many fluorophores",
        1, max_n,
        value=min(4, max_n),
        step=1,
        key="n_pick_pool",
    )
    groups = {"Pool": pool}
    use_pool = True

elif source_mode == "All fluorophores":
    pool = inventory_pool[:]
    if not pool:
        st.error("No fluorophores found in probe_fluor_map.yaml that also exist in dyes.yaml.")
        st.stop()
    max_n = len(pool)
    N_pick = st.number_input(
        "How many fluorophores",
        1, max_n,
        value=min(4, max_n),
        step=1,
        key="n_pick_inv",
    )
    groups = {"Pool": pool}
    use_pool = True

elif source_mode == "EUB338 only":
    pool = _get_eub338_pool()
    if not pool:
        st.error("No candidates found for EUB 338 in probe_fluor_map.yaml.")
        st.stop()
    max_n = len(pool)
    N_pick = st.number_input(
        "How many fluorophores",
        1, max_n,
        value=min(4, max_n),
        step=1,
        key="n_pick_eub338",
    )
    groups = {"Pool": pool}
    use_pool = True

else:  # "By probes"
    all_probes = sorted(probe_map.keys())
    picked = st.multiselect("Probes", options=all_probes, key="picked_probes")
    if not picked:
        st.info("Select at least one probe to proceed.")
        st.stop()
    groups = {}
    for p in picked:
        cands = [f for f in probe_map.get(p, []) if f in dye_db]
        if cands:
            groups[p] = cands
    if not groups:
        st.error("No valid candidates with spectra in dyes.yaml.")
        st.stop()


# -------------------- Main run logic --------------------
def run(groups, mode, laser_strategy, laser_list):
    required_count = (N_pick if use_pool else None)

    # ---------- EMISSION-ONLY MODE ----------
    if mode == "Emission spectra":
        E_norm, labels, idx_groups = build_emission_only_matrix(wl, dye_db, groups)
        if E_norm.shape[1] == 0:
            st.error("No spectra.")
            st.stop()

        sel_idx, _ = solve_lexicographic_k(
            E_norm,
            idx_groups,
            labels,
            levels=10,
            enforce_unique=True,
            required_count=required_count,
        )

        # sort by emission peak wavelength
        sel_labels = [labels[j] for j in sel_idx]
        order = _sort_by_emission_peak(wl, sel_labels)
        sel_idx = [sel_idx[i] for i in order]
        sel_labels = [sel_labels[i] for i in order]

        colors = _ensure_colors(len(sel_idx))

        # --- Selected fluorophores table ---
        if use_pool:
            fluors = [lab.split(" – ", 1)[1] if " – " in lab else lab for lab in sel_labels]
            st.subheader("Selected Fluorophores")
            _html_two_row_table(
                "Slot", "Fluorophore",
                [f"Slot {i+1}" for i in range(len(fluors))],
                fluors,
            )
        else:
            st.subheader("Selected Fluorophores")
            _html_two_row_table(
                "Probe", "Fluorophore",
                [lab.split(" – ", 1)[0] for lab in sel_labels],
                [lab.split(" – ", 1)[1] for lab in sel_labels],
            )

        # --- Pairwise similarities ---
        S = cosine_similarity_matrix(E_norm[:, sel_idx])
        tops = top_k_pairwise(S, [labels[j] for j in sel_idx], k=k_show)
        st.subheader("Top pairwise similarities")
        _html_two_row_table(
            "Pair", "Similarity",
            [_pair_only_fluor(a, b) for _, a, b in tops],
            [val for val, _, _ in tops],
            color_second_row=True,
            color_thresh=0.9,
            fmt2=True,
        )

        # --- Spectra viewer (emission-only, peak-normalized) ---
        st.subheader("Spectra viewer")
        fig = go.Figure()
        for t, j in enumerate(sel_idx):
            y = E_norm[:, j]
            y = y / (np.max(y) + 1e-12)
            fig.add_trace(go.Scatter(
                x=wl,
                y=y,
                mode="lines",
                name=labels[j],
                line=dict(color=_rgb01_to_plotly(colors[t]), width=2),
            ))
        fig.update_layout(
            xaxis_title="Wavelength (nm)",
            yaxis_title="Normalized intensity",
            yaxis=dict(range=[0, 1.05]),
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- Simulated rods + unmixing ---
        C = 23
        chan = 494.0 + 8.9 * np.arange(C)
        E = cached_interpolate_E_on_channels(wl, E_norm[:, sel_idx], chan)

        Atrue, Ahat = simulate_rods_and_unmix(E, rods_per=3)

        colL, colR = st.columns(2)
        true_rgb = (colorize_composite(Atrue, colors) * 255).astype(np.uint8)
        labelmap_rgb = _argmax_labelmap(Ahat, colors)
        with colL:
            st.image(true_rgb, use_container_width=True, clamp=True)
            st.caption("True")
        with colR:
            st.image(labelmap_rgb, use_container_width=True, clamp=True)
            st.caption("Unmixing results")

        names = [_prettify_name(labels[j]) for j in sel_idx]
        unmix_bw = [_to_uint8_gray(Ahat[:, :, r]) for r in range(Ahat.shape[2])]

        st.divider()
        _show_bw_grid(
            "Per-fluorophore (Unmixing, grayscale)",
            unmix_bw,
            names,
            cols_per_row=6,
        )

        # --- Per-fluorophore RMSE ---
        rmse_vals = []
        for r in range(len(names)):
            rmse_vals.append(
                np.sqrt(np.mean((Ahat[:, :, r] - Atrue[:, :, r]) ** 2))
            )
        st.subheader("Per-fluorophore RMSE")
        _html_two_row_table(
            row0_label="Fluorophore",
            row1_label="RMSE",
            row0_vals=names,
            row1_vals=rmse_vals,
            fmt2=True,
        )

        return  # end emission-only mode

    # ---------- PREDICTED SPECTRA MODE ----------
    else:
        if not laser_list:
            st.error("Please specify laser wavelengths.")
            st.stop()

        # ---- Round A: provisional selection on emission-only ----
        E0, labels0, idx0 = build_emission_only_matrix(wl, dye_db, groups)
        if E0.shape[1] == 0:
            st.error("No spectra available for provisional selection.")
            st.stop()

        sel0, _ = solve_lexicographic_k(
            E0,
            idx0,
            labels0,
            levels=10,
            enforce_unique=True,
            required_count=required_count,
        )
        A_labels = [labels0[j] for j in sel0]

        # ---- (1) Compute laser powers from provisional selection ----
        if laser_strategy == "Simultaneous":
            powers_A, _ = derive_powers_simultaneous(wl, dye_db, A_labels, laser_list)
        else:
            powers_A, _ = derive_powers_separate(wl, dye_db, A_labels, laser_list)

        # ---- Build effective spectra for ALL candidates using powers_A ----
        E_raw_all, E_norm_all, labels_all, idx_all = cached_build_effective_with_lasers(
            wl,
            dye_db,
            groups,
            laser_list,
            laser_strategy,
            powers_A,
        )

        if E_norm_all.shape[1] == 0:
            st.error("No effective spectra for predicted mode.")
            st.stop()

        # ---- Final selection on effective spectra ----
        sel_idx, _ = solve_lexicographic_k(
            E_norm_all,
            idx_all,
            labels_all,
            levels=10,
            enforce_unique=True,
            required_count=required_count,
        )
        final_labels = [labels_all[j] for j in sel_idx]

        # ---- (2) Recalibrate powers based on the final selection ----
        if laser_strategy == "Simultaneous":
            powers, B = derive_powers_simultaneous(wl, dye_db, final_labels, laser_list)
        else:
            powers, B = derive_powers_separate(wl, dye_db, final_labels, laser_list)

        # ---- Build effective spectra only for the final selected set ----
        if use_pool:
            small_groups = {"Pool": [s.split(" – ", 1)[1] for s in final_labels]}
        else:
            small_groups = {}
            for s in final_labels:
                p, f = s.split(" – ", 1)
                small_groups.setdefault(p, []).append(f)

        E_raw_sel, E_norm_sel, labels_sel, _ = cached_build_effective_with_lasers(
            wl,
            dye_db,
            small_groups,
            laser_list,
            laser_strategy,
            powers,
        )

        # ---- Sort final selection by emission peak wavelength ----
        order = _sort_by_emission_peak(wl, labels_sel)
        E_raw_sel = E_raw_sel[:, order]
        E_norm_sel = E_norm_sel[:, order]
        labels_sel = [labels_sel[i] for i in order]

        colors = _ensure_colors(len(labels_sel))

        # ---- Selected fluorophores table ----
        st.subheader("Selected Fluorophores (with lasers)")
        fluors = [s.split(" – ", 1)[1] for s in labels_sel]
        _html_two_row_table(
            "Slot",
            "Fluorophore",
            [f"Slot {i+1}" for i in range(len(fluors))],
            fluors,
        )

        # ---- Pairwise similarities ----
        S = cosine_similarity_matrix(E_norm_sel)
        tops = top_k_pairwise(S, labels_sel, k=k_show)
        st.subheader("Top pairwise similarities")
        _html_two_row_table(
            "Pair", "Similarity",
            [_pair_only_fluor(a, b) for _, a, b in tops],
            [val for val, _, _ in tops],
            color_second_row=True,
            color_thresh=0.9,
            fmt2=True,
        )

        # ---- Spectra viewer (single viewer, final selection only) ----
        st.subheader("Spectra viewer")
        fig = go.Figure()
        for t in range(len(labels_sel)):
            y = E_raw_sel[:, t] / (B + 1e-12)
            fig.add_trace(go.Scatter(
                x=wl,
                y=y,
                mode="lines",
                name=labels_sel[t],
                line=dict(color=_rgb01_to_plotly(colors[t]), width=2),
            ))
        fig.update_layout(
            xaxis_title="Wavelength (nm)",
            yaxis_title="Normalized intensity (relative to B)",
            yaxis=dict(range=[0, 1.05]),
        )
        st.plotly_chart(fig, use_container_width=True)

        # ---- Simulated rods + unmixing (based on effective spectra) ----
        C = 23
        chan = 494.0 + 8.9 * np.arange(C)
        E = cached_interpolate_E_on_channels(
            wl,
            E_raw_sel / (B + 1e-12),
            chan,
        )

        Atrue, Ahat = simulate_rods_and_unmix(E, rods_per=3)

        colL, colR = st.columns(2)
        true_rgb = (colorize_composite(Atrue, colors) * 255).astype(np.uint8)
        labelmap_rgb = _argmax_labelmap(Ahat, colors)
        with colL:
            st.image(true_rgb, use_container_width=True, clamp=True)
            st.caption("True")
        with colR:
            st.image(labelmap_rgb, use_container_width=True, clamp=True)
            st.caption("Unmixing results")

        names = [_prettify_name(s) for s in labels_sel]
        unmix_bw = [_to_uint8_gray(Ahat[:, :, r]) for r in range(Ahat.shape[2])]

        st.divider()
        _show_bw_grid(
            "Per-fluorophore (Unmixing, grayscale)",
            unmix_bw,
            names,
            cols_per_row=6,
        )

        # ---- Per-fluorophore RMSE ----
        rmse_vals = []
        for r in range(len(names)):
            rmse_vals.append(
                np.sqrt(np.mean((Ahat[:, :, r] - Atrue[:, :, r]) ** 2))
            )
        st.subheader("Per-fluorophore RMSE")
        _html_two_row_table(
            row0_label="Fluorophore",
            row1_label="RMSE",
            row0_vals=names,
            row1_vals=rmse_vals,
            fmt2=True,
        )

        return


# -------------------- Execute --------------------
if __name__ == "__main__":
    run(groups, mode, laser_strategy, laser_list)
