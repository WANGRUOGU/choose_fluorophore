import os
import yaml
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from utils import load_dyes_yaml
from optimizer import solve_minimax_inventory

st.set_page_config(page_title="Probe–Dye Optimizer", layout="wide")

# Optional: show a logo if you store it in assets/
# from PIL import Image
# logo = Image.open("assets/valm_lab_logo.png")
# st.image(logo, width=160)

st.title("Probe–Dye Selection via Spectral Similarity Minimization")

# ---- Sidebar: load spectral DB and options ----
with st.sidebar:
    st.header("Data & Options")
    yaml_path = st.text_input("Spectra YAML path", "spectra/dyes.yaml")
    wl, dye_db = load_dyes_yaml(yaml_path)
    all_dyes = sorted(dye_db.keys())

    st.markdown("---")
    st.subheader("Purchasable dyes (optional)")
    allow_purchase = st.checkbox("Allow purchasable dyes", value=False)
    purch_pool = []
    max_new = 0
    if allow_purchase:
        purch_pool = st.multiselect("Purchasable dye pool", options=all_dyes, default=[])
        max_new = st.number_input("Max new dyes to purchase", min_value=0, value=1, step=1)

st.subheader("Probe–Fluor Mapping (from repository)")

# Load mapping from YAML committed to the repo
mapping_file = "data/probe_fluor_map.yaml"
if not os.path.exists(mapping_file):
    st.error(f"Mapping file not found: {mapping_file}. Please add it to the repository.")
    st.stop()

with open(mapping_file, "r", encoding="utf-8") as f:
    pf = yaml.safe_load(f)

# Build mapping: probe -> [fluors...]
probe_to_fluors = {entry["name"]: entry["fluors"] for entry in pf.get("probes", [])}

# Warn if some fluors in mapping do not exist in spectra/dyes.yaml
missing_pairs = []
filtered_mapping = {}
for probe, fls in probe_to_fluors.items():
    keep = [f for f in fls if f in dye_db]
    if len(keep) < len(fls):
        for f in fls:
            if f not in dye_db:
                missing_pairs.append((probe, f))
    if keep:
        filtered_mapping[probe] = keep

if missing_pairs:
    st.warning(f"{len(missing_pairs)} fluor(s) in mapping have no spectra in {yaml_path} and were ignored.")

# Preview mapping
if not filtered_mapping:
    st.error("No valid probe→fluor mapping after filtering by spectra. Please update mapping or spectra.")
    st.stop()

preview = {k: ", ".join(v) for k, v in list(filtered_mapping.items())[:12]}
st.write("Detected probe → available fluor mapping (filtered):")
st.json(preview)

# Let the user select which probes to optimize
all_probes = sorted(filtered_mapping.keys())
default_n = min(5, len(all_probes))
selected_probes = st.multiselect("Pick probes to optimize", options=all_probes, default=all_probes[:default_n])

# Prepare optimizer inputs
if selected_probes:
    probe_names = selected_probes
    group_options = [filtered_mapping[p] for p in selected_probes]
else:
    probe_names = []
    group_options = []

st.divider()
run = st.button("Optimize")

if run:
    if len(probe_names) < 2:
        st.error("Please select at least two probes.")
        st.stop()

    # Build emission dictionary from spectral DB
    E_by_dye = {name: dye_db[name]["emission"] for name in dye_db.keys()}

    # Call optimizer (single-level minimax)
    res = solve_minimax_inventory(
        probe_names=probe_names,
        group_options=group_options,
        emission_by_dye=E_by_dye,
        allow_purchase=allow_purchase,
        purchasable_pool=purch_pool,
        max_new_dyes=max_new
    )

    st.success(f"Optimal max similarity t* = {res['t']:.3f}")

    # Show chosen assignment
    df_assign = pd.DataFrame({
        "Probe": probe_names,
        "Chosen (Probe / Dye)": res["chosen_by_probe"]
    })
    st.dataframe(df_assign, use_container_width=True)

    # Heatmap of pairwise cosine among chosen
    sim = np.array(res["sim_pp"])
    fig = px.imshow(
    sim, 
    x=probe_names, 
    y=probe_names, 
    zmin=0, 
    zmax=1,
    color_continuous_scale="RdBu_r", 
    aspect="auto",
    title="Cosine Similarity (Chosen Set)"
)

    st.plotly_chart(fig, use_container_width=True)

    # Top similarities
    top_df = pd.DataFrame([
        {"Pair": f"{probe_names[i]}–{probe_names[j]}", "Cosine": s}
        for (i, j), s in res["top_pairs"]
    ])
    st.write("Top pairwise similarities (descending):")
    st.dataframe(top_df, use_container_width=True)

    # Purchases (if any)
    if res["purchased"]:
        st.info("Purchased dyes: " + ", ".join(res["purchased"]))

    # Export
    st.download_button(
        "Download assignment CSV",
        df_assign.to_csv(index=False),
        file_name="assignment.csv",
        mime="text/csv"
    )
