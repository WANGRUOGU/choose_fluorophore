# app.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

from PIL import Image
from utils import load_dyes_yaml, build_emission_dict
from optimizer import solve_minimax_inventory

st.set_page_config(page_title="Probe–Dye Optimizer", layout="wide")

logo = Image.open("assets/lab logo.jpg")
st.image(logo, width=180)

st.title("Choosing Fluorophores via Spectral Similarity Minimization")

# ---- Sidebar: data & settings ----
with st.sidebar:
    st.header("Data")
    yaml_path = st.text_input("Dye spectra YAML", "spectra/dyes.yaml")
    wl, dye_db = load_dyes_yaml(yaml_path)
    all_dyes = sorted(list(dye_db.keys()))

    st.header("Optimization Mode")
    allow_purchase = st.checkbox("Allow purchasable dyes (extend beyond inventory)", value=False)
    purch_pool = []
    max_new = 0
    if allow_purchase:
        purch_pool = st.multiselect("Purchasable dye pool", options=all_dyes, default=[])
        max_new = st.number_input("Max new dyes to purchase", min_value=0, value=1, step=1)

st.subheader("Inventory Inputs")

# Let user define number of probes and pick inventory options
P = st.number_input("Number of probes", min_value=2, max_value=24, value=3, step=1)

probe_names = []
group_options = []
colL, colR = st.columns(2)

for g in range(P):
    with (colL if g % 2 == 0 else colR):
        pname = st.text_input(f"Probe {g+1} name", value=f"Probe{g+1}", key=f"pname_{g}")
        opts  = st.multiselect(f"Inventory dyes for {pname}", options=all_dyes, default=[], key=f"opts_{g}")
        probe_names.append(pname)
        group_options.append(opts)

st.divider()
run = st.button("Optimize")

# ---- Run optimizer ----
if run:
    # Validate inputs
    if any(len(opts)==0 for opts in group_options):
        st.error("Each probe must have at least one inventory dye option.")
    else:
        # Build emission dict
        E_by_dye = {name: dye_db[name]["emission"] for name in all_dyes}

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
        df = pd.DataFrame({
            "Probe": probe_names,
            "Chosen (Probe / Dye)": res["chosen_by_probe"]
        })
        st.dataframe(df, use_container_width=True)

        # Heatmap of pairwise cosine among chosen
        sim = np.array(res["sim_pp"])
        fig = px.imshow(sim, x=probe_names, y=probe_names, zmin=0, zmax=1,
                        color_continuous_scale="RdBu_r", aspect="auto",
                        title="Cosine Similarity (Chosen Set)")
        st.plotly_chart(fig, use_container_width=True)

        # Top similarities list
        top_df = pd.DataFrame([
            {"Pair": f"{probe_names[i]}–{probe_names[j]}", "Cosine": s}
            for (i,j), s in res["top_pairs"]
        ])
        st.write("Top pairwise similarities (descending):")
        st.dataframe(top_df, use_container_width=True)

        # Purchases (if any)
        if res["purchased"]:
            st.info("Purchased dyes: " + ", ".join(res["purchased"]))

        # Export
        st.download_button(
            "Download assignment CSV",
            df.to_csv(index=False),
            file_name="assignment.csv",
            mime="text/csv"
        )
