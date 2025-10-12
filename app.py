# app.py
import pandas as pd
import streamlit as st
from utils import load_dyes_yaml, build_probe_to_fluors, intersect_fluors_with_spectra

st.subheader("Inventory Inputs (from your spreadsheet)")

# 1) Upload your Excel or CSV file
uploaded = st.file_uploader("Upload probe–fluor table (Excel/CSV)", type=["xlsx", "xls", "csv"])

probe_to_fluors = {}
parse_hint = r'^([^-\s]+)'  # Extract probe name from “Probe-Fluor” format, adjust if needed

if uploaded is not None:
    # 2) Read file
    if uploaded.name.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded)
    else:
        df = pd.read_csv(uploaded)

    st.write("Preview of uploaded table:")
    st.dataframe(df.head(10), use_container_width=True)

    # 3) Let user select which columns correspond to probe and fluorophore
    cols = list(df.columns)
    c1, c2 = st.columns(2)
    with c1:
        probe_col = st.selectbox("Select the column for probe (2nd col in your description)", options=cols, index=min(1, len(cols)-1))
    with c2:
        fluor_col = st.selectbox("Select the column for fluorophore (3rd col)", options=cols, index=min(2, len(cols)-1))

    # 4) Build probe → fluor mapping
    try:
        mapping = build_probe_to_fluors(df, probe_col=probe_col, fluor_col=fluor_col, probe_parse=parse_hint)
        # 5) Filter out fluors that have no spectra in dyes.yaml
        filtered = intersect_fluors_with_spectra(mapping, dye_db)
        missing = {p: [f for f in fls if f not in filtered.get(p, [])] for p, fls in mapping.items()}
        missing_total = sum(len(v) for v in missing.values())
        if missing_total > 0:
            st.warning(f"{missing_total} fluor(s) found in the table have no spectra in dyes.yaml; they were ignored.")

        probe_to_fluors = filtered

        # Display a few probes and their candidate fluors for verification
        st.write("Detected probe → candidate fluor mapping (filtered by spectra availability):")
        preview = {k: ", ".join(v) for k, v in list(probe_to_fluors.items())[:10]}
        st.json(preview)
    except Exception as e:
        st.error(f"Failed to parse the table: {e}")

# 6) Let user choose which probes to include in optimization
selected_probes = []
if probe_to_fluors:
    all_probes = sorted(probe_to_fluors.keys())
    selected_probes = st.multiselect("Pick probes to optimize", options=all_probes, default=all_probes[:min(5, len(all_probes))])

# 7) Build inputs for optimizer
if probe_to_fluors and selected_probes:
    probe_names = selected_probes
    group_options = [probe_to_fluors[p] for p in selected_probes]
else:
    probe_names = []
    group_options = []

st.divider()
run = st.button("Optimize")

if run:
    if not probe_names:
        st.error("Please upload the table and select at least two probes.")
    else:
        # Build emission dictionary from YAML (unchanged logic)
        E_by_dye = {name: dye_db[name]["emission"] for name in dye_db.keys()}

        # Call the optimizer (no modification needed)
        res = solve_minimax_inventory(
            probe_names=probe_names,
            group_options=group_options,
            emission_by_dye=E_by_dye,
            allow_purchase=allow_purchase,
            purchasable_pool=purch_pool,
            max_new_dyes=max_new
        )

        # Display optimization results (keep your original visualization)
        # ...
