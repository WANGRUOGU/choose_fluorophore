# utils.py
import yaml
import numpy as np
from typing import Dict, Tuple

def load_dyes_yaml(path: str) -> Tuple[np.ndarray, Dict[str, dict]]:
    """
    Load wavelengths and dye records from a YAML file.
    Returns:
      wl: np.ndarray of wavelengths (length W)
      dyes: dict name -> {emission: np.ndarray(W), excitation: np.ndarray(W), qy, ec}
    """
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    wl = np.array(data["wavelengths"], dtype=float)
    W = len(wl)

    dyes = {}
    for name, rec in data["dyes"].items():
        em = np.array(rec.get("emission", []), dtype=float)
        ex = np.array(rec.get("excitation", []), dtype=float)
        # pad/trim to match wavelength length to be robust
        if em.size == 0:
            em = np.zeros(W, dtype=float)
        if em.size != W:
            em = np.pad(em[:W], (0, max(0, W - em.size)), constant_values=0.0)

        if ex.size == 0:
            ex = np.zeros(W, dtype=float)
        if ex.size != W:
            ex = np.pad(ex[:W], (0, max(0, W - ex.size)), constant_values=0.0)

        # replace NaNs just in case
        em = np.nan_to_num(em, nan=0.0, posinf=0.0, neginf=0.0)
        ex = np.nan_to_num(ex, nan=0.0, posinf=0.0, neginf=0.0)

        dyes[name] = dict(
            emission=em,
            excitation=ex,
            qy=rec.get("quantum_yield", None),
            ec=rec.get("extinction_coeff", None),
        )
    return wl, dyes

def build_emission_dict(dyes: Dict[str, dict]) -> Dict[str, np.ndarray]:
    """
    Convenience: return dye -> emission vector dict from the loaded dye records.
    """
    return {name: rec["emission"] for name, rec in dyes.items()}

# utils.py (append these at the end)
import re
import pandas as pd
from typing import Dict, List, Optional

def build_probe_to_fluors(
    df: pd.DataFrame,
    probe_col: str,
    fluor_col: str,
    probe_parse: Optional[str] = None,
) -> Dict[str, List[str]]:
    """
    Convert a table into a mapping {probe: [fluor, ...]}.
    - probe_col: name of the column containing probe information 
                 (sometimes combined with fluor name, e.g., 'Probe-Fluor')
    - fluor_col: name of the column containing fluorophore names
    - probe_parse: optional regex or delimiter used to extract the probe name
                   if the probe column contains combined names.
                   Example: r'^([^-\s]+)' extracts the part before the first '-' or space.
                   If None, the column is used as-is for probe names.
    """
    if probe_col not in df.columns or fluor_col not in df.columns:
        raise ValueError(f"Columns not found: {probe_col}, {fluor_col}")

    def extract_probe(v: str) -> str:
        s = str(v).strip()
        if probe_parse:
            m = re.match(probe_parse, s)
            if m:
                return m.group(1).strip()
        return s

    mapping: Dict[str, List[str]] = {}
    for _, row in df[[probe_col, fluor_col]].dropna().iterrows():
        probe = extract_probe(row[probe_col])
        fluor = str(row[fluor_col]).strip()
        if len(probe) == 0 or len(fluor) == 0:
            continue
        mapping.setdefault(probe, [])
        if fluor not in mapping[probe]:
            mapping[probe].append(fluor)
    return mapping


def intersect_fluors_with_spectra(
    mapping: Dict[str, List[str]],
    dye_db: Dict[str, dict],
) -> Dict[str, List[str]]:
    """
    Remove fluorophores that are not present in the YAML spectral database.
    Returns a filtered mapping with only available fluorophores.
    """
    available = set(dye_db.keys())
    filtered: Dict[str, List[str]] = {}
    for probe, fls in mapping.items():
        keep = [f for f in fls if f in available]
        if keep:
            filtered[probe] = keep
    return filtered

