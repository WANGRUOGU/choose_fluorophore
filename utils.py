# utils.py
import yaml
import numpy as np
from typing import Dict, List, Tuple

def load_dyes_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    wl = np.array(data["wavelengths"], dtype=float)
    dyes = {}
    for name, rec in data["dyes"].items():
        dyes[name] = dict(
            emission=np.array(rec["emission"], dtype=float),
            excitation=np.array(rec.get("excitation") or [0]*len(wl), dtype=float),
            qy=rec.get("quantum_yield"),
            ec=rec.get("extinction_coeff"),
        )
    return wl, dyes

def build_emission_dict(dyes: Dict[str, dict]) -> Dict[str, np.ndarray]:
    return {name: rec["emission"] for name, rec in dyes.items()}
