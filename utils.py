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
