import yaml
import numpy as np

# ------------- I/O -------------

def load_dyes_yaml(path):
    """Load dyes.yaml -> (wavelengths, dye_db).
    dye_db: dict[name] = {"emission": np.array, "excitation": np.array, "quantum_yield": float|None, "extinction_coeff": float|None}
    """
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    wl = np.array(data["wavelengths"], dtype=float)
    dye_db = {}
    for name, rec in data["dyes"].items():
        em = np.array(rec.get("emission", []), dtype=float)
        ex = np.array(rec.get("excitation", []), dtype=float)
        qy = rec.get("quantum_yield", None)
        ec = rec.get("extinction_coeff", None)
        dye_db[name] = dict(emission=em, excitation=ex, quantum_yield=qy, extinction_coeff=ec)
    return wl, dye_db


def load_probe_fluor_map(path):
    """Load mapping: supports two YAML schemas:
       1) {mapping: {ProbeA: [AF488, ...], ...}}
       2) {ProbeA: [AF488, ...], ProbeB: [...]}
       Returns a dict[str, list[str]].
    """
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}

    # Accept both schemas
    if isinstance(data, dict) and "mapping" in data and isinstance(data["mapping"], dict):
        mapping = data["mapping"]
    elif isinstance(data, dict):
        mapping = data  # assume direct mapping at the top level
    else:
        raise ValueError(
            "probe_fluor_map.yaml must be a mapping. "
            "Either use {'mapping': {...}} or put the probe→fluor list at the top level."
        )

    # Normalize to dict[str, list[str]]
    clean = {}
    for k, v in mapping.items():
        key = str(k)
        if v is None:
            clean[key] = []
        elif isinstance(v, (list, tuple)):
            clean[key] = [str(x) for x in v]
        else:
            # allow single string -> wrap into list
            clean[key] = [str(v)]
    return clean




# ------------- Spectra builders -------------

def _safe_norm(x):
    n = np.linalg.norm(x)
    return x / (n + 1e-12)

def build_effective_emission_emission_only(wl, dye_db, groups):
    """Return (E, labels) for emission-only mode.
    E: [W x N] matrix, concatenating candidates in the given group order.
    Each column is L2-normalized (cosine similarity setting).
    """
    W = len(wl)
    cols = []
    labels = []
    for g, cands in enumerate(groups):
        for name in cands:
            em = np.array(dye_db[name]["emission"], dtype=float)
            if em.size != W:
                # if dimension mismatch, skip
                continue
            cols.append(_safe_norm(em))
            labels.append(name)
    if not cols:
        return np.zeros((W, 0)), []
    E = np.stack(cols, axis=1)
    return E, labels


def build_effective_emission_with_lasers(
    wl, dye_db, candidates, laser_wavelengths, mode, powers, selection_for_base=None
):
    """Build effective spectra under lasers for a list of candidate fluorophores.
    - mode: "Simultaneous" or "Separate"
    - powers: list of laser powers (already determined)
    - selection_for_base: (unused in this builder; included for API symmetry)
    Returns [W x N] matrix (not normalized), labels (candidates).
    """
    W = len(wl)
    if len(candidates) == 0:
        return np.zeros((W, 0)), []

    lam = np.array(sorted(laser_wavelengths), dtype=float)
    pw = np.array([powers[laser_wavelengths.index(l)] for l in lam], dtype=float)  # align power to sorted lasers

    segments = []
    for i in range(len(lam)):
        lo = lam[i]
        hi = lam[i+1] if i+1 < len(lam) else wl[-1] + 1  # last segment goes to end
        segments.append((lo, hi))

    cols = []
    labels = []
    for name in candidates:
        rec = dye_db[name]
        em = np.array(rec["emission"], dtype=float)
        ex = np.array(rec["excitation"], dtype=float)
        qy = rec["quantum_yield"]
        ec = rec["extinction_coeff"]

        if any(v is None for v in (qy, ec)) or em.size != W or ex.size != W:
            # missing data => cannot build in laser mode
            cols.append(np.zeros(W))
            labels.append(name)
            continue

        # precompute excitation at laser wavelengths
        # (nearest index; wavelengths assumed integer grid)
        eff = np.zeros(W)
        for i, l in enumerate(lam):
            idx = int(l - wl[0])
            if idx < 0 or idx >= W:
                continue
            k_i = ex[idx] * qy * ec * pw[i]
            if mode == "Separate":
                # whole spectrum scaled (separate firing)
                eff += em * k_i
            else:
                # Simultaneous: only add to its segment (piecewise)
                lo, hi = segments[i]
                loi = int(max(lo - wl[0], 0))
                hii = int(min(hi - wl[0], W))
                eff[loi:hii] += em[loi:hii] * k_i

        cols.append(eff)
        labels.append(name)

    if not cols:
        return np.zeros((W, 0)), []
    M = np.stack(cols, axis=1)
    return M, labels


# ------------- Laser power derivation (per your rule set) -------------

def _segment_indices(wl, lam_sorted, s_idx):
    """Return slice indices [loi:hii) for segment s_idx defined by lam_sorted."""
    W = len(wl)
    lo = lam_sorted[s_idx]
    hi = lam_sorted[s_idx+1] if s_idx+1 < len(lam_sorted) else wl[-1] + 1
    loi = int(max(lo - wl[0], 0))
    hii = int(min(hi - wl[0], W))
    return loi, hii

def _max_segment_emission(em, loi, hii):
    """max of emission on [loi:hii)."""
    if loi >= hii:
        return 0.0
    return float(np.max(em[loi:hii]))

def derive_powers_simultaneous(wl, dye_db, selection, laser_wavelengths):
    """Simultaneous lasers: compute powers so each segment’s max across the selected set equals B,
    where B is the max peak of the first informative segment (as you described)."""
    lam = np.array(sorted(laser_wavelengths), dtype=float)
    W = len(wl)

    # Gather for selected A
    recs = [dye_db[name] for name in selection]

    # Find the first segment that has any nonzero emission peak among A to start (its power = 1)
    start_seg = 0
    while start_seg < len(lam):
        loi, hii = _segment_indices(wl, lam, start_seg)
        # if any selected has positive peak in this segment:
        has_peak = False
        for rec in recs:
            em = np.array(rec["emission"], dtype=float)
            if em.size != W:
                continue
            if _max_segment_emission(em, loi, hii) > 0:
                has_peak = True
                break
        if has_peak:
            break
        start_seg += 1
    if start_seg >= len(lam):
        # degenerate: no emission in any segment; set all powers to 1
        return [1.0] * len(lam)

    # Helper: a_i = ex(l1)*qy*ec ; b_i = ex(l2)*qy*ec ; etc.
    def coef_at_l(rec, l):
        em = np.array(rec["emission"], dtype=float)
        ex = np.array(rec["excitation"], dtype=float)
        qy = rec["quantum_yield"]
        ec = rec["extinction_coeff"]
        if any(v is None for v in (qy, ec)) or ex.size != W or em.size != W:
            return None
        idx = int(l - wl[0])
        if idx < 0 or idx >= W:
            return 0.0
        return float(ex[idx] * qy * ec)

    # Segment maxima M_i(s)
    def seg_peak(rec, loi, hii):
        em = np.array(rec["emission"], dtype=float)
        if em.size != W:
            return 0.0
        return _max_segment_emission(em, loi, hii)

    # (1) set P[start_seg] = 1, compute B = max_i a_i * M_i(start_seg)
    P = np.zeros(len(lam))
    P[start_seg] = 1.0

    loi, hii = _segment_indices(wl, lam, start_seg)
    a = np.array([coef_at_l(r, lam[start_seg]) or 0.0 for r in recs])
    Mseg = np.array([seg_peak(r, loi, hii) for r in recs])
    B = float(np.max(a * Mseg)) if np.any(Mseg > 0) else 0.0

    # (2) march forward: for segment s>start, require max_i (M_i(s)*(sum_{m<=s} c_i(m)*P[m])) = B
    for s in range(start_seg + 1, len(lam)):
        loi, hii = _segment_indices(wl, lam, s)
        Mseg = np.array([seg_peak(r, loi, hii) for r in recs])
        # pre-sum (already set powers) contribution
        pre = np.zeros(len(recs))
        for m in range(start_seg, s):
            c = np.array([coef_at_l(r, lam[m]) or 0.0 for r in recs])
            pre += c * P[m]
        # now solve for P[s] ≤ min_i (B / (M_i * c_i(s)) - pre_i / c_i(s)) over those with M_i>0 and c_i(s)>0
        c_s = np.array([coef_at_l(r, lam[s]) or 0.0 for r in recs])
        feasible = (Mseg > 0) & (c_s > 0)
        if not np.any(feasible):
            P[s] = 0.0
            continue
        bounds = (B / (Mseg[feasible] * c_s[feasible])) - (pre[feasible] / c_s[feasible])
        P[s] = max(0.0, float(np.min(bounds)))

    return [float(p) for p in P]


def derive_powers_separate(wl, dye_db, selection, laser_wavelengths):
    """Separate lasers: each laser produces full-spectrum emission scaled by ex(l)*qy*ec*P_l.
    Set P_l so that each laser’s global peak across selection equals a common B (the maximum of the “raw” peaks with P=1)."""
    lam = np.array(sorted(laser_wavelengths), dtype=float)
    W = len(wl)

    recs = [dye_db[name] for name in selection]

    def coef_at_l(rec, l):
        em = np.array(rec["emission"], dtype=float)
        ex = np.array(rec["excitation"], dtype=float)
        qy = rec["quantum_yield"]
        ec = rec["extinction_coeff"]
        if any(v is None for v in (qy, ec)) or ex.size != W or em.size != W:
            return None
        idx = int(l - wl[0])
        if idx < 0 or idx >= W:
            return 0.0
        return float(ex[idx] * qy * ec)

    # M_l(A) = max_i (max_w em_i(w)) * coef_i(l)
    M_l = []
    for l in lam:
        peaks = []
        for rec in recs:
            em = np.array(rec["emission"], dtype=float)
            c = coef_at_l(rec, l)
            if em.size != W or c is None:
                continue
            peaks.append(np.max(em) * c)
        M_l.append(max(peaks) if peaks else 0.0)
    M_l = np.array(M_l, dtype=float)

    if np.all(M_l <= 0):
        return [1.0] * len(lam)  # degenerate

    B = float(np.max(M_l))
    P = []
    for val in M_l:
        if val <= 0:
            P.append(0.0)
        else:
            P.append(float(B / val))
    return P


# ------------- Iteration wrapper -------------

def iterate_selection_with_lasers(
    wl, dye_db, groups, laser_wavelengths, mode, init_selection, max_iter=8
):
    """Iterate:
      - Start from selection A (from emission-only).
      - Compute laser powers from A (per your rule).
      - Build effective spectra for ALL candidates using those powers.
      - Re-solve selection (minimax by default here).
      - Repeat until selection stabilizes or reaches max_iter.
    Returns: (selected_labels, powers, iters, converged, E_final, labels_final)
    """
    selected = list(init_selection)
    powers = None
    converged = False
    labels_final = None
    E_final = None

    for it in range(1, max_iter + 1):
        # Derive powers from current selection
        if mode == "Simultaneous":
            powers = derive_powers_simultaneous(wl, dye_db, selected, laser_wavelengths)
        else:
            powers = derive_powers_separate(wl, dye_db, selected, laser_wavelengths)

        # Build effective spectra for all candidates across all groups
        all_candidates = [name for group in groups for name in group]
        E_all, labels_all = build_effective_emission_with_lasers(
            wl, dye_db, all_candidates, laser_wavelengths, mode, powers
        )

        # Normalize columns for cosine selection
        E_norm = E_all / (np.linalg.norm(E_all, axis=0, keepdims=True) + 1e-12)

        # Rebuild groups indices in this concatenated label list
        idx_groups = []
        start = 0
        for group in groups:
            idxs = []
            for name in group:
                if name in labels_all:
                    idxs.append(labels_all.index(name))
            idx_groups.append(idxs)

        # Re-select (single-level minimax; you can swap to lexicographic if desired)
        sel_idx, t_star = select_minimax_subset_grouped(E_norm, idx_groups, preindexed=True)
        new_selected = [labels_all[i] for i in sel_idx]

        if new_selected == selected:
            converged = True
            E_final, labels_final = E_all, labels_all
            break
        selected = new_selected
        E_final, labels_final = E_all, labels_all

    return selected, powers, it, converged, E_final, labels_final


# ------------- Optimization (grouped) -------------

def cosine_similarity_matrix(E):
    """Cosine similarity matrix of columns of E (E: W x N)."""
    N = E.shape[1]
    norms = np.linalg.norm(E, axis=0) + 1e-12
    G = (E.T @ E) / np.outer(norms, norms)
    np.fill_diagonal(G, 0.0)
    return G

def select_minimax_subset_grouped(E, groups, preindexed=False):
    """Greedy minimax selection under group partition.
    groups:
      - if preindexed=False: list of lists of strings (ignored here), we deduce group sizes
      - if preindexed=True : list of lists of column indices into E
    Returns (sel_indices, t_star).
    """
    # Build index groups
    if preindexed:
        idx_groups = groups
    else:
        # when groups are lists of names, E built in the same order; each subgroup contiguous
        idx_groups = []
        cur = 0
        for cands in groups:
            idx_groups.append(list(range(cur, cur + len(cands))))
            cur += len(cands)

    # Greedy per group: choose candidate minimizing the current worst similarity
    chosen = []
    for g, idxs in enumerate(idx_groups):
        best_idx = None
        best_worst = np.inf
        for j in idxs:
            trial = chosen + [j]
            S = cosine_similarity_matrix(E[:, trial])
            worst = float(np.max(S))
            if worst < best_worst:
                best_worst = worst
                best_idx = j
        chosen.append(best_idx)
    S_final = cosine_similarity_matrix(E[:, chosen])
    return chosen, float(np.max(S_final))


def select_lexicographic_grouped(E, groups):
    """Placeholder for your existing lexicographic grouped solver (not re-implemented here).
    If you need the full MILP version wired back, we can insert it; the GUI already toggles this option.
    """
    return select_minimax_subset_grouped(E, groups)


# ------------- Display helpers -------------

def top_k_pairwise(S, k=10):
    """Return top-k largest off-diagonal similarities and their pairs."""
    N = S.shape[0]
    tri = np.triu_indices(N, k=1)
    vals = S[tri]
    if vals.size == 0:
        return [], []
    order = np.argsort(-vals)  # descending
    order = order[: min(k, vals.size)]
    top_vals = vals[order].tolist()
    pairs = list(zip(tri[0][order], tri[1][order]))
    return top_vals, pairs

def normalize_probe_mapping(mapping_raw, dye_db):
    """Clean and filter the probe→fluor mapping:
       - strip whitespace on probe & fluor names
       - coerce non-list values to list
       - deduplicate candidates
       - keep ONLY candidates that appear in dye_db
       - drop probes that end up with empty candidate lists
       Returns dict[str, list[str]].
    """
    clean = {}
    for probe, vals in (mapping_raw or {}).items():
        p = str(probe).strip()
        if not p:
            continue
        if isinstance(vals, (list, tuple)):
            lst = [str(v).strip() for v in vals if str(v).strip()]
        elif vals is None:
            lst = []
        else:
            lst = [str(vals).strip()]
        # unique, keep only dyes present in dye_db
        lst = sorted({v for v in lst if v in dye_db})
        if lst:
            clean[p] = lst
    return clean
