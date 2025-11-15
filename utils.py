import yaml
import numpy as np
import pulp

# --- CBC solver: no time/gap limits (guarantee optimality), quiet log ---
def _make_cbc_exact():
    try:
        return pulp.PULP_CBC_CMD(msg=False, mip=True)
    except TypeError:
        return pulp.PULP_CBC_CMD(msg=False)

_SOLVER = _make_cbc_exact()


# ====================== I/O ======================

def load_dyes_yaml(path):
    """
    Load dyes.yaml -> (wavelengths, dye_db).
    dye_db[name] = {
        "emission": np.array(W,),
        "excitation": np.array(W,),
        "quantum_yield": float|None,
        "extinction_coeff": float|None
    }
    Missing QY will be filled with the mean of available QYs.
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
        dye_db[name] = dict(
            emission=em, excitation=ex, quantum_yield=qy, extinction_coeff=ec
        )

    qys = [v["quantum_yield"] for v in dye_db.values() if v.get("quantum_yield") is not None]
    mean_qy = float(np.mean(qys)) if len(qys) else 1.0
    for v in dye_db.values():
        if v.get("quantum_yield") is None:
            v["quantum_yield"] = mean_qy

    return wl, dye_db


def load_probe_fluor_map(path):
    """
    Accepts:
      - top-level list of {name: <probe>, fluors: [...]}
      - or {probes: [...] } with same structure.
    Returns dict[probe] -> list[fluor].
    """
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if isinstance(data, list):
        items = data
    elif isinstance(data, dict) and isinstance(data.get("probes"), list):
        items = data["probes"]
    else:
        return {}

    mapping = {}
    for it in items:
        if not isinstance(it, dict):
            continue
        name = str(it.get("name", "")).strip()
        fls = it.get("fluors", []) or []
        if not name:
            continue
        if not isinstance(fls, (list, tuple)):
            fls = [str(fls).strip()] if str(fls).strip() else []
        mapping[name] = [str(x).strip() for x in fls if str(x).strip()]
    return mapping


# =================== Linear-algebra helpers ===================

def _safe_l2norm_cols(E):
    """L2-normalize each column with numerical guard."""
    denom = np.linalg.norm(E, axis=0, keepdims=True) + 1e-12
    return E / denom

def cosine_similarity_matrix(E):
    """
    Cosine similarity among columns of E.
    Diagonal is set to 0 so it's easy to take pairwise maxima.
    """
    norms = np.linalg.norm(E, axis=0) + 1e-12
    G = (E.T @ E) / np.outer(norms, norms)
    np.fill_diagonal(G, 0.0)
    return G

def top_k_pairwise(S, labels_pair, k=10):
    """
    Given NxN cosine matrix S (diag=0), return top-k pairs.
    Each item: (value, label_i, label_j), sorted descending by value.
    """
    N = S.shape[0]
    iu = np.triu_indices(N, k=1)
    vals = S[iu]
    if vals.size == 0:
        return []
    order = np.argsort(-vals)[: min(k, vals.size)]
    out = []
    for idx in order:
        i = iu[0][idx]; j = iu[1][idx]
        out.append((float(vals[idx]), labels_pair[i], labels_pair[j]))
    return out


# =================== Spectra builders ===================

def build_emission_only_matrix(wl, dye_db, groups):
    """
    Build peak-normalized emission-only matrix for optimization.
    Returns:
      E_norm: W x N (L2-normalized by column)
      labels_pair: list[str] "Probe – Fluor"
      idx_groups: list[list[int]] column indices per probe group, in order.
    """
    W = len(wl)
    cols, labels, idx_groups = [], [], []
    col_id = 0
    for probe, cand_list in groups.items():
        idxs = []
        for fluor in cand_list:
            rec = dye_db.get(fluor)
            if rec is None:
                continue
            em = np.array(rec["emission"], dtype=float)
            if em.size != W:
                continue
            m = np.max(em) if np.max(em) > 0 else 1.0
            em_peak = em / m
            cols.append(em_peak)
            labels.append(f"{probe} – {fluor}")
            idxs.append(col_id)
            col_id += 1
        if idxs:
            idx_groups.append(idxs)

    if not cols:
        return np.zeros((W, 0)), [], []
    E = np.stack(cols, axis=1)
    E_norm = _safe_l2norm_cols(E)
    return E_norm, labels, idx_groups


def _nearest_idx_from_grid(wl, lam):
    """Assuming 1 nm grid starting at wl[0]; pick nearest index."""
    idx = int(round(lam - wl[0]))
    if idx < 0: idx = 0
    if idx >= len(wl): idx = len(wl) - 1
    return idx

def _segments_from_lasers(wl, lasers_sorted):
    """Segments [lo, hi) defined by sorted laser wavelengths."""
    segs = []
    for i, l in enumerate(lasers_sorted):
        lo = l
        hi = lasers_sorted[i+1] if i+1 < len(lasers_sorted) else wl[-1] + 1
        segs.append((lo, hi))
    return segs

def _interp_at(w, y, x):
    """1D linear interpolation on a discrete grid (clamped)."""
    if x <= w[0]: return float(y[0])
    if x >= w[-1]: return float(y[-1])
    i = int(np.searchsorted(w, x)) - 1
    t = (x - w[i]) / (w[i+1] - w[i])
    return float(y[i]*(1-t) + y[i+1]*t)


def derive_powers_simultaneous(wl, dye_db, selection_labels, laser_wavelengths):
    """
    Simultaneous firing with 'useful-segment gating':
      - A segment [lam[i], lam[i+1]) is 'useful' iff at least one selected dye has its
        global emission peak located inside that segment. Otherwise set P[i]=0 and skip it.
      - Among useful segments, find the first one in wavelength order; set its power=1 and
        define B = max_i( seg_peak_i * coef_i(lam[start]) ).
      - For each subsequent useful segment s, choose the smallest nonnegative P[s] so that
        the segment's peak does not exceed B when adding cumulative contributions from all
        lasers with indices <= s (inside that segment we sum coefficients of all m<=s).
    Returns (powers_sorted_by_wavelength, B).
    """
    lam = np.array(sorted(laser_wavelengths), dtype=float)
    segs = _segments_from_lasers(wl, lam)
    W = len(wl)

    # Selected dye records
    fls = [s.split(" – ", 1)[1] for s in selection_labels]
    recs = [dye_db[f] for f in fls if f in dye_db]

    # Helper: excitation*QY*EC at laser l (nearest-grid)
    def coef(rec, l):
        ex = rec["excitation"]; qy = rec["quantum_yield"]; ec = rec["extinction_coeff"]
        if ex is None or len(ex) != W or qy is None or ec is None:
            return 0.0
        idx = _nearest_idx_from_grid(wl, l)
        return float(ex[idx] * qy * (ec if ec is not None else 1.0))

    # Helper: peak emission inside [lo,hi)
    def seg_peak(rec, lo, hi):
        em = rec["emission"]
        if em is None or len(em) != W:
            return 0.0
        loi = _nearest_idx_from_grid(wl, lo)
        hii = _nearest_idx_from_grid(wl, hi - 1) + 1
        if loi >= hii:
            return 0.0
        return float(np.max(em[loi:hii]))

    # ---- decide which segments are 'useful' by global emission peak location ----
    useful = [False] * len(segs)
    # precompute each dye's global-peak wavelength
    for rec in recs:
        em = rec.get("emission", None)
        if em is None or len(em) != W:
            continue
        jmax = int(np.argmax(em))  # if ties, first max is fine
        lam_peak = wl[jmax]
        # mark the segment containing lam_peak
        for s, (lo, hi) in enumerate(segs):
            if lo <= lam_peak < hi:
                useful[s] = True
                break

    # if none useful, trivial
    if not any(useful):
        return [0.0] * len(lam), 0.0

    # ---- find first useful segment and initialize B ----
    start = None
    for s, u in enumerate(useful):
        if u:
            start = s
            break

    P = np.zeros(len(lam), dtype=float)
    P[start] = 1.0

    lo0, hi0 = segs[start]
    M0 = np.array([seg_peak(r, lo0, hi0) for r in recs])
    a0 = np.array([coef(r, lam[start]) for r in recs])
    B = float(np.max(a0 * M0)) if np.any(M0 > 0) else 0.0

    # ---- march forward over later segments ----
    for s in range(start + 1, len(lam)):
        if not useful[s]:
            P[s] = 0.0
            continue
        lo, hi = segs[s]
        M = np.array([seg_peak(r, lo, hi) for r in recs])
        c_s = np.array([coef(r, lam[s]) for r in recs])

        # cumulative contribution from earlier lasers (only those we kept)
        pre = np.zeros(len(recs))
        for m in range(start, s):
            if P[m] <= 0.0:
                continue
            c_m = np.array([coef(r, lam[m]) for r in recs])
            pre += c_m * P[m]

        feasible = (M > 0) & (c_s > 0)
        if not np.any(feasible):
            P[s] = 0.0
        else:
            # ensure: (pre + c_s*P[s]) * M <= B  =>  P[s] <= B/(M*c_s) - pre/c_s
            bounds = (B / (M[feasible] * c_s[feasible])) - (pre[feasible] / c_s[feasible])
            P[s] = max(0.0, float(np.min(bounds)))

    return [float(x) for x in P], float(B)



def derive_powers_separate(wl, dye_db, selection_labels, laser_wavelengths):
    """
    Separate mode calibration (per your spec):
    - Optimize on emission-only to get a set A.
    - Set the smallest-wavelength laser to power=1 and define B by that laser's reachable peak.
    - For each other laser, scale so its reachable peak equals B.
    Returns (powers_sorted, B).
    """
    lam = np.array(sorted(laser_wavelengths), dtype=float)
    W = len(wl)
    fls = [s.split(" – ", 1)[1] for s in selection_labels]
    recs = [dye_db[f] for f in fls if f in dye_db]

    def coef(rec, l):
        ex = rec["excitation"]; qy = rec["quantum_yield"]; ec = rec["extinction_coeff"]
        if ex is None or len(ex) != W or qy is None:
            return 0.0
        ex_l = _interp_at(wl, ex, l)
        return float(ex_l * qy * (ec if ec is not None else 1.0))

    # reachable peak per laser across selected dyes
    M = []
    for l in lam:
        peaks = []
        for r in recs:
            em = r["emission"]
            if em is None or len(em) != W:
                continue
            peaks.append(np.max(em) * coef(r, l))
        M.append(max(peaks) if peaks else 0.0)
    M = np.array(M, dtype=float)

    P = np.zeros_like(M)
    if M.size == 0:
        return [1.0]*0, 1.0
    P[0] = 1.0
    B = float(M[0])

    for i in range(1, len(M)):
        P[i] = float(B / M[i]) if M[i] > 0 else 0.0

    return [float(x) for x in P], B


def build_effective_with_lasers(wl, dye_db, groups, laser_wavelengths, mode, powers):
    """
    Build effective spectra for ALL candidates.
    - Simultaneous: additive inside each laser's segment; cumulative coefficient within segment.
    - Separate: build a block per laser and horizontally concatenate the blocks per dye.
    Returns:
      E_raw:  W x N  (Simultaneous)  OR  (W*L) x N (Separate)
      E_norm: same shape, L2 norm by column for optimization
      labels_pair, idx_groups
    """
    W = len(wl)
    lam = np.array(sorted(laser_wavelengths), dtype=float)
    pw = np.array(powers, dtype=float)

    def _nearest_idx_from_grid_local(wl, lam):
        idx = int(round(lam - wl[0]))
        if idx < 0: idx = 0
        if idx >= len(wl): idx = len(wl) - 1
        return idx

    def _segments_from_lasers_local(wl, lasers_sorted):
        segs = []
        for i, l in enumerate(lasers_sorted):
            lo = l
            hi = lasers_sorted[i+1] if i+1 < len(lasers_sorted) else wl[-1] + 1
            segs.append((lo, hi))
        return segs

    cols, labels, idx_groups = [], [], []
    col_id = 0

    if mode == "Separate":
        for probe, cand_list in groups.items():
            idxs = []
            for fluor in cand_list:
                rec = dye_db.get(fluor)
                if rec is None:
                    continue
                em = rec["emission"]; ex = rec["excitation"]
                qy = rec["quantum_yield"]; ec = rec["extinction_coeff"]
                if em is None or ex is None or len(em) != W or len(ex) != W:
                    continue

                per_laser_blocks = []
                for i, l in enumerate(lam):
                    k = _interp_at(wl, ex, l) * qy * (ec if ec is not None else 1.0) * pw[i]
                    per_laser_blocks.append(em * k)
                eff_concat = np.concatenate(per_laser_blocks, axis=0)
                cols.append(eff_concat)
                labels.append(f"{probe} – {fluor}")
                idxs.append(col_id)
                col_id += 1
            if idxs:
                idx_groups.append(idxs)

        if not cols:
            Z = np.zeros((W * max(1, len(lam)), 0))
            return Z, Z, [], []

        E_raw = np.stack(cols, axis=1)
        denom = np.linalg.norm(E_raw, axis=0, keepdims=True) + 1e-12
        E_norm = E_raw / denom
        return E_raw, E_norm, labels, idx_groups

    else:
        segs = _segments_from_lasers_local(wl, lam)
        for probe, cand_list in groups.items():
            idxs = []
            for fluor in cand_list:
                rec = dye_db.get(fluor)
                if rec is None:
                    continue
                em = rec["emission"]; ex = rec["excitation"]
                qy = rec["quantum_yield"]; ec = rec["extinction_coeff"]
                if em is None or ex is None or len(em) != W or len(ex) != W:
                    continue

                eff = np.zeros(W, dtype=float)
                for i, (lo, hi) in enumerate(segs):
                    loi = _nearest_idx_from_grid_local(wl, lo)
                    hii = _nearest_idx_from_grid_local(wl, hi - 1) + 1
                    # cumulative coefficient within the segment (0..i lasers)
                    total_k = 0.0
                    for m in range(i + 1):
                        total_k += _interp_at(wl, ex, lam[m]) * qy * (ec if ec is not None else 1.0) * pw[m]
                    eff[loi:hii] += em[loi:hii] * total_k

                cols.append(eff)
                labels.append(f"{probe} – {fluor}")
                idxs.append(col_id)
                col_id += 1
            if idxs:
                idx_groups.append(idxs)

        if not cols:
            Z = np.zeros((W, 0))
            return Z, Z, [], []
if fluor == "Pacific Blue":
    val_405 = _interp_at(wl, ex, 405)
    val_488 = _interp_at(wl, ex, 488)
    import streamlit as st
    st.write("PB ex(405) = ", float(val_405))
    st.write("PB ex(488) = ", float(val_488))
        E_raw = np.stack(cols, axis=1)
        denom = np.linalg.norm(E_raw, axis=0, keepdims=True) + 1e-12
        E_norm = E_raw / denom
        return E_raw, E_norm, labels, idx_groups


# =================== Global-unique constraint ===================

def _unique_dye_constraints(prob, x_vars, labels_pair, groups, fluor_names):
    """
    Enforce "each fluorophore can be used at most once globally".
    Works for the 'by probe' mode; not applied in pool mode.
    """
    dye_to_cols = {}
    for j, d in enumerate(fluor_names):
        dye_to_cols.setdefault(d, []).append(j)
    for d, cols in dye_to_cols.items():
        prob += pulp.lpSum(x_vars[j] for j in cols) <= 1, f"Unique_{d}"


# =================== Optimization: lexicographic ===================

def _pick_integral_from_relaxed(x_vars, idx_groups):
    """
    Safety: if a MIP solution leaves fractional x, pick argmax within each group.
    """
    xvals = np.array([(v.value() or 0.0) for v in x_vars], dtype=float)
    sel = []
    for idxs in idx_groups:
        if not idxs:
            continue
        j_local = int(np.argmax(xvals[idxs]))
        sel.append(int(idxs[j_local]))
    return sel


def solve_minimax_layer(E_norm, idx_groups, labels_pair,
                        enforce_unique=True, required_count: int | None = None):
    """
    Minimize the maximum pairwise cosine among selected columns.

    Two modes:
      - required_count is None: "by probes" (pick exactly 1 per group, with global-unique dyes).
      - required_count is int : "pool mode" (pick exactly N from the union set).
    """
    N = E_norm.shape[1]
    C = cosine_similarity_matrix(E_norm)
    prob = pulp.LpProblem("minimax", pulp.LpMinimize)
    x = [pulp.LpVariable(f"x_{j}", lowBound=0, upBound=1, cat="Binary") for j in range(N)]
    # y can be continuous in [0,1]; with binary x the relaxation is tight
    y = {}
    for i in range(N):
        for j in range(i+1, N):
            y[(i,j)] = pulp.LpVariable(f"y_{i}_{j}", lowBound=0, upBound=1)
    t = pulp.LpVariable("t", lowBound=0)

    # selection constraints
    if required_count is None:
        for g, idxs in enumerate(idx_groups):
            prob += pulp.lpSum(x[j] for j in idxs) == 1, f"OnePerGroup_{g}"
        if enforce_unique:
            fluor_names = [s.split(" – ", 1)[1] for s in labels_pair]
            _unique_dye_constraints(prob, x, labels_pair, idx_groups, fluor_names)
    else:
        prob += pulp.lpSum(x) == int(required_count), "PickN"

    # y linking + t >= C_ij * y_ij
    for (i,j), yij in y.items():
        prob += yij <= x[i]; prob += yij <= x[j]; prob += yij >= x[i] + x[j] - 1
        prob += t >= float(C[i,j]) * yij

    prob += t
    _ = prob.solve(_SOLVER)

    if required_count is None:
        x_star = _pick_integral_from_relaxed(x, idx_groups)
    else:
        xv = np.array([(v.value() or 0.0) for v in x])
        x_star = list(np.argsort(-xv)[:required_count])
    t_val = float(t.value() or 0.0)
    return x_star, t_val


def solve_lexicographic_k(E_norm, idx_groups, labels_pair, levels: int = 10,
                          enforce_unique: bool = True,
                          required_count: int | None = None):
    """
    True lexicographic minimization over pairwise cosines:
      level-1: minimize the maximum pairwise cosine (t1)
      level-2..K: at each level k, fix all previous levels to their optimal values
                  (within a tiny epsilon) and minimize the next largest term using
                  a standard lambda/mu linearization.

    Modes:
      - required_count=None: "by probes" (1 per group + global unique).
      - required_count=int : "pool mode" (select N from the union set).

    Returns (selected_indices, [t1*, lam2*, lam3*, ... up to K]).
    """
    N = E_norm.shape[1]
    if N == 0:
        return [], []
    C = cosine_similarity_matrix(E_norm)
    pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    P = len(pairs)
    if P == 0:
        if required_count is None:
            sel = [g[0] for g in idx_groups if g]
        else:
            sel = list(range(min(required_count, N)))
        return sel, []

    K = int(max(1, min(levels, P)))
    eps = 1e-6
    best_layers, x_star_last = [], None

    # single-group view for reading/writing x uniformly
    single_group = idx_groups if required_count is None else [list(range(N))]

    for k in range(1, K + 1):
        prob = pulp.LpProblem(f"lexi_k{k}", pulp.LpMinimize)

        x = [pulp.LpVariable(f"x_{j}", lowBound=0, upBound=1, cat="Binary") for j in range(N)]
        y = { (i,j): pulp.LpVariable(f"y_{i}_{j}", lowBound=0, upBound=1) for (i,j) in pairs }
        z = { (i,j): pulp.LpVariable(f"z_{i}_{j}", lowBound=0) for (i,j) in pairs }

        # selection constraints
        if required_count is None:
            for g, idxs in enumerate(idx_groups):
                prob += pulp.lpSum(x[j] for j in idxs) == 1, f"OnePerGroup_{g}"
            if enforce_unique:
                fluor_names = [s.split(" – ", 1)[1] for s in labels_pair]
                _unique_dye_constraints(prob, x, labels_pair, idx_groups, fluor_names)
        else:
            prob += pulp.lpSum(x) == int(required_count), "PickN"

        # linking
        for (i,j) in pairs:
            yij = y[(i,j)]
            prob += yij <= x[i]; prob += yij <= x[j]; prob += yij >= x[i] + x[j] - 1
            cij = float(C[i,j]); zij = z[(i,j)]
            prob += zij <= cij * yij; prob += zij >= cij * yij

        # level-1
        t1 = pulp.LpVariable("t1", lowBound=0)
        for (i,j) in pairs:
            prob += t1 >= z[(i,j)]
        if 0 < len(best_layers):
            prob += t1 <= float(best_layers[0]) + eps

        # levels 2..k
        lam, mu = {}, {}
        if k >= 2:
            for r in range(2, k + 1):
                lam[r] = pulp.LpVariable(f"lam{r}", lowBound=0)
                mu[r] = { (i,j): pulp.LpVariable(f"mu{r}_{i}_{j}", lowBound=0) for (i,j) in pairs }
                for (i,j) in pairs:
                    prob += mu[r][(i,j)] >= z[(i,j)] - lam[r]
                if (r - 1) < len(best_layers):
                    prob += lam[r] <= float(best_layers[r - 1]) + eps

        # objective (current level only)
        if k == 1:
            prob += t1
        else:
            prob += lam[k] + (1.0 / max(1, P)) * pulp.lpSum(mu[k][p] for p in pairs)

        _ = prob.solve(_SOLVER)

        # record layer value
        best_layers.append(float((t1 if k == 1 else lam[k]).value() or 0.0))

        # read current x (robust integralization)
        if required_count is None:
            x_star_last = _pick_integral_from_relaxed(x, single_group)
        else:
            xv = np.array([(v.value() or 0.0) for v in x])
            x_star_last = list(np.argsort(-xv)[:required_count])

        # early exit if value is essentially zero
        if best_layers[-1] <= 1e-8:
            break

    return x_star_last, best_layers
