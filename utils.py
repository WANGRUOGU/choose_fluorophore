import yaml
import numpy as np

# MILP
import pulp


# ---------------- I/O ----------------

import yaml
import numpy as np

def load_dyes_yaml(path):
    """Load dyes.yaml -> (wavelengths, dye_db).

    - Normalize every emission to its own max (peak) = 1.0  (if max>0)
    - Keep excitation原样 (不做归一化)
    - Fill missing quantum_yield with the mean of available QYs
    """
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    wl = np.array(data["wavelengths"], dtype=float)
    raw_db = data["dyes"]

    # 先收一遍 QY，计算均值
    qy_list = []
    for name, rec in raw_db.items():
        qy = rec.get("quantum_yield", None)
        if qy is not None:
            try:
                qy_val = float(qy)
                if np.isfinite(qy_val):
                    qy_list.append(qy_val)
            except Exception:
                pass
    qy_mean = float(np.mean(qy_list)) if len(qy_list) else 1.0  # 没有的话用 1.0 兜底

    dye_db = {}
    for name, rec in raw_db.items():
        em = np.array(rec.get("emission", []), dtype=float)
        ex = np.array(rec.get("excitation", []), dtype=float)
        qy = rec.get("quantum_yield", None)
        ec = rec.get("extinction_coeff", None)

        # emission：峰值归一化到 1
        if em.size > 0:
            m = np.max(em)
            if m > 0:
                em = em / m

        # QY 缺失回填为均值
        if qy is None or (isinstance(qy, float) and not np.isfinite(qy)):
            qy = qy_mean
        else:
            try:
                qy = float(qy)
            except Exception:
                qy = qy_mean

        # 允许 EC 缺失（算法里会照常参与乘法；缺失可视需求再决定是否设为 1）
        if ec is not None:
            try:
                ec = float(ec)
            except Exception:
                ec = None

        dye_db[name] = dict(
            emission=em, excitation=ex, quantum_yield=qy, extinction_coeff=ec
        )
    return wl, dye_db




def read_probes_and_mapping(path):
    """
    Parse ONLY the `probes:` array from probe_fluor_map.yaml.
    Returns:
      names_sorted: list[str]
      mapping: dict[str, list[str]]
    """
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    plist = data if isinstance(data, list) else data.get("probes", [])
    names = []
    mapping = {}
    if isinstance(plist, list):
        for item in plist:
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


# ------------- Helpers -------------

def _safe_norm(x):
    n = np.linalg.norm(x)
    return x / (n + 1e-12)


def cosine_similarity_matrix(E):
    """Cosine similarity matrix of columns of E (E: W x N)."""
    N = E.shape[1]
    norms = np.linalg.norm(E, axis=0) + 1e-12
    G = (E.T @ E) / np.outer(norms, norms)
    np.fill_diagonal(G, 0.0)
    return G


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


# ------------- Spectra builders -------------

def build_effective_emission_emission_only(wl, dye_db, groups_as_lists_of_names):
    """(Deprecated for MILP building directly) — kept for compatibility."""
    W = len(wl)
    cols, labels = [], []
    for cands in groups_as_lists_of_names:
        for name in cands:
            em = np.array(dye_db[name]["emission"], dtype=float)
            if em.size != W:
                continue
            cols.append(_safe_norm(em))
            labels.append(name)
    if not cols:
        return np.zeros((W, 0)), []
    return np.stack(cols, axis=1), labels


def build_E_and_groups_from_mapping(
    wl, dye_db, picked_probes, probe_to_fluors, mode="emission_only",
    laser_wavelengths=None, laser_mode=None, laser_powers=None
):
    """
    Build a big matrix E (W x N) across all picked probes & their candidates,
    along with `labels` (length N) and `groups_idx` (list of list of indices into columns of E).
    mode:
      - "emission_only"
      - "lasers" (requires wavelengths, mode, and powers to be provided; E columns are effective spectra)
    """
    W = len(wl)
    cols = []
    labels = []
    groups_idx = []
    group_names = []
    cur = 0
    for probe in picked_probes:
        cands = list(probe_to_fluors.get(probe, []))
        # filter by spectra availability
        cands = [f for f in cands if f in dye_db]
        if not cands:
            continue
        idxs = []
        for name in cands:
            rec = dye_db[name]
            if mode == "emission_only":
                em = np.array(rec["emission"], dtype=float)
                if em.size != W:
                    continue
                cols.append(em.copy())  # raw emission; we will normalize when computing cosine constants
                labels.append(name)
                idxs.append(cur)
                cur += 1
            else:
                # lasers mode: need effective spectrum built from powers
                em = np.array(rec["emission"], dtype=float)
                ex = np.array(rec["excitation"], dtype=float)
                qy = rec["quantum_yield"]
                ec = rec["extinction_coeff"]
                if any(v is None for v in (qy, ec)) or em.size != W or ex.size != W:
                    continue
                lam = np.array(sorted(laser_wavelengths), dtype=float)
                pw = np.array([laser_powers[laser_wavelengths.index(l)] for l in lam], dtype=float)
                eff = np.zeros(W)
                if laser_mode == "Separate":
                    for i, l in enumerate(lam):
                        idx = int(l - wl[0])
                        if 0 <= idx < W:
                            k_i = ex[idx] * qy * ec * pw[i]
                            eff += em * k_i
                else:
                    # Simultaneous: piecewise segments
                    for i, l in enumerate(lam):
                        lo = l
                        hi = lam[i+1] if i+1 < len(lam) else wl[-1] + 1
                        loi = int(max(lo - wl[0], 0))
                        hii = int(min(hi - wl[0], W))
                        idx_l = int(l - wl[0])
                        if 0 <= idx_l < W and loi < hii:
                            k_i = ex[idx_l] * qy * ec * pw[i]
                            eff[loi:hii] += em[loi:hii] * k_i
                cols.append(eff)
                labels.append(name)
                idxs.append(cur)
                cur += 1
        if idxs:
            groups_idx.append(idxs)
            group_names.append(probe)

    if not cols:
        return np.zeros((W, 0)), [], [], []

    E = np.stack(cols, axis=1)
    return E, labels, groups_idx, group_names


# ------------- Laser power derivation & iteration -------------

def derive_powers_simultaneous(wl, dye_db, selection, laser_wavelengths):
    """Compute laser powers for 'Simultaneous' per the segment-wise leveling rule.

    Steps:
      1) Find first informative segment start_seg; set P[start_seg]=1.
      2) Let B = max over dyes in selection of peak(em_segment * ex(λ_start)*QY*EC).
      3) For each later segment s, choose the *smallest* P[s] ≥ 0 such that
         for all dyes i in selection and all wavelength samples k in segment s:
           pre_i,k + em_i[k]*ex_i(λ_s)*QY_i*EC_i*P[s] ≤ B,
         where pre_i,k = em_i[k] * sum_{m< s} ex_i(λ_m)*QY_i*EC_i*P[m].
    """
    W = len(wl)
    lam = np.array(sorted(laser_wavelengths), dtype=float)

    # Build segments as index ranges
    seg_ranges = []
    for i in range(len(lam)):
        lo = lam[i]
        hi = lam[i+1] if i+1 < len(lam) else wl[-1] + 1
        loi = int(max(lo - wl[0], 0))
        hii = int(min(hi - wl[0], W))
        seg_ranges.append((loi, hii))

    recs = [dye_db[nm] for nm in selection]

    # helper: coefficient at laser λ for a record
    def coef_at(rec, L):
        ex = np.array(rec.get("excitation", []), dtype=float)
        qy = rec.get("quantum_yield", None)
        ec = rec.get("extinction_coeff", None)
        if ex.size != W or qy is None or ec is None:
            return None
        idx = int(L - wl[0])
        if 0 <= idx < W:
            return float(ex[idx] * qy * ec)
        return 0.0

    # find first informative segment
    start_seg = None
    for s, (loi, hii) in enumerate(seg_ranges):
        informative = False
        for rec in recs:
            em = np.array(rec.get("emission", []), dtype=float)
            c0 = coef_at(rec, lam[s])
            if em.size == W and c0 not in (None, 0.0) and np.max(em[loi:hii]) > 0:
                informative = True
                break
        if informative:
            start_seg = s
            break

    if start_seg is None:
        # degenerate case: no emission anywhere
        return [1.0] * len(lam)

    # init powers
    P = np.zeros(len(lam), dtype=float)
    P[start_seg] = 1.0

    # compute B on start segment using *true* segment vector peaks
    loi, hii = seg_ranges[start_seg]
    peaks = []
    for rec in recs:
        em = np.array(rec.get("emission", []), dtype=float)
        c = coef_at(rec, lam[start_seg])
        if em.size != W or c in (None, 0.0):
            continue
        vec = em[loi:hii] * c * P[start_seg]
        if vec.size:
            peaks.append(np.max(vec))
    B = float(max(peaks) if peaks else 0.0)

    # accumulate pre-sums per record as a scalar “sum of coefficients so far”
    # BUT constraints用逐点：pre_i,k = em_i[k] * sum_prev
    sum_prev = np.zeros(len(recs), dtype=float)
    sum_prev += np.array([coef_at(rec, lam[start_seg]) or 0.0 for rec in recs]) * P[start_seg]

    # iterate later segments
    for s in range(start_seg + 1, len(lam)):
        loi, hii = seg_ranges[s]
        if loi >= hii:
            P[s] = 0.0
            continue

        # collect feasible upper bounds for P[s]
        bounds = []
        for i, rec in enumerate(recs):
            em = np.array(rec.get("emission", []), dtype=float)
            c_s = coef_at(rec, lam[s])
            if em.size != W or c_s in (None, 0.0):
                continue
            em_seg = em[loi:hii]
            if em_seg.size == 0 or np.all(em_seg <= 0):
                continue
            # per-sample constraint: pre_i,k + em_i[k]*c_s*P[s] ≤ B
            # where pre_i,k = em_i[k] * sum_prev[i]
            # rearrange (em_i[k]*c_s) * P[s] ≤ B - em_i[k]*sum_prev[i]
            denom = em_seg * c_s
            numer = B - em_seg * sum_prev[i]
            mask = (denom > 0) & (numer >= 0)
            if np.any(mask):
                ub = np.min(numer[mask] / denom[mask])  # smallest upper bound across k
                bounds.append(ub)

        if bounds:
            P[s] = max(0.0, float(min(bounds)))
        else:
            P[s] = 0.0

        # update cumulative coefficient sum for later segments
        sum_prev += np.array([coef_at(rec, lam[s]) or 0.0 for rec in recs]) * P[s]

    return [float(x) for x in P]



def derive_powers_separate(wl, dye_db, selection, laser_wavelengths):
    lam = np.array(sorted(laser_wavelengths), dtype=float)
    W = len(wl)
    recs = [dye_db[name] for name in selection]

    def coef_at_l(rec, l):
        em = np.array(rec["emission"], dtype=float)
        ex = np.array(rec["excitation"], dtype=float)
        qy = rec["quantum_yield"]; ec = rec["extinction_coeff"]
        if any(v is None for v in (qy, ec)) or ex.size != W or em.size != W:
            return None
        idx = int(l - wl[0])
        if idx < 0 or idx >= W:
            return 0.0
        return float(ex[idx] * qy * ec)

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
        return [1.0] * len(lam)
    B = float(np.max(M_l))
    P = [0.0 if val <= 0 else float(B / val) for val in M_l]
    return P


def iterate_selection_with_lasers(
    wl, dye_db, picked_probes, probe_to_fluors, laser_wavelengths, mode,
    solver="lexi_milp", max_iter=8
):
    """
    Start from emission-only selection A, derive laser powers, rebuild effective spectra for ALL candidates,
    re-solve with chosen solver (lexicographic MILP), repeat to fixed point or max_iter.
    Returns: (selected_labels, powers, iters, converged, E_final, labels_final, groups_idx_final)
    """
    # Build emission-only E for initial A
    E0, labels0, groups0, _ = build_E_and_groups_from_mapping(
        wl, dye_db, picked_probes, probe_to_fluors, mode="emission_only"
    )
    # Normalize for cosine constants internally in MILP; here just pass raw emission
    A_idx = lexicographic_select_grouped_milp(E0, groups0)
    A = [labels0[i] for i in A_idx]

    selected = list(A)
    powers = None
    converged = False
    E_final = None
    labels_final = None
    groups_idx_final = None

    for it in range(1, max_iter + 1):
        # powers from current selection
        if mode == "Simultaneous":
            powers = derive_powers_simultaneous(wl, dye_db, selected, laser_wavelengths)
        else:
            powers = derive_powers_separate(wl, dye_db, selected, laser_wavelengths)

        # Build effective spectra of all candidates
        E_all, labels_all, groups_all, _ = build_E_and_groups_from_mapping(
            wl, dye_db, picked_probes, probe_to_fluors,
            mode="lasers",
            laser_wavelengths=laser_wavelengths,
            laser_mode=mode,
            laser_powers=powers
        )

        # Re-select with lexicographic MILP
        sel_idx = lexicographic_select_grouped_milp(E_all, groups_all)
        new_selected = [labels_all[i] for i in sel_idx]

        E_final, labels_final, groups_idx_final = E_all, labels_all, groups_all

        if new_selected == selected:
            converged = True
            break
        selected = new_selected

    return selected, powers, it, converged, E_final, labels_final, groups_idx_final


# ------------- Lexicographic MILP (layer-by-layer) -------------

def _build_cosine_constants(E):
    """Return normalized columns and pairwise cosine constants matrix C (N x N, diag 0)."""
    En = E / (np.linalg.norm(E, axis=0, keepdims=True) + 1e-12)
    C = (En.T @ En)
    np.fill_diagonal(C, 0.0)
    return En, C


def lexicographic_select_grouped_milp(E, groups_idx, tol_eq=1e-6):
    """
    Layer-1: minimize t, s.t. choose exactly one per group, and for every cross-group pair (a,b):
        t >= c_ab * y_ab, with y_ab = 1 if both a and b are selected
    Layer-2: with t fixed to t*, minimize the "second largest" via the standard convexification:
        min  2*lambda2 + sum mu_p
        s.t. mu_p >= c_p y_p - lambda2, mu_p >= 0
             (carry over Layer-1 constraints, and fix t <= t*)
    Implementation notes:
      - We use PuLP + CBC.
      - We only build pair variables for pairs across different groups.
    """
    _, C = _build_cosine_constants(E)
    N = E.shape[1]

    # Build a reverse map: for each candidate index -> group id
    cand_to_group = {}
    for g, idxs in enumerate(groups_idx):
        for j in idxs:
            cand_to_group[j] = g

    # List all cross-group pairs once (i < j and group different)
    pairs = []
    for i in range(N):
        gi = cand_to_group.get(i, None)
        if gi is None:
            continue
        for j in range(i+1, N):
            gj = cand_to_group.get(j, None)
            if gj is None or gj == gi:
                continue
            pairs.append((i, j))

    # ------------ Layer 1 ------------
    prob1 = pulp.LpProblem("Layer1_Minimax", sense=pulp.LpMinimize)

    # x_j: select candidate j
    x = {j: pulp.LpVariable(f"x_{j}", lowBound=0, upBound=1, cat=pulp.LpBinary) for j in range(N)}
    # y_ij: both i and j selected
    y = { (i,j): pulp.LpVariable(f"y_{i}_{j}", lowBound=0, upBound=1, cat=pulp.LpBinary) for (i,j) in pairs }
    # t: maximum similarity among selected pairs
    t = pulp.LpVariable("t", lowBound=0)

    # group constraints: pick exactly one per group
    for g, idxs in enumerate(groups_idx):
        prob1 += (pulp.lpSum(x[j] for j in idxs) == 1), f"group_{g}_one"

    # linking y with x
    for (i,j) in pairs:
        prob1 += (y[(i,j)] <= x[i])
        prob1 += (y[(i,j)] <= x[j])
        prob1 += (y[(i,j)] >= x[i] + x[j] - 1)

    # t >= c_ij * y_ij
    for (i,j) in pairs:
        cij = float(C[i, j])
        if cij <= 0:
            # c<=0 不会影响最大值，仍然保留个下界 t>=0
            continue
        prob1 += (t >= cij * y[(i,j)])

    # objective
    prob1 += t

    # solve
    prob1.solve(pulp.PULP_CBC_CMD(msg=False))
    t_star = pulp.value(t)

    # record chosen x to warm-start Layer 2 (optional)
    x_val1 = {j: pulp.value(var) for j, var in x.items()}

    # ------------ Layer 2 ------------
    prob2 = pulp.LpProblem("Layer2_SecondLargest", sense=pulp.LpMinimize)

    # re-create variables (or clone) to keep clean model
    x2 = {j: pulp.LpVariable(f"x2_{j}", lowBound=0, upBound=1, cat=pulp.LpBinary) for j in range(N)}
    y2 = { (i,j): pulp.LpVariable(f"y2_{i}_{j}", lowBound=0, upBound=1, cat=pulp.LpBinary) for (i,j) in pairs }
    t2 = pulp.LpVariable("t2", lowBound=0)  # will be constrained to <= t_star
    lam2 = pulp.LpVariable("lambda2", lowBound=0)
    mu = { (i,j): pulp.LpVariable(f"mu_{i}_{j}", lowBound=0) for (i,j) in pairs }

    # group constraints
    for g, idxs in enumerate(groups_idx):
        prob2 += (pulp.lpSum(x2[j] for j in idxs) == 1), f"group2_{g}_one"

    # link y2 and x2
    for (i,j) in pairs:
        prob2 += (y2[(i,j)] <= x2[i])
        prob2 += (y2[(i,j)] <= x2[j])
        prob2 += (y2[(i,j)] >= x2[i] + x2[j] - 1)

    # t2 constraints (same as layer-1) and fix t2 <= t_star (+ small tol)
    for (i,j) in pairs:
        cij = float(C[i, j])
        if cij <= 0:
            continue
        prob2 += (t2 >= cij * y2[(i,j)])
    prob2 += (t2 <= t_star + tol_eq)

    # lambda2 / mu formulation to minimize the second largest:
    #   mu_p >= c_p*y_p - lambda2
    #   mu_p >= 0
    for (i,j) in pairs:
        cij = float(C[i, j])
        prob2 += (mu[(i,j)] >= cij * y2[(i,j)] - lam2)

    # objective: 2*lambda2 + sum mu
    prob2 += (2 * lam2 + pulp.lpSum(mu.values()))

    # (optional) warm start x2 with layer1 solution
    for j, v in x2.items():
        if x_val1.get(j, 0) > 0.5:
            v.setInitialValue(1)
            v.fixValue()  # fix to layer-1 choice ensures the same t* face; comment this line to allow tie-breaking within t*
    # Note: 如果你希望在 t*=const 的可行面上重新自由优化第二大值，可以去掉 fixValue()

    # solve
    prob2.solve(pulp.PULP_CBC_CMD(msg=False))

    x_sol = {j: int(round(pulp.value(var))) for j, var in x2.items()}
    sel = []
    for g, idxs in enumerate(groups_idx):
        # find chosen j in this group
        picked = sorted(idxs, key=lambda j: -x_sol.get(j, 0))
        sel.append(picked[0])
    return sel
def build_effective_emission_with_lasers(
    wl, dye_db, candidates, laser_wavelengths, mode, powers, selection_for_base=None
):
    """Build effective spectra under lasers for a list of candidate fluorophores.

    Simultaneous:
      - Divide spectrum into segments [λ1,λ2), [λ2,λ3), ..., [λK, end)
      - In segment s, sum contributions from *all lasers with index m ≤ s*:
          eff_segment += emission_segment * (sum_m≤s ex(λ_m)*QY*EC*P_m)

    Separate:
      - Each laser produces a full-spectrum emission scaled by ex(λ)*QY*EC*P.
        We sum the lasers’ full spectra.
    """
    W = len(wl)
    if len(candidates) == 0:
        return np.zeros((W, 0)), []

    lam_sorted = np.array(sorted(laser_wavelengths), dtype=float)
    # map original powers to sorted order
    pw_sorted = np.array([powers[laser_wavelengths.index(L)] for L in lam_sorted], dtype=float)

    # precompute segment index ranges
    segments = []
    for i in range(len(lam_sorted)):
        lo = lam_sorted[i]
        hi = lam_sorted[i+1] if i+1 < len(lam_sorted) else wl[-1] + 1
        loi = int(max(lo - wl[0], 0))
        hii = int(min(hi - wl[0], W))
        segments.append((loi, hii))

    cols, labels = [], []
    for name in candidates:
        rec = dye_db[name]
        em = np.array(rec.get("emission", []), dtype=float)
        ex = np.array(rec.get("excitation", []), dtype=float)
        qy = rec.get("quantum_yield", None)
        ec = rec.get("extinction_coeff", None)

        if em.size != W or ex.size != W or qy is None or ec is None:
            cols.append(np.zeros(W))
            labels.append(name)
            continue

        # coefficients at laser wavelengths
        c = np.zeros(len(lam_sorted))
        for i, L in enumerate(lam_sorted):
            idx = int(L - wl[0])
            if 0 <= idx < W:
                c[i] = ex[idx] * qy * ec * pw_sorted[i]

        eff = np.zeros(W)
        if mode == "Separate":
            # full-spectrum sum over lasers
            scale = np.sum(c)  # sum_i ex(L_i)*QY*EC*P_i
            eff += em * scale
        else:
            # Simultaneous (cumulative inside each segment)
            for s, (loi, hii) in enumerate(segments):
                if loi >= hii:
                    continue
                # sum contributions from all lasers up to s (inclusive)
                scale = np.sum(c[:s+1])
                if scale > 0:
                    eff[loi:hii] += em[loi:hii] * scale

        cols.append(eff)
        labels.append(name)

    if not cols:
        return np.zeros((W, 0)), []
    return np.stack(cols, axis=1), labels
