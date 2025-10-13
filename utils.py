import yaml
import numpy as np

# MILP
import pulp


# ---------------- I/O ----------------

import yaml
import numpy as np

def load_dyes_yaml(path):
    """Load dyes.yaml -> (wavelengths, dye_db).
    - Emission is normalized by its own max (peak=1) if max>0.
    - If quantum_yield is missing -> impute by mean of available QY.
    """
    import yaml, numpy as np

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    wl = np.array(data["wavelengths"], dtype=float)

    # first pass: collect QY
    raw = {}
    qy_list = []
    for name, rec in data["dyes"].items():
        em = np.array(rec.get("emission", []), dtype=float)
        ex = np.array(rec.get("excitation", []), dtype=float)
        qy = rec.get("quantum_yield", None)
        ec = rec.get("extinction_coeff", None)

        # emission peak-normalize
        if em.size:
            m = float(np.max(em))
            if m > 0:
                em = em / m

        raw[name] = dict(emission=em, excitation=ex, quantum_yield=qy, extinction_coeff=ec)
        if isinstance(qy, (int, float)) and np.isfinite(qy):
            qy_list.append(float(qy))

    qy_mean = float(np.mean(qy_list)) if qy_list else 1.0

    # second pass: impute missing QY
    dye_db = {}
    for name, rec in raw.items():
        qy = rec["quantum_yield"]
        if (qy is None) or (not np.isfinite(qy)):
            qy = qy_mean
        dye_db[name] = dict(
            emission=rec["emission"],
            excitation=rec["excitation"],
            quantum_yield=float(qy),
            extinction_coeff=rec["extinction_coeff"],
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

def derive_powers_simultaneous(wl, dye_db, selection, laser_wavelengths, eps=1e-12):
    """
    严格分段等峰：
    - 设 L1<L2<...<LS。把谱分成 S 段：[L1,L2), [L2,L3), ..., [LS, +inf)
    - 第一段设 P1=1，B = max_i{ M_i(1) * c_i(L1) }，其中 M_i(s) 是第 s 段 emission 的最大值，
      c_i(Lk)=ex_i(Lk)*QY_i*EC_i。
    - 对 s=2..S：令 pre_i = sum_{m=1..s-1} c_i(Lm) Pm。
      约束：forall i,  M_i(s)*(pre_i + c_i(Ls) Ps) ≤ B。
      取满足不等式的 **最小上界** Ps = min_i  (B/M_i(s) - pre_i)/c_i(Ls) （仅在 M_i(s)>0 且 c_i>0 且分子>0 上取）。
      这个最小上界对应的 i 会“卡紧”，从而该段的最大值 = B（数值误差除外）。
    """
    import numpy as np

    lam = np.array(sorted(laser_wavelengths), dtype=float)
    W = len(wl)

    # handy helpers
    def seg_slice(lo_nm, hi_nm):
        lo_i = int(max(lo_nm - wl[0], 0))
        hi_i = int(min((hi_nm if np.isfinite(hi_nm) else wl[-1] + 1) - wl[0], W))
        return lo_i, hi_i

    def Mseg(rec, lo_i, hi_i):
        em = np.array(rec["emission"], dtype=float)
        if em.size != W or lo_i >= hi_i:
            return 0.0
        return float(np.max(em[lo_i:hi_i]))

    def coef(rec, lnm):
        ex = np.array(rec["excitation"], dtype=float)
        if ex.size != W:
            return 0.0
        qy = float(rec["quantum_yield"])
        ec = rec["extinction_coeff"]
        if ec is None or not np.isfinite(ec):
            return 0.0
        idx = int(lnm - wl[0])
        if idx < 0 or idx >= W:
            return 0.0
        return float(ex[idx]) * qy * float(ec)

    # build selected records
    A = [dye_db[n] for n in selection]
    if len(A) == 0:
        return [1.0] * len(lam)

    # segments
    seg_bounds = []
    for s in range(len(lam)):
        lo = lam[s]
        hi = lam[s+1] if s+1 < len(lam) else float("inf")
        seg_bounds.append((lo, hi))

    # 找到第一个“可用”的起始段：该段存在 M_i(s)>0 且 c_i(Ls)>0
    start = None
    for s, (lo, hi) in enumerate(seg_bounds):
        lo_i, hi_i = seg_slice(lo, hi)
        feas = False
        for rec in A:
            if Mseg(rec, lo_i, hi_i) > 0 and coef(rec, lam[s]) > 0:
                feas = True
                break
        if feas:
            start = s
            break
    if start is None:
        return [1.0] * len(lam)  # 无法驱动任何段

    # P 初始化
    P = np.zeros(len(lam), dtype=float)
    P[start] = 1.0

    # 定义 B
    lo_i, hi_i = seg_slice(*seg_bounds[start])
    B = 0.0
    for rec in A:
        Mi = Mseg(rec, lo_i, hi_i)
        ci = coef(rec, lam[start])
        B = max(B, Mi * ci)
    # 若 B=0（极端），直接全 1
    if B <= eps:
        return [1.0] * len(lam)

    # 逐段推进
    # 为了简洁，我们从 start+1 开始；之前段的 P 已定
    for s in range(start + 1, len(lam)):
        lo, hi = seg_bounds[s]
        lo_i, hi_i = seg_slice(lo, hi)

        # pre_i 和 c_s
        pre = []
        c_s = []
        M_s = []
        for rec in A:
            # 该段的发射峰
            Mi = Mseg(rec, lo_i, hi_i)
            M_s.append(Mi)
            # 该段对应激发系数
            cs = coef(rec, lam[s])
            c_s.append(cs)
            # 累积前面段的贡献系数和（注意这些系数在“这一段”的放大倍数一样乘以 M_i(s)）
            acc = 0.0
            for m in range(start, s):
                cm = coef(rec, lam[m])
                acc += cm * P[m]
            pre.append(acc)

        M_s = np.array(M_s, dtype=float)
        c_s = np.array(c_s, dtype=float)
        pre  = np.array(pre, dtype=float)

        # 可行 i：M_i(s)>0, c_i>0, 且 B/M_i(s) - pre_i > 0
        feas = (M_s > eps) & (c_s > eps) & ((B / np.maximum(M_s, eps) - pre) > eps)
        if not np.any(feas):
            # 这一段任何选中染料都无法把峰抬到 B（比如全是 c_i=0），按规则功率设 0
            P[s] = 0.0
            continue

        # 取最小上界：Ps = min_i (B/M_i - pre_i)/c_i
        bounds = (B / M_s[feas] - pre[feas]) / c_s[feas]
        Ps = float(np.min(bounds))
        if Ps < 0:
            Ps = 0.0
        P[s] = Ps

        # 数值校正（把该段真实峰值钉到 B±eps）
        # 真实峰值 = max_i M_i(s)*(pre_i + c_i Ps)
        real = float(np.max(M_s * (pre + c_s * P[s])))
        if real > eps:
            P[s] *= (B / real)

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
    """Return [W x N] effective spectra.
    - Emission 已在 load 时做了 peak=1 归一；此处不再改动。
    - Simultaneous: 每段只加对应段的贡献，段边界按激光波长划分。
    - Separate: 整段全谱都加（每束激光都产生全谱发射）。
    """
    import numpy as np

    W = len(wl)
    if len(candidates) == 0:
        return np.zeros((W, 0)), []

    lam_sorted = np.array(sorted(laser_wavelengths), dtype=float)
    # 对应功率按排序对齐
    P = np.array([powers[laser_wavelengths.index(L)] for L in lam_sorted], dtype=float)

    def seg_slice(lo_nm, hi_nm):
        lo_i = int(max(lo_nm - wl[0], 0))
        hi_i = int(min((hi_nm if np.isfinite(hi_nm) else wl[-1] + 1) - wl[0], W))
        return lo_i, hi_i

    # 预先切好段
    segs = []
    for s in range(len(lam_sorted)):
        lo = lam_sorted[s]
        hi = lam_sorted[s+1] if s+1 < len(lam_sorted) else float("inf")
        segs.append((lo, hi))

    M = np.zeros((W, len(candidates)), dtype=float)
    labels = []

    for j, name in enumerate(candidates):
        rec = dye_db[name]
        em = np.array(rec["emission"], dtype=float)  # 已 peak-normalized
        ex = np.array(rec["excitation"], dtype=float)
        qy = float(rec["quantum_yield"])
        ec = rec["extinction_coeff"]
        if em.size != W or ex.size != W or ec is None or not np.isfinite(ec):
            labels.append(name)
            continue

        y = np.zeros(W, dtype=float)
        if mode == "Separate":
            # 每束激光：全谱贡献
            for s, L in enumerate(lam_sorted):
                idx = int(L - wl[0])
                if 0 <= idx < W:
                    k = ex[idx] * qy * float(ec) * P[s]
                    y += em * k
        else:
            # Simultaneous：分段贡献
            for s, L in enumerate(lam_sorted):
                lo, hi = segs[s]
                lo_i, hi_i = seg_slice(lo, hi)
                idx = int(L - wl[0])
                if 0 <= idx < W and lo_i < hi_i:
                    k = ex[idx] * qy * float(ec) * P[s]
                    y[lo_i:hi_i] += em[lo_i:hi_i] * k

        M[:, j] = y
        labels.append(name)

    return M, labels

