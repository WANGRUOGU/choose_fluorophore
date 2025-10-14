# utils.py
import yaml
import numpy as np
import pulp

# =============== I/O =================

def load_dyes_yaml(path):
    """
    Load dyes.yaml -> (wavelengths, dye_db).
    dye_db[name] = {
        "emission": np.array(W,),
        "excitation": np.array(W,),
        "quantum_yield": float|None,
        "extinction_coeff": float|None
    }
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

    # Fill missing QY with mean of available ones (used later in builders)
    qys = [v["quantum_yield"] for v in dye_db.values() if v.get("quantum_yield") is not None]
    mean_qy = float(np.mean(qys)) if len(qys) else 1.0
    for v in dye_db.values():
        if v.get("quantum_yield") is None:
            v["quantum_yield"] = mean_qy

    return wl, dye_db


def load_probe_fluor_map(path):
    """
    Expect YAML either:
      - a list of {name: <probe>, fluors: [..]} at top-level
      - or {probes: [...] } with同样的结构
    Return: dict[probe] -> list[fluor]
    """
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if isinstance(data, list):
        items = data
    elif isinstance(data, dict) and isinstance(data.get("probes"), list):
        items = data["probes"]
    else:
        # Fallback: empty
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


# =============== Helpers ===============

def _safe_l2norm_cols(E):
    """L2-normalize each column; avoid divide by zero."""
    denom = np.linalg.norm(E, axis=0, keepdims=True) + 1e-12
    return E / denom

def cosine_similarity_matrix(E):
    """Cosine similarity of columns; diagonal set to 0."""
    norms = np.linalg.norm(E, axis=0) + 1e-12
    G = (E.T @ E) / np.outer(norms, norms)
    np.fill_diagonal(G, 0.0)
    return G

def top_k_pairwise(S, labels_pair, k=10):
    """
    S: NxN cosine similarity (diag=0)
    labels_pair: list[str] length N, each like "Probe – Fluor"
    Return list of (value, label_i, label_j) sorted desc by value.
    """
    N = S.shape[0]
    iu = np.triu_indices(N, k=1)
    vals = S[iu]
    if vals.size == 0:
        return []
    order = np.argsort(-vals)
    order = order[: min(k, vals.size)]
    out = []
    for idx in order:
        i = iu[0][idx]
        j = iu[1][idx]
        out.append((float(vals[idx]), labels_pair[i], labels_pair[j]))
    return out


# =============== Spectra builders ===============

def build_emission_only_matrix(wl, dye_db, groups):
    """
    Build emission-only spectra matrix for optimization.
    Rule: BEFORE any calculation, normalize emission by its own max (peak=1).
    Return:
      E_norm: W x N (L2-normalized columns for cosine),
      labels_pair: ["Probe – Fluor", ...],
      idx_groups: list[list] column indices per probe group in same order as E.
    """
    W = len(wl)
    cols = []
    labels = []
    idx_groups = []
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
            # normalize by max peak first
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
    """Assume integer or 1nm grid; pick nearest index of wavelength lam."""
    idx = int(round(lam - wl[0]))
    if idx < 0: idx = 0
    if idx >= len(wl): idx = len(wl) - 1
    return idx

def _segments_from_lasers(wl, lasers_sorted):
    """
    Build segments [lo, hi) in wavelength for simultaneous mode.
    """
    segs = []
    for i, l in enumerate(lasers_sorted):
        lo = l
        hi = lasers_sorted[i+1] if i+1 < len(lasers_sorted) else wl[-1] + 1
        segs.append((lo, hi))
    return segs

def derive_powers_simultaneous(wl, dye_db, selection_labels, laser_wavelengths):
    """
    Your 'B-leveling' rule when lasers fire simultaneously.
    selection_labels: list like ["Probe – Fluor", ...] for A
    Return powers aligned to sorted lasers.
    """
    lam = np.array(sorted(laser_wavelengths), dtype=float)
    segs = _segments_from_lasers(wl, lam)
    W = len(wl)

    # Extract fluor names from "Probe – Fluor"
    fls = [s.split(" – ", 1)[1] for s in selection_labels]
    recs = [dye_db[f] for f in fls if f in dye_db]

    # Helper coefficients c_i(l) = ex(l)*QY*EC
    def coef(rec, l):
        ex = rec["excitation"]; qy = rec["quantum_yield"]; ec = rec["extinction_coeff"]
        if ex is None or len(ex) != W or qy is None or ec is None:
            return 0.0
        idx = _nearest_idx_from_grid(wl, l)
        return float(ex[idx] * qy * (ec if ec is not None else 1.0))

    # Peak in segment for emission
    def seg_peak(rec, lo, hi):
        em = rec["emission"]
        if em is None or len(em) != W:
            return 0.0
        loi = _nearest_idx_from_grid(wl, lo)
        hii = _nearest_idx_from_grid(wl, hi-1) + 1
        if loi >= hii:
            return 0.0
        return float(np.max(em[loi:hii]))

    # Find first informative segment and set its power = 1, define B
    start = None
    for s, (lo, hi) in enumerate(segs):
        ok = any(seg_peak(r, lo, hi) > 0 for r in recs)
        if ok:
            start = s; break
    if start is None:
        return [1.0] * len(lam)  # degenerate

    P = np.zeros(len(lam))
    P[start] = 1.0

    lo, hi = segs[start]
    M = np.array([seg_peak(r, lo, hi) for r in recs])
    a = np.array([coef(r, lam[start]) for r in recs])
    B = float(np.max(a * M)) if np.any(M > 0) else 0.0

    # March forward segments
    for s in range(start+1, len(lam)):
        lo, hi = segs[s]
        M = np.array([seg_peak(r, lo, hi) for r in recs])
        c_s = np.array([coef(r, lam[s]) for r in recs])
        # contribution from already-set lasers
        pre = np.zeros(len(recs))
        for m in range(start, s):
            c_m = np.array([coef(r, lam[m]) for r in recs])
            pre += c_m * P[m]
        feasible = (M > 0) & (c_s > 0)
        if not np.any(feasible):
            P[s] = 0.0
        else:
            bounds = (B / (M[feasible] * c_s[feasible])) - (pre[feasible] / c_s[feasible])
            P[s] = max(0.0, float(np.min(bounds)))
    return [float(x) for x in P]


def derive_powers_separate(wl, dye_db, selection_labels, laser_wavelengths):
    """
    Lasers fired separately; each laser power set so its global peak across A equals a common B.
    """
    lam = np.array(sorted(laser_wavelengths), dtype=float)
    W = len(wl)
    fls = [s.split(" – ", 1)[1] for s in selection_labels]
    recs = [dye_db[f] for f in fls if f in dye_db]

    def coef(rec, l):
        ex = rec["excitation"]; qy = rec["quantum_yield"]; ec = rec["extinction_coeff"]
        if ex is None or len(ex) != W or qy is None or ec is None:
            return 0.0
        idx = _nearest_idx_from_grid(wl, l)
        return float(ex[idx] * qy * (ec if ec is not None else 1.0))

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
    if np.all(M <= 0):  # degenerate
        return [1.0] * len(lam)
    B = float(np.max(M))
    P = [float(B / v) if v > 0 else 0.0 for v in M]
    return P

def _interp_at(wl, y, lam):
    # 简单线性插值（边界截断）
    if lam <= wl[0]: return float(y[0])
    if lam >= wl[-1]: return float(y[-1])
    i = int(np.searchsorted(wl, lam)) - 1
    t = (lam - wl[i]) / (wl[i+1] - wl[i])
    return float(y[i] * (1 - t) + y[i+1] * t)


def build_effective_with_lasers(wl, dye_db, groups, laser_wavelengths, mode, powers):
    """
    Build effective spectra for ALL candidates under lasers.
    mode: "Simultaneous" or "Separate"
    powers: aligned to sorted(laser_wavelengths)
    Return:
      E_raw: W x N (not normalized,用于展示),
      E_norm: W x N (列向量 L2 归一化，用于相似度/优化),
      labels_pair, idx_groups
    """
    W = len(wl)
    lam = np.array(sorted(laser_wavelengths), dtype=float)
    pw = np.array(powers, dtype=float)
    segs = _segments_from_lasers(wl, lam)

    cols = []
    labels = []
    idx_groups = []
    col_id = 0

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
            if mode == "Separate":
                for i, l in enumerate(lam):
                    idx_l = _nearest_idx_from_grid(wl, l)
                    k = ex[idx_l] * qy * (ec if ec is not None else 1.0) * pw[i]
                    eff += em * k
            else:
                for i, (lo, hi) in enumerate(segs):
                    loi = _nearest_idx_from_grid(wl, lo)
                    hii = _nearest_idx_from_grid(wl, hi - 1) + 1
                    total_k = 0.0
                    for m in range(i + 1):
                        k_m = _interp_at(wl, ex, lam[m]) * qy * (ec if ec is not None else 1.0) * pw[m]
                        total_k += k_m
                    eff[loi:hii] += em[loi:hii] * total_k

            cols.append(eff)
            labels.append(f"{probe} – {fluor}")
            idxs.append(col_id)
            col_id += 1
        if idxs:
            idx_groups.append(idxs)

    if not cols:
        return np.zeros((W, 0)), np.zeros((W, 0)), [], []
    E_raw = np.stack(cols, axis=1)          # for plotting
    E_norm = _safe_l2norm_cols(E_raw)       # for cosine similarity
    return E_raw, E_norm, labels, idx_groups


# =============== Layer-by-layer MILP (with unique-fluor constraint) ===============

def _unique_dye_constraints(prob, x_vars, labels_pair, groups, fluor_names):
    """
    Add 'each fluor ≤ 1 time globally' constraints.
    fluor_names: list[str] aligned with labels_pair, where labels are "Probe – Fluor".
    """
    # Map dye -> indices of columns using that dye
    dye_to_cols = {}
    for j, d in enumerate(fluor_names):
        dye_to_cols.setdefault(d, []).append(j)
    for d, cols in dye_to_cols.items():
        prob += pulp.lpSum(x_vars[j] for j in cols) <= 1, f"Unique_{d}"


def solve_minimax_layer(E_norm, idx_groups, labels_pair, enforce_unique=True):
    """
    Minimize max pairwise cosine among selected one per group.
    Decision: pick exactly one column per group.
    y_ij = AND(x_i, x_j)
    t >= c_ij * y_ij
    Return (x_star_indices, t_star)
    """
    N = E_norm.shape[1]
    # Precompute pairwise cosine constants
    C = cosine_similarity_matrix(E_norm)
    # Variables
    prob = pulp.LpProblem("minimax", pulp.LpMinimize)
    x = [pulp.LpVariable(f"x_{j}", lowBound=0, upBound=1, cat="Binary") for j in range(N)]
    y = {}
    for i in range(N):
        for j in range(i+1, N):
            y[(i,j)] = pulp.LpVariable(f"y_{i}_{j}", lowBound=0, upBound=1, cat="Binary")
    t = pulp.LpVariable("t", lowBound=0)

    # One per group
    for g, idxs in enumerate(idx_groups):
        prob += pulp.lpSum(x[j] for j in idxs) == 1, f"OnePerGroup_{g}"

    # Unique-dye constraint (global)
    if enforce_unique:
        fluor_names = [s.split(" – ", 1)[1] for s in labels_pair]
        _unique_dye_constraints(prob, x, labels_pair, idx_groups, fluor_names)

    # y-AND linking
    for (i,j), yij in y.items():
        prob += yij <= x[i]
        prob += yij <= x[j]
        prob += yij >= x[i] + x[j] - 1

    # t constraints
    for (i,j), yij in y.items():
        cij = float(C[i,j])
        prob += t >= cij * yij

    prob += t
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    x_star = [j for j, var in enumerate(x) if var.value() >= 0.5]
    return x_star, float(t.value())


def solve_lexicographic(E_norm, idx_groups, labels_pair, levels=3, enforce_unique=True):
    """
    Layer-by-layer per论文思路：
      第1层：min t1 = max c_ij*y_ij
      第2层：在保留 t1*=常数 的同时，最小化第二大 —— 我们用常见的“阈值-偏差”表达。
    这里实现：逐层固定上一层的最大值（允许一个很小松弛），然后最小化
      sum_k (z_ij - lambda_k)^+ 形式，等价于线性引入mu_ij >= z_ij - lambda_k, mu>=0
    简化实现：每一层都新解一个LP，在上一层基础上添加“ t <= t_prev + eps ”并引入新的lambda/mu。
    """
    # 先做第1层
    sel, t1 = solve_minimax_layer(E_norm, idx_groups, labels_pair, enforce_unique=enforce_unique)

    # 若只要第一层
    if levels <= 1:
        return sel, [t1]

    # 为更高层，重新建立模型
    N = E_norm.shape[1]
    C = cosine_similarity_matrix(E_norm)

    # 通用变量
    prob = pulp.LpProblem("lexi", pulp.LpMinimize)
    x = [pulp.LpVariable(f"x_{j}", lowBound=0, upBound=1, cat="Binary") for j in range(N)]
    y = {}
    for i in range(N):
        for j in range(i+1, N):
            y[(i,j)] = pulp.LpVariable(f"y_{i}_{j}", lowBound=0, upBound=1, cat="Binary")
    # z_ij = c_ij*y_ij （常数*binary 可直接通过目标/约束使用，无需新变量）
    # 但为了后续层方便，我们定义 z 连续变量并强制 z = c*y
    z = {}
    for i in range(N):
        for j in range(i+1, N):
            z[(i,j)] = pulp.LpVariable(f"z_{i}_{j}", lowBound=0)

    # 组约束
    for g, idxs in enumerate(idx_groups):
        prob += pulp.lpSum(x[j] for j in idxs) == 1, f"OnePerGroup_{g}"

    if enforce_unique:
        fluor_names = [s.split(" – ", 1)[1] for s in labels_pair]
        _unique_dye_constraints(prob, x, labels_pair, idx_groups, fluor_names)

    # y-AND 约束
    for (i,j), yij in y.items():
        prob += yij <= x[i]
        prob += yij <= x[j]
        prob += yij >= x[i] + x[j] - 1

    # z = c * y
    for (i,j), zij in z.items():
        cij = float(C[i,j])
        # 等式：z == c*y  （c>=0）可用两条不等式夹住
        prob += zij <= cij * y[(i,j)]
        prob += zij >= cij * y[(i,j)]

    # 第1层值固定（给一点松弛 eps 避免数值问题）
    eps = 1e-6
    # t1 = max z_ij
    t1_var = pulp.LpVariable("t1", lowBound=0)
    for (i,j), zij in z.items():
        prob += t1_var >= zij
    prob += t1_var <= t1 + eps

    # 第2层：最小化第二大
    # 标准技巧：min lambda2 + (1/|P|)*sum mu_ij  近似收缩
    lam2 = pulp.LpVariable("lambda2", lowBound=0)
    mu2 = {k: pulp.LpVariable(f"mu2_{k[0]}_{k[1]}", lowBound=0) for k in z.keys()}
    for k, zij in z.items():
        prob += mu2[k] >= zij - lam2
        prob += mu2[k] >= 0

    # 如果需要第三层，可继续引入 lam3/mu3 等；这里实现到第二层（常用）
    prob += lam2 + (1.0 / max(1, len(z))) * pulp.lpSum(mu2.values())

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    x_star = [j for j, var in enumerate(x) if var.value() >= 0.5]

    # 返回各层最优值（第一层用 t1 的数值，第二层用 lam2）
    return x_star, [float(t1), float(lam2.value())]
