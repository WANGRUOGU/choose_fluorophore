# utils.py
import yaml
import numpy as np
import pulp

# 统一使用 CBC（MIP）
_SOLVER = pulp.PULP_CBC_CMD(msg=False, mip=True)

def _pick_integral_from_relaxed(x_vars, idx_groups):
    """
    从（可能分数的）x 解里做“组内 argmax”，确保每个 probe 组只选一个。
    """
    xvals = np.array([(v.value() or 0.0) for v in x_vars], dtype=float)
    sel = []
    for idxs in idx_groups:
        if not idxs:
            continue
        j_local = int(np.argmax(xvals[idxs]))
        sel.append(int(idxs[j_local]))
    return sel

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
    'B-leveling' rule when lasers fire simultaneously.
    selection_labels: list like ["Probe – Fluor", ...] for A
    Return powers aligned to sorted lasers, and B.
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
        return [1.0] * len(lam), 0.0  # degenerate

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
    return [float(x) for x in P], float(B)


def derive_powers_separate(wl, dye_db, selection_labels, laser_wavelengths):
    """
    Separate 模式（新定义）：
    - 先在 emission-only 选集 A 上标定功率；
    - 令最小波长激光 power=1；
    - 其它激光把“可达峰值”拉到与最小波长相同；
    - 返回 (powers_sorted_by_wavelength, B) ，其中 B=M_min（仅作标尺，可用于可视化归一）。
    """
    lam = np.array(sorted(laser_wavelengths), dtype=float)
    W = len(wl)
    # 选中的染料名
    fls = [s.split(" – ", 1)[1] for s in selection_labels]
    recs = [dye_db[f] for f in fls if f in dye_db]

    # 线性插值取 excitation(l)
    def _interp_at(w, y, x):
        if x <= w[0]: return float(y[0])
        if x >= w[-1]: return float(y[-1])
        i = int(np.searchsorted(w, x)) - 1
        t = (x - w[i]) / (w[i+1] - w[i])
        return float(y[i]*(1-t) + y[i+1]*t)

    def coef(rec, l):
        ex = rec["excitation"]; qy = rec["quantum_yield"]; ec = rec["extinction_coeff"]
        if ex is None or len(ex) != W or qy is None:
            return 0.0
        ex_l = _interp_at(wl, ex, l)
        return float(ex_l * qy * (ec if ec is not None else 1.0))

    # 每束激光的“可达峰值” M_l（在选集 A 上）
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

    # 锚定最小波长束为 1，其 B=M_min
    P = np.zeros_like(M)
    if M.size == 0:
        return [1.0]*0, 1.0
    P[0] = 1.0
    B = float(M[0])

    # 其它束：把各自可达峰值拉到 B
    for i in range(1, len(M)):
        P[i] = float(B / M[i]) if M[i] > 0 else 0.0

    return [float(x) for x in P], B


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
      E_raw:  W x N (Simultaneous)  或  (W*L) x N（Separate，横向拼接）,
      E_norm: 同形状，列向量 L2 归一（用于相似度/优化）,
      labels_pair, idx_groups
    """
    W = len(wl)
    lam = np.array(sorted(laser_wavelengths), dtype=float)
    pw = np.array(powers, dtype=float)

    def _nearest_idx_from_grid(wl, lam):
        idx = int(round(lam - wl[0]))
        if idx < 0: idx = 0
        if idx >= len(wl): idx = len(wl) - 1
        return idx

    def _segments_from_lasers(wl, lasers_sorted):
        segs = []
        for i, l in enumerate(lasers_sorted):
            lo = l
            hi = lasers_sorted[i+1] if i+1 < len(lasers_sorted) else wl[-1] + 1
            segs.append((lo, hi))
        return segs

    # 线性插值取 excitation(l)
    def _interp_at(w, y, x):
        if x <= w[0]: return float(y[0])
        if x >= w[-1]: return float(y[-1])
        i = int(np.searchsorted(w, x)) - 1
        t = (x - w[i]) / (w[i+1] - w[i])
        return float(y[i]*(1-t) + y[i+1]*t)

    cols = []
    labels = []
    idx_groups = []
    col_id = 0

    if mode == "Separate":
        # ------- 横向拼接：每束激光的整条谱单独算，再 concat -------
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

                # 每束激光的整条谱
                per_laser_blocks = []
                for i, l in enumerate(lam):
                    k = _interp_at(wl, ex, l) * qy * (ec if ec is not None else 1.0) * pw[i]
                    per_laser_blocks.append(em * k)  # 整条谱，不做分段

                eff_concat = np.concatenate(per_laser_blocks, axis=0)  # (W*L,)
                cols.append(eff_concat)
                labels.append(f"{probe} – {fluor}")
                idxs.append(col_id)
                col_id += 1
            if idxs:
                idx_groups.append(idxs)

        if not cols:
            return np.zeros((W * max(1, len(lam)), 0)), np.zeros((W * max(1, len(lam)), 0)), [], []

        E_raw = np.stack(cols, axis=1)          # [(W*L) x N]
        # 列 L2 归一
        denom = np.linalg.norm(E_raw, axis=0, keepdims=True) + 1e-12
        E_norm = E_raw / denom
        return E_raw, E_norm, labels, idx_groups

    else:
        # ------- Simultaneous：逐段累计叠加 -------
        segs = _segments_from_lasers(wl, lam)
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
                    loi = _nearest_idx_from_grid(wl, lo)
                    hii = _nearest_idx_from_grid(wl, hi - 1) + 1
                    # 累计 0..i 束
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

        E_raw = np.stack(cols, axis=1)          # [W x N]
        denom = np.linalg.norm(E_raw, axis=0, keepdims=True) + 1e-12
        E_norm = E_raw / denom
        return E_raw, E_norm, labels, idx_groups



# =============== 优化器（含字典序与 Top-M 目标） ===============

def _unique_dye_constraints(prob, x_vars, labels_pair, groups, fluor_names):
    """
    Add 'each fluor ≤ 1 time globally' constraints.
    fluor_names: list[str] aligned with labels_pair, where labels are "Probe – Fluor".
    """
    dye_to_cols = {}
    for j, d in enumerate(fluor_names):
        dye_to_cols.setdefault(d, []).append(j)
    for d, cols in dye_to_cols.items():
        prob += pulp.lpSum(x_vars[j] for j in cols) <= 1, f"Unique_{d}"


def solve_minimax_layer(E_norm, idx_groups, labels_pair, enforce_unique=True):
    """
    Minimize max pairwise cosine among selected one per group.
    """
    N = E_norm.shape[1]
    C = cosine_similarity_matrix(E_norm)
    prob = pulp.LpProblem("minimax", pulp.LpMinimize)
    x = [pulp.LpVariable(f"x_{j}", lowBound=0, upBound=1, cat="Binary") for j in range(N)]
    y = {}
    for i in range(N):
        for j in range(i+1, N):
            y[(i,j)] = pulp.LpVariable(f"y_{i}_{j}", lowBound=0, upBound=1)
    t = pulp.LpVariable("t", lowBound=0)

    # 一组选一个
    for g, idxs in enumerate(idx_groups):
        prob += pulp.lpSum(x[j] for j in idxs) == 1, f"OnePerGroup_{g}"

    # 染料全局唯一（可选）
    if enforce_unique:
        fluor_names = [s.split(" – ", 1)[1] for s in labels_pair]
        _unique_dye_constraints(prob, x, labels_pair, idx_groups, fluor_names)

    # y = AND(x_i, x_j)
    for (i,j), yij in y.items():
        prob += yij <= x[i]
        prob += yij <= x[j]
        prob += yij >= x[i] + x[j] - 1

    # t >= c_ij * y_ij
    for (i,j), yij in y.items():
        cij = float(C[i,j])
        prob += t >= cij * yij

    # 目标
    prob += t
    status = prob.solve(_SOLVER)

    x_star = _pick_integral_from_relaxed(x, idx_groups)
    t_val = float(t.value() or 0.0)
    return x_star, t_val


def solve_lexicographic(E_norm, idx_groups, labels_pair, levels=3, enforce_unique=True):
    """
    先做第一层 minimax；若 levels>1，则在固定 t1 的前提下收缩第二大。
    """
    # 第一层
    sel, t1 = solve_minimax_layer(E_norm, idx_groups, labels_pair, enforce_unique=enforce_unique)
    if levels <= 1:
        return sel, [t1]

    N = E_norm.shape[1]
    C = cosine_similarity_matrix(E_norm)

    prob = pulp.LpProblem("lexi", pulp.LpMinimize)
    x = [pulp.LpVariable(f"x_{j}", lowBound=0, upBound=1, cat="Binary") for j in range(N)]
    y = {}
    for i in range(N):
        for j in range(i+1, N):
            y[(i,j)] = pulp.LpVariable(f"y_{i}_{j}", lowBound=0, upBound=1)
    z = {}
    for i in range(N):
        for j in range(i+1, N):
            z[(i,j)] = pulp.LpVariable(f"z_{i}_{j}", lowBound=0)

    # 一组选一个
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
        prob += zij <= cij * y[(i,j)]
        prob += zij >= cij * y[(i,j)]

    # 固定第一层最优值（加一点 eps 防数值问题）
    eps = 1e-6
    t1_var = pulp.LpVariable("t1", lowBound=0)
    for (i,j), zij in z.items():
        prob += t1_var >= zij
    prob += t1_var <= t1 + eps

    # 第二层：lambda2 + 平均超额
    lam2 = pulp.LpVariable("lambda2", lowBound=0)
    mu2 = {k: pulp.LpVariable(f"mu2_{k[0]}_{k[1]}", lowBound=0) for k in z.keys()}
    for k, zij in z.items():
        prob += mu2[k] >= zij - lam2
        prob += mu2[k] >= 0

    prob += lam2 + (1.0 / max(1, len(z))) * pulp.lpSum(mu2.values())
    status = prob.solve(_SOLVER)

    x_star = _pick_integral_from_relaxed(x, idx_groups)
    lam2_val = float(lam2.value() or 0.0)
    return x_star, [float(t1), lam2_val]


def solve_min_top_m(E_norm, idx_groups, labels_pair, top_m: int = 10, enforce_unique: bool = True):
    """
    一次性最小化“前 top_m 大”的 pairwise 相似度之和：
        min  M*t + sum mu_ij
        s.t. mu_ij >= z_ij - t, mu_ij >= 0
             z_ij = c_ij * y_ij,  y_ij = AND(x_i, x_j)
             每组恰选 1，且（可选）同一染料全局唯一
    返回: (x_star_indices, objective_value)
    """
    N = E_norm.shape[1]
    if N == 0:
        return [], 0.0

    C = cosine_similarity_matrix(E_norm)

    prob = pulp.LpProblem("min_topM_cosine", pulp.LpMinimize)
    x = [pulp.LpVariable(f"x_{j}", lowBound=0, upBound=1, cat="Binary") for j in range(N)]
    y, z, mu = {}, {}, {}

    pairs = []
    for i in range(N):
        for j in range(i+1, N):
            pairs.append((i, j))
            y[(i, j)] = pulp.LpVariable(f"y_{i}_{j}", lowBound=0, upBound=1, cat="Binary")
            z[(i, j)] = pulp.LpVariable(f"z_{i}_{j}", lowBound=0)
            mu[(i, j)] = pulp.LpVariable(f"mu_{i}_{j}", lowBound=0)

    # 一组选一个
    for g, idxs in enumerate(idx_groups):
        prob += pulp.lpSum(x[j] for j in idxs) == 1, f"OnePerGroup_{g}"

    # 全局唯一染料（从 "Probe – Fluor" 取 fluor 名）
    if enforce_unique:
        fluor_names = [s.split(" – ", 1)[1] for s in labels_pair]
        _unique_dye_constraints(prob, x, labels_pair, idx_groups, fluor_names)

    # y = AND(x_i, x_j)
    for (i, j), yij in y.items():
        prob += yij <= x[i]
        prob += yij <= x[j]
        prob += yij >= x[i] + x[j] - 1

    # z = C * y
    for (i, j), zij in z.items():
        cij = float(C[i, j])
        prob += zij <= cij * y[(i, j)]
        prob += zij >= cij * y[(i, j)]

    # Sum of Top-M：min M*t + sum mu_ij,  mu_ij >= z_ij - t
    M = int(max(1, top_m))
    t = pulp.LpVariable("t", lowBound=0)
    for (i, j), zij in z.items():
        prob += mu[(i, j)] >= zij - t
        prob += mu[(i, j)] >= 0

    prob += M * t + pulp.lpSum(mu.values())

    status = prob.solve(_SOLVER)

    # 读取：即便有分数解，也用组内 argmax 取可用解
    x_star = _pick_integral_from_relaxed(x, idx_groups)
    obj_val = float(pulp.value(M * t + pulp.lpSum(mu.values())))
    return x_star, obj_val

def solve_lexicographic_k(E_norm, idx_groups, labels_pair, levels: int = 10, enforce_unique: bool = True):
    """
    渐进式字典序最小化（前 K 层）：
      第1层：最小化 t1 = max z_ij
      第2层：在 t1 固定为最优的基础上，最小化“第二大”：
             用标准的 lambda/mu 线性化（mu_ij >= z_ij - lambda），最小化 lambda + 平均超额
      第k层：在前 (k-1) 层最优值都被“固定”的基础上，最小化第 k 大

    做法：逐层“重建”同一类模型，但在第 k 次求解时，一并引入 lam_1..lam_k、mu^(1..k)，
         并对 r<k 的 lam_r 加上上次的最优封顶 lam_r <= best[r-1] + eps，保证层层递进。
         目标函数只用当前层的 lam_k + avg(mu^(k))。

    返回：
      (x_star_indices, layer_values) 其中 layer_values = [t1*, lam2*, lam3*, ...] 最多 levels 项
    """
    N = E_norm.shape[1]
    if N == 0:
        return [], []
    # 常量
    C = cosine_similarity_matrix(E_norm)
    pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    P = len(pairs)
    if P == 0:
        # 只有一个组时，不存在 pair
        # 仍保证每组选1
        # 简单选每组第一个
        sel = [g[0] for g in idx_groups if g]
        return sel, []

    K = int(max(1, min(levels, P)))  # 最多到对数 P
    eps = 1e-6
    best_layers = []   # 记录每层的最优值 [t1*, lam2*, lam3*, ...]

    # 逐层求解（每次重建模型，加入前面层的caps）
    for k in range(1, K + 1):
        prob = pulp.LpProblem(f"lexi_k{k}", pulp.LpMinimize)

        # 变量
        x = [pulp.LpVariable(f"x_{j}", lowBound=0, upBound=1, cat="Binary") for j in range(N)]
        y = { (i,j): pulp.LpVariable(f"y_{i}_{j}", lowBound=0, upBound=1) for (i,j) in pairs }
        z = { (i,j): pulp.LpVariable(f"z_{i}_{j}", lowBound=0) for (i,j) in pairs }

        # 组约束
        for g, idxs in enumerate(idx_groups):
            prob += pulp.lpSum(x[j] for j in idxs) == 1, f"OnePerGroup_{g}"

        # 全局唯一（可选）
        if enforce_unique:
            fluor_names = [s.split(" – ", 1)[1] for s in labels_pair]
            _unique_dye_constraints(prob, x, labels_pair, idx_groups, fluor_names)

        # y = AND(x_i, x_j)
        for (i,j) in pairs:
            yij = y[(i,j)]
            prob += yij <= x[i]
            prob += yij <= x[j]
            prob += yij >= x[i] + x[j] - 1

        # z = C * y
        for (i,j) in pairs:
            cij = float(C[i,j])
            zij = z[(i,j)]
            prob += zij <= cij * y[(i,j)]
            prob += zij >= cij * y[(i,j)]

        # 第1层：t1 = max z_ij
        t1 = pulp.LpVariable("t1", lowBound=0)
        for (i,j) in pairs:
            prob += t1 >= z[(i,j)]

        # 若已有前一层最优，给 t1 加cap（保持前层最优不被破坏）
        if len(best_layers) >= 1:
            prob += t1 <= best_layers[0] + eps

        # 为了到第 k 层，需要定义 lam_r/mu_r (r=2..k)，并对 r<k 的 lam_r 施加cap
        lam = {}
        mu = {}
        if k >= 2:
            for r in range(2, k + 1):
                lam[r] = pulp.LpVariable(f"lam{r}", lowBound=0)
                # mu^(r) 为每个 pair 定义
                mu[r] = { (i,j): pulp.LpVariable(f"mu{r}_{i}_{j}", lowBound=0) for (i,j) in pairs }
                # 连接 mu^(r) 与 z、lam_r：mu^(r)_ij >= z_ij - lam_r
                for (i,j) in pairs:
                    prob += mu[r][(i,j)] >= z[(i,j)] - lam[r]
                    prob += mu[r][(i,j)] >= 0
                # 如果 r-1 层已有最优，给 lam_r 加cap
                if len(best_layers) >= (r - 1):
                    prob += lam[r] <= best_layers[r - 1] + eps

        # 目标：当前层
        if k == 1:
            prob += t1
        else:
            # 最小化 lam_k + 平均超额
            prob += lam[k] + (1.0 / max(1, P)) * pulp.lpSum(mu[k][p] for p in pairs)

        # 求解
        _ = prob.solve(_SOLVER)
        # 记录本层最优值
        if k == 1:
            best_layers.append(float(t1.value() or 0.0))
        else:
            best_layers.append(float(lam[k].value() or 0.0))

        # 最后一次的 x 解用于返回（做一次组内 argmax 兜底）
        x_star = _pick_integral_from_relaxed(x, idx_groups)

    return x_star, best_layers
