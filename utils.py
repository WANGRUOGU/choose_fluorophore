import yaml
import numpy as np
# utils.py 顶部
import pulp

def _make_cbc_exact():
    # 不设置 time limit / gap，保证返回最优解
    # 仍然关闭日志，启用 MIP
    try:
        return pulp.PULP_CBC_CMD(msg=False, mip=True)
    except TypeError:
        # 极老版本兜底
        return pulp.PULP_CBC_CMD(msg=False)

_SOLVER = _make_cbc_exact()


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
    denom = np.linalg.norm(E, axis=0, keepdims=True) + 1e-12
    return E / denom

def cosine_similarity_matrix(E):
    norms = np.linalg.norm(E, axis=0) + 1e-12
    G = (E.T @ E) / np.outer(norms, norms)
    np.fill_diagonal(G, 0.0)
    return G

def top_k_pairwise(S, labels_pair, k=10):
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

# 简单线性插值（边界截断）
def _interp_at(wl, y, lam):
    if lam <= wl[0]: return float(y[0])
    if lam >= wl[-1]: return float(y[-1])
    i = int(np.searchsorted(wl, lam)) - 1
    t = (lam - wl[i]) / (wl[i+1] - wl[i])
    return float(y[i] * (1 - t) + y[i+1] * t)

# —— 贪心 warm-start —— #
def _greedy_lexi_seed(E_norm: np.ndarray, idx_groups):
    C = cosine_similarity_matrix(E_norm)
    chosen = []
    chosen_set = set()
    for g_idxs in idx_groups:
        best_j, best_score = None, 1e9
        for j in g_idxs:
            score = 0.0 if not chosen_set else max(C[j, k] for k in chosen_set)
            if score < best_score:
                best_score, best_j = score, j
        chosen.append(best_j)
        chosen_set.add(best_j)
    return chosen

def _warm_start_x_vars(x_vars, idx_groups, seed_indices):
    seed = set(seed_indices or [])
    for j, var in enumerate(x_vars):
        var.setInitialValue(1 if j in seed else 0)

# =============== Spectra builders ===============

def build_emission_only_matrix(wl, dye_db, groups):
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

def derive_powers_simultaneous(wl, dye_db, selection_labels, laser_wavelengths):
    lam = np.array(sorted(laser_wavelengths), dtype=float)
    segs = _segments_from_lasers(wl, lam)
    W = len(wl)

    fls = [s.split(" – ", 1)[1] for s in selection_labels]
    recs = [dye_db[f] for f in fls if f in dye_db]

    def coef(rec, l):
        ex = rec["excitation"]; qy = rec["quantum_yield"]; ec = rec["extinction_coeff"]
        if ex is None or len(ex) != W or qy is None or ec is None:
            return 0.0
        idx = _nearest_idx_from_grid(wl, l)
        return float(ex[idx] * qy * (ec if ec is not None else 1.0))

    def seg_peak(rec, lo, hi):
        em = rec["emission"]
        if em is None or len(em) != W:
            return 0.0
        loi = _nearest_idx_from_grid(wl, lo)
        hii = _nearest_idx_from_grid(wl, hi-1) + 1
        if loi >= hii:
            return 0.0
        return float(np.max(em[loi:hii]))

    start = None
    for s, (lo, hi) in enumerate(segs):
        ok = any(seg_peak(r, lo, hi) > 0 for r in recs)
        if ok:
            start = s; break
    if start is None:
        return [1.0] * len(lam), 0.0

    P = np.zeros(len(lam))
    P[start] = 1.0

    lo, hi = segs[start]
    M = np.array([seg_peak(r, lo, hi) for r in recs])
    a = np.array([coef(r, lam[start]) for r in recs])
    B = float(np.max(a * M)) if np.any(M > 0) else 0.0

    for s in range(start+1, len(lam)):
        lo, hi = segs[s]
        M = np.array([seg_peak(r, lo, hi) for r in recs])
        c_s = np.array([coef(r, lam[s]) for r in recs])
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
    lam = np.array(sorted(laser_wavelengths), dtype=float)
    W = len(wl)
    fls = [s.split(" – ", 1)[1] for s in selection_labels]
    recs = [dye_db[f] for f in fls if f in dye_db]

    def _interp_at_local(w, y, x):
        if x <= w[0]: return float(y[0])
        if x >= w[-1]: return float(y[-1])
        i = int(np.searchsorted(w, x)) - 1
        t = (x - w[i]) / (w[i+1] - w[i])
        return float(y[i]*(1-t) + y[i+1]*t)

    def coef(rec, l):
        ex = rec["excitation"]; qy = rec["quantum_yield"]; ec = rec["extinction_coeff"]
        if ex is None or len(ex) != W or qy is None:
            return 0.0
        ex_l = _interp_at_local(wl, ex, l)
        return float(ex_l * qy * (ec if ec is not None else 1.0))

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

    def _interp_at_local(w, y, x):
        if x <= w[0]: return float(y[0])
        if x >= w[-1]: return float(y[-1])
        i = int(np.searchsorted(w, x)) - 1
        t = (x - w[i]) / (w[i+1] - w[i])
        return float(y[i]*(1-t) + y[i+1]*t)

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
                    k = _interp_at_local(wl, ex, l) * qy * (ec if ec is not None else 1.0) * pw[i]
                    per_laser_blocks.append(em * k)
                eff_concat = np.concatenate(per_laser_blocks, axis=0)
                cols.append(eff_concat)
                labels.append(f"{probe} – {fluor}")
                idxs.append(col_id)
                col_id += 1
            if idxs:
                idx_groups.append(idxs)

        if not cols:
            return np.zeros((W * max(1, len(lam)), 0)), np.zeros((W * max(1, len(lam)), 0)), [], []

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
                    total_k = 0.0
                    for m in range(i + 1):
                        k_m = _interp_at_local(wl, ex, lam[m]) * qy * (ec if ec is not None else 1.0) * pw[m]
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

        E_raw = np.stack(cols, axis=1)
        denom = np.linalg.norm(E_raw, axis=0, keepdims=True) + 1e-12
        E_norm = E_raw / denom
        return E_raw, E_norm, labels, idx_groups


# =============== 约束与两种求解器 ===============

def _unique_dye_constraints(prob, x_vars, labels_pair, groups, fluor_names):
    dye_to_cols = {}
    for j, d in enumerate(fluor_names):
        dye_to_cols.setdefault(d, []).append(j)
    for d, cols in dye_to_cols.items():
        prob += pulp.lpSum(x_vars[j] for j in cols) <= 1, f"Unique_{d}"


def solve_minimax_layer(E_norm, idx_groups, labels_pair, enforce_unique=True):
    N = E_norm.shape[1]
    C = cosine_similarity_matrix(E_norm)
    prob = pulp.LpProblem("minimax", pulp.LpMinimize)
    x = [pulp.LpVariable(f"x_{j}", lowBound=0, upBound=1, cat="Binary") for j in range(N)]
    # y 改成连续 [0,1]，显著提速
    y = {}
    for i in range(N):
        for j in range(i+1, N):
            y[(i,j)] = pulp.LpVariable(f"y_{i}_{j}", lowBound=0, upBound=1)  # 连续
    t = pulp.LpVariable("t", lowBound=0)

    for g, idxs in enumerate(idx_groups):
        prob += pulp.lpSum(x[j] for j in idxs) == 1, f"OnePerGroup_{g}"

    if enforce_unique:
        fluor_names = [s.split(" – ", 1)[1] for s in labels_pair]
        _unique_dye_constraints(prob, x, labels_pair, idx_groups, fluor_names)

    for (i,j), yij in y.items():
        prob += yij <= x[i]
        prob += yij <= x[j]
        prob += yij >= x[i] + x[j] - 1

    for (i,j), yij in y.items():
        cij = float(C[i,j])
        prob += t >= cij * yij

    # warm start
    seed = _greedy_lexi_seed(E_norm, idx_groups)
    _warm_start_x_vars(x, idx_groups, seed)

    prob += t
    status = prob.solve(_SOLVER)

    x_star = _pick_integral_from_relaxed(x, idx_groups)
    t_val = float(t.value() or 0.0)
    return x_star, t_val


def solve_lexicographic_k(E_norm, idx_groups, labels_pair, levels: int = 10, enforce_unique: bool = True):
    """
    渐进式字典序最小化（前 K 层）：
      第1层：最小化 t1 = max z_ij
      第2层：在 t1 最优基础上，最小化“第二大”
      ...
      逐层递进。每层都对前层最优值加 cap（+eps），并 warm-start。
    返回: (x_star_indices, [t1*, lam2*, lam3*, ...])
    """
    N = E_norm.shape[1]
    if N == 0:
        return [], []
    C = cosine_similarity_matrix(E_norm)
    pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    P = len(pairs)
    if P == 0:
        sel = [g[0] for g in idx_groups if g]
        return sel, []

    K = int(max(1, min(levels, P)))
    eps = 1e-6
    best_layers = []
    x_star_last = None

    for k in range(1, K + 1):
        prob = pulp.LpProblem(f"lexi_k{k}", pulp.LpMinimize)

        x = [pulp.LpVariable(f"x_{j}", lowBound=0, upBound=1, cat="Binary") for j in range(N)]
        # y 连续
        y = { (i,j): pulp.LpVariable(f"y_{i}_{j}", lowBound=0, upBound=1) for (i,j) in pairs }
        z = { (i,j): pulp.LpVariable(f"z_{i}_{j}", lowBound=0) for (i,j) in pairs }

        for g, idxs in enumerate(idx_groups):
            prob += pulp.lpSum(x[j] for j in idxs) == 1, f"OnePerGroup_{g}"

        if enforce_unique:
            fluor_names = [s.split(" – ", 1)[1] for s in labels_pair]
            _unique_dye_constraints(prob, x, labels_pair, idx_groups, fluor_names)

        for (i,j) in pairs:
            yij = y[(i,j)]
            prob += yij <= x[i]
            prob += yij <= x[j]
            prob += yij >= x[i] + x[j] - 1

        for (i,j) in pairs:
            cij = float(C[i,j])
            zij = z[(i,j)]
            prob += zij <= cij * y[(i,j)]
            prob += zij >= cij * y[(i,j)]

        # 第1层：t1 = max z_ij
        t1 = pulp.LpVariable("t1", lowBound=0)
        for (i,j) in pairs:
            prob += t1 >= z[(i,j)]

        # 对已完成的前层加入cap
        if len(best_layers) >= 1:
            prob += t1 <= best_layers[0] + eps

        lam = {}
        mu = {}
        if k >= 2:
            for r in range(2, k + 1):
                lam[r] = pulp.LpVariable(f"lam{r}", lowBound=0)
                mu[r] = { (i,j): pulp.LpVariable(f"mu{r}_{i}_{j}", lowBound=0) for (i,j) in pairs }
                for (i,j) in pairs:
                    prob += mu[r][(i,j)] >= z[(i,j)] - lam[r]
                    prob += mu[r][(i,j)] >= 0
                if (r - 1) < len(best_layers):
                    prob += lam[r] <= float(best_layers[r - 1]) + eps

        # warm start（上一层的 x 或贪心）
        if x_star_last is not None:
            _warm_start_x_vars(x, idx_groups, x_star_last)
        else:
            seed = _greedy_lexi_seed(E_norm, idx_groups)
            _warm_start_x_vars(x, idx_groups, seed)

        # 目标：当前层
        if k == 1:
            prob += t1
        else:
            prob += lam[k] + (1.0 / max(1, P)) * pulp.lpSum(mu[k][p] for p in pairs)

        _ = prob.solve(_SOLVER)

        # 记录层值 + 早停
        if k == 1:
            best_layers.append(float(t1.value() or 0.0))
        else:
            best_layers.append(float(lam[k].value() or 0.0))
        if best_layers[-1] <= 1e-8:
            x_star_last = _pick_integral_from_relaxed(x, idx_groups)
            break

        x_star_last = _pick_integral_from_relaxed(x, idx_groups)

    return x_star_last, best_layers
