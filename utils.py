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
    Simultaneous 模式下，严格按照下面逻辑确定各激光功率：

    1. 将激光波长排序得到 lam[0] < lam[1] < ...，并定义波段
       [lam[i], lam[i+1])，最后一段为 [lam[-1], wl[-1]+1)。

    2. 只用 selection_labels 对应的 fluorophore 来标定功率：
       - 找出每个染料全局 emission 峰值所在的波段；
       - 某个波段只要有一个峰值落在其中，就称为“有峰值波段”。

    3. 对于“没有峰值”的波段，其左边界激光功率设为 0。

    4. 在最左的“有峰值波段” s0：
       - 设该段左边界激光的功率 P[s0] = 1；
       - 对每个染料 j 计算：
            seg_peak_j = 该段内 emission 的最大值 M_{j,s0}
            k_j        = excitation(L_s0)*QY*EC
            value_j    = seg_peak_j * k_j
         取所有 value_j 的最大值作为全局亮度上限 B。

    5. 对后续每个“有峰值波段” s：
       - 对每个染料 j：
            pre_j = sum_{m < s} excitation(L_m)*QY*EC * P[m]
            k_js  = excitation(L_s)*QY*EC
            seg_peak_j = 该段内 emission 的最大值 M_{j,s}
            需要满足：seg_peak_j * (pre_j + k_js * P[s]) <= B
            令 f_j(c) = seg_peak_j * (pre_j + k_js * c)，
            解 f_j(c) = B 得到：
                c_j = (B/seg_peak_j - pre_j) / k_js
            只保留 c_j > 0 作为有效上界。
         在所有有效 c_j 中取最小值，作为该段左边界激光的功率 P[s]。

    返回值：
        powers_sorted_by_wavelength: len(lam) 的列表，和升序的 lam 一一对应
        B: 初始段定义的亮度上限
    """
    import numpy as np

    # 1) 激光排序
    lam = np.array(sorted(set(float(l) for l in laser_wavelengths)), dtype=float)
    W = len(wl)

    if lam.size == 0:
        return [0.0] * 0, 0.0

    # 2) 只用 selection_labels 对应的染料
    fluor_names = [s.split(" – ", 1)[1] for s in selection_labels]
    recs = []
    for f in fluor_names:
        rec = dye_db.get(f)
        if rec is None:
            continue
        em = rec.get("emission")
        ex = rec.get("excitation")
        if em is None or ex is None or len(em) != W or len(ex) != W:
            continue
        recs.append(rec)

    if not recs:
        # 没有可用染料，就全部功率设 0
        return [0.0] * len(lam), 0.0

    # 分段 helper（和 build_effective_with_lasers 保持一致）
    segs = _segments_from_lasers(wl, lam)

    def seg_peak(rec, seg_index):
        """该染料 rec 在第 seg_index 段内 emission 最大值。"""
        lo, hi = segs[seg_index]
        loi = _nearest_idx_from_grid(wl, lo)
        hii = _nearest_idx_from_grid(wl, hi - 1) + 1
        if loi >= hii:
            return 0.0
        em = rec["emission"]
        return float(np.max(em[loi:hii]))

    def coef_at(rec, l):
        """在激光波长 l 下的 excitation * QY * EC（用插值方式取 ex(l)）。"""
        ex = rec["excitation"]
        qy = rec.get("quantum_yield", None)
        ec = rec.get("extinction_coeff", None)
        if ex is None or len(ex) != W or qy is None:
            return 0.0
        ex_l = _interp_at(wl, ex, l)
        return float(ex_l * qy * (ec if ec is not None else 1.0))

    # 3) 标记每个波段是否有峰值
    seg_has_peak = [False] * len(segs)
    for rec in recs:
        em = rec["emission"]
        if em is None or len(em) != W:
            continue
        jmax = int(np.argmax(em))
        lam_peak = wl[jmax]
        for s, (lo, hi) in enumerate(segs):
            if lo <= lam_peak < hi:
                seg_has_peak[s] = True
                break

    peak_segs = [s for s, u in enumerate(seg_has_peak) if u]

    # 初始化功率（默认 0）
    P = np.zeros(len(lam), dtype=float)

    if not peak_segs:
        # 没有任何有峰值的段，保持全 0
        return P.tolist(), 0.0

    # 4) 最左的有峰值段 s0
    s0 = peak_segs[0]
    # 没有峰值的段左边界激光 power = 0 已经由 P 的初始值保证
    P[s0] = 1.0

    # 用 s0 段定义 B
    B = 0.0
    for rec in recs:
        m0 = seg_peak(rec, s0)  # 该段内 emission 峰值
        if m0 <= 0:
            continue
        k0 = coef_at(rec, lam[s0]) * P[s0]
        val = m0 * k0
        if val > B:
            B = val

    if B <= 0.0:
        # 理论上不太会出现，退化情况：所有东西都是 0
        return P.tolist(), 0.0

    # 5) 对后续每个有峰值段，依次求该段的激光功率
    for s in peak_segs[1:]:
        cand_c = []

        for rec in recs:
            # 该段 emission 峰值
            m_seg = seg_peak(rec, s)
            if m_seg <= 0.0:
                continue

            # pre_j = 之前所有已经确定的激光在该染料上的总系数
            pre_j = 0.0
            for m_idx in range(s):
                if P[m_idx] == 0.0:
                    continue
                k_prev = coef_at(rec, lam[m_idx])
                pre_j += k_prev * P[m_idx]

            # 当前段左边界激光的系数 k_js
            k_js = coef_at(rec, lam[s])
            if k_js <= 0.0:
                # 这一段该染料对当前激光不敏感，不会给出上界
                continue

            # 解 m_seg * (pre_j + k_js * c) = B
            c_j = (B / m_seg - pre_j) / k_js

            # 只保留 c_j > 0 的解作为有效上界
            if c_j > 0.0:
                cand_c.append(c_j)

        if cand_c:
            P[s] = float(max(0.0, min(cand_c)))
        else:
            # 没有任何染料给出正的上界，就保持 0
            P[s] = 0.0

    # 返回和 lam（升序）一一对应的功率列表，以及 B
    return P.tolist(), float(B)






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

    def _nearest_idx_from_grid_local(wl_, lam_):
        idx = int(round(lam_ - wl_[0]))
        if idx < 0:
            idx = 0
        if idx >= len(wl_):
            idx = len(wl_) - 1
        return idx

    def _segments_from_lasers_local(wl_, lasers_sorted):
        segs_ = []
        for i_, l_ in enumerate(lasers_sorted):
            lo_ = l_
            hi_ = lasers_sorted[i_ + 1] if i_ + 1 < len(lasers_sorted) else wl_[-1] + 1
            segs_.append((lo_, hi_))
        return segs_

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
