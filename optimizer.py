# optimizer.py
from ortools.linear_solver import pywraplp
import numpy as np
from typing import Dict, List, Tuple

def l2_normalize_columns(M: np.ndarray) -> np.ndarray:
    M = M.copy()
    norms = np.linalg.norm(M, axis=0)
    norms[norms == 0.0] = 1.0
    return M / norms

def cosine_matrix(E: np.ndarray) -> np.ndarray:
    # E: [Wavelength x N] already normalized by column
    return E.T @ E  # cosine because columns are unit L2

def build_pair_index(groups: List[List[int]]) -> List[Tuple[int,int]]:
    """All cross-group pairs of column indices."""
    pairs = []
    for g in range(len(groups)):
        for h in range(g+1, len(groups)):
            for i in groups[g]:
                for j in groups[h]:
                    pairs.append((i, j))
    return pairs

def solve_minimax_inventory(
    probe_names: List[str],
    group_options: List[List[str]],
    emission_by_dye: Dict[str, np.ndarray],
    allow_purchase: bool = False,
    purchasable_pool: List[str] = None,
    max_new_dyes: int = 0,
) -> Dict:
    """
    Single-level minimax with grouped assignment and optional purchase.
    - probe_names: ["probe1", "probe2", ...]
    - group_options: list per probe: [["ATTO 488", "AF405"], ["ATTO 647N", ...], ...] (inventory options)
    - emission_by_dye: dict dye -> emission vector (same length wavelengths)
    - allow_purchase: if True, we can add dyes from purchasable_pool for any probe
    - purchasable_pool: list of dye names we can buy (subset of keys in emission_by_dye)
    - max_new_dyes: cap on number of purchased dyes (0 = inventory only)
    Returns dict with chosen assignment, t*, similarity matrix & top similarities.
    """
    P = len(group_options)
    # 1) Build global list of candidate (probe, dye)
    #    Inventory candidates first
    cand_names = []  # length N: "probe/dye"
    cand_probe  = [] # probe index per candidate
    cand_dye    = [] # dye name per candidate
    candidate_is_inventory = []

    for g, opts in enumerate(group_options):
        for dye in opts:
            cand_names.append(f"{probe_names[g]} / {dye}")
            cand_probe.append(g)
            cand_dye.append(dye)
            candidate_is_inventory.append(True)

    # If allowed, append purchasable entries for each probe
    purchasable_pool = purchasable_pool or []
    if allow_purchase and max_new_dyes > 0 and len(purchasable_pool) > 0:
        for g in range(P):
            for dye in purchasable_pool:
                cand_names.append(f"{probe_names[g]} / {dye}")
                cand_probe.append(g)
                cand_dye.append(dye)
                candidate_is_inventory.append(False)

    N = len(cand_names)
    if N == 0:
        raise ValueError("No candidates provided.")

    # 2) Assemble emission matrix [W x N] and normalize columns
    W = len(next(iter(emission_by_dye.values())))
    E = np.zeros((W, N), dtype=float)
    for j, dye in enumerate(cand_dye):
        E[:, j] = emission_by_dye[dye]
    E = l2_normalize_columns(E)
    C = cosine_matrix(E)  # NxN, cosine similarities

    # 3) Build groups of column indices per probe
    groups = [[] for _ in range(P)]
    for j, g in enumerate(cand_probe):
        groups[g].append(j)

    pairs = build_pair_index(groups)  # list of (i,j) with cross-probe pairs
    m = len(pairs)

    solver = pywraplp.Solver.CreateSolver('CBC')
    if solver is None:
        raise RuntimeError("Could not create CBC solver.")

    # 4) Variables
    x = [solver.IntVar(0, 1, f"x_{j}") for j in range(N)]            # choose candidate j
    y = [solver.IntVar(0, 1, f"y_{p}") for p in range(m)]            # select pair
    t = solver.NumVar(0.0, 1.0, "t")                                 # maximum similarity

    # purchasable flags per dye (NOT per candidate): u_dye (buy once, reuse)
    dye_to_uid = {}
    u = []
    purch_names = []
    if allow_purchase and max_new_dyes > 0:
        # Only create u for dyes that actually appear as purchasable in candidates
        purch_set = set([cand_dye[j] for j in range(N) if not candidate_is_inventory[j]])
        for dname in sorted(purch_set):
            dye_to_uid[dname] = len(u)
            u.append(solver.IntVar(0, 1, f"u_{dname}"))
            purch_names.append(dname)

    # 5) Constraints
    # 5.1: One candidate per probe
    for g in range(P):
        solver.Add(sum(x[j] for j in groups[g]) == 1)

    # 5.2: Pair linking: y_ij = 1 iff x_i = x_j = 1  (big-M free via three inequalities)
    for pidx, (i, j) in enumerate(pairs):
        solver.Add(y[pidx] <= x[i])
        solver.Add(y[pidx] <= x[j])
        solver.Add(y[pidx] >= x[i] + x[j] - 1)

    # 5.3: t >= c_ij * y_ij
    for pidx, (i, j) in enumerate(pairs):
        solver.Add(t >= C[i, j] * y[pidx])

    # 5.4: Purchase linking & cap
    if allow_purchase and max_new_dyes > 0:
        # link v (implicit via x) to u: if a chosen candidate j uses a purchasable dye d, then u_d=1
        for j in range(N):
            if not candidate_is_inventory[j]:
                dname = cand_dye[j]
                solver.Add(x[j] <= u[dye_to_uid[dname]])
        # cap number of purchased dyes
        solver.Add(sum(u) <= max_new_dyes)

    # 6) Objective: minimize t  (minimax)
    solver.Minimize(t)

    status = solver.Solve()
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        raise RuntimeError("Solver did not find a feasible solution.")

    chosen = [j for j in range(N) if x[j].solution_value() > 0.5]
    chosen_names = [cand_names[j] for j in chosen]
    # Build selected similarity matrix for display (P x P with chosen per probe)
    # Map: per probe which candidate was chosen
    chosen_by_probe = {}
    for j in chosen:
        g = cand_probe[j]
        chosen_by_probe[g] = j
    # P x P matrix of pairwise cosine between chosen candidates
    sim_pp = np.zeros((P, P))
    for g in range(P):
        for h in range(P):
            if g == h:
                sim_pp[g, h] = 1.0
            else:
                i = chosen_by_probe[g]
                j = chosen_by_probe[h]
                sim_pp[g, h] = C[i, j]

    # top similarities among cross-probe pairs
    top_pairs = []
    for g in range(P):
        for h in range(g+1, P):
            i = chosen_by_probe[g]
            j = chosen_by_probe[h]
            top_pairs.append(((g, h), float(C[i, j])))
    top_pairs.sort(key=lambda z: z[1], reverse=True)

    purchased = []
    if allow_purchase and max_new_dyes > 0:
        for dname, uid in dye_to_uid.items():
            if u[uid].solution_value() > 0.5:
                purchased.append(dname)

    return dict(
        t=float(t.solution_value()),
        chosen_names=chosen_names,
        chosen_by_probe=[cand_names[chosen_by_probe[g]] for g in range(P)],
        sim_pp=sim_pp,
        top_pairs=top_pairs,
        purchased=purchased,
    )
