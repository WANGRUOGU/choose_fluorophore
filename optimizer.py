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
    # E: [Wavelength x N], columns already unit L2
    return E.T @ E

def build_pair_index(groups: List[List[int]]) -> List[Tuple[int,int]]:
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
    P = len(group_options)

    # 1) build candidates
    cand_names, cand_probe, cand_dye, is_inventory = [], [], [], []
    for g, opts in enumerate(group_options):
        for dye in opts:
            cand_names.append(f"{probe_names[g]} / {dye}")
            cand_probe.append(g)
            cand_dye.append(dye)
            is_inventory.append(True)

    purchasable_pool = purchasable_pool or []
    if allow_purchase and max_new_dyes > 0 and len(purchasable_pool) > 0:
        for g in range(P):
            for dye in purchasable_pool:
                cand_names.append(f"{probe_names[g]} / {dye}")
                cand_probe.append(g)
                cand_dye.append(dye)
                is_inventory.append(False)

    N = len(cand_names)
    if N == 0:
        raise ValueError("No candidates provided.")

    # 2) emission matrix
    W = len(next(iter(emission_by_dye.values())))
    E = np.zeros((W, N), dtype=float)
    for j, dye in enumerate(cand_dye):
        E[:, j] = emission_by_dye[dye]
    E = l2_normalize_columns(E)
    C = cosine_matrix(E)

    # 3) groups of indices
    groups = [[] for _ in range(P)]
    for j, g in enumerate(cand_probe):
        groups[g].append(j)
    pairs = build_pair_index(groups)

    solver = pywraplp.Solver.CreateSolver('CBC')
    if solver is None:
        raise RuntimeError("Could not create CBC solver.")

    # 4) variables
    x = [solver.IntVar(0, 1, f"x_{j}") for j in range(N)]
    y = [solver.IntVar(0, 1, f"y_{p}") for p in range(len(pairs))]
    t = solver.NumVar(0.0, 1.0, "t")

    # purchasable dye usage flags
    dye_to_uid, u, purch_names = {}, [], []
    if allow_purchase and max_new_dyes > 0:
        purch_set = set([cand_dye[j] for j in range(N) if not is_inventory[j]])
        for dname in sorted(purch_set):
            dye_to_uid[dname] = len(u)
            u.append(solver.IntVar(0, 1, f"u_{dname}"))
            purch_names.append(dname)

    # 5) constraints
    # one candidate per probe
    for g in range(P):
        solver.Add(sum(x[j] for j in groups[g]) == 1)

    # pair linking
    for pidx, (i, j) in enumerate(pairs):
        solver.Add(y[pidx] <= x[i])
        solver.Add(y[pidx] <= x[j])
        solver.Add(y[pidx] >= x[i] + x[j] - 1)

    # t >= c_ij * y_ij
    for pidx, (i, j) in enumerate(pairs):
        solver.Add(t >= C[i, j] * y[pidx])

    # purchase linking & cap
    if allow_purchase and max_new_dyes > 0:
        for j in range(N):
            if not is_inventory[j]:
                dname = cand_dye[j]
                solver.Add(x[j] <= u[dye_to_uid[dname]])
        solver.Add(sum(u) <= max_new_dyes)

    # 6) objective
    solver.Minimize(t)
    status = solver.Solve()
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        raise RuntimeError("Solver failed to find a feasible solution.")

    chosen = [j for j in range(N) if x[j].solution_value() > 0.5]
    chosen_by_probe = {cand_probe[j]: j for j in chosen}
    chosen_names = [cand_names[chosen_by_probe[g]] for g in range(P)]

    # P x P chosen similarity
    sim_pp = np.zeros((P, P))
    for g in range(P):
        for h in range(P):
            if g == h:
                sim_pp[g, h] = 1.0
            else:
                i = chosen_by_probe[g]; j = chosen_by_probe[h]
                sim_pp[g, h] = C[i, j]

    top_pairs = []
    for g in range(P):
        for h in range(g+1, P):
            i = chosen_by_probe[g]; j = chosen_by_probe[h]
            top_pairs.append(((g, h), float(C[i, j])))
    top_pairs.sort(key=lambda z: z[1], reverse=True)

    purchased = []
    if allow_purchase and max_new_dyes > 0:
        for dname, uid in dye_to_uid.items():
            if u[uid].solution_value() > 0.5:
                purchased.append(dname)

    return dict(
        t=float(t.solution_value()),
        chosen_by_probe=chosen_names,
        sim_pp=sim_pp,
        top_pairs=top_pairs,
        purchased=purchased,
    )
