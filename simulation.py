# simulation.py
import numpy as np


def _to_uint8_gray(img2d):
    z = np.asarray(img2d, dtype=float)
    m = float(z.max())
    if m > 0:
        z = z / m
    return (np.clip(z, 0, 1) * 255).astype(np.uint8)


def _argmax_labelmap(Ahat, colors, rescale_global=False):
    """
    Colored label map:
      - hue from the channel with maximum abundance per pixel
      - brightness from that maximum abundance
    """
    H, W, R = Ahat.shape
    idx = np.argmax(Ahat, axis=2)  # (H,W)
    mx  = np.max(Ahat, axis=2)     # (H,W)
    if rescale_global:
        m = float(mx.max())
        if m > 0:
            mx = mx / m
    cols = np.asarray(colors, dtype=float)   # (R,3)
    rgb = cols[idx] * mx[:, :, None]         # (H,W,3)
    rgb = np.clip(rgb, 0, 1)
    return (rgb * 255).astype(np.uint8)


def nls_unmix(Timg, E, iters=2000, tol=1e-6):
    """Fast MU-style NLS with per-pixel normalization. Timg(H,W,C), E(C,R) -> A(H,W,R)."""
    H, W, C = Timg.shape
    E = np.asarray(E, dtype=np.float32)
    if E.ndim != 2 or E.shape[0] != C:
        raise ValueError(f"E shape {E.shape} mismatch with Timg channels {C}")
    M = Timg.reshape(-1, C).astype(np.float32, copy=False)
    scale = np.sqrt(np.mean(M**2, axis=1, keepdims=True)); scale[scale<=0]=1.0
    Mn = M/scale
    EtE = E.T @ E
    A = Mn @ E @ np.linalg.pinv(EtE)
    A[A<0]=0
    for _ in range(iters):
        numer = Mn @ E
        denom = (A @ EtE) + 1e-12
        A *= numer / denom
        if np.max(numer / (denom + 1e-12)) < 1 + tol:
            break
    A *= scale
    mA = float(np.max(A))
    if mA>0: 
        A /= mA
    return A.reshape(H, W, E.shape[1])


def colorize_single(A_r, color):
    z = np.clip(A_r, 0, 1); m = float(z.max())
    if m > 0: 
        z /= m
    return z[:, :, None] * np.asarray(color)[None, None, :]


def colorize_composite(A, colors):
    rgb = np.zeros((A.shape[0], A.shape[1], 3), dtype=float)
    for r in range(A.shape[2]):
        rgb += colorize_single(A[:, :, r], colors[r])
    m = float(rgb.max())
    if m > 0: 
        rgb /= m
    return rgb


def _capsule_profile(H, W, cx, cy, length, width, theta):
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    X = xx - cx; Y = yy - cy
    c, s = np.cos(theta), np.sin(theta)
    xp =  c*X + s*Y
    yp = -s*X + c*Y
    half_L = 0.5*length
    r = 0.5*width
    rect = (np.abs(xp) <= half_L) & (np.abs(yp) <= r)
    val = np.zeros((H, W))
    if np.any(rect): 
        val[rect] = 1 - np.abs(yp[rect])/(r+1e-12)
    for side in (-1, 1):
        rho = np.sqrt((xp + side*half_L)**2 + yp**2)
        cap = rho <= r
        if np.any(cap): 
            val[cap] = np.maximum(val[cap], 1 - rho[cap]/(r+1e-12))
    return np.clip(val, 0, 1), val > 0


def _place_rods_scene(H, W, R, rods_per=3, rng=None, max_trials_per_class=1200):
    """
    Non-overlapping rods (capsules). Shorter & thicker to resemble E. coli.
    Returns Atrue(H,W,R) and per-class placed_counts(R,).
    """
    rng = np.random.default_rng() if rng is None else rng
    Atrue = np.zeros((H, W, R), dtype=np.float32)
    occ = np.zeros((H, W), dtype=bool)
    placed_counts = np.zeros(R, dtype=int)

    # E. coli–like: shorter & thicker
    Lmin, Lmax = 18, 30
    Wmin, Wmax = 10, 16

    for r in range(R):
        placed = 0
        tries = 0
        while placed < rods_per and tries < max_trials_per_class:
            tries += 1
            length = int(rng.integers(Lmin, Lmax + 1))
            width  = int(rng.integers(Wmin, Wmax + 1))
            theta  = float(rng.uniform(0, np.pi))
            margin = 6 + int(max(length, width) / 2)
            if W - 2*margin <= 2 or H - 2*margin <= 2:
                break
            cx = int(rng.integers(margin, W - margin))
            cy = int(rng.integers(margin, H - margin))
            prof, mask = _capsule_profile(H, W, cx, cy, length, width, theta)
            if not np.any(mask): 
                continue
            if np.any(occ & mask):
                continue
            m = float(prof.max())
            if m > 0:
                prof /= m
            Atrue[:, :, r] = np.maximum(Atrue[:, :, r], prof.astype(np.float32))
            occ |= mask
            placed += 1
        placed_counts[r] = placed

    return np.clip(Atrue, 0, 1), placed_counts


def _capsule_expected_area(Lmin=18, Lmax=30, Wmin=10, Wmax=16):
    L = 0.5 * (Lmin + Lmax)
    W = 0.5 * (Wmin + Wmax)
    r = 0.5 * W
    return 2 * r * L + np.pi * r * r


def _suggest_canvas_size(R, rods_per, target_density=0.22, min_side=160):
    area_one = _capsule_expected_area()
    total_obj_area = R * rods_per * area_one
    canvas_area = total_obj_area / max(1e-6, target_density)
    side = int(np.ceil(np.sqrt(canvas_area)))
    side = max(min_side, side)
    return side, side


def simulate_rods_and_unmix(E, H=None, W=None, rods_per=3, rng=None):
    """
    Forward: T = Atrue ⊗ E; scale to peak=255; Poisson; NLS unmix.
    Auto-resize canvas so each fluorophore can place 'rods_per' rods if possible.
    """
    rng = np.random.default_rng() if rng is None else rng
    E = np.asarray(E, dtype=float)
    if E.ndim != 2:
        raise ValueError(f"E must be 2D, got {E.shape}")
    C, R = E.shape

    if H is None or W is None:
        H, W = _suggest_canvas_size(R, rods_per, target_density=0.22, min_side=160)

    scale_attempts = 0
    while True:
        Atrue, placed = _place_rods_scene(H, W, R, rods_per, rng)
        if np.all(placed >= rods_per):
            break
        if scale_attempts >= 4:
            # give best-effort result
            break
        H = int(np.ceil(H * 1.25))
        W = int(np.ceil(W * 1.25))
        scale_attempts += 1

    Tclean = np.zeros((H, W, C), dtype=float)
    for c in range(C):
        Tclean[:, :, c] = np.tensordot(Atrue, E[c, :], axes=([2], [0]))

    peak = 255.0
    Tmax = float(np.max(Tclean))
    if Tmax <= 0:
        Tnoisy = np.zeros_like(Tclean)
    else:
        lam = Tclean * (peak / Tmax)
        lam = np.nan_to_num(lam, nan=0.0, posinf=1e6, neginf=0.0)
        lam = np.clip(lam, 0.0, 1e6)
        Tnoisy = rng.poisson(lam).astype(float) / peak

    Ahat = nls_unmix(Tnoisy, E, iters=1500, tol=1e-6)
    return Atrue, Ahat
