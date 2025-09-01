import cv2
import numpy as np
from typing import Dict, Tuple, List, Optional


# ====================== Utilities ======================

def _largest_contour(mask: np.ndarray) -> Optional[np.ndarray]:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    return max(cnts, key=cv2.contourArea)


def _polyline_arclen(P: np.ndarray) -> Tuple[np.ndarray, float]:
    if P.shape[0] < 2:
        return np.zeros((P.shape[0],), dtype=np.float32), 0.0
    d = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    return s, float(s[-1])


def _resample_polyline(P: np.ndarray, M: int) -> np.ndarray:
    if P.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if P.shape[0] == 1 or M <= 1:
        return np.repeat(P[:1, :].astype(np.float32), M, axis=0)
    s, L = _polyline_arclen(P)
    if L <= 1e-6:
        return np.repeat(P[:1, :].astype(np.float32), M, axis=0)
    t = np.linspace(0.0, L, M, dtype=np.float32)
    idx = np.searchsorted(s, t, side="right") - 1
    idx = np.clip(idx, 0, len(s) - 2)
    seg_len = (s[idx + 1] - s[idx]); seg_len[seg_len < 1e-9] = 1.0
    alpha = (t - s[idx]) / seg_len
    P0 = P[idx].astype(np.float32); P1 = P[idx + 1].astype(np.float32)
    return P0 + (P1 - P0) * alpha[:, None]


def _neighbors8(y: int, x: int, H: int, W: int):
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            yy = y + dy; xx = x + dx
            if 0 <= yy < H and 0 <= xx < W:
                yield yy, xx


# ====================== Skeletonization ======================

def _skeletonize_morph(bin_img: np.ndarray) -> np.ndarray:
    """Morphological skeleton (no ximgproc dependency)."""
    img = (bin_img > 0).astype(np.uint8) * 255
    skel = np.zeros_like(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False
    eroded = img.copy()
    while not done:
        opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(eroded, opened)
        skel = cv2.bitwise_or(skel, temp)
        eroded = cv2.erode(eroded, element)
        done = cv2.countNonZero(eroded) == 0
    return (skel > 0).astype(np.uint8) * 255


def _skeleton_longest_path(skel: np.ndarray) -> np.ndarray:
    """Extract the longest 8-connected path on the skeleton as an ordered polyline."""
    ys, xs = np.where(skel > 0)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    H, W = skel.shape
    # Build index map
    idx_map = -np.ones((H, W), dtype=np.int32)
    coords = np.stack([ys, xs], axis=1)
    for i, (y, x) in enumerate(coords):
        idx_map[y, x] = i

    # adjacency
    adj = [[] for _ in range(len(coords))]
    deg1 = []
    for i, (y, x) in enumerate(coords):
        deg = 0
        for ny, nx in _neighbors8(y, x, H, W):
            j = idx_map[ny, nx]
            if j >= 0:
                adj[i].append(j)
                deg += 1
        if deg == 1:
            deg1.append(i)

    # BFS helper
    def bfs(start):
        from collections import deque
        q = deque([start])
        dist = -np.ones((len(coords),), dtype=np.int32)
        prev = -np.ones((len(coords),), dtype=np.int32)
        dist[start] = 0
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] < 0:
                    dist[v] = dist[u] + 1
                    prev[v] = u
                    q.append(v)
        end = int(np.argmax(dist))
        return end, dist, prev

    # pick endpoints
    if len(deg1) >= 1:
        start = deg1[0]
    else:
        start = 0  # closed loop: pick arbitrary

    a, _, _ = bfs(start)
    b, dist, prev = bfs(a)

    # reconstruct path a->b
    path_idx = []
    cur = b
    while cur >= 0:
        path_idx.append(cur)
        cur = prev[cur]
    path_idx = path_idx[::-1]
    P = coords[path_idx][:, ::-1].astype(np.float32)  # swap to (x,y)
    return P


# ====================== Perpendicular (legacy alt) ======================

def draw_stitching_pattern(
    bgr: np.ndarray,
    mask: np.ndarray,
    info: Dict,
    spacing: int = 25,
    length_scale: float = 2.0,
    color: Tuple[int, int, int] = (30, 220, 30),
    thickness: int = 2,
):
    """Perpendicular 'staple-like' visualization using PCA axis (kept as alternative)."""
    out = bgr.copy()
    c = _largest_contour(mask)
    if c is None or len(c) < 5:
        return out, []
    pts = c.reshape(-1, 2).astype(np.float32)
    mu = np.mean(pts, axis=0)
    cov = np.cov(pts.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]
    (ux, uy) = eigvecs[:, 0]; (vx, vy) = eigvecs[:, 1]
    (cx, cy) = (float(mu[0]), float(mu[1]))

    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return out, []
    rel = np.stack([xs.astype(np.float32) - cx, ys.astype(np.float32) - cy], axis=1)
    t = rel @ np.array([[ux], [uy]], dtype=np.float32)
    tmin, tmax = float(np.min(t)), float(np.max(t))
    seg_half = max(6, int(length_scale * np.sqrt(max(1.0, float(np.min(eigvals))))))
    segments = []
    t_vals = np.arange(tmin, tmax + 1e-3, max(6, int(spacing)), dtype=np.float32)
    for ti in t_vals:
        px = cx + ux * ti; py = cy + uy * ti
        x1 = int(round(px - vx * seg_half)); y1 = int(round(py - vy * seg_half))
        x2 = int(round(px + vx * seg_half)); y2 = int(round(py + vy * seg_half))
        cv2.line(out, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
        segments.append(((x1, y1), (x2, y2)))
    return out, segments


# ====================== Centerline-based continuous suture ======================

def _binary_search_boundary(contour: np.ndarray, a: np.ndarray, b: np.ndarray, iters: int = 12) -> np.ndarray:
    """Refine intersection with polygon boundary between inside point a and outside point b."""
    lo = a.astype(np.float32).copy()
    hi = b.astype(np.float32).copy()
    for _ in range(iters):
        mid = (lo + hi) * 0.5
        res = cv2.pointPolygonTest(contour, (float(mid[0]), float(mid[1])), False)
        if res > 0:  # inside
            lo = mid
        else:
            hi = mid
    return lo


def _intersect_normal_with_boundary(contour: np.ndarray, p: np.ndarray, n: np.ndarray, max_dist: int = 120) -> Tuple[np.ndarray, np.ndarray, float]:
    """Shoot rays +/- n from p until outside, then binary search back to the boundary."""
    n = n / (np.linalg.norm(n) + 1e-6)
    step = 1.5
    # forward
    inside = p.copy()
    q = p.copy()
    for d in np.arange(0, max_dist + step, step):
        q = p + n * d
        res = cv2.pointPolygonTest(contour, (float(q[0]), float(q[1])), False)
        if res <= 0:  # crossed boundary
            q = _binary_search_boundary(contour, inside, q)
            break
        inside = q
    p_plus = q.copy()
    # backward
    inside = p.copy()
    q = p.copy()
    for d in np.arange(0, max_dist + step, step):
        q = p - n * d
        res = cv2.pointPolygonTest(contour, (float(q[0]), float(q[1])), False)
        if res <= 0:
            q = _binary_search_boundary(contour, inside, q)
            break
        inside = q
    p_minus = q.copy()
    width = float(np.linalg.norm(p_plus - p_minus))
    return p_plus, p_minus, width


def _curvature_along_polyline(P: np.ndarray) -> np.ndarray:
    """Discrete curvature magnitude along polyline P (Nx2)."""
    N = P.shape[0]
    if N < 3:
        return np.zeros((N,), dtype=np.float32)
    v_prev = P[1:-1] - P[:-2]
    v_next = P[2:] - P[1:-1]
    # normalize
    def _norm(v):
        n = np.linalg.norm(v, axis=1, keepdims=True)
        n[n < 1e-6] = 1.0
        return v / n
    t1 = _norm(v_prev)
    t2 = _norm(v_next)
    # angle between t1 and t2
    cross = t1[:, 0]*t2[:, 1] - t1[:, 1]*t2[:, 0]
    dot = np.clip((t1 * t2).sum(axis=1), -1.0, 1.0)
    dtheta = np.abs(np.arctan2(cross, dot))
    # arc-length approx
    ds = (np.linalg.norm(v_prev, axis=1) + np.linalg.norm(v_next, axis=1)) * 0.5 + 1e-6
    kappa = dtheta / ds
    # pad ends
    kappa = np.concatenate([[kappa[0]], kappa, [kappa[-1]]]).astype(np.float32)
    return kappa


def _adaptive_samples(P: np.ndarray, kappa: np.ndarray, alpha: float, s_min: float, s_max: float) -> List[int]:
    """Return indices into P chosen with curvature-adaptive spacing."""
    s, L = _polyline_arclen(P)
    if L <= 1e-6:
        return [0]
    idxs = [0]
    acc = 0.0
    i = 1
    while i < len(P):
        ds = float(np.linalg.norm(P[i] - P[i-1]))
        acc += ds
        # local target spacing
        kk = float(kappa[i])
        s_target = float(np.clip(alpha / (kk + 1e-6), s_min, s_max))
        if acc >= s_target:
            idxs.append(i)
            acc = 0.0
        i += 1
    if idxs[-1] != len(P)-1:
        idxs.append(len(P)-1)
    return idxs


def draw_running_suture_centerline(
    bgr: np.ndarray,
    mask: np.ndarray,
    alpha: float = 20.0,     # curvature weight: larger = generally wider spacing
    s_min: int = 8,          # min spacing in px
    s_max: int = 60,         # max spacing in px
    bite_frac: float = 0.9,  # fraction of width from center to edge (<=1 inside wound; >1 outside)
    max_probe: int = 120,
    color_thread: Tuple[int, int, int] = (30, 200, 255),
    thickness: int = 2,
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Continuous running suture based on centerline:
      1) skeletonize mask and extract longest path
      2) sample along path with curvature-adaptive spacing
      3) compute normals; binary search to boundary on both sides
      4) place entry/exit at +/- bite_frac of half-width
    """
    out = bgr.copy()
    bin_mask = (mask >= 128).astype(np.uint8) * 255
    c = _largest_contour(bin_mask)
    if c is None or len(c) < 5:
        return out, []

    # skeleton & longest path
    skel = _skeletonize_morph(bin_mask)
    centerline = _skeleton_longest_path(skel)
    if centerline.shape[0] < 5:
        # fallback: use contour midline (rare)
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect).astype(np.float32)
        box = np.vstack([box, box[:1]])
        centerline = _resample_polyline(box, 40)

    # dense resample for smooth normals
    P_dense = _resample_polyline(centerline, max(80, int(max(bin_mask.shape) * 2)))
    kappa = _curvature_along_polyline(P_dense)

    # adaptive indices
    idxs = _adaptive_samples(P_dense, kappa, alpha=float(alpha), s_min=float(s_min), s_max=float(s_max))
    if len(idxs) < 2:
        idxs = list(range(0, len(P_dense), max(5, int(s_min))))

    # polygon for point-in test
    contour_poly = c.reshape(-1, 2).astype(np.float32)

    thread_pts: List[Tuple[int, int]] = []
    for ii in idxs:
        p = P_dense[ii]
        # tangent from neighbors
        j0 = max(0, ii - 3); j1 = min(len(P_dense) - 1, ii + 3)
        v = P_dense[j1] - P_dense[j0]
        nrm = float(np.linalg.norm(v)) + 1e-6
        t = v / nrm
        n = np.array([-t[1], t[0]], dtype=np.float32)

        p_plus, p_minus, width = _intersect_normal_with_boundary(contour_poly, p, n, max_dist=int(max_probe))
        if width <= 3.0:
            continue

        mid = (p_plus + p_minus) * 0.5
        dir_vec = (p_plus - p_minus)
        L = float(np.linalg.norm(dir_vec)) + 1e-6
        dir_u = dir_vec / L
        half = 0.5 * width * float(np.clip(bite_frac, 0.05, 1.8))
        entry = mid + dir_u * half
        exitp = mid - dir_u * half

        e = (int(round(entry[0])), int(round(entry[1])))
        x = (int(round(exitp[0])), int(round(exitp[1])))

        cv2.line(out, e, x, color_thread, thickness, cv2.LINE_AA)
        cv2.circle(out, e, 2, color_thread, -1, cv2.LINE_AA)
        cv2.circle(out, x, 2, color_thread, -1, cv2.LINE_AA)

        thread_pts.append(e)
        thread_pts.append(x)

    # connect successive points for a single continuous thread
    for u, v in zip(thread_pts[:-1], thread_pts[1:]):
        cv2.line(out, u, v, color_thread, thickness, cv2.LINE_AA)

    return out, thread_pts
