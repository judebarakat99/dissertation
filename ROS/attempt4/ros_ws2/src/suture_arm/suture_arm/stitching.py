import cv2
import numpy as np
from typing import Dict, Tuple, List, Optional

# ============================================================
# Mask ingestion & hygiene
# ============================================================

def _as_binary_mask(mask: np.ndarray, thresh: float = 0.5) -> np.ndarray:
    """Convert any mask to 0/255 uint8 (accepts float/bool/uint8/RGB)."""
    m = mask
    if m is None or m.size == 0:
        return np.zeros((1, 1), np.uint8)
    if m.dtype == np.bool_:
        m = m.astype(np.uint8) * 255
    elif np.issubdtype(m.dtype, np.floating):
        mx = float(np.nanmax(m)) if m.size else 1.0
        if mx <= 1.5:
            m = (m >= float(thresh)).astype(np.uint8) * 255
        else:
            m = (m >= 128.0).astype(np.uint8) * 255
    else:  # integer
        mx = int(m.max()) if m.size else 0
        if mx == 1:
            m = m.astype(np.uint8) * 255
        else:
            m = (m >= 128).astype(np.uint8) * 255
    if m.ndim == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    return m

def _clean_mask(mask_bin: np.ndarray) -> np.ndarray:
    """Gentle clean + close + fill holes via flood-fill (safe for thin wounds)."""
    m = mask_bin.copy()
    if m.ndim == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), 1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), 1)
    h, w = m.shape[:2]
    ff = np.zeros((h + 2, w + 2), np.uint8)
    inv = cv2.bitwise_not(m)
    cv2.floodFill(inv, ff, (0, 0), 255)
    holes = cv2.bitwise_not(inv)
    m = cv2.bitwise_or(m, holes)
    return m

def _fallback_if_empty(cleaned: np.ndarray, raw_bin: np.ndarray) -> np.ndarray:
    return cleaned if cv2.countNonZero(cleaned) > 0 else raw_bin

def _debug_safe_return(out: np.ndarray, why: str) -> Tuple[np.ndarray, list]:
    dbg = out.copy()
    cv2.putText(dbg, why, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2, cv2.LINE_AA)
    return dbg, []


# ============================================================
# Small utilities
# ============================================================

def _largest_contour(mask: np.ndarray) -> Optional[np.ndarray]:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    return max(cnts, key=cv2.contourArea)

def _neighbors8(y: int, x: int, H: int, W: int):
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            yy, xx = y + dy, x + dx
            if 0 <= yy < H and 0 <= xx < W:
                yield yy, xx

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
        return np.repeat(P[:1, :].astype(np.float32), max(1, M), axis=0)
    s, L = _polyline_arclen(P)
    if L <= 1e-6:
        return np.repeat(P[:1, :].astype(np.float32), max(1, M), axis=0)
    t = np.linspace(0.0, L, M, dtype=np.float32)
    idx = np.searchsorted(s, t, side="right") - 1
    idx = np.clip(idx, 0, len(s) - 2)
    seg_len = (s[idx + 1] - s[idx]); seg_len[seg_len < 1e-9] = 1.0
    alpha = (t - s[idx]) / seg_len
    P0 = P[idx].astype(np.float32); P1 = P[idx + 1].astype(np.float32)
    return P0 + (P1 - P0) * alpha[:, None]

def _sample_along_polyline_by_step(P: np.ndarray, step: float) -> np.ndarray:
    if P.shape[0] < 2:
        return P.copy()
    s, L = _polyline_arclen(P)
    if L <= 1e-6:
        return P[:1].copy()
    nseg = max(2, int(np.floor(L / max(1e-3, step))) + 1)
    t = np.linspace(0.0, L, nseg, dtype=np.float32)
    idx = np.searchsorted(s, t, side="right") - 1
    idx = np.clip(idx, 0, len(s) - 2)
    seg_len = (s[idx + 1] - s[idx]); seg_len[seg_len < 1e-9] = 1.0
    alpha = (t - s[idx]) / seg_len
    P0 = P[idx].astype(np.float32); P1 = P[idx + 1].astype(np.float32)
    Q = P0 + (P1 - P0) * alpha[:, None]
    keep = [0]
    for i in range(1, len(Q)):
        if np.linalg.norm(Q[i] - Q[keep[-1]]) >= 0.75:
            keep.append(i)
    return Q[keep]


# ============================================================
# Skeleton build + pruning
# ============================================================

def _skeletonize_morph(bin_img: np.ndarray) -> np.ndarray:
    img = (bin_img > 0).astype(np.uint8) * 255
    skel = np.zeros_like(img)
    elem = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    eroded = img.copy()
    while True:
        opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, elem)
        temp = cv2.subtract(eroded, opened)
        skel = cv2.bitwise_or(skel, temp)
        eroded = cv2.erode(eroded, elem)
        if cv2.countNonZero(eroded) == 0:
            break
    return (skel > 0).astype(np.uint8) * 255

def _prune_skeleton_spurs(skel: np.ndarray, min_len: int = 4) -> np.ndarray:
    s = skel.copy()
    H, W = s.shape
    changed = True
    while changed:
        changed = False
        ys, xs = np.where(s > 0)
        for y, x in zip(ys, xs):
            y0, y1 = max(0, y - 1), min(H, y + 2)
            x0, x1 = max(0, x - 1), min(W, x + 2)
            nb = s[y0:y1, x0:x1]
            deg = int(np.count_nonzero(nb) - 1)
            if deg == 1:
                s[y, x] = 0
                changed = True
    kcross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    s = cv2.morphologyEx(s, cv2.MORPH_OPEN, kcross, 1)
    return s

def _skeleton_graph(skel: np.ndarray):
    ys, xs = np.where(skel > 0)
    H, W = skel.shape
    if len(xs) == 0:
        return np.empty((0,2), np.int32), [], np.full((H,W), -1, np.int32), np.zeros((0,), np.int32)
    coords = np.stack([ys, xs], axis=1).astype(np.int32)
    idx_map = -np.ones((H, W), dtype=np.int32)
    for i, (y, x) in enumerate(coords):
        idx_map[y, x] = i
    adj = [[] for _ in range(len(coords))]
    deg = np.zeros((len(coords),), np.int32)
    for i, (y, x) in enumerate(coords):
        for ny, nx in _neighbors8(y, x, H, W):
            j = idx_map[ny, nx]
            if j >= 0:
                adj[i].append(j)
        deg[i] = len(adj[i])
    return coords, adj, idx_map, deg

def _bfs_parents(adj: List[List[int]], start: int):
    from collections import deque
    n = len(adj)
    dist = [-1]*n; prev = [-1]*n
    q = deque([start]); dist[start] = 0
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] < 0:
                dist[v] = dist[u] + 1
                prev[v] = u
                q.append(v)
    return dist, prev

def _reconstruct_path(prev: List[int], end: int) -> List[int]:
    path = []
    cur = end
    while cur >= 0:
        path.append(cur)
        cur = prev[cur]
    return path[::-1]

def _poly_from_idx_list(coords: np.ndarray, seq: List[int]) -> np.ndarray:
    if len(seq) == 0:
        return np.zeros((0,2), np.float32)
    P = coords[np.array(seq, dtype=np.int32)][:, ::-1].astype(np.float32)  # -> (x,y)
    return P


# ============================================================
# Curve geometry: normals & curvature
# ============================================================

def _normals_from_polyline(P: np.ndarray, k: int = 3) -> np.ndarray:
    N = len(P)
    normals = np.zeros_like(P, dtype=np.float32)
    for i in range(N):
        j0 = max(0, i - k)
        j1 = min(N - 1, i + k)
        v = P[j1] - P[j0]
        nrm = float(np.linalg.norm(v)) + 1e-6
        t = v / nrm
        normals[i] = np.array([-t[1], t[0]], dtype=np.float32)
    return normals

def _signed_curvature(P: np.ndarray) -> np.ndarray:
    N = len(P)
    if N < 3:
        return np.zeros((N,), np.float32)
    v_prev = P[1:-1] - P[:-2]
    v_next = P[2:] - P[1:-1]
    def _norm(v):
        n = np.linalg.norm(v, axis=1, keepdims=True)
        n[n < 1e-6] = 1.0
        return v / n
    t1 = _norm(v_prev); t2 = _norm(v_next)
    cross = t1[:,0]*t2[:,1] - t1[:,1]*t2[:,0]
    dot = np.clip((t1*t2).sum(axis=1), -1.0, 1.0)
    dtheta = np.arctan2(cross, dot)
    ds = (np.linalg.norm(v_prev, axis=1) + np.linalg.norm(v_next, axis=1))*0.5 + 1e-6
    kappa = dtheta / ds
    kappa = np.concatenate([[kappa[0]], kappa, [kappa[-1]]]).astype(np.float32)
    return kappa


# ============================================================
# Shape classification
# ============================================================

def _classify_shape(cnt: np.ndarray) -> str:
    rect = cv2.minAreaRect(cnt)
    (_, _), (w, h), _ang = rect
    w = max(w, 1e-3); h = max(h, 1e-3)
    ar = max(w, h) / min(w, h)
    rect_area = float(w*h)
    area = float(cv2.contourArea(cnt))
    fill = area / rect_area if rect_area > 1e-6 else 0.0
    if ar >= 2.5 and fill > 0.15:
        return "rect"
    if ar <= 1.6:
        return "boomerang"
    return "snake"


# ============================================================
# Centerlines (single & arms)
# ============================================================

def _centerline_from_mask(bin_mask: np.ndarray, spur_min_px: int = 4) -> np.ndarray:
    skel = _skeletonize_morph(bin_mask)
    if spur_min_px > 0:
        skel = _prune_skeleton_spurs(skel, min_len=int(spur_min_px))
    coords, adj, _, deg = _skeleton_graph(skel)
    if len(coords) == 0:
        return np.zeros((0,2), np.float32)
    ends = np.where(deg == 1)[0]
    start = int(ends[0]) if len(ends) else 0
    a_dist, _ = _bfs_parents(adj, start)
    a = int(np.argmax(a_dist))
    b_dist, prev = _bfs_parents(adj, a)
    b = int(np.argmax(b_dist))
    path_idx = _reconstruct_path(prev, b)
    return _poly_from_idx_list(coords, path_idx)

def _boomerang_arms_centerlines(bin_mask: np.ndarray, cnt: np.ndarray, spur_min_px: int = 4) -> List[np.ndarray]:
    skel = _skeletonize_morph(bin_mask)
    if spur_min_px > 0:
        skel = _prune_skeleton_spurs(skel, min_len=int(spur_min_px))
    coords, adj, _, deg = _skeleton_graph(skel)
    if len(coords) == 0:
        return []
    junctions = np.where(deg >= 3)[0]
    if len(junctions) == 0:
        return []
    M = cv2.moments(cnt)
    if abs(M['m00']) < 1e-6:
        return []
    cx = float(M['m10']/M['m00']); cy = float(M['m01']/M['m00'])
    c_yx = np.array([cy, cx], dtype=np.float32)
    dists = [np.linalg.norm(coords[j].astype(np.float32) - c_yx) for j in junctions]
    j_best = int(junctions[int(np.argmin(dists))])
    ends = np.where(deg == 1)[0]
    if len(ends) == 0:
        return []
    dist, prev = _bfs_parents(adj, j_best)
    arms: List[np.ndarray] = []
    for e in ends:
        if dist[e] < 0:
            continue
        path_idx = _reconstruct_path(prev, e)
        P = _poly_from_idx_list(coords, path_idx)
        if P.shape[0] >= 6:
            arms.append(P)
    arms.sort(key=lambda P: _polyline_arclen(P)[1], reverse=True)
    return arms[:3]


# ============================================================
# Mask-based ray intersections (ROBUST)
# ============================================================

def _intersect_normal_with_mask(bin_mask: np.ndarray, p: np.ndarray, n: np.ndarray, max_dist: int = 240):
    """
    Ray-march along ±n on a binary mask (0/255). Returns boundary hits (p_plus, p_minus) + chord width.
    """
    H, W = bin_mask.shape[:2]
    n = n / (np.linalg.norm(n) + 1e-6)
    def march(sign: float):
        step = 1.0
        prev_inside = None
        last_inside_xy = p.copy()
        q = p.copy()
        for d in np.arange(0.0, float(max_dist)+step, step):
            q = p + n * (sign * d)
            x, y = int(round(q[0])), int(round(q[1]))
            if x < 0 or x >= W or y < 0 or y >= H:
                # refine between last inside and outside by binary search on mask value
                lo = last_inside_xy.copy()
                hi = q.copy()
                for _ in range(10):
                    mid = (lo + hi) * 0.5
                    mx, my = int(round(mid[0])), int(round(mid[1]))
                    inside = (0 <= mx < W and 0 <= my < H and bin_mask[my, mx] > 0)
                    if inside: lo = mid
                    else:       hi = mid
                return lo
            inside = bin_mask[y, x] > 0
            if prev_inside is None:
                prev_inside = inside
            if prev_inside and not inside:
                # crossed boundary: refine
                lo = last_inside_xy.copy()
                hi = q.copy()
                for _ in range(10):
                    mid = (lo + hi) * 0.5
                    mx, my = int(round(mid[0])), int(round(mid[1]))
                    if 0 <= mx < W and 0 <= my < H and bin_mask[my, mx] > 0:
                        lo = mid
                    else:
                        hi = mid
                return lo
            if inside:
                last_inside_xy = q.copy()
        return last_inside_xy
    p_plus  = march(+1.0)
    p_minus = march(-1.0)
    width = float(np.linalg.norm(p_plus - p_minus))
    return p_plus, p_minus, width

def _nudge_inside_mask(bin_mask: np.ndarray, p: np.ndarray) -> np.ndarray:
    """If p is outside/edge, move it to the nearest inside pixel."""
    H, W = bin_mask.shape[:2]
    x, y = int(round(p[0])), int(round(p[1]))
    if 0 <= x < W and 0 <= y < H and bin_mask[y, x] > 0:
        return p
    r = 3
    best = None; best_d = 1e9
    for dy in range(-r, r+1):
        for dx in range(-r, r+1):
            xx, yy = x+dx, y+dy
            if 0 <= xx < W and 0 <= yy < H and bin_mask[yy, xx] > 0:
                d = dx*dx + dy*dy
                if d < best_d:
                    best_d = d; best = np.array([float(xx), float(yy)], dtype=np.float32)
    return best if best is not None else p


# ============================================================
# Adaptive sampling along curve
# ============================================================

def _sample_curve_adaptive(P: np.ndarray, base_step: float, alpha: float = 18.0) -> np.ndarray:
    """
    Curvature-adaptive arclength sampling; fewer rows in tight bends.
    SAFE for empty/1-point polylines.
    """
    P = np.asarray(P, dtype=np.float32)
    # --- NEW: guard empties & single-point inputs so np.ptp never sees size 0 ---
    if P.ndim != 2 or P.shape[0] == 0:
        return np.zeros((0, 2), np.float32)
    if P.shape[0] == 1:
        return P.copy()

    # --- NEW: compute spans safely (no reductions on empty) ---
    span_x = float(np.ptp(P[:, 0]))  # P has >=2 rows here
    span_y = float(np.ptp(P[:, 1]))
    M = max(240, int(2 * max(span_x, span_y, 40.0)))

    P_dense = _resample_polyline(P, M)
    if P_dense.shape[0] < 3:
        return P_dense

    kappa = np.abs(_signed_curvature(P_dense))
    s, L = _polyline_arclen(P_dense)

    t = [0.0]
    while t[-1] < L:
        i = np.searchsorted(s, t[-1], side="right") - 1
        i = np.clip(i, 0, len(s) - 2)
        step_here = float(base_step * (1.0 + alpha * kappa[i]))
        step_here = float(np.clip(step_here, 3.0, 60.0))
        nxt = t[-1] + step_here
        if nxt >= L:
            break
        t.append(nxt)

    tt = np.array(t, dtype=np.float32)
    idx = np.searchsorted(s, tt, side="right") - 1
    idx = np.clip(idx, 0, len(s) - 2)
    seg_len = (s[idx + 1] - s[idx]); seg_len[seg_len < 1e-9] = 1.0
    alpha_t = (tt - s[idx]) / seg_len
    P0 = P_dense[idx].astype(np.float32); P1 = P_dense[idx + 1].astype(np.float32)
    Q = P0 + (P1 - P0) * alpha_t[:, None]

    keep = [0]
    for i in range(1, len(Q)):
        if np.linalg.norm(Q[i] - Q[keep[-1]]) >= 1.0:
            keep.append(i)
    return Q[keep]


# ============================================================
# Path helpers: anti-crossing & progression checks
# ============================================================

def _segments_cross(a, b, c, d):
    def ccw(p1, p2, p3):
        return (p3[1]-p1[1])*(p2[0]-p1[0]) > (p2[1]-p1[1])*(p3[0]-p1[0])
    return (ccw(a, c, d) != ccw(b, c, d)) and (ccw(a, b, c) != ccw(a, b, d))

def _avoid_crossing(path: List[Tuple[int,int]]) -> List[Tuple[int,int]]:
    if len(path) < 4:
        return path
    out = [path[0]]
    for i in range(1, len(path)):
        p = path[i]
        q = out[-1]
        if len(out) >= 2 and _segments_cross(out[-2], q, q, p):
            v = np.array(q) - np.array(out[-2])
            nrm = np.linalg.norm(v)
            if nrm > 0:
                v = v / nrm
                p = (int(round(p[0] + 2*v[0])), int(round(p[1] + 2*v[1])))
        out.append(p)
    return out

def _ensure_progressive(path: List[Tuple[int,int]]) -> List[Tuple[int,int]]:
    if len(path) < 6:
        return path
    y = np.array([p[1] for p in path], dtype=np.float32)
    if np.std(y) < 2.0:
        return path[::-1]
    return path


# ============================================================
# Entry distance (mm) → pixels helper
# ============================================================

def _entry_offset_px(entry_mm: float = 6.0,
                     px_per_mm: Optional[float] = None,
                     width_px: float = 0.0,
                     outside_scale: float = 0.12,
                     outside_px: float = 3.0,
                     min_px: float = 3.0) -> float:
    """
    Returns a safe offset (pixels) to place needle entry points away from the wound edge.

    Rules:
    - Enforce clinical band: 5–10 mm (minimum 5 mm, maximum 10 mm).
    - If px_per_mm is provided (>0), convert mm→px and use it as a MINIMUM.
    - Also respect legacy safeguards: outside_scale * chord width and outside_px.
    """
    # Clamp to requested band
    entry_mm = float(min(10.0, max(5.0, entry_mm)))

    entry_px = 0.0
    if px_per_mm is not None and px_per_mm > 0:
        # Convert and keep within 5–10 mm in pixel space
        entry_px = float(entry_mm) * float(px_per_mm)
        min_allowed = 5.0 * float(px_per_mm)
        max_allowed = 10.0 * float(px_per_mm)
        if entry_px < min_allowed:
            entry_px = min_allowed
        elif entry_px > max_allowed:
            entry_px = max_allowed

    # Final offset is the maximum of: mm-based minimum, fixed px floor, and proportional-to-width.
    return float(max(min_px, float(outside_px), float(outside_scale) * float(width_px), entry_px))


# ============================================================
# Quadratic Bézier smoothing for the centerline
# ============================================================

def _quad_bezier(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Evaluate a quadratic Bézier B(t) = (1−t)^2 p0 + 2(1−t)t p1 + t^2 p2 for vector t."""
    t = t.astype(np.float32)
    omt = 1.0 - t
    return (omt*omt)[:, None] * p0 + (2.0*omt*t)[:, None] * p1 + (t*t)[:, None] * p2

def _bezier_chain_from_polyline(P: np.ndarray, samples_per_seg: int = 16) -> np.ndarray:
    """
    Build a C1-ish smooth chain of quadratic Béziers through a polyline:
    Each internal vertex becomes a control point; segment endpoints are midpoints.
    """
    P = np.asarray(P, dtype=np.float32)
    n = P.shape[0]
    if n <= 2:
        return _resample_polyline(P, max(2, (n-1) * samples_per_seg))
    out = []
    for i in range(n - 1):
        if i == 0 and n >= 3:
            a = P[0]
            b = P[1]
            c = 0.5 * (P[1] + P[2])
        elif i < n - 2:
            a = 0.5 * (P[i] + P[i+1])
            b = P[i+1]
            c = 0.5 * (P[i+1] + P[i+2])
        else:  # last span
            a = 0.5 * (P[i] + P[i+1])
            b = P[i+1]
            c = P[i+1]
        t = np.linspace(0.0, 1.0, max(3, int(samples_per_seg)), dtype=np.float32)
        curve = _quad_bezier(a, b, c, t)
        if len(out) > 0:
            curve = curve[1:]  # de-duplicate joint
        out.append(curve)
    S = np.vstack(out) if out else P
    keep = [0]
    for i in range(1, len(S)):
        if np.linalg.norm(S[i] - S[keep[-1]]) >= 0.75:
            keep.append(i)
    return S[keep].astype(np.float32)


# ============================================================
# PUBLIC: Perpendicular (curve-hugging staples)
# ============================================================

def draw_stitching_pattern(
    bgr: np.ndarray,
    mask: np.ndarray,
    info: Dict,
    spacing: int = 25,
    length_scale: float = 2.0,
    color: Tuple[int, int, int] = (30, 220, 30),
    thickness: int = 2,
    curvature_gain: float = 18.0,
    outside_scale: float = 0.12,
    spur_min_px: int = 4,
    entry_mm: float = 4.0,
    px_per_mm: Optional[float] = None,
):
    out = bgr.copy()
    raw_bin = _as_binary_mask(mask)
    bin_mask = _fallback_if_empty(_clean_mask(raw_bin), raw_bin)

    cnt = _largest_contour(bin_mask)
    if cnt is None or len(cnt) < 5:
        cnt = _largest_contour(raw_bin)
        if cnt is None or len(cnt) < 5:
            return _debug_safe_return(out, "no contour")

    shape = _classify_shape(cnt)
    base_step = float(max(6, int(spacing)) if shape == "rect" else max(3, int(spacing)))

    min_dim = min(bin_mask.shape[:2])
    spur_px = max(0, min(int(min_dim * 0.01), int(spur_min_px)))
    P = _centerline_from_mask(bin_mask, spur_min_px=int(spur_px))

    if len(P) < 2:
        ys, xs = np.where(bin_mask > 0)
        if xs.size < 8:
            return _debug_safe_return(out, "no centerline")
        pts = np.vstack([xs, ys]).T.astype(np.float32)
        mu = pts.mean(axis=0)
        cov = np.cov((pts - mu).T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        u = eigvecs[:, np.argmax(eigvals)]
        t = np.linspace(-50, 50, 20, dtype=np.float32)
        P = np.stack([mu[0] + u[0]*t, mu[1] + u[1]*t], axis=1)

    C = _sample_curve_adaptive(P, base_step=base_step, alpha=float(curvature_gain))
    if len(C) < 2:
        return _debug_safe_return(out, "too few samples")

    N = _normals_from_polyline(C, k=3)
    max_probe = int(2.0 * np.hypot(*bin_mask.shape))

    segments = []
    for p, n in zip(C, N):
        if np.linalg.norm(n) < 1e-6:
            continue
        p_plus, p_minus, width = _intersect_normal_with_mask(bin_mask, p, n, max_dist=max_probe)
        if width <= 2.0:
            continue
        outside_eff = _entry_offset_px(entry_mm=entry_mm, px_per_mm=px_per_mm,
                                       width_px=width, outside_scale=outside_scale,
                                       outside_px=3.0, min_px=3.0)
        e1 = p_plus + n * outside_eff
        e2 = p_minus - n * outside_eff
        a = (int(round(e1[0])), int(round(e1[1])))
        b = (int(round(e2[0])), int(round(e2[1])))
        cv2.line(out, a, b, color, int(thickness), cv2.LINE_AA)
        cv2.circle(out, a, 2, color, -1, cv2.LINE_AA)
        cv2.circle(out, b, 2, color, -1, cv2.LINE_AA)
        segments.append((a, b))

    if not segments:
        return _debug_safe_return(out, "no pattern segments")
    return out, segments


# ============================================================
# PUBLIC: Continuous zig-zag (shape-aware)
# ============================================================

def draw_running_suture_auto(
    bgr: np.ndarray,
    mask: np.ndarray,
    spacing_px: float = 20.0,
    outside_px: float = 3.0,
    rect_min_step: float = 6.0,
    max_probe: int = 240,
    color_thread: Tuple[int,int,int] = (30, 200, 255),
    thickness: int = 2,
    debug: bool = False,
    curvature_gain: float = 18.0,
    outside_scale: float = 0.12,
    spur_min_px: int = 4,
    entry_mm: float = 4.0,
    px_per_mm: Optional[float] = None,
) -> Tuple[np.ndarray, List[Tuple[int,int]]]:
    out = bgr.copy()
    raw_bin = _as_binary_mask(mask)
    bin_mask = _fallback_if_empty(_clean_mask(raw_bin), raw_bin)

    cnt = _largest_contour(bin_mask)
    if cnt is None or len(cnt) < 5:
        cnt = _largest_contour(raw_bin)
        if cnt is None or len(cnt) < 5:
            return _debug_safe_return(out, "no contour")

    contour_xy = cnt.reshape(-1, 2).astype(np.float32)
    shape = _classify_shape(cnt)

    def _draw_over_centerline(P: np.ndarray, step_min: float) -> List[Tuple[int,int]]:
        if P is None or P.size == 0:
            return []
        base_step = max(step_min, float(spacing_px))
        C = _sample_curve_adaptive(P, base_step=base_step, alpha=float(curvature_gain))
        if C.shape[0] < 2:
            return []
        N = _normals_from_polyline(C, k=3)

        L_pts: List[Tuple[int,int]] = []
        R_pts: List[Tuple[int,int]] = []
        local_probe = int(max(max_probe, 2.0 * np.hypot(*bin_mask.shape)))
        for p, n in zip(C, N):
            if np.linalg.norm(n) < 1e-6:
                continue
            p = _nudge_inside_mask(bin_mask, p)
            p_plus, p_minus, width = _intersect_normal_with_mask(bin_mask, p, n, max_dist=local_probe)
            if width <= 1.0:
                continue
            offset = _entry_offset_px(entry_mm=entry_mm, px_per_mm=px_per_mm,
                                      width_px=width, outside_scale=outside_scale,
                                      outside_px=outside_px, min_px=3.0)
            offset = float(min(offset, 0.45 * width))  # clamp so we never overshoot
            eR = p_plus + n * offset
            eL = p_minus - n * offset
            L_pts.append((int(round(eL[0])), int(round(eL[1]))))
            R_pts.append((int(round(eR[0])), int(round(eR[1]))))

        if len(L_pts) < 2 or len(R_pts) < 2:
            return []
        path: List[Tuple[int,int]] = []
        m = min(len(L_pts), len(R_pts))
        for i in range(m):
            path.append(L_pts[i]); path.append(R_pts[i])
        path = _avoid_crossing(path)
        path = _ensure_progressive(path)
        for a, b in zip(path[:-1], path[1:]):
            cv2.line(out, a, b, color_thread, thickness, cv2.LINE_AA)
        for pxy in path:
            cv2.circle(out, pxy, 2, color_thread, -1, cv2.LINE_AA)
        if debug:
            cv2.polylines(out, [contour_xy.astype(np.int32)], True, (120,220,220), 1, cv2.LINE_AA)
        return path

    min_dim = min(bin_mask.shape[:2])
    spur_px = max(0, min(int(min_dim * 0.01), int(spur_min_px)))

    all_path: List[Tuple[int,int]] = []
    if shape == "boomerang":
        arms = _boomerang_arms_centerlines(bin_mask, cnt, spur_min_px=int(spur_px))
        if not arms:
            P = _centerline_from_mask(bin_mask, spur_min_px=int(spur_px))
            all_path += _draw_over_centerline(P, step_min=3.0)
        else:
            for P in arms:
                all_path += _draw_over_centerline(P, step_min=3.0)
    elif shape == "rect":
        P = _centerline_from_mask(bin_mask, spur_min_px=int(spur_px))
        all_path += _draw_over_centerline(P, step_min=max(6.0, float(rect_min_step)))
    else:
        P = _centerline_from_mask(bin_mask, spur_min_px=int(spur_px))
        all_path += _draw_over_centerline(P, step_min=3.0)

    if not all_path:
        P = _centerline_from_mask(bin_mask, spur_min_px=int(spur_px))
        for factor in (0.75, 0.5):
            all_path = _draw_over_centerline(P, step_min=max(3.0, float(spacing_px) * factor))
            if all_path:
                break
    if not all_path:
        k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
        for msk in (cv2.dilate(bin_mask, k1, 1), cv2.dilate(bin_mask, k2, 1)):
            P = _centerline_from_mask(msk, spur_min_px=int(spur_px))
            all_path = _draw_over_centerline(P, step_min=max(3.0, float(spacing_px) * 0.66))
            if all_path:
                break

    # LAST RESORT: scanline zig-zag across bbox (guaranteed non-empty for non-empty masks)
    if not all_path and cv2.countNonZero(bin_mask) > 0:
        all_path = _scanline_zigzag(bin_mask, step=int(max(6, spacing_px)))
        for a, b in zip(all_path[:-1], all_path[1:]):
            cv2.line(out, a, b, color_thread, thickness, cv2.LINE_AA)
        for pxy in all_path:
            cv2.circle(out, pxy, 2, color_thread, -1, cv2.LINE_AA)

    if not all_path:
        return _debug_safe_return(out, "no stitch path")
    return out, all_path


# ============================================================
# PUBLIC: Mold-border zig-zag 
# ============================================================

def _zigzag_from_lr(L_pts: List[Tuple[int,int]], R_pts: List[Tuple[int,int]]) -> List[Tuple[int,int]]:
    m = min(len(L_pts), len(R_pts))
    path: List[Tuple[int,int]] = []
    for i in range(m):
        path.append(L_pts[i]); path.append(R_pts[i])
    path = _avoid_crossing(path)
    path = _ensure_progressive(path)
    return path

def _scanline_zigzag(bin_mask: np.ndarray, step: int = 12) -> List[Tuple[int,int]]:
    """Last-resort: horizontal scanlines across the mask bbox to produce L/R pairs."""
    ys, xs = np.where(bin_mask > 0)
    if xs.size == 0: return []
    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())
    L_pts, R_pts = [], []
    for y in range(y_min, y_max + 1, max(3, int(step))):
        row = bin_mask[y, x_min:x_max+1]
        on = np.where(row > 0)[0]
        if on.size < 2:  # need two sides
            continue
        lx = x_min + int(on.min())
        rx = x_min + int(on.max())
        L_pts.append((lx, y))
        R_pts.append((rx, y))
    return _zigzag_from_lr(L_pts, R_pts)

def draw_stitching_mold_border(
    bgr: np.ndarray,
    mask: np.ndarray,
    spacing_px: float = 20.0,
    grow_px: int = 3,
    max_probe: int = 240,
    color_thread: Tuple[int,int,int] = (50, 230, 70),
    dot_color: Tuple[int,int,int] = (255, 220, 80),
    thickness: int = 2,
    draw_dots: bool = True,
    debug: bool = False,
    curvature_gain: float = 18.0,
    spur_min_px: int = 4,
    border_push_px: float = 0.0,
    entry_mm: float = 4.0,
    px_per_mm: Optional[float] = None,
    outside_scale: float = 0.12,          
    **kwargs,                              
) -> Tuple[np.ndarray, List[Tuple[int,int]]]:

    out = bgr.copy()
    raw_bin = _as_binary_mask(mask)
    bin_mask = _fallback_if_empty(_clean_mask(raw_bin), raw_bin)

    if cv2.countNonZero(bin_mask) == 0:
        return _debug_safe_return(out, "empty mask")

    cnt = _largest_contour(bin_mask)
    shape = _classify_shape(cnt) if cnt is not None else "snake"
    base_step = float(max(6.0, spacing_px) if shape == "rect" else max(3.0, spacing_px))

    min_dim = min(bin_mask.shape[:2])
    spur_px = max(0, min(int(min_dim * 0.01), int(spur_min_px)))
    polylines: List[np.ndarray] = []
    if cnt is not None and shape == "boomerang":
        arms = _boomerang_arms_centerlines(bin_mask, cnt, spur_min_px=int(spur_px))
        if arms: polylines = arms
    if not polylines:
        P = _centerline_from_mask(bin_mask, spur_min_px=int(spur_px))
        if len(P) >= 2: polylines = [P]

    def attempt(mold_mask: np.ndarray, step_val: float) -> List[Tuple[int,int]]:
        adaptive_probe = int(max(max_probe, 2.0 * np.hypot(*mold_mask.shape)))
        all_path: List[Tuple[int,int]] = []
        if not polylines:
            return []
        for P in polylines:
            C = _sample_curve_adaptive(P, base_step=step_val, alpha=float(curvature_gain))
            if len(C) < 2:
                continue
            N = _normals_from_polyline(C, k=3)
            L_pts: List[Tuple[int,int]] = []; R_pts: List[Tuple[int,int]] = []
            for p, n in zip(C, N):
                if np.linalg.norm(n) < 1e-6:
                    continue
                p_fixed = _nudge_inside_mask(mold_mask, p)
                p_plus, p_minus, width = _intersect_normal_with_mask(mold_mask, p_fixed, n, max_dist=adaptive_probe)
                if width <= 1.0:
                    continue
                base_off = _entry_offset_px(entry_mm=entry_mm, px_per_mm=px_per_mm,
                                            width_px=0.0, outside_scale=0.0,
                                            outside_px=0.0, min_px=0.0)
                offset = base_off + float(border_push_px)
                p_plus  = p_plus  + n * offset
                p_minus = p_minus - n * offset
                L_pts.append((int(round(p_minus[0])), int(round(p_minus[1]))))
                R_pts.append((int(round(p_plus[0])),  int(round(p_plus[1]))))
            if len(L_pts) >= 2 and len(R_pts) >= 2:
                all_path.extend(_zigzag_from_lr(L_pts, R_pts))
        return all_path

    def enlarge(m: np.ndarray, gp: int) -> np.ndarray:
        ksz = max(1, int(gp) * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
        return cv2.dilate(m, kernel, iterations=1)

    mold_A = enlarge(bin_mask, grow_px)
    path = attempt(mold_A, base_step)
    if not path: path = attempt(mold_A, max(3.0, base_step * 0.66))
    if not path: path = attempt(mold_A, max(3.0, base_step * 0.5))
    if not path: path = attempt(bin_mask, base_step)
    if not path:
        mold_D = enlarge(bin_mask, grow_px * 2)
        path = attempt(mold_D, max(3.0, base_step * 0.66))

    if not path:
        path = _scanline_zigzag(bin_mask, step=int(max(6, spacing_px)))

    for a, b in zip(path[:-1], path[1:]):
        cv2.line(out, a, b, color_thread, int(thickness), cv2.LINE_AA)
    if draw_dots:
        for p in path:
            cv2.circle(out, p, 2, dot_color, -1, cv2.LINE_AA)

    if not path:
        return _debug_safe_return(out, "no border path (after all retries)")
    if debug:
        show_cnt = _largest_contour(mold_A)
        if show_cnt is not None:
            cv2.polylines(out, [show_cnt.reshape(-1,2)], True, (200,200,200), 1, cv2.LINE_AA)
    return out, path


# ============================================================
# PUBLIC: Continuous (Bézier–smoothed centerline)
# ============================================================

def draw_stitching_bezier(
    bgr: np.ndarray,
    mask: np.ndarray,
    spacing_px: float = 20.0,
    outside_px: float = 3.0,
    rect_min_step: float = 6.0,
    max_probe: int = 240,
    color_thread: Tuple[int,int,int] = (30, 200, 255),
    thickness: int = 2,
    debug: bool = False,
    curvature_gain: float = 14.0,
    outside_scale: float = 0.12,
    spur_min_px: int = 4,
    bezier_samples_per: int = 16,
    entry_mm: float = 4.0,
    px_per_mm: Optional[float] = None,
    **kwargs,
) -> Tuple[np.ndarray, List[Tuple[int,int]]]:
    """
    Bézier-smoothed continuous suture with robust retries:
      1) build centerline(s), smooth with quadratic Bézier chain
      2) sample adaptively, cast normals, clamp offset to ≤45% width
      3) retry with denser sampling / mask dilation if needed
      4) final fallback → classic continuous renderer
    """
    out = bgr.copy()
    raw_bin = _as_binary_mask(mask)
    bin_mask = _fallback_if_empty(_clean_mask(raw_bin), raw_bin)

    cnt = _largest_contour(bin_mask)
    if cnt is None or len(cnt) < 5:
        cnt = _largest_contour(raw_bin)
        if cnt is None or len(cnt) < 5:
            return _debug_safe_return(out, "no contour")

    shape = _classify_shape(cnt)
    min_dim = min(bin_mask.shape[:2])
    spur_px = max(0, min(int(min_dim * 0.01), int(spur_min_px)))

    # collect one or more centerlines (boomerang arms supported)
    polylines: List[np.ndarray] = []
    if shape == "boomerang":
        arms = _boomerang_arms_centerlines(bin_mask, cnt, spur_min_px=int(spur_px))
        if arms:
            polylines = arms
    if not polylines:
        P = _centerline_from_mask(bin_mask, spur_min_px=int(spur_px))
        if P.shape[0] >= 2:
            polylines = [P]

    if not polylines:
        # fallback to classic continuous early if no polyline at all
        return draw_running_suture_auto(
            bgr, mask, spacing_px, outside_px, rect_min_step, max_probe,
            color_thread, thickness, debug, curvature_gain, outside_scale, spur_min_px,
            entry_mm=entry_mm, px_per_mm=px_per_mm
        )

    def _attempt(m: np.ndarray, base_step: float, bez_samp: int, relax_offset: bool = False) -> List[Tuple[int,int]]:
        """Try to build a path on mask m with sampling params."""
        probe = int(max(max_probe, 2.0 * np.hypot(*m.shape)))
        L_pts: List[Tuple[int,int]] = []
        R_pts: List[Tuple[int,int]] = []

        for P in polylines:
            if P.shape[0] < 2:
                continue
            # Bézier smooth then adaptive sample
            S = _bezier_chain_from_polyline(P, samples_per_seg=int(max(6, bez_samp)))
            C = _sample_curve_adaptive(
                S,
                base_step=max(rect_min_step if shape == "rect" else 3.0, float(base_step)),
                alpha=float(curvature_gain)
            )
            if C.shape[0] < 2:
                continue

            N = _normals_from_polyline(C, k=3)
            for p, n in zip(C, N):
                if np.linalg.norm(n) < 1e-6:
                    continue
                p = _nudge_inside_mask(m, p)
                p_plus, p_minus, width = _intersect_normal_with_mask(m, p, n, max_dist=probe)
                # need a real chord
                if width <= 1.0:
                    continue

                # offset in px (mm->px minimum), then clamp to ≤45% of local width
                offset = _entry_offset_px(
                    entry_mm=entry_mm,
                    px_per_mm=px_per_mm,
                    width_px=(0.0 if relax_offset else width),  # in relax mode ignore width scaling
                    outside_scale=(0.0 if relax_offset else outside_scale),
                    outside_px=outside_px,
                    min_px=3.0
                )
                offset = float(min(offset, 0.45 * width))  # <- critical clamp

                eR = p_plus + n * offset
                eL = p_minus - n * offset
                L_pts.append((int(round(eL[0])), int(round(eL[1]))))
                R_pts.append((int(round(eR[0])), int(round(eR[1]))))

        if len(L_pts) < 2 or len(R_pts) < 2:
            return []

        # weave zig-zag
        path: List[Tuple[int,int]] = []
        mlen = min(len(L_pts), len(R_pts))
        for i in range(mlen):
            path.append(L_pts[i]); path.append(R_pts[i])
        path = _avoid_crossing(path)
        path = _ensure_progressive(path)
        return path

    base = float(spacing_px)
    tries: List[Tuple[np.ndarray, float, int, bool]] = []

    # 1) nominal
    tries.append((bin_mask, base, int(bezier_samples_per), False))
    # 2) denser along curve
    tries.append((bin_mask, max(3.0, base * 0.75), int(bezier_samples_per * 1.25), False))
    tries.append((bin_mask, max(3.0, base * 0.50), int(bezier_samples_per * 1.5), False))
    # 3) light dilations for thin masks
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    tries.append((cv2.dilate(bin_mask, k1, iterations=1), max(3.0, base * 0.66), int(bezier_samples_per * 1.25), False))
    tries.append((cv2.dilate(bin_mask, k2, iterations=1), max(3.0, base * 0.66), int(bezier_samples_per * 1.5), False))
    # 4) relax offset (ignore width scaling, just enforce mm min and clamp)
    tries.append((bin_mask, max(3.0, base * 0.66), int(bezier_samples_per * 1.5), True))

    path: List[Tuple[int,int]] = []
    for m, step, bsamp, relax in tries:
        path = _attempt(m, step, bsamp, relax_offset=relax)
        if path:
            break

    if not path:
        # last fallback → classic continuous
        return draw_running_suture_auto(
            bgr, mask, spacing_px, outside_px, rect_min_step, max_probe,
            color_thread, thickness, debug, curvature_gain, outside_scale, spur_min_px,
            entry_mm=entry_mm, px_per_mm=px_per_mm
        )

    # draw
    for a, b in zip(path[:-1], path[1:]):
        cv2.line(out, a, b, color_thread, int(thickness), cv2.LINE_AA)
    for pxy in path:
        cv2.circle(out, pxy, 2, color_thread, -1, cv2.LINE_AA)

    if debug:
        cv2.polylines(out, [cnt.reshape(-1,2)], True, (120,220,220), 1, cv2.LINE_AA)

    return out, path


# ============================================================
# Back-compat + UI aliases
# ============================================================

def draw_running_suture_centerline(*args, **kwargs):
    return draw_running_suture_auto(*args, **kwargs)

def draw_running_suture_contour_spline(*args, **kwargs):
    return draw_running_suture_auto(*args, **kwargs)

def draw_running_suture_spline(*args, **kwargs):
    return draw_running_suture_auto(*args, **kwargs)

def draw_stitching_perp(*args, **kwargs):
    return draw_stitching_pattern(*args, **kwargs)

def draw_stitching_continuous(*args, **kwargs):
    return draw_running_suture_auto(*args, **kwargs)

def draw_stitching_mold(*args, **kwargs):
    return draw_stitching_mold_border(*args, **kwargs)

def draw_stitching_bezier_mode(*args, **kwargs):
    return draw_stitching_bezier(*args, **kwargs)
