import cv2
import numpy as np
from typing import Dict, Tuple, List, Optional


# =============== Core utilities ===============

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

def _skeletonize_morph(bin_img: np.ndarray) -> np.ndarray:
    """Morphological skeleton, no extra deps."""
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

def _skeleton_longest_path(skel: np.ndarray) -> np.ndarray:
    """Longest 8-connected path from skeleton as an ordered (x,y) polyline."""
    ys, xs = np.where(skel > 0)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    H, W = skel.shape
    idx_map = -np.ones((H, W), dtype=np.int32)
    coords = np.stack([ys, xs], axis=1)
    for i, (y, x) in enumerate(coords):
        idx_map[y, x] = i

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

    start = deg1[0] if len(deg1) >= 1 else 0
    a, _, _ = bfs(start)
    b, dist, prev = bfs(a)

    path_idx = []
    cur = b
    while cur >= 0:
        path_idx.append(cur)
        cur = prev[cur]
    path_idx = path_idx[::-1]
    P = coords[path_idx][:, ::-1].astype(np.float32)  # (x,y)
    return P

def _binary_search_boundary(contour_xy: np.ndarray, a: np.ndarray, b: np.ndarray, iters: int = 14) -> np.ndarray:
    """Find boundary point between inside a and outside b."""
    lo = a.astype(np.float32).copy()
    hi = b.astype(np.float32).copy()
    for _ in range(iters):
        mid = (lo + hi) * 0.5
        res = cv2.pointPolygonTest(contour_xy, (float(mid[0]), float(mid[1])), False)
        if res > 0:
            lo = mid
        else:
            hi = mid
    return lo

def _intersect_normal_with_boundary(contour_xy: np.ndarray, p: np.ndarray, n: np.ndarray, max_dist: int = 240) -> Tuple[np.ndarray, np.ndarray, float]:
    """Shoot rays +/- n from p to boundary, refine by binary search. Returns (p_plus, p_minus, width)."""
    n = n / (np.linalg.norm(n) + 1e-6)
    step = 1.5
    # +n
    inside = p.copy(); q = p.copy()
    for d in np.arange(0, max_dist + step, step):
        q = p + n * d
        r = cv2.pointPolygonTest(contour_xy, (float(q[0]), float(q[1])), False)
        if r <= 0:
            q = _binary_search_boundary(contour_xy, inside, q)
            break
        inside = q
    p_plus = q.copy()
    # -n
    inside = p.copy(); q = p.copy()
    for d in np.arange(0, max_dist + step, step):
        q = p - n * d
        r = cv2.pointPolygonTest(contour_xy, (float(q[0]), float(q[1])), False)
        if r <= 0:
            q = _binary_search_boundary(contour_xy, inside, q)
            break
        inside = q
    p_minus = q.copy()
    width = float(np.linalg.norm(p_plus - p_minus))
    return p_plus, p_minus, width


# =============== Shape classification ===============

def _classify_shape(cnt: np.ndarray) -> str:
    """
    Heuristics:
      - LONG-RECT (slit): minAreaRect aspect >= 2.5 and fill ratio decent
      - BOOMERANG: aspect <= 1.6 (nearly square-ish)
      - SNAKE: otherwise (curvy)
    """
    rect = cv2.minAreaRect(cnt)
    (cx, cy), (w, h), angle = rect
    w = max(w, 1e-3); h = max(h, 1e-3)
    ar = max(w, h) / min(w, h)
    box = cv2.boxPoints(rect).astype(np.float32)
    rect_area = float(w * h)
    area = float(cv2.contourArea(cnt))
    fill = area / rect_area if rect_area > 1e-6 else 0.0

    if ar >= 2.5 and fill > 0.15:
        return "rect"
    if ar <= 1.6:
        return "boomerang"
    return "snake"


# =============== Continuous stitch generators ===============

def _sample_along_polyline(P: np.ndarray, step: float) -> np.ndarray:
    """Return points sampled along P with at least 'step' pixels between samples."""
    if P.shape[0] < 2 or step <= 0:
        return P.copy()
    s, L = _polyline_arclen(P)
    if L <= 1e-6:
        return P[:1].copy()
    t = [0.0]
    while t[-1] + step < L:
        t.append(t[-1] + step)
    t.append(L)
    t = np.array(t, dtype=np.float32)
    idx = np.searchsorted(s, t, side="right") - 1
    idx = np.clip(idx, 0, len(s) - 2)
    seg_len = (s[idx + 1] - s[idx]); seg_len[seg_len < 1e-9] = 1.0
    alpha = (t - s[idx]) / seg_len
    P0 = P[idx].astype(np.float32); P1 = P[idx + 1].astype(np.float32)
    return P0 + (P1 - P0) * alpha[:, None]

def _centerline_from_mask(mask: np.ndarray) -> np.ndarray:
    skel = _skeletonize_morph(mask)
    P = _skeleton_longest_path(skel)
    if P.shape[0] < 4:
        # fallback to contour axis ordering if skeleton too small
        cnt = _largest_contour(mask)
        if cnt is None:
            return P
        pts = cnt.reshape(-1, 2).astype(np.float32)
        mu = np.mean(pts, axis=0); cov = np.cov(pts.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        u = eigvecs[:, order][:, 0]
        proj = ((pts - mu) @ u.reshape(2, 1)).ravel()
        idx = np.argsort(proj)
        P = pts[idx].astype(np.float32)
    return P

def _normals_from_polyline(P: np.ndarray, k: int = 3) -> np.ndarray:
    """Approximate tangent using neighbors +/-k; return unit normals at each point."""
    N = len(P)
    normals = np.zeros_like(P, dtype=np.float32)
    for i in range(N):
        j0 = max(0, i - k)
        j1 = min(N - 1, i + k)
        v = P[j1] - P[j0]
        nrm = float(np.linalg.norm(v)) + 1e-6
        t = v / nrm
        n = np.array([-t[1], t[0]], dtype=np.float32)
        normals[i] = n
    return normals

def _outside_points(contour_xy: np.ndarray, centers: np.ndarray, normals: np.ndarray, max_probe: int, outside_px: float) -> Tuple[List[Tuple[int,int]], List[Tuple[int,int]], List[float]]:
    """For each (center, normal), compute boundary hits and push 'outside_px' outside the cut."""
    L_pts: List[Tuple[int,int]] = []
    R_pts: List[Tuple[int,int]] = []
    widths: List[float] = []
    for p, n in zip(centers, normals):
        p_plus, p_minus, width = _intersect_normal_with_boundary(contour_xy, p, n, max_probe=max_probe)
        if width <= 2.0:
            continue
        # outward points: +n side goes further +n; -n side goes further -n
        eR = p_plus + n * outside_px
        eL = p_minus - n * outside_px
        L_pts.append((int(round(eL[0])), int(round(eL[1]))))
        R_pts.append((int(round(eR[0])), int(round(eR[1]))))
        widths.append(width)
    return L_pts, R_pts, widths

def _zigzag_path(L_pts: List[Tuple[int,int]], R_pts: List[Tuple[int,int]]) -> List[Tuple[int,int]]:
    """
    Build continuous path: L0 -> R1 -> L2 -> R3 ... (diagonal running pattern).
    Requires at least 2 rows. If not enough points, falls back to simple chain.
    """
    m = min(len(L_pts), len(R_pts))
    if m < 2:
        # fallback: interleave same indices if possible
        path = []
        for i in range(m):
            path.append(L_pts[i])
            path.append(R_pts[i])
        return path

    path: List[Tuple[int,int]] = []
    # We use up to m-1 steps in the zig-zag
    for i in range(m - 1):
        if i % 2 == 0:
            # L_i -> R_{i+1}
            path.append(L_pts[i])
            path.append(R_pts[i + 1])
        else:
            # R_i -> L_{i+1}
            path.append(R_pts[i])
            path.append(L_pts[i + 1])
    return path

def draw_running_suture_auto(
    bgr: np.ndarray,
    mask: np.ndarray,
    spacing_px: float = 20.0,     # user "spacing" slider; we enforce mins below
    outside_px: float = 3.0,      # >= 3 px outside the cut
    rect_min_step: float = 6.0,   # rectangle vertical spacing >= 6 px
    max_probe: int = 240,
    color_thread: Tuple[int,int,int] = (30, 200, 255),
    thickness: int = 2,
    debug: bool = False,
) -> Tuple[np.ndarray, List[Tuple[int,int]]]:
    """
    Shape-aware, simple continuous stitch:
      - classify: rect / boomerang / snake
      - build centerline, sample with min spacing
      - compute +/- normals, intersect to boundary, push outside by 'outside_px'
      - continuous zig-zag path: L0->R1->L2->R3...
    """
    out = bgr.copy()
    bin_mask = (mask >= 128).astype(np.uint8) * 255
    cnt = _largest_contour(bin_mask)
    if cnt is None or len(cnt) < 5:
        return out, []

    shape = _classify_shape(cnt)
    # centerline (works for all)
    P = _centerline_from_mask(bin_mask)
    if P.shape[0] < 2:
        return out, []

    # sampling step constraints
    if shape == "rect":
        step = max(rect_min_step, float(spacing_px), 6.0)  # >= 6 px
    else:
        step = max(3.0, float(spacing_px))                 # >= 3 px

    # densify then sample by step
    P_dense = _resample_polyline(P, max(160, int(max(bin_mask.shape) * 2)))
    C = _sample_along_polyline(P_dense, step=float(step))
    if C.shape[0] < 2:
        C = P_dense[::max(1, int(step))]

    # normals + boundary hits
    N = _normals_from_polyline(P_dense, k=4)
    # generate normals for sampled points by nearest neighbor on dense path
    from bisect import bisect_left
    s_dense, _ = _polyline_arclen(P_dense)
    s_samp, _ = _polyline_arclen(C)
    # map each C[i] to nearest index in P_dense by cumulative length
    idx_dense = [min(len(s_dense)-1, bisect_left(s_dense, si)) for si in s_samp]
    N_samp = np.array([N[j] for j in idx_dense], dtype=np.float32)

    contour_xy = cnt.reshape(-1, 2).astype(np.float32)
    L_pts, R_pts, widths = _outside_points(contour_xy, C, N_samp, max_probe=max_probe, outside_px=float(outside_px))
    if len(L_pts) < 2 or len(R_pts) < 2:
        return out, []

    # build zig-zag path
    path = _zigzag_path(L_pts, R_pts)
    if len(path) < 2:
        return out, []

    # draw thread as a single continuous polyline
    for a, b in zip(path[:-1], path[1:]):
        cv2.line(out, a, b, color_thread, thickness, cv2.LINE_AA)
    # small markers
    for p in path:
        cv2.circle(out, p, 2, color_thread, -1, cv2.LINE_AA)

    if debug:
        # show centerline and normals
        for a, b in zip(P_dense[:-1], P_dense[1:]):
            cv2.line(out, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (40, 140, 40), 1, cv2.LINE_AA)
        for c, n in zip(C[::max(1, len(C)//40 + 1)], N_samp[::max(1, len(N_samp)//40 + 1)]):
            p1 = (int(round(c[0])), int(round(c[1])))
            p2 = (int(round(c[0] + n[0]*12)), int(round(c[1] + n[1]*12)))
            cv2.line(out, p1, p2, (200, 80, 80), 1, cv2.LINE_AA)

    return out, path


# -------- Legacy names kept for compatibility --------
def draw_stitching_pattern(
    bgr: np.ndarray,
    mask: np.ndarray,
    info: Dict,
    spacing: int = 25,
    length_scale: float = 2.0,
    color: Tuple[int, int, int] = (30, 220, 30),
    thickness: int = 2,
):
    # simple “staple-like” alt kept just in case
    out = bgr.copy()
    bin_mask = (mask >= 128).astype(np.uint8) * 255
    cnt = _largest_contour(bin_mask)
    if cnt is None or len(cnt) < 5:
        return out, []
    # principal axis approach
    pts = cnt.reshape(-1, 2).astype(np.float32)
    mu = np.mean(pts, axis=0)
    cov = np.cov(pts.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    u = eigvecs[:, order][:, 0]; v = eigvecs[:, order][:, 1]
    cx, cy = float(mu[0]), float(mu[1])

    ys, xs = np.where(bin_mask > 0)
    if xs.size == 0:
        return out, []
    rel = np.stack([xs.astype(np.float32) - cx, ys.astype(np.float32) - cy], axis=1)
    t = rel @ u.reshape(2, 1)
    tmin, tmax = float(np.min(t)), float(np.max(t))
    seg_half = max(6, int(length_scale * np.sqrt(max(1.0, float(np.min(eigvals))))))
    for ti in np.arange(tmin, tmax + 1e-3, max(6, int(spacing)), dtype=np.float32):
        px, py = cx + u[0]*ti, cy + u[1]*ti
        x1 = int(round(px - v[0]*seg_half)); y1 = int(round(py - v[1]*seg_half))
        x2 = int(round(px + v[0]*seg_half)); y2 = int(round(py + v[1]*seg_half))
        cv2.line(out, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    return out, []


# Backwards compatibility adapters (older imports in vision_web point here)
def draw_running_suture_centerline(*args, **kwargs):
    return draw_running_suture_auto(*args, **kwargs)

def draw_running_suture_contour_spline(*args, **kwargs):
    return draw_running_suture_auto(*args, **kwargs)

def draw_running_suture_spline(*args, **kwargs):
    return draw_running_suture_auto(*args, **kwargs)
