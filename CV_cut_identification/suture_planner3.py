import cv2
import numpy as np
from scipy.interpolate import splprep, splev

def get_evenly_spaced_points(contour, num_points):
    # Convert contour shape (N,1,2) to (N,2)
    pts = contour[:, 0, :].astype(np.float32)
    # Fit a periodic spline through contour points with smoothing
    tck, u = splprep(pts.T, s=5.0, per=True)
    # Sample evenly spaced parametric points along spline
    u_new = np.linspace(0, 1, num_points, endpoint=False)
    x_new, y_new = splev(u_new, tck)
    return np.vstack([x_new, y_new]).T, tck, u_new

def estimate_local_width(contour, center_point, normal, max_dist=40):
    contour = contour.astype(np.float32)
    p1 = center_point + normal * max_dist
    p2 = center_point - normal * max_dist

    for d in np.linspace(0, max_dist, 100):
        pt_a = center_point + normal * d
        a = (int(round(pt_a[0])), int(round(pt_a[1])))
        if cv2.pointPolygonTest(contour, a, False) <= 0:
            p1 = center_point + normal * (d - 1)
            break

    for d in np.linspace(0, max_dist, 100):
        pt_b = center_point - normal * d
        b = (int(round(pt_b[0])), int(round(pt_b[1])))
        if cv2.pointPolygonTest(contour, b, False) <= 0:
            p2 = center_point - normal * (d - 1)
            break

    return np.linalg.norm(p1 - p2), p1, p2

def generate_suture_points(contour, num_stitches=10, max_width=100):
    contour = contour.astype(np.float32)
    points, tck, u_vals = get_evenly_spaced_points(contour, num_stitches)

    sutures = []
    for i, (pt, u) in enumerate(zip(points, u_vals)):
        # Calculate derivative of spline at u to get tangent vector
        dx, dy = splev(u, tck, der=1)
        tangent = np.array([dx, dy])
        tangent /= np.linalg.norm(tangent) + 1e-6
        # Normal vector is perpendicular to tangent
        normal = np.array([-tangent[1], tangent[0]])

        # Estimate local wound width and entry/exit points
        width, entry, exit = estimate_local_width(contour, pt, normal, max_dist=max_width)

        if width > 5:  # filter out very small widths/no valid width found
            sutures.append({
                "entry": entry,
                "exit": exit,
                "width": width,
                "center": pt,
                "normal": normal
            })

    return sutures

