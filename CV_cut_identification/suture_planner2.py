# suture_planner.py
import numpy as np
import cv2
import json
import csv
from scipy.interpolate import splprep, splev
import pyrealsense2 as rs


def generate_suture_points(contour, depth_frame=None, intrinsics=None, num_stitches=5, pixel_to_mm=1.0):
    if len(contour) < 5:
        return []

    # Flatten and sort contour for spline
    contour = contour[:, 0, :]  # shape: (N, 2)

    # Fit a spline through the contour's centerline
    epsilon = 5.0  # curve approximation accuracy
    approx = cv2.approxPolyDP(contour, epsilon, True)
    if len(approx) < 5:
        approx = contour

    # Fit spline through the simplified contour
    try:
        tck, u = splprep(approx.T, s=5.0)
        u_fine = np.linspace(0, 1, num_stitches + 2)[1:-1]  # exclude ends
        spline_points = np.array(splev(u_fine, tck)).T  # shape (N, 2)
    except:
        return []

    suture_data = []

    for pt in spline_points:
        pt = pt.astype(np.float32)
        # Get local direction (tangent)
        delta = 1e-2
        try:
            tangent = np.array(splev(u_fine[0] + delta, tck)) - np.array(splev(u_fine[0] - delta, tck))
            tangent = tangent / np.linalg.norm(tangent)
        except:
            tangent = np.array([1, 0])
        normal = np.array([-tangent[1], tangent[0]])

        # Estimate wound width perpendicular to center
        w = estimate_local_width(contour, pt, normal)

        entry = (pt + normal * (w / 2)).astype(int)
        exit = (pt - normal * (w / 2)).astype(int)

        entry_3d, exit_3d = None, None
        spacing_mm = None

        if depth_frame is not None and intrinsics is not None:
            z1 = depth_frame.get_distance(entry[0], entry[1])
            z2 = depth_frame.get_distance(exit[0], exit[1])
            entry_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [int(entry[0]), int(entry[1])], z1)
            exit_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [int(exit[0]), int(exit[1])], z2)
            spacing_mm = np.linalg.norm(np.array(entry_3d) - np.array(exit_3d)) * 1000  # mm

        suture_data.append({
            "entry": tuple(entry),
            "exit": tuple(exit),
            "entry_3d": entry_3d,
            "exit_3d": exit_3d,
            "spacing_mm": spacing_mm
        })

    return suture_data

def estimate_local_width(contour, center_point, normal, max_dist=40):
    """
    Raycast in both directions along normal to find wound edges.
    contour: np.array shape (N,2) float or int
    center_point: np.array shape(2,)
    normal: np.array shape(2,)
    """
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

    return np.linalg.norm(p1 - p2)


def export_sutures_to_json(suture_data, filename="sutures.json"):
    with open(filename, "w") as f:
        json.dump(suture_data, f, indent=2)

def export_sutures_to_csv(suture_data, filename="sutures.csv"):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["entry_x", "entry_y", "exit_x", "exit_y", "spacing_mm"])
        for s in suture_data:
            e = s['entry']
            x = s['exit']
            writer.writerow([e[0], e[1], x[0], x[1], s['spacing_mm']])
