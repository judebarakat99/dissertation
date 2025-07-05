# suture_planner.py
import numpy as np

def generate_suture_points(contour, num_stitches=5, wound_width=20):
    """
    Given a wound contour, returns suture (entry, exit) point pairs.
    """
    if len(contour) < 2:
        return []

    # Find leftmost and rightmost points
    leftmost = tuple(contour[contour[:, :, 0].argmin()][0])
    rightmost = tuple(contour[contour[:, :, 0].argmax()][0])

    # Convert to float numpy arrays
    p1 = np.array(leftmost, dtype=np.float32)
    p2 = np.array(rightmost, dtype=np.float32)
    wound_vec = p2 - p1
    wound_length = np.linalg.norm(wound_vec)

    if wound_length == 0:
        return []

    wound_dir = wound_vec / wound_length
    normal = np.array([-wound_dir[1], wound_dir[0]])

    suture_points = []

    for i in range(num_stitches):
        t = (i + 1) / (num_stitches + 1)
        midpoint = p1 + wound_vec * t

        entry = (midpoint + normal * (wound_width / 2)).astype(int)
        exit = (midpoint - normal * (wound_width / 2)).astype(int)

        suture_points.append((tuple(entry), tuple(exit)))

    return suture_points
