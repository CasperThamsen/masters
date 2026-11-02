import numpy as np


def median_parallax(pts_ref, pts_cur):
    """Median pixel displacement between matched inliers."""
    return float(np.median(np.linalg.norm(pts_cur - pts_ref, axis=1)))
