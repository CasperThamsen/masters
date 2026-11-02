import numpy as np
import cv2


def draw_matches(cur, ref, matches, status):
    img = cv2.drawMatches(
        cur[0], cur[1],
        ref[0], ref[1],
        matches[:20], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    cv2.putText(img, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return img

def draw_tracking(cur, pts, status):
    """Current frame: all keypoints in blue, points tracked against the map in green."""
    img = cv2.cvtColor(cur[0], cv2.COLOR_GRAY2BGR)
    for k in cur[1]:
        cv2.circle(img, (int(k.pt[0]), int(k.pt[1])), 1, (255, 0, 0), -1)
    if pts is not None:
        for p in pts:
            cv2.circle(img, (int(p[0]), int(p[1])), 3, (0, 255, 0), -1)
    cv2.putText(img, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return img

def draw_trajectory(poses, points=None, size=600):
    """Top-down view (x right, z forward): camera path plus local map points in gray."""
    canvas = np.full((size, size, 3), 255, np.uint8)
    xz = np.array([[T[0, 3], T[2, 3]] for T in poses])

    # view bounds: the whole path, plus the bulk of the map points
    bounds = xz
    if points is not None and len(points) > 0:
        pts_xz = points[:, [0, 2]]
        bounds = np.vstack([xz, np.percentile(pts_xz, 10, axis=0), np.percentile(pts_xz, 90, axis=0)])
    lo, hi = bounds.min(axis=0), bounds.max(axis=0)
    scale = (size - 60) / max((hi - lo).max(), 1e-6)

    if points is not None and len(points) > 0:
        pp = (pts_xz - lo) * scale + 30
        pp[:, 1] = size - pp[:, 1]
        inside = (pp[:, 0] >= 0) & (pp[:, 0] < size) & (pp[:, 1] >= 0) & (pp[:, 1] < size)
        for x, y in pp[inside].astype(int):
            cv2.circle(canvas, (int(x), int(y)), 1, (150, 150, 150), -1)

    pix = ((xz - lo) * scale + 30).astype(int)
    pix[:, 1] = size - pix[:, 1]
    for a, b in zip(pix[:-1], pix[1:]):
        cv2.line(canvas, tuple(int(v) for v in a), tuple(int(v) for v in b), (200, 0, 0), 1)
    cv2.circle(canvas, tuple(int(v) for v in pix[0]), 5, (0, 160, 0), -1)
    cv2.circle(canvas, tuple(int(v) for v in pix[-1]), 5, (0, 0, 255), -1)
    return canvas
