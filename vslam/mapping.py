import numpy as np
import cv2
from .config import MAX_REPROJ_ERR, MIN_TRI_ANGLE, KF_BASELINE_RATIO, KF_INLIER_RATIO


def new_map():
    """Empty local map: 3D points (world), their ORB descriptors, and the keyframe id that created them."""
    return {'pts': np.empty((0, 3)), 'des': np.empty((0, 32), np.uint8), 'kf': np.empty(0, int)}

def map_add(local_map, pts3d, des, kf_id):
    local_map['pts'] = np.vstack([local_map['pts'], pts3d])
    local_map['des'] = np.vstack([local_map['des'], des])
    local_map['kf'] = np.concatenate([local_map['kf'], np.full(len(pts3d), kf_id)])

def map_prune(local_map, min_kf_id):
    """Drop points created by keyframes older than min_kf_id (sliding window)."""
    keep = local_map['kf'] >= min_kf_id
    for key in ('pts', 'des', 'kf'):
        local_map[key] = local_map[key][keep]

def triangulate_points(K, T_wc1, T_wc2, pts1, pts2):
    """Triangulate matched pixels from two known camera poses (camera-to-world).
    Returns (pts3d_world (N,3), valid (N,) bool). Valid = in front of both cameras,
    low reprojection error in both views, and enough parallax angle."""
    if len(pts1) == 0:
        return np.empty((0, 3)), np.zeros(0, bool)
    T_cw1, T_cw2 = np.linalg.inv(T_wc1), np.linalg.inv(T_wc2)
    P1, P2 = K @ T_cw1[:3], K @ T_cw2[:3]
    X_h = cv2.triangulatePoints(P1, P2, pts1.T.astype(np.float64), pts2.T.astype(np.float64))

    with np.errstate(divide='ignore', invalid='ignore'):
        X = (X_h[:3] / X_h[3]).T
        valid = np.isfinite(X).all(axis=1)
        for T_cw, pts in ((T_cw1, pts1), (T_cw2, pts2)):
            Xc = T_cw[:3, :3] @ X.T + T_cw[:3, 3:4]
            uv = (K @ Xc)[:2] / Xc[2]
            err = np.linalg.norm(uv.T - pts, axis=1)
            valid &= (Xc[2] > 0) & (err < MAX_REPROJ_ERR)
        v1, v2 = X - T_wc1[:3, 3], X - T_wc2[:3, 3]
        cos = np.sum(v1 * v2, axis=1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1))
        angle = np.degrees(np.arccos(np.clip(cos, -1, 1)))
        valid &= angle > MIN_TRI_ANGLE
    return X, valid

def median_depth(T_wc, pts3d):
    """Median depth of world points as seen from the camera at T_wc."""
    if len(pts3d) == 0:
        return 1.0
    T_cw = np.linalg.inv(T_wc)
    z = T_cw[2, :3] @ pts3d.T + T_cw[2, 3]
    return float(np.median(z))

def need_keyframe(state, T_wc, n_inliers, map_idx):
    """New keyframe when we moved far enough (relative to scene depth) or tracking is thinning out."""
    kf = state['keyframes'][-1]
    baseline = np.linalg.norm(T_wc[:3, 3] - kf['T_wc'][:3, 3])
    depth = max(median_depth(T_wc, state['map']['pts'][map_idx]), 1e-6)
    return baseline > KF_BASELINE_RATIO * depth or n_inliers < KF_INLIER_RATIO * state['kf_inliers']
