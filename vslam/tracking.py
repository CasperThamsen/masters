import numpy as np
import cv2
from . import utils
from .config import (MIN_MATCHES, MIN_INLIERS, MIN_PARALLAX, MAX_LOST, MIN_INIT_POINTS,
                     MIN_PNP_MATCHES, MIN_PNP_INLIERS, PNP_REPROJ_ERR, MAP_WINDOW, SCALE)
from .mapping import new_map, map_add, map_prune, triangulate_points, need_keyframe

orb = cv2.ORB_create(nfeatures=2000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING)


def feature_extraction(img):
    """Detect ORB keypoints and descriptors"""
    kp = orb.detect(img,None)
    kp, des = orb.compute(img, kp)
    return kp, des

def feature_matching(des1, des2, ratio=0.75):
    """Match ORB descriptors using Brute-Force Matcher"""
    if des1 is None or des2 is None:
        return []

    knn = bf.knnMatch(des1, des2, k=2)
    good_matches = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good_matches.append(m)
    return sorted(good_matches, key=lambda x: x.distance)

def estimate_pose(kp_ref, kp_cur, matches, K):
    """Essential matrix + RANSAC + cheirality check.
    Returns (R, t, inlier_matches, pts_ref_inliers, pts_cur_inliers) or None."""
    pts_ref = np.float32([kp_ref[m.trainIdx].pt for m in matches])
    pts_cur = np.float32([kp_cur[m.queryIdx].pt for m in matches])

    E, mask = cv2.findEssentialMat(
        pts_ref, pts_cur, K, method=cv2.RANSAC, prob=0.999, threshold=1.0
    )
    if E is None or mask is None:
        return None
    E = E[:3]

    n_pose, R, t, mask_pose = cv2.recoverPose(E, pts_ref, pts_cur, K, mask=mask.copy())
    keep = mask_pose.ravel() > 0
    if n_pose == 0 or keep.sum() == 0:
        return None
    inlier_matches = [m for m, k in zip(matches, keep) if k]
    return R, t.ravel(), inlier_matches, pts_ref[keep], pts_cur[keep]

def update_pose(T_wc, R, t):
    """Chain the relative motion (ref -> cur) onto the camera-to-world pose."""
    T_cur_ref = np.eye(4)
    T_cur_ref[:3, :3] = R
    T_cur_ref[:3, 3] = t #maybe add scale factor here ??
    return T_wc @ np.linalg.inv(T_cur_ref)

def initialize_map(ref, cur, T_wc_ref, K):
    """Two-view initialisation: relative pose from the essential matrix, then triangulate.
    The baseline is fixed to unit length, which defines the scale of the whole map.
    Returns (status, T_wc_cur, pts3d, des, matches, n_inliers)."""
    _, kp_ref, des_ref = ref
    _, kp_cur, des_cur = cur

    matches = feature_matching(des_cur, des_ref)
    if len(matches) < MIN_MATCHES:
        return "FAIL few matches", None, None, None, matches, 0

    result = estimate_pose(kp_ref, kp_cur, matches, K)
    if result is None:
        return "FAIL no pose", None, None, None, matches, 0

    R, t, inl_matches, pts_ref, pts_cur = result
    n_inliers = len(inl_matches)
    if n_inliers < MIN_INLIERS:
        return "FAIL few inliers", None, None, None, matches, n_inliers

    if utils.median_parallax(pts_ref, pts_cur) < MIN_PARALLAX:
        return "SKIP low parallax", None, None, None, matches, n_inliers

    T_wc_cur = update_pose(T_wc_ref, R, t)
    pts3d, valid = triangulate_points(K, T_wc_ref, T_wc_cur, pts_ref, pts_cur)
    if valid.sum() < MIN_INIT_POINTS:
        return "FAIL few map points", None, None, None, matches, n_inliers

    idx = np.array([m.queryIdx for m in inl_matches], dtype=int)
    return "OK", T_wc_cur, pts3d[valid], des_cur[idx][valid], matches, n_inliers

def track_map(cur, local_map, K):
    """Match the current frame against the local map and estimate the pose with PnP-RANSAC.
    Returns (status, T_wc, cur_idx, map_idx, n_matches); T_wc is None on failure.
    cur_idx / map_idx are the PnP inliers (indices into the keypoints / the map)."""
    _, kp_cur, des_cur = cur
    if len(local_map['pts']) < 2:
        return "FAIL empty map", None, None, None, 0

    matches = feature_matching(des_cur, local_map['des'])
    n_matches = len(matches)
    if n_matches < MIN_PNP_MATCHES:
        return "FAIL few map matches", None, None, None, n_matches

    cur_idx = np.array([m.queryIdx for m in matches], dtype=int)
    map_idx = np.array([m.trainIdx for m in matches], dtype=int)
    obj = local_map['pts'][map_idx].astype(np.float64)
    img = np.float64([kp_cur[i].pt for i in cur_idx])

    ok, rvec, tvec, inl = cv2.solvePnPRansac(
        obj, img, K, None, iterationsCount=200,
        reprojectionError=PNP_REPROJ_ERR, confidence=0.999, flags=cv2.SOLVEPNP_EPNP
    )
    if not ok or inl is None or len(inl) < MIN_PNP_INLIERS:
        return "FAIL PnP", None, None, None, n_matches

    inl = inl.ravel()
    rvec, tvec = cv2.solvePnPRefineLM(obj[inl], img[inl], K, None, rvec, tvec)

    R, _ = cv2.Rodrigues(rvec)         # PnP gives world -> camera
    T_cw = np.eye(4)
    T_cw[:3, :3] = R
    T_cw[:3, 3] = tvec.ravel()
    return "OK", np.linalg.inv(T_cw), cur_idx[inl], map_idx[inl], n_matches

def insert_keyframe(state, cur, T_wc, tracked_cur_idx):
    """Triangulate new points between the last keyframe and the current frame (skipping keypoints
    that are already tracked), store the frame as a keyframe, and slide the map window.
    Returns the number of new map points."""
    img, kp_cur, des_cur = cur
    kf = state['keyframes'][-1]

    tracked = set(tracked_cur_idx.tolist())
    matches = [m for m in feature_matching(des_cur, kf['des']) if m.queryIdx not in tracked]
    pts_kf = np.float32([kf['kp'][m.trainIdx].pt for m in matches]).reshape(-1, 2)
    pts_cur = np.float32([kp_cur[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    pts3d, valid = triangulate_points(state['K'], kf['T_wc'], T_wc, pts_kf, pts_cur)

    idx = np.array([m.queryIdx for m in matches], dtype=int)
    state['kf_count'] += 1
    kf_id = state['kf_count']
    map_add(state['map'], pts3d[valid], des_cur[idx][valid], kf_id)

    state['keyframes'].append({'id': kf_id, 'img': img, 'kp': kp_cur, 'des': des_cur, 'T_wc': T_wc.copy()})
    state['keyframes'] = state['keyframes'][-MAP_WINDOW:]
    map_prune(state['map'], kf_id - MAP_WINDOW + 1)
    return int(valid.sum())


def new_state(K):
    return {
        'K': K,                # camera matrix
        'T_wc': np.eye(4),     # current camera-to-world pose
        'initialized': False,  # False: looking for a good initial pair, True: tracking the map
        'ref': None,           # candidate reference frame for initialisation: (img, kp, des)
        'keyframes': [],       # last MAP_WINDOW keyframes
        'map': new_map(),
        'kf_count': 0,
        'kf_inliers': 0,       # tracked inliers at the last keyframe
        'lost': 0,             # consecutive failures
    }

def reset_state(state, cur):
    """Throw away the map and start looking for a new initial pair. The last good pose is kept,
    so the trajectory continues from there (with a new, unrelated scale)."""
    state['initialized'] = False
    state['ref'] = cur
    state['keyframes'] = []
    state['map'] = new_map()
    state['kf_inliers'] = 0
    state['lost'] = 0

def make_info(mode, n_matches=0, n_inliers=0, matches=None, ref=None, pts=None, pose_updated=False):
    return {'mode': mode, 'n_matches': n_matches, 'n_inliers': n_inliers, 'matches': matches or [],
            'ref': ref, 'pts': pts, 'pose_updated': pose_updated}

def VO_init(state, cur):
    """No map yet: wait for a reference frame, then for a second frame with enough parallax."""
    if state['ref'] is None:
        state['ref'] = cur
        return "WAIT first frame", make_info('init', ref=cur)

    ref = state['ref']
    status, T_wc_cur, pts3d, des, matches, n_inliers = initialize_map(ref, cur, state['T_wc'], state['K'])
    info = make_info('init', len(matches), n_inliers, matches, ref)

    if status == "OK":
        state['keyframes'] = [
            {'id': 0, 'img': ref[0], 'kp': ref[1], 'des': ref[2], 'T_wc': state['T_wc'].copy()},
            {'id': 1, 'img': cur[0], 'kp': cur[1], 'des': cur[2], 'T_wc': T_wc_cur.copy()},
        ]
        state['kf_count'] = 1
        state['map'] = new_map()
        map_add(state['map'], pts3d, des, 1)
        state['T_wc'] = T_wc_cur
        state['kf_inliers'] = len(pts3d)
        state['initialized'] = True
        state['ref'] = None
        state['lost'] = 0
        info['pose_updated'] = True
        return f"INIT OK (map={len(pts3d)} pts)", info

    if status.startswith("FAIL"):
        state['lost'] += 1
        if state['lost'] > MAX_LOST:
            state['ref'] = cur       # keep the old ref too long and it never overlaps again
            state['lost'] = 0
            status += " -> re-anchored"
    else:                            # SKIP low parallax: keep ref so the baseline grows
        state['lost'] = 0
    return status, info

def VO_track(state, cur):
    """Map exists: PnP against the local map, insert a keyframe when needed."""
    status, T_wc, cur_idx, map_idx, n_matches = track_map(cur, state['map'], state['K'])

    if T_wc is None:
        state['lost'] += 1
        if state['lost'] > MAX_LOST:
            reset_state(state, cur)
            status += " -> re-initialising"
        return status, make_info('track', n_matches)

    state['lost'] = 0
    state['T_wc'] = T_wc
    n_inliers = len(cur_idx)
    pts = np.float32([cur[1][i].pt for i in cur_idx])
    status = "TRACK OK"

    if need_keyframe(state, T_wc, n_inliers, map_idx):
        n_new = insert_keyframe(state, cur, T_wc, cur_idx)
        state['kf_inliers'] = n_inliers
        status += f" + KF (+{n_new} pts)"

    return status, make_info('track', n_matches, n_inliers, pts=pts, pose_updated=True)

def VO(state, cur):
    """One visual odometry step. Updates state in place, returns (status, info)."""
    if not state['initialized']:
        return VO_init(state, cur)
    return VO_track(state, cur)
