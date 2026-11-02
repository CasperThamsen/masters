import numpy as np
import cv2
import time as time
import utils as utils
import os

# ---------------------------------------------------- PARAMETERS ----------------------------------------------------

# VO parameters
FRAME_STEP = 1         # only use every n'th frame
MIN_MATCHES = 40       # matches after ratio test needed to even try a pose
MIN_INLIERS = 30      # RANSAC/cheirality inliers needed to accept a pose
MIN_PARALLAX = 5.0   # median pixel motion needed before trusting a pose
MAX_LOST = 10         # consecutive failures before re-anchoring the reference
WAIT_MS = 1          # 1 = play, 0 = step frame by frame

#PARAMS
IMSHOW = False        # Set to False to disable visualization
TRAJSHOW = True      # Set to True to visualize trajectory in a separate window

# Set to True if using KITTI dataset, False if using custom video
kitti = True
kitti_scale = 0.7897 #estimated monocular scale factor for KITTI sequence 00 (from evaluate.py)

# --- SETUP DATA SOURCE & CALIBRATION ---
if kitti:
    base_path = "data_odometry_gray/dataset/sequences/00"
    image_dir = os.path.join(base_path, "image_0")
    calib_path = os.path.join(base_path, "calib.txt")
    
    def load_kitti_calib(path):
        with open(path, 'r') as f:
            for line in f:
                if line.startswith('P0:'):
                    values = np.array([float(x) for x in line.strip().split()[1:]])
                    return values.reshape(3, 4)[:, :3] # Take top-left 3x3
        raise ValueError("P0 not found in calib.txt")

    camera_matrix = load_kitti_calib(calib_path)
    distortion = None  # KITTI is already rectified
    
    # Get sorted list of images
    image_filenames = sorted([x for x in os.listdir(image_dir) if x.endswith('.png')])

    def frame_stream():
        for img_name in image_filenames:
            img_path = os.path.join(image_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) # Already grayscale
            yield img is not None, img

else:
    calibration_data = np.load('phone_calibration.npz')
    camera_matrix = calibration_data['mtx']
    distortion = calibration_data['dist']
    cap = cv2.VideoCapture("vidtest_1.mp4")

    def frame_stream():
        while cap.isOpened():
            ret, img = cap.read()
            if not ret:
                break
            # Convert custom video to grayscale if it's color
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            yield ret, img


# --- INITIALIZE PARAMETERS ---
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
    """Essential matrix + RANSAC + cheirality check."""
    pts_ref = np.float32([kp_ref[m.trainIdx].pt for m in matches])
    pts_cur = np.float32([kp_cur[m.queryIdx].pt for m in matches])
 
    E, mask = cv2.findEssentialMat(
        pts_ref, pts_cur, K, method=cv2.RANSAC, prob=0.999, threshold=1.0
    )
    if E is None or mask is None:
        return None
 
    n_pose, R, t, mask_pose = cv2.recoverPose(E, pts_ref, pts_cur, K, mask=mask.copy())
    keep = mask_pose.ravel() > 0
    if n_pose == 0 or keep.sum() == 0:
        return None
    return R, t.ravel(), pts_ref[keep], pts_cur[keep]

def update_pose(T_wc, R, t):
    """Chain the relative motion (ref -> cur) onto the camera-to-world pose."""
    T_cur_ref = np.eye(4)
    T_cur_ref[:3, :3] = R
    if kitti: 
        T_cur_ref[:3, 3] = t * kitti_scale
    else:
        T_cur_ref[:3, 3] = t
    return T_wc @ np.linalg.inv(T_cur_ref)

def VO(ref, cur, T_wc):
    """One visual odometry step."""
    _, kp_ref, des_ref = ref
    _, kp_cur, des_cur = cur
 
    matches = feature_matching(des_cur, des_ref)
    if len(matches) < MIN_MATCHES:
        return "FAIL few matches", T_wc, matches, 0
 
    result = estimate_pose(kp_ref, kp_cur, matches, camera_matrix)
    if result is None:
        return "FAIL no pose", T_wc, matches, 0
 
    R, t, pts_ref, pts_cur = result
    n_inliers = len(pts_cur)
    if n_inliers < MIN_INLIERS:
        return "FAIL few inliers", T_wc, matches, n_inliers
 
    if utils.median_parallax(pts_ref, pts_cur) < MIN_PARALLAX:
        return "SKIP low parallax", T_wc, matches, n_inliers
 
    return "OK", update_pose(T_wc, R, t), matches, n_inliers

def draw_matches(cur, ref, matches, status):
    img = cv2.drawMatches(
        cur[0], cur[1],  
        ref[0], ref[1],  
        matches[:20], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    cv2.putText(img, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return img

def draw_trajectory(poses, size=600):
    canvas = np.full((size, size, 3), 255, np.uint8)
    xz = np.array([[T[0, 3], T[2, 3]] for T in poses])
    lo, hi = xz.min(axis=0), xz.max(axis=0)
    scale = (size - 60) / max((hi - lo).max(), 1e-6)
    pix = ((xz - lo) * scale + 30).astype(int)
    pix[:, 1] = size - pix[:, 1]  
    for a, b in zip(pix[:-1], pix[1:]):
        cv2.line(canvas, tuple(int(v) for v in a), tuple(int(v) for v in b), (200, 0, 0), 1)
    cv2.circle(canvas, tuple(int(v) for v in pix[0]), 5, (0, 160, 0), -1)     
    cv2.circle(canvas, tuple(int(v) for v in pix[-1]), 5, (0, 0, 255), -1)   
    return canvas

def main():
    ref = None                
    T_wc = np.eye(4)             
    poses = [T_wc.copy()]
    lost = 0
    frame_count = 0

    # Universal loop using the frame stream generator
    for ret, img in frame_stream():
        if not ret or img is None:
            break
            
        frame_count += 1
        if frame_count % FRAME_STEP != 0:
            continue

        # Undistort only if using custom video (KITTI is already rectified)
        if not kitti and distortion is not None:
            img = cv2.undistort(img, camera_matrix, distortion)

        kp, des = feature_extraction(img)
        cur = (img, kp, des)

        if ref is None:
            ref = cur
            continue

        status, T_wc, matches, n_inliers = VO(ref, cur, T_wc)
        print(f"Frame {frame_count}: {status}, matches={len(matches)}, inliers={n_inliers}")

        if IMSHOW == True:
            cv2.imshow('ORB Features', draw_matches(cur, ref, matches, status))
        if TRAJSHOW == True:
            cv2.imshow('Trajectory', draw_trajectory(poses))

        if status == "OK":
            poses.append(T_wc.copy())
            ref = cur
            lost = 0
        elif status.startswith("FAIL"):
            lost += 1
            if lost > MAX_LOST:
                ref = cur
                lost = 0
                status += "re-anchored"
        else:
            lost = 0
        
        if cv2.waitKey(WAIT_MS) & 0xFF == ord('q'):
            break

    if not kitti:
        cap.release()
    cv2.destroyAllWindows()
    np.savez('vo_poses.npz', poses=np.array(poses))

if __name__ == "__main__":
    main()