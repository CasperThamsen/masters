import cv2
import numpy as np      
from .config import FRAME_STEP, WAIT_MS, kitti
from .data import load_source
from .tracking import feature_extraction, new_state, VO
from .visualize import draw_matches, draw_tracking, draw_trajectory


def main():
    camera_matrix, distortion, frame_stream = load_source()
    state = new_state(camera_matrix)
    poses = [state['T_wc'].copy()]
    frame_count = 0
    ids = [FRAME_STEP - 1]                           # new: KITTI frame index of each pose

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

        status, info = VO(state, cur)
        if info['pose_updated']:
            poses.append(state['T_wc'].copy())
            ids.append(frame_count - 1)

        print(f"Frame {frame_count}: {status}, matches={info['n_matches']}, "
              f"inliers={info['n_inliers']}, map={len(state['map']['pts'])}")

        if info['mode'] == 'init':
            cv2.imshow('ORB Features', draw_matches(cur, info['ref'], info['matches'], status))
        else:
            cv2.imshow('ORB Features', draw_tracking(cur, info['pts'], status))
        cv2.imshow('Trajectory', draw_trajectory(poses, state['map']['pts']))

        if cv2.waitKey(WAIT_MS) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    np.savez('poses.npz', T_wc=np.array(poses), frame=np.array(ids))   # new

if __name__ == "__main__":
    main()
