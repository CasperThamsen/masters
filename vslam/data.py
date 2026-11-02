import os
import numpy as np
import cv2
from .config import kitti, kitti_dir, video_path, calibration_file


def load_kitti_calib(path):
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('P0:'):
                values = np.array([float(x) for x in line.strip().split()[1:]])
                return values.reshape(3, 4)[:, :3] # Take top-left 3x3
    raise ValueError("P0 not found in calib.txt")


def load_source():
    """Returns (camera_matrix, distortion, frame_stream) for the data source set in the config."""
    if kitti:
        image_dir = os.path.join(kitti_dir, "image_0")
        camera_matrix = load_kitti_calib(os.path.join(kitti_dir, "calib.txt"))
        distortion = None  # KITTI is already rectified

        # Get sorted list of images
        image_filenames = sorted([x for x in os.listdir(image_dir) if x.endswith('.png')])

        def frame_stream():
            for img_name in image_filenames:
                img_path = os.path.join(image_dir, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) # Already grayscale
                yield img is not None, img

    else:
        calibration_data = np.load(calibration_file)
        camera_matrix = calibration_data['mtx']
        distortion = calibration_data['dist']
        cap = cv2.VideoCapture(video_path)

        def frame_stream():
            try:
                while cap.isOpened():
                    ret, img = cap.read()
                    if not ret:
                        break
                    # Convert custom video to grayscale if it's color
                    if len(img.shape) == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    yield ret, img
            finally:
                cap.release()

    return camera_matrix, distortion, frame_stream
