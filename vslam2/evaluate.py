import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# 1. Load poses and their KITTI frame IDs
with np.load('poses.npz') as pose_data:
    vposes = np.asarray(pose_data['T_wc'])
    pose_frames = np.asarray(pose_data['frame'], dtype=int)

gt_raw = np.loadtxt('data_odometry_gray/dataset/sequences/00/00.txt')

if vposes.ndim != 3 or vposes.shape[1:] != (4, 4):
    raise ValueError(f"Expected T_wc to have shape (N, 4, 4), got {vposes.shape}")
if len(vposes) != len(pose_frames):
    raise ValueError("T_wc and frame must contain the same number of poses")
if len(vposes) < 2:
    raise ValueError("At least two poses are required for evaluation")
if np.any(pose_frames < 0) or np.any(pose_frames >= len(gt_raw)):
    raise ValueError("poses.npz contains frame IDs outside the ground-truth trajectory")

# Extract (x, y, z) for the recorded frames.
gt_xyz = np.stack([gt_raw[pose_frames, 3], gt_raw[pose_frames, 7], gt_raw[pose_frames, 11]], axis=-1)
vo_xyz = vposes[:, :3, 3]

# 2. Handle Monocular Scale Alignment
gt_path_length = np.sum(np.linalg.norm(np.diff(gt_xyz, axis=0), axis=1))
vo_path_length = np.sum(np.linalg.norm(np.diff(vo_xyz, axis=0), axis=1))
scale_factor = gt_path_length / (vo_path_length + 1e-6)
print(f"Estimated Monocular Scale Factor: {scale_factor:.4f}")

vo_xyz_scaled = vo_xyz * scale_factor

# 3. CRITICAL: Align starting points (Zero out the origin)
# This forces both trajectories to start at (0, 0, 0)
gt_xyz = gt_xyz - gt_xyz[0]
vo_xyz_scaled = vo_xyz_scaled - vo_xyz_scaled[0]

# 4. Calculate RMSE on the aligned translation
rmse = np.sqrt(np.mean((vo_xyz_scaled - gt_xyz) ** 2))
print(f"Aligned Translation RMSE: {rmse:.4f} meters")

# Load the trajectory produced by the vslam2 implementation.
with np.load('vo_poses.npz') as pose_data:
    vo2_poses = np.asarray(pose_data['poses'])

if vo2_poses.ndim != 3 or vo2_poses.shape[1:] != (4, 4):
    raise ValueError(f"Expected vo_poses.npz poses to have shape (N, 4, 4), got {vo2_poses.shape}")
if len(vo2_poses) < 2:
    raise ValueError("At least two poses are required in vo_poses.npz")

vo2_xyz = vo2_poses[:, :3, 3]
vo2_count = min(len(vo2_xyz), len(gt_raw))
vo2_xyz = vo2_xyz[:vo2_count]
vo2_gt_xyz = np.stack(
    [gt_raw[:vo2_count, 3], gt_raw[:vo2_count, 7], gt_raw[:vo2_count, 11]],
    axis=-1,
)
vo2_scale = np.sum(np.linalg.norm(np.diff(vo2_gt_xyz, axis=0), axis=1))
vo2_scale /= np.sum(np.linalg.norm(np.diff(vo2_xyz, axis=0), axis=1)) + 1e-6
vo2_xyz = vo2_xyz * vo2_scale
vo2_xyz -= vo2_xyz[0]
vo2_gt_xyz -= vo2_gt_xyz[0]
print(f"vo_poses.npz Scale Factor: {vo2_scale:.4f}")

# 5. Plot all trajectories in the top-down x/z plane.
plt.figure(figsize=(10, 8))
plt.plot(gt_xyz[:, 0], gt_xyz[:, 2], label="Ground truth", linewidth=2)
poses_line, = plt.plot(vo_xyz_scaled[:, 0], vo_xyz_scaled[:, 2], label="poses.npz", linewidth=2)
vo_poses_line, = plt.plot(vo2_xyz[:, 0], vo2_xyz[:, 2], label="vo_poses.npz", linewidth=2)
plt.scatter(gt_xyz[0, 0], gt_xyz[0, 2], color="green", label="Start", zorder=3)
plt.scatter(gt_xyz[-1, 0], gt_xyz[-1, 2], color="red", label="End", zorder=3)
plt.xlabel("x (m)")
plt.ylabel("z (m)")
plt.title("Estimated and Ground-Truth Trajectories")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.subplots_adjust(bottom=0.2)

poses_button_axis = plt.axes((0.25, 0.05, 0.2, 0.075))
vo_poses_button_axis = plt.axes((0.55, 0.05, 0.2, 0.075))
poses_button = Button(poses_button_axis, "Hide poses.npz")
vo_poses_button = Button(vo_poses_button_axis, "Hide vo_poses.npz")


def toggle_line(line, button, label):
    visible = not line.get_visible()
    line.set_visible(visible)
    button.label.set_text(("Hide " if visible else "Show ") + label)
    plt.draw()


poses_button.on_clicked(lambda event: toggle_line(poses_line, poses_button, "poses.npz"))
vo_poses_button.on_clicked(lambda event: toggle_line(vo_poses_line, vo_poses_button, "vo_poses.npz"))

plt.savefig("trajectory_comparison.png", dpi=150)
plt.show()