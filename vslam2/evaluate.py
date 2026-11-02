import numpy as np

# 1. Load poses
vposes = np.array(np.load('poses.npy')) 
gt_raw = np.loadtxt('data_odometry_gray/dataset/sequences/00/00.txt')

# Extract (x, y, z)
gt_xyz = np.stack([gt_raw[:, 3], gt_raw[:, 7], gt_raw[:, 11]], axis=-1)
vo_xyz = vposes[:, :3, 3]

# Ensure same length
min_frames = min(len(gt_xyz), len(vo_xyz))
gt_xyz = gt_xyz[:min_frames]
vo_xyz = vo_xyz[:min_frames]

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