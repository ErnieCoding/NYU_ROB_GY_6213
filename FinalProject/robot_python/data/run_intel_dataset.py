"""Run EKF + LiDAR SLAM on the Intel Research Lab dataset.

Usage:
    python data/run_intel_dataset.py
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
from scipy.spatial import KDTree

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROBOT_PYTHON = os.path.dirname(_HERE)
_FINAL_PROJECT = os.path.dirname(_ROBOT_PYTHON)
_NYU_ROOT = os.path.dirname(_FINAL_PROJECT)   # needed for FinalProject.* imports

sys.path.insert(0, _NYU_ROOT)
sys.path.insert(0, _ROBOT_PYTHON)
sys.path.insert(0, _HERE)

from config.config import Config
from data_types import LidarScan, Pose2D, RelativeMotion, normalize_angle
from download_dataset import download_dataset
from frontend.ekf import EKFLocalizer
from frontend.motion_model import DifferentialDriveMotionModel
from intel_dataset_loader import load_intel_dataset

# Use existing LidarMatcher when open3d is available (Python ≤ 3.11).
# Fall back to the built-in numpy/scipy ICP for Python 3.13+.
try:
    from frontend.lidar_matching import LidarMatcher as _LidarMatcher
    _OPEN3D_AVAILABLE = True
except ImportError:
    _OPEN3D_AVAILABLE = False
    print("open3d not available — using built-in numpy/scipy ICP fallback")


# ---------------------------------------------------------------------------
# Fallback ICP (numpy/scipy only)
# ---------------------------------------------------------------------------

def _scan_to_points(scan: LidarScan,
                    min_r: float = 0.05, max_r: float = 12.0, step: int = 2) -> np.ndarray:
    r = np.array(scan.ranges)
    a = np.array(scan.angles)
    valid = (r > min_r) & (r < max_r)
    r, a = r[valid][::step], a[valid][::step]
    return np.column_stack([r * np.cos(a), r * np.sin(a)])


def _icp_fallback(prev_scan: LidarScan, curr_scan: LidarScan,
                  max_iter: int = 25, max_dist: float = 0.35) -> RelativeMotion | None:
    """Point-to-point ICP using numpy + scipy KDTree."""
    src = _scan_to_points(curr_scan)
    tgt = _scan_to_points(prev_scan)

    if len(src) < 10 or len(tgt) < 10:
        return None

    T = np.eye(3)
    mask = np.ones(len(src), dtype=bool)

    for _ in range(max_iter):
        src_t = (T[:2, :2] @ src.T).T + T[:2, 2]
        dists, idx = KDTree(tgt).query(src_t)
        mask = dists < max_dist
        if mask.sum() < 5:
            return None

        s = src_t[mask]
        t = tgt[idx[mask]]
        s_c, t_c = s - s.mean(0), t - t.mean(0)
        U, _, Vt = np.linalg.svd(s_c.T @ t_c)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T
        tr = t.mean(0) - R @ s.mean(0)

        T[:2, :2] = R @ T[:2, :2]
        T[:2, 2] = R @ T[:2, 2] + tr

        if np.linalg.norm(tr) < 1e-4:
            break

    quality = mask.sum() / len(src)
    if quality < 0.1:
        return None

    rmse = float(np.sqrt((dists[mask] ** 2).mean()))
    sigma = max(0.01, rmse)
    cov = [[sigma ** 2, 0, 0], [0, sigma ** 2, 0], [0, 0, (sigma * 0.5) ** 2]]

    return RelativeMotion(
        dx=float(T[0, 2]),
        dy=float(T[1, 2]),
        dtheta=float(math.atan2(T[1, 0], T[0, 0])),
        covariance=cov,
        quality=quality,
        source="icp_fallback",
    )


# ---------------------------------------------------------------------------
# EKF predict workaround
# ---------------------------------------------------------------------------

def _simulate_predict(ekf: EKFLocalizer, odom_motion: RelativeMotion) -> None:
    """
    Advance EKF state using odometry without calling the broken predict() method.

    predict() crashes because motion_model.RelativeMotion_change passes delta_distance=
    which is not a field in RelativeMotion. We replicate all three state updates:
      1. save current pose as _prev_pose (needed by correct_with_lidar)
      2. advance pose via propagate_pose (world-frame dx/dy addition)
      3. propagate covariance P = F @ P @ F^T + Q
         Without step 3, P collapses to zero after the first correction and
         K → 0 for all future frames, making ICP corrections no-ops.
    """
    prev_pose = ekf._state.pose
    ekf._prev_pose = prev_pose
    ekf._state.pose = prev_pose.propagate_pose(odom_motion)
    ekf.last_odom_motion = odom_motion

    delta_s = math.sqrt(odom_motion.dx ** 2 + odom_motion.dy ** 2)
    theta_mid = prev_pose.theta + 0.5 * odom_motion.dtheta

    F = np.array([
        [1.0, 0.0, -delta_s * math.sin(theta_mid)],
        [0.0, 1.0,  delta_s * math.cos(theta_mid)],
        [0.0, 0.0,  1.0],
    ])
    xy_var = max(1e-4, 0.02 * delta_s + 0.005 * abs(odom_motion.dtheta))
    th_var = max(1e-4, 0.01 * abs(odom_motion.dtheta) + 0.002 * delta_s)
    Q = np.diag([xy_var, xy_var, th_var])

    P = np.asarray(ekf._state.covariance)
    ekf._state.covariance = F @ P @ F.T + Q


# ---------------------------------------------------------------------------
# Map building helper
# ---------------------------------------------------------------------------

def _build_map_points(frames: list[dict], poses: list[Pose2D],
                      every_n: int = 10,
                      min_r: float = 0.05, max_r: float = 12.0) -> tuple[np.ndarray, np.ndarray]:
    """Project LiDAR scans into world frame using the given poses."""
    wx_all, wy_all = [], []
    for i in range(0, len(frames), every_n):
        scan = frames[i]["lidar"]
        pose = poses[i]
        r = np.array(scan.ranges)
        a = np.array(scan.angles)
        valid = (r > min_r) & (r < max_r)
        r, a = r[valid], a[valid]
        lx = r * np.cos(a)
        ly = r * np.sin(a)
        c, s = math.cos(pose.theta), math.sin(pose.theta)
        wx_all.append(c * lx - s * ly + pose.x)
        wy_all.append(s * lx + c * ly + pose.y)
    return np.concatenate(wx_all), np.concatenate(wy_all)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_intel_dataset() -> None:
    clf_path = download_dataset()
    frames = load_intel_dataset(clf_path)
    print(f"Loaded {len(frames)} frames")

    config = Config()
    motion_model = DifferentialDriveMotionModel(config.robot)
    ekf = EKFLocalizer(config.frontend, motion_model)

    if _OPEN3D_AVAILABLE:
        _matcher = _LidarMatcher(config.robot, config.frontend)
        def match_scans(p, c): return _matcher.match_scans(p, c)
    else:
        def match_scans(p, c): return _icp_fallback(p, c)

    estimated_poses: list[Pose2D] = []   # EKF SLAM output
    ground_truth_poses: list[Pose2D] = []  # directly from CLF embedded robot pose
    prev_scan = None

    for i, frame in enumerate(frames):
        # Ground truth: the robot_pose embedded directly in each FLASER line
        ground_truth_poses.append(frame["ground_truth"])

        # Step 1: EKF predict using odometry
        if i > 0:
            _simulate_predict(ekf, frame["odom_motion"])

        # Step 2: LiDAR scan matching + EKF correction
        if prev_scan is not None:
            lidar_motion = match_scans(prev_scan, frame["lidar"])
            if lidar_motion is not None:
                ekf.correct_with_lidar(lidar_motion)

        prev_scan = frame["lidar"]
        estimated_poses.append(ekf.get_state().pose)

        if i % 500 == 0:
            print(f"  frame {i}/{len(frames)}")

    _evaluate(estimated_poses, ground_truth_poses)
    _plot(frames, estimated_poses, ground_truth_poses)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _evaluate(estimated: list[Pose2D], ground_truth: list[Pose2D]) -> None:
    trans_errors = [
        math.sqrt((e.x - g.x) ** 2 + (e.y - g.y) ** 2)
        for e, g in zip(estimated, ground_truth)
    ]
    rot_errors = [
        abs(normalize_angle(e.theta - g.theta))
        for e, g in zip(estimated, ground_truth)
    ]
    n = len(trans_errors)
    trans_rmse = math.sqrt(sum(e ** 2 for e in trans_errors) / n)
    mean_rot_deg = math.degrees(sum(rot_errors) / n)
    max_trans = max(trans_errors)
    print(f"\n--- Evaluation ({n} frames) ---")
    print(f"Translation RMSE (EKF vs CLF ground truth) : {trans_rmse:.4f} m")
    print(f"Max translation error                      : {max_trans:.4f} m")
    print(f"Mean rotation error                        : {mean_rot_deg:.4f} deg")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot(frames: list[dict], estimated: list[Pose2D], ground_truth: list[Pose2D]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plots")
        return

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))

    gx = [p.x for p in ground_truth]
    gy = [p.y for p in ground_truth]
    ex = [p.x for p in estimated]
    ey = [p.y for p in estimated]

    # ── Left: trajectory comparison ──────────────────────────────────────────
    ax = axes[0]
    ax.plot(gx, gy, "g-", linewidth=1.5, label="Ground Truth (CLF robot pose)", zorder=2)
    ax.plot(ex, ey, "r--", linewidth=1.0, label="EKF SLAM estimate", zorder=3)
    ax.scatter([gx[0]], [gy[0]], c="green", s=100, zorder=5, label="Start")
    ax.scatter([gx[-1]], [gy[-1]], c="darkgreen", s=100, marker="*", zorder=5, label="End")
    ax.legend(fontsize=9)
    ax.set_title("Trajectory: Ground Truth vs EKF SLAM")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)

    # ── Right: 2D LiDAR map ───────────────────────────────────────────────────
    ax = axes[1]

    # Map from ground truth poses (green dots)
    wx_gt, wy_gt = _build_map_points(frames, ground_truth, every_n=5)
    ax.scatter(wx_gt, wy_gt, s=0.3, c="green", alpha=0.3, linewidths=0, label="Map (GT poses)")

    # Map from SLAM estimated poses (red dots) — overlaid to show difference
    wx_est, wy_est = _build_map_points(frames, estimated, every_n=5)
    ax.scatter(wx_est, wy_est, s=0.3, c="red", alpha=0.3, linewidths=0, label="Map (SLAM poses)")

    # Trajectory overlay
    ax.plot(gx, gy, "g-", linewidth=0.8, alpha=0.6)
    ax.plot(ex, ey, "r-", linewidth=0.8, alpha=0.6)

    ax.legend(fontsize=9, markerscale=10)
    ax.set_title("2D LiDAR Map")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.axis("equal")
    ax.grid(True, alpha=0.2)

    fig.suptitle("Intel Research Lab Dataset — EKF SLAM", fontsize=14)
    plt.tight_layout()

    out = os.path.join(_HERE, "trajectory.png")
    plt.savefig(out, dpi=150)
    print(f"Plot saved to {out}")
    plt.show()


if __name__ == "__main__":
    run_intel_dataset()
