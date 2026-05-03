"""Parse a CARMEN .clf file (ODOM / FLASER / TRUEPOS) into aligned SLAM frames."""

from __future__ import annotations

import math
import os
import sys

import numpy as np

_ROBOT_PYTHON = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_NYU_ROOT = os.path.dirname(os.path.dirname(_ROBOT_PYTHON))
sys.path.insert(0, _NYU_ROOT)
sys.path.insert(0, _ROBOT_PYTHON)

from data_types import LidarScan, Pose2D, RelativeMotion, normalize_angle


def _relative_motion(prev: Pose2D, curr: Pose2D) -> RelativeMotion:
    """World-frame delta between two absolute ODOM poses (matches propagate_pose convention)."""
    return RelativeMotion(
        dx=curr.x - prev.x,
        dy=curr.y - prev.y,
        dtheta=normalize_angle(curr.theta - prev.theta),
        source="odom",
    )


def load_intel_dataset(clf_path: str) -> list[dict]:
    """
    Parse a CARMEN .clf file and return one dict per laser scan.

    Each dict: {"lidar": LidarScan, "odom_motion": RelativeMotion, "ground_truth": Pose2D}

    odom_motion is the world-frame delta from the ODOM pose at the previous laser
    frame to the ODOM pose at the current laser frame (zero vector on the first frame).

    FLASER format (CARMEN v2):
      FLASER N r0..rN-1 robot_x robot_y robot_theta odom_x odom_y odom_theta ts host log_ts
    The embedded robot_pose is used as ground truth (best-known corrected pose from dataset).
    """
    frames: list[dict] = []
    curr_odom: Pose2D | None = None
    odom_at_prev_laser: Pose2D | None = None

    with open(clf_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            tag = parts[0]

            if tag == "ODOM":
                # ODOM x y theta tv rv accel timestamp hostname log_ts
                curr_odom = Pose2D(
                    x=float(parts[1]),
                    y=float(parts[2]),
                    theta=float(parts[3]),
                )

            elif tag == "FLASER":
                # FLASER N r0..rN-1 robot_x robot_y robot_theta odom_x odom_y odom_theta ts host log_ts
                n = int(parts[1])
                ranges = [float(parts[2 + i]) for i in range(n)]
                robot_x = float(parts[2 + n])
                robot_y = float(parts[2 + n + 1])
                robot_theta = float(parts[2 + n + 2])
                ts = float(parts[2 + n + 6])  # Unix timestamp before hostname
                angles = np.linspace(-math.pi / 2, math.pi / 2, n).tolist()

                lidar = LidarScan(ranges=ranges, angles=angles, timestamp=ts)
                gt = Pose2D(x=robot_x, y=robot_y, theta=robot_theta)
                odom_pose = curr_odom  # ODOM seen just before this FLASER

                if odom_pose is None:
                    continue  # skip if no ODOM received yet

                if odom_at_prev_laser is None:
                    odom_motion = RelativeMotion(dx=0.0, dy=0.0, dtheta=0.0, source="odom")
                else:
                    odom_motion = _relative_motion(odom_at_prev_laser, odom_pose)

                frames.append({
                    "lidar": lidar,
                    "odom_motion": odom_motion,
                    "ground_truth": gt,
                })
                odom_at_prev_laser = odom_pose

    return frames
