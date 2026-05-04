"""Shared data types and tiny pose helpers for the indoor SLAM scaffold."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def normalize_angle(angle: float) -> float:
    """Normalize an angle in radians to the interval [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class Pose2D:
    """Planar robot pose in an SE(2)-like representation."""

    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0

    # def moved_by(self, motion: RelativeMotion) -> Pose2D:
    #     """Apply a relative motion in the world frame."""
    #     cos_t = math.cos(self.theta)
    #     sin_t = math.sin(self.theta)
    #     world_dx = cos_t * motion.dx - sin_t * motion.dy
    #     world_dy = sin_t * motion.dx + cos_t * motion.dy
    #     return Pose2D(
    #         x=self.x + world_dx,
    #         y=self.y + world_dy,
    #         theta=normalize_angle(self.theta + motion.dtheta),
    #     )

    def propagate_pose(self, motion: RelativeMotion) -> Pose2D:
        """Update pose in robot-local frame (no world transform)."""
        return Pose2D(
            x=self.x + motion.dx,
            y=self.y + motion.dy,
            theta=normalize_angle(self.theta + motion.dtheta),
    )

    def distance_to(self, other: Pose2D) -> float:
        """Return Euclidean distance between pose translations."""
        return math.hypot(self.x - other.x, self.y - other.y)


@dataclass
class PoseEstimate:
    """Pose estimate with timestamp and covariance placeholder."""

    pose: Pose2D = field(default_factory=Pose2D)
    timestamp: float = 0.0
    covariance: list[list[float]] = field(
        default_factory=lambda: [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    source: str = "unknown"


@dataclass
class LidarScan:
    """Single 2D LiDAR scan."""

    ranges: list[float]
    angles: list[float]
    timestamp: float
    frame_id: str = "laser"
    intensities: list[float] | None = None


@dataclass
class EncoderState:
    """Wheel encoder state from the differential-drive base."""
    left_ticks: int
    right_ticks: int
    timestamp: float

@dataclass
class RobotFrame:
    """Synchronized robot-side packet decoded from the UDP stream."""

    timestamp: float
    encoder: EncoderState
    lidar_scan: LidarScan | None = None
    raw_packet: bytes | None = None


@dataclass
class CameraFrame:
    """Camera frame from the ESP32-CAM stream."""

    timestamp: float
    image: Any
    frame_id: str = "camera"


@dataclass
class RelativeMotion:
    """Relative planar motion between two robot poses."""

    dx: float = 0.0
    dy: float = 0.0
    dtheta: float = 0.0
    covariance: list[list[float]] | None = None
    quality: float | None = None
    source: str = "unknown"


@dataclass
class LandmarkObservation:
    """Processed landmark measurement consumed by the shared SLAM core."""

    timestamp: float
    marker_id: int
    robot_pose_meas: Pose2D
    covariance: np.ndarray
    quality: dict | None = None


@dataclass
class FrontendOutput:
    """Result of one local front-end update."""

    timestamp: float
    pose_estimate: PoseEstimate
    odom_motion: RelativeMotion | None = None
    lidar_motion: RelativeMotion | None = None
    created_keyframe: bool = False
    keyframe_id: int | None = None


@dataclass
class LocalMapState:
    """Front-end local map state built from corrected poses and recent scans."""

    trajectory: list[PoseEstimate] = field(default_factory=list)
    keyframe_scans: dict[int, LidarScan] = field(default_factory=dict)
    landmark_observations: list[LandmarkObservation] = field(default_factory=list)


@dataclass
class OptimizationResult:
    """Container for the latest back-end optimization output."""

    optimized_poses: dict[int, Pose2D]
    timestamp: float
    converged: bool = False
    summary: dict[str, Any] = field(default_factory=dict)
