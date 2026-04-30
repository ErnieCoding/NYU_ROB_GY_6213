"""ArUco marker observation pipeline for asynchronous camera frames."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from FinalProject.robot_python.config.config import CameraConfig
from FinalProject.robot_python.data_types import CameraFrame, LandmarkObservation, Pose2D


class LandmarkObserver:
    """Extract processed landmark observations from raw camera frames."""

    def __init__(self, config: CameraConfig) -> None:
        self.config = config

    def observe(self, camera_frame: CameraFrame) -> list[LandmarkObservation]:
        """Run ArUco detection and pose conversion for one camera frame."""
        observations: list[LandmarkObservation] = []
        detections = self.detect_markers(camera_frame.image)

        for detection in detections:
            marker_pose = self.estimate_marker_pose(detection)
            marker_id = int(detection["id"])
            observation = self.convert_to_robot_observation(marker_pose, marker_id, camera_frame.timestamp)
            if observation is not None:
                observations.append(observation)

        return observations

    def detect_markers(self, image: Any) -> list[dict[str, Any]]:
        """Detect ArUco markers in an image."""
        _ = image
        # TODO: Use cv2.aruco detection with the configured dictionary and
        # return marker ids, image corners, and detection confidence.
        return []

    def estimate_marker_pose(self, detection: dict[str, Any]) -> dict[str, Any] | None:
        """Estimate marker pose in the camera frame from one detection."""
        _ = detection
        # TODO: Use solvePnP with camera intrinsics, distortion coefficients,
        # and configured physical marker size.
        return None

    def convert_to_robot_observation(
        self,
        marker_pose: dict[str, Any] | None,
        marker_id: int,
        timestamp: float,
    ) -> LandmarkObservation | None:
        """Convert a camera-frame marker pose into a robot-relative observation."""
        if marker_pose is None:
            return None

        # TODO: Apply camera-to-base extrinsics and quality-filter observations.
        # If marker_world_map is populated, use it to convert marker-relative
        # camera measurements into a robot pose measurement in the map frame.
        x = float(marker_pose.get("x", 0.0))
        y = float(marker_pose.get("y", 0.0))
        robot_pose_meas = self.config.marker_world_map.get(marker_id, Pose2D(x=x, y=y, theta=0.0))
        covariance = np.asarray(marker_pose.get("covariance", np.eye(3)), dtype=float)
        return LandmarkObservation(
            timestamp=timestamp,
            marker_id=marker_id,
            robot_pose_meas=robot_pose_meas,
            covariance=covariance,
            quality={
                "confidence": float(marker_pose.get("confidence", 1.0)),
                "range_m": math.hypot(x, y),
                "bearing_rad": math.atan2(y, x),
            },
        )
