"""ArUco marker observation pipeline for asynchronous camera frames."""

from __future__ import annotations

import math
from typing import Any

import cv2
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
        if image is None:
            return []

        # ArUco detection works best on grayscale images.
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # Use a common 5x5 dictionary for the generated project tags.
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
        detector = cv2.aruco.ArucoDetector(aruco_dict)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is None:
            return []

        detections: list[dict[str, Any]] = []
        for index, marker_id in enumerate(ids.flatten()):
            detections.append(
                {
                    "id": int(marker_id),
                    "corners": corners[index],
                    "confidence": 1.0,
                }
            )

        return detections

    def estimate_marker_pose(self, detection: dict[str, Any]) -> dict[str, Any] | None:
        """Estimate marker pose in the camera frame from one detection."""
        corners = detection.get("corners")
        if corners is None:
            return None

        # Keep config compatibility with both scaffold and prompt names.
        marker_length = getattr(self.config, "marker_length_m", self.config.aruco_marker_size_m)
        dist_coeffs = getattr(self.config, "dist_coeffs", self.config.distortion_coefficients)

        camera_matrix = np.asarray(self.config.camera_matrix, dtype=float)
        dist_coeffs = np.asarray(dist_coeffs, dtype=float)

        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            [corners],
            marker_length,
            camera_matrix,
            dist_coeffs,
        )

        if rvecs is None or tvecs is None:
            return None

        return {
            "tvec": np.asarray(tvecs[0][0], dtype=float),
            "rvec": np.asarray(rvecs[0][0], dtype=float),
            "confidence": float(detection.get("confidence", 1.0)),
        }

    def convert_to_robot_observation(
        self,
        marker_pose: dict[str, Any] | None,
        marker_id: int,
        timestamp: float,
    ) -> LandmarkObservation | None:
        """Convert a camera-frame marker pose into a global robot pose measurement."""
        if marker_pose is None:
            return None

        tvec = marker_pose.get("tvec")
        if tvec is None:
            return None

        # Step 1: extract marker position in the camera frame.
        x_cam = float(tvec[0])
        y_cam = float(tvec[1])
        z_cam = float(tvec[2])
        _ = y_cam

        # Step 2: approximate camera-frame position as robot-frame 2D offset.
        dx_robot = z_cam
        dy_robot = -x_cam

        # Step 3: use only known global marker poses.
        marker_pose_world = self.config.marker_world_map.get(marker_id)
        if marker_pose_world is None:
            return None

        # Step 4: estimate robot pose in world frame.
        x_robot = marker_pose_world.x - dx_robot
        y_robot = marker_pose_world.y - dy_robot
        theta_robot = 0.0

        # Step 5: simple range-based covariance placeholder.
        range_m = math.hypot(dx_robot, dy_robot)
        sigma_xy = 0.05 + 0.1 * range_m
        sigma_theta = 0.1
        covariance = np.diag([sigma_xy, sigma_xy, sigma_theta])

        # Step 6: reject distant or low-confidence detections.
        confidence = float(marker_pose.get("confidence", 1.0))
        if range_m > 3.0 or confidence < 0.2:
            return None

        robot_pose_meas = Pose2D(x=x_robot, y=y_robot, theta=theta_robot)
        return LandmarkObservation(
            timestamp=timestamp,
            marker_id=marker_id,
            robot_pose_meas=robot_pose_meas,
            covariance=covariance,
            quality={
                "confidence": confidence,
                "range_m": range_m,
                "bearing_rad": math.atan2(dy_robot, dx_robot),
            },
        )
