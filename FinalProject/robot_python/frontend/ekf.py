"""EKF localizer using wheel odometry and LiDAR scan matching updates."""

from __future__ import annotations

import math
import time
import numpy as np

from FinalProject.robot_python.config.config import FrontendConfig
from FinalProject.robot_python.data_types import EncoderState, Pose2D, PoseEstimate, RelativeMotion
from FinalProject.robot_python.frontend.motion_model import DifferentialDriveMotionModel
from FinalProject.robot_python.data_types import normalize_angle

from FinalProject.robot_python import parameters

class EKFLocalizer:
    """Maintain a local pose estimate with EKF predict/correct steps."""

    def __init__(self, config: FrontendConfig, motion_model: DifferentialDriveMotionModel) -> None:
        self.config = config
        self.motion_model = motion_model
        self.last_odom_motion: RelativeMotion | None = None
        

        self._state = PoseEstimate(
            pose=Pose2D(),
            timestamp=0.0,
            covariance=self._make_initial_covariance(),
            source="ekf_initial",
        )
        self._prev_pose: Pose2D | None = None

    def predict(self, prev_encoder: EncoderState, curr_encoder: EncoderState) -> PoseEstimate:
        """Run the EKF prediction step using wheel odometry."""

        # 1. Get motion increment
        motion,delta_s = self.motion_model.RelativeMotion_change(prev_encoder, curr_encoder)
        self.last_odom_motion = motion

        # 2. Update pose:
        prev_pose = self._state.pose
        self._prev_pose = prev_pose
        new_pose = prev_pose.propagate_pose(motion)

        # 3. Compute Jacobian F with respect to previous pose
        theta = prev_pose.theta
        theta_m = theta + 0.5 * motion.dtheta

        F = np.array([
            [1.0, 0.0, -delta_s * math.sin(theta_m)],
            [0.0, 1.0,  delta_s * math.cos(theta_m)],
            [0.0, 0.0,  1.0],
        ])

        G = np.array([
            [0.5 * np.cos(theta_m) + (delta_s / (2*parameters.wheel_base)) * np.sin(theta_m),
            0.5 * np.cos(theta_m) - (delta_s / (2*parameters.wheel_base)) * np.sin(theta_m),
        ],
        [
            0.5 * np.sin(theta_m) - (delta_s / (2 * parameters.wheel_base)) * np.cos(theta_m),
            0.5 * np.sin(theta_m) + (delta_s / (2 * parameters.wheel_base)) * np.cos(theta_m),
        ],
        [
            -1.0 / parameters.wheel_base,
            1.0 / parameters.wheel_base,
        ]
        ])

        # Compute jacobian G with resepct to previous pose

        # 4. Covariance propagation
        P_prev = self._state.covariance
        Q = Q = G @ motion.covariance @ G.T

        P_new = F @ P_prev @ F.T + Q

        # 5. Update state
        self._state.pose = new_pose
        self._state.covariance = P_new
        self._state.timestamp = time.time()
        self._state.source = "ekf_predict"

        return self.get_state()

    def correct_with_lidar(self, lidar_measurement: RelativeMotion | PoseEstimate) -> PoseEstimate:
        """Run the EKF correction step from LiDAR scan matching."""
       

        if not isinstance(lidar_measurement, RelativeMotion):
            self._state.source = "ekf_lidar_placeholder"
            return self.get_state()

        if self.last_odom_motion is None:
            self._state.source = "ekf_lidar_no_previous_pose"
            return self.get_state()

        predicted_pose = self._state.pose
        prev_pose = self._prev_pose

        dx = lidar_measurement.dx
        dy = lidar_measurement.dy
        dtheta = lidar_measurement.dtheta

        z_x = prev_pose.x + math.cos(prev_pose.theta) * dx - math.sin(prev_pose.theta) * dy
        z_y = prev_pose.y + math.sin(prev_pose.theta) * dx + math.cos(prev_pose.theta) * dy
        z_theta = normalize_angle(prev_pose.theta + dtheta)
        z = np.array([z_x, z_y, z_theta])

        x = np.array([predicted_pose.x, predicted_pose.y, predicted_pose.theta])
        Residual = z - x
        Residual[2] = normalize_angle(Residual[2])

        P = np.asarray(self._state.covariance)
        R = np.asarray(lidar_measurement.covariance if lidar_measurement.covariance is not None else self._zero_covariance())

        H = np.eye(3)

        S = H @ P @ H.T + R
        K = P @ np.linalg.inv(S)

        x_new = x + K @ Residual
        P_new = (np.eye(3) - K @ H) @ P

        self._state.pose = Pose2D(
            float(x_new[0]),
            float(x_new[1]),
            normalize_angle(float(x_new[2])),
        )
        self._state.covariance = P_new
        self._state.source = "ekf_lidar"
        
        return self.get_state()

    def get_state(self) -> PoseEstimate:
        """Return the current pose estimate."""
        return PoseEstimate(
            pose=Pose2D(
                self._state.pose.x,
                self._state.pose.y,
                self._state.pose.theta,
            ),
            timestamp=self._state.timestamp,
            covariance=self._state.covariance.copy(),
            source=self._state.source,
        )

    def reset(self, initial_pose: Pose2D | None = None) -> None:
        """Reset the filter."""
        self._state = PoseEstimate(
            pose=initial_pose or Pose2D(),
            timestamp=time.time(),
            covariance=self._make_initial_covariance(),
            source="ekf_reset",
        )

    def _make_initial_covariance(self) -> np.ndarray:
        """Create diagonal initial covariance."""
        value = self.config.initial_covariance
        return np.diag([value, value, value])

    def _zero_covariance(self) -> np.ndarray:
        """Create zero covariance matrix."""
        return np.zeros((3, 3))
