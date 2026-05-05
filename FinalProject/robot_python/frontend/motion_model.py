"""Differential-drive wheel odometry model for local front-end prediction."""

from __future__ import annotations

import math
import numpy as np

from FinalProject.robot_python.config.config import RobotConfig
from FinalProject.robot_python.data_types import (
    EncoderState,
    Pose2D,
    RelativeMotion,
    normalize_angle,
)
from FinalProject.robot_python import parameters


class DifferentialDriveMotionModel:
    """Compute relative motion and pose predictions from wheel encoder states."""

    def __init__(self, config: RobotConfig) -> None:
        self.config = config

    def _wheel_base_m(self) -> float:
        if hasattr(self.config, "wheel_base_m"):
            return float(self.config.wheel_base_m)
        return float(parameters.wheel_base)

    def left_encoder_to_distance(self, delta_encoder: int) -> float:
        s_L_slope = 0.00035698051573807086
        return delta_encoder * s_L_slope

    def right_encoder_to_distance(self, delta_encoder: int) -> float:
        s_R_slope = 0.0003658445193934484
        return delta_encoder * s_R_slope

    # These fitted functions appear to return variance, not sigma.
    def distance_to_sigma_L(self, s_L_pred: float) -> float:
        return max(
            0.00047296 * abs(s_L_pred) ** 2 + 0.00428335 * abs(s_L_pred) + 0.00742903,
            1e-6,
        )

    def distance_to_sigma_R(self, s_R_pred: float) -> float:
        return max(
            0.00679702 * abs(s_R_pred) ** 2 - 0.01167738 * abs(s_R_pred) + 0.00900622,
            1e-6,
        )

    def _wheel_motion_components(
        self,
        prev_encoder: EncoderState,
        curr_encoder: EncoderState,
    ) -> tuple[RelativeMotion, float, np.ndarray]:
        left_delta_ticks = curr_encoder.left_ticks - prev_encoder.left_ticks
        right_delta_ticks = curr_encoder.right_ticks - prev_encoder.right_ticks

        left_distance = self.left_encoder_to_distance(left_delta_ticks)
        right_distance = self.right_encoder_to_distance(right_delta_ticks)

        sigma_L2 = self.distance_to_sigma_L(left_distance)
        sigma_R2 = self.distance_to_sigma_R(right_distance)

        wheel_base = self._wheel_base_m()

        delta_s = 0.5 * (left_distance + right_distance)
        delta_theta = (right_distance - left_distance) / wheel_base

        # Local body-frame increment using midpoint approximation.
        dx = delta_s * math.cos(0.5 * delta_theta)
        dy = delta_s * math.sin(0.5 * delta_theta)
        dtheta = normalize_angle(delta_theta)

        wheel_covariance = np.array(
            [
                [sigma_L2, 0.0],
                [0.0, sigma_R2],
            ],
            dtype=float,
        )

        motion = RelativeMotion(
            dx=dx,
            dy=dy,
            dtheta=dtheta,
            covariance=None,
            source="wheel_odometry",
        )
        return motion, delta_s, wheel_covariance

    def RelativeMotion_change(
        self,
        prev_encoder: EncoderState,
        curr_encoder: EncoderState,
    ) -> tuple[RelativeMotion, float]:
        """
        Backward-compatible interface for older scripts.

        Returns a RelativeMotion whose covariance is now a 3x3 pose covariance,
        not the old 2x2 wheel covariance.
        """
        motion, delta_s, wheel_covariance = self._wheel_motion_components(
            prev_encoder,
            curr_encoder,
        )

        # Use theta = 0 local-frame linearization for compatibility.
        dummy_pose = Pose2D(x=0.0, y=0.0, theta=0.0)
        G = self.control_jacobian(dummy_pose, motion, delta_s)
        motion.covariance = self.process_covariance(G, wheel_covariance).tolist()
        return motion, delta_s

    def state_jacobian(
        self,
        prev_pose: Pose2D,
        motion: RelativeMotion,
        delta_s: float,
    ) -> np.ndarray:
        theta_m = prev_pose.theta + 0.5 * motion.dtheta
        return np.array(
            [
                [1.0, 0.0, -delta_s * math.sin(theta_m)],
                [0.0, 1.0,  delta_s * math.cos(theta_m)],
                [0.0, 0.0,  1.0],
            ],
            dtype=float,
        )

    def control_jacobian(
        self,
        prev_pose: Pose2D,
        motion: RelativeMotion,
        delta_s: float,
    ) -> np.ndarray:
        theta_m = prev_pose.theta + 0.5 * motion.dtheta
        wheel_base = self._wheel_base_m()

        return np.array(
            [
                [
                    0.5 * math.cos(theta_m) + (delta_s / (2.0 * wheel_base)) * math.sin(theta_m),
                    0.5 * math.cos(theta_m) - (delta_s / (2.0 * wheel_base)) * math.sin(theta_m),
                ],
                [
                    0.5 * math.sin(theta_m) - (delta_s / (2.0 * wheel_base)) * math.cos(theta_m),
                    0.5 * math.sin(theta_m) + (delta_s / (2.0 * wheel_base)) * math.cos(theta_m),
                ],
                [
                    -1.0 / wheel_base,
                    1.0 / wheel_base,
                ],
            ],
            dtype=float,
        )

    def process_covariance(self, G: np.ndarray, wheel_covariance: np.ndarray) -> np.ndarray:
        return G @ wheel_covariance @ G.T

    def predict_motion(
        self,
        prev_pose: Pose2D,
        prev_encoder: EncoderState,
        curr_encoder: EncoderState,
    ) -> tuple[RelativeMotion, np.ndarray]:
        motion, delta_s, wheel_covariance = self._wheel_motion_components(
            prev_encoder,
            curr_encoder,
        )
        F = self.state_jacobian(prev_pose, motion, delta_s)
        G = self.control_jacobian(prev_pose, motion, delta_s)
        motion.covariance = self.process_covariance(G, wheel_covariance).tolist()
        return motion, F

    def propagate(self, prev_pose: Pose2D, motion: RelativeMotion) -> Pose2D:
        return prev_pose.propagate_pose(motion)