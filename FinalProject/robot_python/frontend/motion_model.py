"""Differential-drive wheel odometry model for local front-end prediction."""

from __future__ import annotations

import math

from FinalProject.robot_python.config.config import RobotConfig
from FinalProject.robot_python.data_types import EncoderState, Pose2D, RelativeMotion, normalize_angle


class DifferentialDriveMotionModel:
    """Compute relative motion and pose predictions from wheel encoder states."""

    def __init__(self, config: RobotConfig) -> None:
        self.config = config

    def compute_encoder_increment(self, prev_encoder: EncoderState, curr_encoder: EncoderState) -> RelativeMotion:
        """Estimate robot-frame motion between two encoder readings."""
        left_delta_ticks = curr_encoder.left_ticks - prev_encoder.left_ticks
        right_delta_ticks = curr_encoder.right_ticks - prev_encoder.right_ticks
        left_distance = left_delta_ticks * self.config.meters_per_tick
        right_distance = right_delta_ticks * self.config.meters_per_tick

        # TODO: Replace this simple midpoint integration with the final
        # differential-drive equations and Jacobians needed by the EKF.
        distance = 0.5 * (left_distance + right_distance)
        dtheta = (right_distance - left_distance) / self.config.wheel_base_m
        dx = distance * math.cos(0.5 * dtheta)
        dy = distance * math.sin(0.5 * dtheta)

        return RelativeMotion(
            dx=dx,
            dy=dy,
            dtheta=normalize_angle(dtheta),
            covariance=None,
            quality=None,
            source="wheel_odometry",
        )

    def predict_pose(self, prev_pose: Pose2D, motion: RelativeMotion) -> Pose2D:
        """Apply a relative odometry increment to a pose estimate."""
        # TODO: Add uncertainty propagation when this helper is folded into the
        # EKF predict step.
        return prev_pose.moved_by(motion)
