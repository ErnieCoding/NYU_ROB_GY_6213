"""Differential-drive wheel odometry model for local front-end prediction."""

from __future__ import annotations

import math

from FinalProject.robot_python.config.config import RobotConfig
from FinalProject.robot_python.data_types import EncoderState, Pose2D, RelativeMotion, normalize_angle


class DifferentialDriveMotionModel:
    """Compute relative motion and pose predictions from wheel encoder states."""

    def __init__(self, config: RobotConfig) -> None:
        self.config = config

    def left_encoder_to_distance(self, delta_encoder: int) -> float:
        """Convert a left-wheel encoder tick increment into travel distance."""
        # TODO: Replace this linear placeholder with a calibrated fitted
        # function from left encoder counts to left wheel travel distance.
        return delta_encoder * self.config.meters_per_tick

    def right_encoder_to_distance(self, delta_encoder: int) -> float:
        """Convert a right-wheel encoder tick increment into travel distance."""
        # TODO: Replace this linear placeholder with a calibrated fitted
        # function from right encoder counts to right wheel travel distance.
        return delta_encoder * self.config.meters_per_tick

    def RelativeMotion_change(self, prev_encoder: EncoderState, curr_encoder: EncoderState) -> RelativeMotion:
        """Estimate robot-frame motion between two encoder readings.

        Flow:
        encoder tick differences -> fitted wheel conversion ->
        left/right wheel distances -> differential-drive increment.
        """
        left_delta_ticks = curr_encoder.left_ticks - prev_encoder.left_ticks
        right_delta_ticks = curr_encoder.right_ticks - prev_encoder.right_ticks
        #Getting distance of each wheels
        left_distance = self.left_encoder_to_distance(left_delta_ticks)
        right_distance = self.right_encoder_to_distance(right_delta_ticks)
        #based on differential drive model, we can compute the distance traveled and change in heading
        delta_s = 0.5 * (left_distance + right_distance)
        delta_theta = (right_distance - left_distance) / self.config.wheel_base_m

        dx = delta_s * math.cos(0.5 * delta_theta)
        dy = delta_s * math.sin(0.5 * delta_theta)
        dtheta = normalize_angle(delta_theta)

        covariance = self._estimate_motion_covariance(delta_s, dtheta)

        return RelativeMotion(
            dx=dx,
            dy=dy,
            dtheta=dtheta,
            delta_distance=delta_s,
            covariance=covariance,
            source="wheel_odometry",
        )

    def propagate(self, prev_pose: Pose2D, motion: RelativeMotion) -> Pose2D:
        """Apply a relative odometry increment to a pose estimate."""
        return prev_pose.propagate_pose(motion)

    def _estimate_motion_covariance(self, distance: float, dtheta: float) -> list[list[float]]:
        """Return a simple diagonal covariance placeholder for odometry motion."""
        # TODO: Replace these hand-tuned growth rules with covariance calibrated
        # from repeated drive tests and encoder residual statistics.
        # We need to do model fittign to get how the covariance grows with distance and heading change
        distance_mag = abs(distance)
        heading_mag = abs(dtheta)
        xy_variance = 1e-4 + 0.02 * distance_mag + 0.005 * heading_mag
        theta_variance = 1e-4 + 0.01 * heading_mag + 0.002 * distance_mag
        return [
            [xy_variance, 0.0, 0.0],
            [0.0, xy_variance, 0.0],
            [0.0, 0.0, theta_variance],
        ]
