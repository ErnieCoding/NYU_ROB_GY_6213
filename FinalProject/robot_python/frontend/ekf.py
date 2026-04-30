"""EKF localizer scaffold for wheel odometry and LiDAR scan matching updates."""

from __future__ import annotations

import time

from FinalProject.robot_python.config.config import FrontendConfig
from FinalProject.robot_python.data_types import EncoderState, Pose2D, PoseEstimate, RelativeMotion
from FinalProject.robot_python.frontend.motion_model import DifferentialDriveMotionModel


class EKFLocalizer:
    """Maintain a local pose estimate with EKF predict/correct placeholders."""

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

    def predict(self, prev_encoder: EncoderState, curr_encoder: EncoderState) -> PoseEstimate:
        """Run the EKF prediction step from wheel odometry."""
        odom_motion = self.motion_model.compute_encoder_increment(prev_encoder, curr_encoder)
        self.last_odom_motion = odom_motion
        # TODO: Define the state vector explicitly, compute the motion-model
        # Jacobian, propagate covariance, and add process noise.
        self._state.pose = self._state.pose.moved_by(odom_motion)
        self._state.timestamp = time.time()
        self._state.source = "ekf_predict"
        return self.get_state()

    def correct_with_lidar(self, lidar_measurement: RelativeMotion | PoseEstimate) -> PoseEstimate:
        """Run the EKF correction step from LiDAR scan matching."""
        # TODO: Add measurement model, innovation calculation, Kalman gain, and
        # covariance update. For now, keep the predicted pose unchanged.
        _ = lidar_measurement
        self._state.source = "ekf_lidar_placeholder"
        return self.get_state()

    def get_state(self) -> PoseEstimate:
        """Return the current local pose estimate."""
        return PoseEstimate(
            pose=Pose2D(self._state.pose.x, self._state.pose.y, self._state.pose.theta),
            timestamp=self._state.timestamp,
            covariance=[row[:] for row in self._state.covariance],
            source=self._state.source,
        )

    def reset(self, initial_pose: Pose2D | None = None) -> None:
        """Reset the filter to an initial pose and covariance."""
        self._state = PoseEstimate(
            pose=initial_pose or Pose2D(),
            timestamp=time.time(),
            covariance=self._make_initial_covariance(),
            source="ekf_reset",
        )

    def _make_initial_covariance(self) -> list[list[float]]:
        """Create a diagonal initial covariance matrix."""
        value = self.config.initial_covariance
        return [
            [value, 0.0, 0.0],
            [0.0, value, 0.0],
            [0.0, 0.0, value],
        ]
