"""Differential-drive wheel odometry model for local front-end prediction."""

from __future__ import annotations

import math
import numpy as np
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
        s_L_slope = 0.00035210
        return delta_encoder * s_L_slope    

    def right_encoder_to_distance(self, delta_encoder: int) -> float:
        """Convert a right-wheel encoder tick increment into travel distance."""
        # TODO: Replace this linear placeholder with a calibrated fitted
        # function from right encoder counts to right wheel travel distance.
        s_R_slope = 0.00035651
        return delta_encoder * s_R_slope
    
    ##Distance to sigma functions for left and right wheel distance predictions. We can use the residuals from our fitted functions to get an estimate of the variance of the distance predictions, which we can then use to construct the covariance matrix for the motion model.
    # sigma_L^2 = 0.00047296*|s_L_pred|^2 + 0.00428335*|s_L_pred| + 0.00742903
    # sigma_R^2 = 0.00679702*|s_R_pred|^2 + -0.01167738*|s_R_pred| + 0.00900622
    def distance_to_sigma_L(self,s_L_pred: float) -> float:
        return 0.00047296*abs(s_L_pred)**2 + 0.00428335*abs(s_L_pred) + 0.00742903

    def distance_to_sigma_R(self,s_R_pred: float) -> float:
        return 0.00679702*abs(s_R_pred)**2 + -0.01167738*abs(s_R_pred) + 0.00900622


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

        #getting sigma for each predicted distance:
        sigma_L2 = self.distance_to_sigma_L(left_distance)
        sigma_R2 = self.distance_to_sigma_R(right_distance)

        #based on differential drive model, we can compute the distance traveled and change in heading
        delta_s = 0.5 * (left_distance + right_distance)
        delta_theta = (right_distance - left_distance) / self.config.wheel_base_m

        dx = delta_s * math.cos(0.5 * delta_theta)
        dy = delta_s * math.sin(0.5 * delta_theta)
        dtheta = normalize_angle(delta_theta)

        #motion model inputs noise matrix:
        Motion_inputs_covariance = np.array([
                [sigma_L2, 0.0],
                [0.0, sigma_R2],
            ])

        

        #return relative motion and also the distance traveled
        return RelativeMotion(
            dx=dx,
            dy=dy,
            dtheta=dtheta,
            covariance=Motion_inputs_covariance,
            source="wheel_odometry",
        ), delta_s

    def propagate(self, prev_pose: Pose2D, motion: RelativeMotion) -> Pose2D:
        """Apply a relative odometry increment to a pose estimate."""
        return prev_pose.propagate_pose(motion)
