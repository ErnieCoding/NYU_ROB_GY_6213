"""2D LiDAR preprocessing and scan matching placeholders for the front-end."""

from __future__ import annotations

import math

import numpy as np
import open3d as o3d

from FinalProject.robot_python.config.config import FrontendConfig, RobotConfig
from FinalProject.robot_python.data_types import LidarScan, Pose2D, RelativeMotion
from FinalProject.robot_python.parameters import LIDAR_CALIB_BIAS, LIDAR_CALIB_DIST, LIDAR_COVARIANCE_FLOOR, C_LINEAR, B_LINEAR


class LidarMatcher:
    """Prepare scans and estimate relative motion with future ICP logic."""

    def __init__(self, robot_config: RobotConfig, frontend_config: FrontendConfig) -> None:
        self.robot_config = robot_config
        self.frontend_config = frontend_config

    def preprocess_scan(self, scan: LidarScan) -> LidarScan:
        """Filter obvious invalid ranges and optionally downsample a scan."""
        valid_ranges: list[float] = []
        valid_angles: list[float] = []
        step = max(1, self.frontend_config.lidar_downsample_step)

        for index in range(0, len(scan.ranges), step):

            # Get a single measurement in mm, preprocess with bias, convert to m
            range_mm = scan.ranges[index]
            corrected_measurement = self.correct_measurement(range_mm)
            range_m = corrected_measurement / 1000
            
            if self.robot_config.lidar_min_range_m <= range_m <= self.robot_config.lidar_max_range_m:
                valid_ranges.append(range_m)
                valid_angles.append(scan.angles[index])

        # TODO: Add robot body/self-hit filtering, spatial downsampling, and
        # optional conversion to Cartesian point clouds.
        return LidarScan(
            ranges=valid_ranges,
            angles=valid_angles,
            timestamp=scan.timestamp,
            frame_id=scan.frame_id,
            intensities=None,
        )
    
    def correct_measurement(self, z_mm):
        """
        Given a raw LiDAR measurement z_mm,
        return the bias-corrected measurement.

        Uses linear interpolation between calibration points.
        Clamps to the nearest known value outside the calibration range.
        """
        bias = float(np.interp(z_mm, LIDAR_CALIB_DIST, LIDAR_CALIB_BIAS))
        
        return z_mm - bias

    def match_scans(
        self,
        prev_scan: LidarScan,
        curr_scan: LidarScan,
        init_guess: Pose2D | None = None,
    ) -> RelativeMotion:
        """Estimate relative motion between consecutive LiDAR scans."""
        # First clean both scans using the same range filtering and
        # downsampling rules used by the rest of the front-end.
        prev_scan = self.preprocess_scan(prev_scan)
        curr_scan = self.preprocess_scan(curr_scan)

        # Convert polar LiDAR scans to Nx3 point arrays for Open3D.
        # Open3D expects 3D points, so this 2D LiDAR data uses z = 0.
        prev_points = np.array(
            [[r * math.cos(a), r * math.sin(a), 0.0] for r, a in zip(prev_scan.ranges, prev_scan.angles)],
            dtype=float,
        )
        curr_points = np.array(
            [[r * math.cos(a), r * math.sin(a), 0.0] for r, a in zip(curr_scan.ranges, curr_scan.angles)],
            dtype=float,
        )

        # ICP needs at least a few points to estimate a meaningful transform.
        # If either scan is too small, return a low-quality zero motion.
        if len(prev_points) < 3 or len(curr_points) < 3:
            return RelativeMotion(
                dx=0.0,
                dy=0.0,
                dtheta=0.0,
                covariance=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                quality=0.0,
                source="lidar_icp",
            )

        # Wrap the numpy point arrays in Open3D point cloud objects.
        # The current scan will be aligned onto the previous scan.
        prev_cloud = o3d.geometry.PointCloud()
        curr_cloud = o3d.geometry.PointCloud()
        prev_cloud.points = o3d.utility.Vector3dVector(prev_points)
        curr_cloud.points = o3d.utility.Vector3dVector(curr_points)

        # Start ICP from identity unless the caller provides a pose guess.
        # A good initial guess usually helps ICP converge to the correct match.
        init_transform = np.eye(4)
        if init_guess is not None:
            cos_t = math.cos(init_guess.theta)
            sin_t = math.sin(init_guess.theta)
            # Convert the 2D pose guess into a 4x4 SE(3)-style matrix because
            # Open3D's ICP API works with 3D homogeneous transforms.
            init_transform = np.array(
                [
                    [cos_t, -sin_t, 0.0, init_guess.x],
                    [sin_t, cos_t, 0.0, init_guess.y],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=float,
            )

        # Run point-to-point ICP using the configured max correspondence
        # distance and iteration/tolerance settings.
        threshold = self.frontend_config.icp_max_correspondence_distance_m
        result = o3d.pipelines.registration.registration_icp(
            curr_cloud,
            prev_cloud,
            threshold,
            init_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=self.frontend_config.icp_max_iterations,
                relative_fitness=self.frontend_config.icp_convergence_tolerance,
                relative_rmse=self.frontend_config.icp_convergence_tolerance,
            ),
        )

        # Fitness is the fraction of points with valid correspondences.
        # RMSE measures how far matched points are after alignment.
        quality = float(result.fitness)
        rmse = float(result.inlier_rmse)
        min_fitness = 0.1
        max_rmse = threshold

        # Reject clearly bad ICP matches. Returning zero motion is safer than
        # feeding an unreliable transform into the EKF and pose graph.
        if quality < min_fitness or rmse > max_rmse:
            sigma = max(float(rmse), 0.5)
            covariance = [
                [sigma, 0.0, 0.0], 
                [0.0, sigma, 0.0], 
                [0.0, 0.0, 0.1]
            ]
            return RelativeMotion(dx=0.0, dy=0.0, dtheta=0.0, covariance=covariance, quality=quality, source="lidar_icp")

        # Extract the 2D translation and yaw angle from the 4x4 ICP transform.
        transform = result.transformation
        dx = float(transform[0, 3])
        dy = float(transform[1, 3])
        dtheta = math.atan2(float(transform[1, 0]), float(transform[0, 0]))

        # TODO: Change implementation to the actual paper implementation with calculated residuals
        mean_range_mm = float(np.mean(curr_scan.ranges)) * 1000.0 if curr_scan.ranges else 700.0
        mean_range_m = mean_range_mm / 1000.0

        sigma2_range = max(C_LINEAR * mean_range_mm + B_LINEAR, LIDAR_COVARIANCE_FLOOR)
        sigma_xy = max(rmse, 1e-3) * (sigma2_range ** 0.5)
        sigma_theta = sigma_xy / mean_range_m
        covariance = [
            [sigma_xy, 0.0, 0.0], 
            [0.0, sigma_xy, 0.0], 
            [0.0, 0.0, sigma_theta]
        ]

        # Return the relative motion from current scan to previous scan,
        # along with ICP quality so later stages can gate weak matches.
        return RelativeMotion(
            dx=dx,
            dy=dy,
            dtheta=dtheta,
            covariance=covariance,
            quality=quality,
            source="lidar_icp",
        )
