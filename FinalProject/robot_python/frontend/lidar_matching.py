"""2D LiDAR preprocessing and correlative scan matching for the front-end."""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from FinalProject.robot_python.config.config import FrontendConfig, RobotConfig
from FinalProject.robot_python.data_types import LidarScan, Pose2D, RelativeMotion, normalize_angle
from FinalProject.robot_python.parameters import (
    LIDAR_CALIB_BIAS,
    LIDAR_CALIB_DIST,
    LIDAR_COVARIANCE_FLOOR,
    C_LINEAR,
    B_LINEAR,
)


class LidarMatcher:
    """Prepare scans, maintain a small submap, and estimate LiDAR-relative motion."""

    def __init__(self, robot_config: RobotConfig, frontend_config: FrontendConfig) -> None:
        self.robot_config = robot_config
        self.frontend_config = frontend_config

        self.max_corr_distance_m = float(
            getattr(frontend_config, "icp_max_correspondence_distance_m", 0.12)
        )
        self.search_translation_window_m = float(
            getattr(frontend_config, "submap_search_translation_window_m", 0.08)
        )
        self.search_translation_step_m = float(
            getattr(frontend_config, "submap_search_translation_step_m", 0.02)
        )
        self.search_rotation_window_rad = float(
            getattr(frontend_config, "submap_search_rotation_window_rad", math.radians(6.0))
        )
        self.search_rotation_step_rad = float(
            getattr(frontend_config, "submap_search_rotation_step_rad", math.radians(1.0))
        )
        self.min_match_quality = float(
            getattr(frontend_config, "submap_min_match_quality", 0.20)
        )
        self.min_points = int(getattr(frontend_config, "submap_min_points", 12))
        self.submap_max_scans = int(getattr(frontend_config, "submap_max_scans", 8))
        self.submap_max_points = int(getattr(frontend_config, "submap_max_points", 2500))

        self._submap_scans_world: deque[np.ndarray] = deque(maxlen=self.submap_max_scans)
        self._submap_points_world = np.empty((0, 2), dtype=float)

        self.lidar_yaw_offset_rad = float(
            getattr(robot_config, "lidar_yaw_offset_rad", 0.0)
        )
        self.match_max_query_points = int(
            getattr(frontend_config, "match_max_query_points", 160)
        )
        self.match_max_reference_points = int(
            getattr(frontend_config, "match_max_reference_points", 900)
        )
        self.coarse_translation_step_m = float(
            getattr(
                frontend_config,
                "submap_coarse_translation_step_m",
                max(0.03, 2.0 * self.search_translation_step_m),
            )
        )
        self.coarse_rotation_step_rad = float(
            getattr(
                frontend_config,
                "submap_coarse_rotation_step_rad",
                max(math.radians(2.0), 2.0 * self.search_rotation_step_rad),
            )
        )

    def reset_submap(self) -> None:
        """Clear the rolling local submap."""
        self._submap_scans_world.clear()
        self._submap_points_world = np.empty((0, 2), dtype=float)

    def has_submap(self) -> bool:
        """Return True when enough points exist for submap matching."""
        return len(self._submap_points_world) >= self.min_points

    def preprocess_scan(self, scan: LidarScan) -> LidarScan:
        """Filter obvious invalid ranges and optionally downsample a scan."""
        valid_ranges: list[float] = []
        valid_angles: list[float] = []
        step = max(1, self.frontend_config.lidar_downsample_step)

        for index in range(0, len(scan.ranges), step):
            range_m = scan.ranges[index]
            raw_mm = range_m * 1000.0
            corrected_mm = self.correct_measurement(raw_mm)
            range_m = corrected_mm / 1000.0

            if self.robot_config.lidar_min_range_m <= range_m <= self.robot_config.lidar_max_range_m:
                valid_ranges.append(range_m)
                valid_angles.append(scan.angles[index])

        return LidarScan(
            ranges=valid_ranges,
            angles=valid_angles,
            timestamp=scan.timestamp,
            frame_id=scan.frame_id,
            intensities=None,
        )

    def correct_measurement(self, z_mm: float) -> float:
        """Bias-correct one LiDAR range sample in millimeters."""
        bias = float(np.interp(z_mm, LIDAR_CALIB_DIST, LIDAR_CALIB_BIAS))
        return z_mm - bias

    # def update_submap(self, scan: LidarScan, pose: Pose2D) -> None:
    #     """Insert one corrected scan into the rolling world-frame submap."""
    #     processed_scan = self.preprocess_scan(scan)
    #     points_local = self._scan_to_points_2d(processed_scan)
    #     if len(points_local) < self.min_points:
    #         return

    #     points_world = self._transform_points(points_local, pose)
    #     self._submap_scans_world.append(points_world)
    #     self._rebuild_submap()

    def update_submap(self, scan: LidarScan, pose: Pose2D) -> None:
        processed_scan, points_world = self.scan_to_world_points(
            scan,
            pose,
            assume_preprocessed=False,
        )
        if len(points_world) < self.min_points:
            return

        self._submap_scans_world.append(points_world)
        self._rebuild_submap()

    # def match_scan_to_submap(
    #     self,
    #     curr_scan: LidarScan,
    #     predicted_pose: Pose2D,
    #     prev_pose: Pose2D,
    # ) -> RelativeMotion | None:
    #     """
    #     Match the current scan against the rolling submap.

    #     predicted_pose is the EKF-predicted current global pose.
    #     prev_pose is the previous corrected global pose and is used to convert
    #     the matched absolute pose back into a relative motion for the EKF.
    #     """
    #     if not self.has_submap():
    #         return None

    #     processed_scan = self.preprocess_scan(curr_scan)
    #     curr_points_local = self._scan_to_points_2d(processed_scan)
    #     if len(curr_points_local) < self.min_points:
    #         return None

    #     best_pose, quality, rmse = self._search_absolute_pose_against_reference(
    #         curr_points_local,
    #         reference_points=self._submap_points_world,
    #         seed_pose=predicted_pose,
    #     )

    #     if best_pose is None or quality < self.min_match_quality:
    #         return None

    #     rel_motion = self._relative_motion_from_poses(prev_pose, best_pose)
    #     rel_motion.covariance = self._covariance_from_match(processed_scan, rmse, quality)
    #     rel_motion.quality = quality
    #     rel_motion.source = "lidar_submap"
    #     return rel_motion

    def match_scan_to_submap(
            self,
            curr_scan: LidarScan,
            predicted_pose: Pose2D,
            prev_pose: Pose2D,
        ) -> RelativeMotion | None:
        if not self.has_submap():
            return None

        processed_scan = self.preprocess_scan(curr_scan)
        curr_points_local = self._scan_to_points_2d(processed_scan)
        if len(curr_points_local) < self.min_points:
            return None

        curr_points_local = self._cap_points(curr_points_local, self.match_max_query_points)
        reference_points = self._cap_points(
            self._submap_points_world,
            self.match_max_reference_points,
        )

        coarse_pose, coarse_quality, coarse_rmse = self._search_absolute_pose_against_reference(
            curr_points_local=curr_points_local,
            reference_points=reference_points,
            seed_pose=predicted_pose,
            translation_window_m=self.search_translation_window_m,
            translation_step_m=self.coarse_translation_step_m,
            rotation_window_rad=self.search_rotation_window_rad,
            rotation_step_rad=self.coarse_rotation_step_rad,
        )

        if coarse_pose is None or coarse_quality < self.min_match_quality:
            return None

        best_pose, quality, rmse = self._search_absolute_pose_against_reference(
            curr_points_local=curr_points_local,
            reference_points=reference_points,
            seed_pose=coarse_pose,
            translation_window_m=max(self.coarse_translation_step_m, self.search_translation_step_m),
            translation_step_m=self.search_translation_step_m,
            rotation_window_rad=max(self.coarse_rotation_step_rad, self.search_rotation_step_rad),
            rotation_step_rad=self.search_rotation_step_rad,
        )

        if best_pose is None or quality < self.min_match_quality:
            return None

        rel_motion = self._relative_motion_from_poses(prev_pose, best_pose)
        rel_motion.covariance = self._covariance_from_match(processed_scan, rmse, quality)
        rel_motion.quality = quality
        rel_motion.source = "lidar_submap"
        return rel_motion

    # def match_scans(
    #     self,
    #     prev_scan: LidarScan,
    #     curr_scan: LidarScan,
    #     init_guess: Pose2D | None = None,
    # ) -> RelativeMotion | None:
    #     """
    #     Correlative scan-to-scan fallback.

    #     This keeps your keyframe and loop-closure code working without Open3D ICP.
    #     The returned motion is in the previous scan frame.
    #     """
    #     prev_scan = self.preprocess_scan(prev_scan)
    #     curr_scan = self.preprocess_scan(curr_scan)

    #     prev_points = self._scan_to_points_2d(prev_scan)
    #     curr_points = self._scan_to_points_2d(curr_scan)

    #     if len(prev_points) < self.min_points or len(curr_points) < self.min_points:
    #         return None

    #     seed = init_guess or Pose2D()
    #     best_pose, quality, rmse = self._search_relative_pose_against_reference(
    #         curr_points_local=curr_points,
    #         reference_points=prev_points,
    #         seed_motion=seed,
    #     )

    #     if best_pose is None or quality < self.min_match_quality:
    #         return None

    #     return RelativeMotion(
    #         dx=best_pose.x,
    #         dy=best_pose.y,
    #         dtheta=best_pose.theta,
    #         covariance=self._covariance_from_match(curr_scan, rmse, quality),
    #         quality=quality,
    #         source="lidar_correlative",
    #     )

    def match_scans(
            self,
            prev_scan: LidarScan,
            curr_scan: LidarScan,
            init_guess: Pose2D | None = None,
        ) -> RelativeMotion | None:
        prev_scan = self.preprocess_scan(prev_scan)
        curr_scan = self.preprocess_scan(curr_scan)

        prev_points = self._cap_points(
            self._scan_to_points_2d(prev_scan),
            self.match_max_reference_points,
        )
        curr_points = self._cap_points(
            self._scan_to_points_2d(curr_scan),
            self.match_max_query_points,
        )

        if len(prev_points) < self.min_points or len(curr_points) < self.min_points:
            return None

        seed = init_guess or Pose2D()
        if init_guess is None:
            trans_window = self.search_translation_window_m
            rot_window = self.search_rotation_window_rad
        else:
            trans_window = min(self.search_translation_window_m, 0.05)
            rot_window = min(self.search_rotation_window_rad, math.radians(4.0))

        coarse_pose, coarse_quality, _ = self._search_relative_pose_against_reference(
            curr_points_local=curr_points,
            reference_points=prev_points,
            seed_motion=seed,
            translation_window_m=trans_window,
            translation_step_m=self.coarse_translation_step_m,
            rotation_window_rad=rot_window,
            rotation_step_rad=self.coarse_rotation_step_rad,
        )

        if coarse_pose is None or coarse_quality < self.min_match_quality:
            return None

        best_pose, quality, rmse = self._search_relative_pose_against_reference(
            curr_points_local=curr_points,
            reference_points=prev_points,
            seed_motion=coarse_pose,
            translation_window_m=max(self.coarse_translation_step_m, self.search_translation_step_m),
            translation_step_m=self.search_translation_step_m,
            rotation_window_rad=max(self.coarse_rotation_step_rad, self.search_rotation_step_rad),
            rotation_step_rad=self.search_rotation_step_rad,
        )

        if best_pose is None or quality < self.min_match_quality:
            return None

        return RelativeMotion(
            dx=best_pose.x,
            dy=best_pose.y,
            dtheta=best_pose.theta,
            covariance=self._covariance_from_match(curr_scan, rmse, quality),
            quality=quality,
            source="lidar_correlative",
        )

    def _rebuild_submap(self) -> None:
        """Flatten the rolling scan buffer into one capped point set."""
        if not self._submap_scans_world:
            self._submap_points_world = np.empty((0, 2), dtype=float)
            return

        stacked = np.vstack(list(self._submap_scans_world))
        if len(stacked) > self.submap_max_points:
            idx = np.linspace(0, len(stacked) - 1, self.submap_max_points, dtype=int)
            stacked = stacked[idx]

        self._submap_points_world = stacked.astype(float)

    # def _scan_to_points_2d(self, scan: LidarScan) -> np.ndarray:
    #     """Convert polar scan data to Nx2 Cartesian points in the sensor frame."""
    #     if not scan.ranges:
    #         return np.empty((0, 2), dtype=float)

    #     return np.array(
    #         [[r * math.cos(a), r * math.sin(a)] for r, a in zip(scan.ranges, scan.angles)],
    #         dtype=float,
    #     )

    def _scan_to_points_2d(self, scan: LidarScan) -> np.ndarray:
        if not scan.ranges:
            return np.empty((0, 2), dtype=float)

        ranges = np.asarray(scan.ranges, dtype=float)
        angles = np.asarray(scan.angles, dtype=float) + self.lidar_yaw_offset_rad
        return np.column_stack((ranges * np.cos(angles), ranges * np.sin(angles)))

    def _transform_points(self, points: np.ndarray, pose: Pose2D) -> np.ndarray:
        """Apply an SE(2)-style transform to Nx2 points."""
        if len(points) == 0:
            return points

        cos_t = math.cos(pose.theta)
        sin_t = math.sin(pose.theta)
        rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=float)
        return points @ rot.T + np.array([pose.x, pose.y], dtype=float)

    def _score_points_against_reference(
        self,
        query_points: np.ndarray,
        reference_points: np.ndarray,
    ) -> tuple[float, float]:
        """Return (quality, rmse) using nearest-neighbor distance gating."""
        if len(query_points) < self.min_points or len(reference_points) < self.min_points:
            return 0.0, float("inf")

        min_sq_dists = self._nearest_sq_dists(query_points, reference_points)
        threshold_sq = self.max_corr_distance_m ** 2
        inliers = min_sq_dists <= threshold_sq

        if not np.any(inliers):
            return 0.0, float("inf")

        fitness = float(np.mean(inliers))
        rmse = float(np.sqrt(np.mean(min_sq_dists[inliers])))
        quality = float(fitness * math.exp(-rmse / max(self.max_corr_distance_m, 1e-6)))
        return quality, rmse

    def _nearest_sq_dists(self, query_points: np.ndarray, reference_points: np.ndarray) -> np.ndarray:
        """Brute-force nearest-neighbor squared distances with chunking."""
        min_sq = np.full(len(query_points), np.inf, dtype=float)
        chunk_size = 256

        for start in range(0, len(reference_points), chunk_size):
            ref_chunk = reference_points[start : start + chunk_size]
            diffs = query_points[:, None, :] - ref_chunk[None, :, :]
            sq = np.sum(diffs * diffs, axis=2)
            min_sq = np.minimum(min_sq, np.min(sq, axis=1))

        return min_sq

    # def _search_absolute_pose_against_reference(
    #     self,
    #     curr_points_local: np.ndarray,
    #     reference_points: np.ndarray,
    #     seed_pose: Pose2D,
    # ) -> tuple[Pose2D | None, float, float]:
    #     """Search around a predicted global pose."""
    #     best_pose: Pose2D | None = None
    #     best_quality = 0.0
    #     best_rmse = float("inf")

    #     dx_offsets = np.arange(
    #         -self.search_translation_window_m,
    #         self.search_translation_window_m + 0.5 * self.search_translation_step_m,
    #         self.search_translation_step_m,
    #     )
    #     dy_offsets = np.arange(
    #         -self.search_translation_window_m,
    #         self.search_translation_window_m + 0.5 * self.search_translation_step_m,
    #         self.search_translation_step_m,
    #     )
    #     dtheta_offsets = np.arange(
    #         -self.search_rotation_window_rad,
    #         self.search_rotation_window_rad + 0.5 * self.search_rotation_step_rad,
    #         self.search_rotation_step_rad,
    #     )

    #     for dtheta in dtheta_offsets:
    #         theta = normalize_angle(seed_pose.theta + dtheta)
    #         cos_t = math.cos(theta)
    #         sin_t = math.sin(theta)
    #         rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=float)
    #         rotated = curr_points_local @ rot.T

    #         for dx in dx_offsets:
    #             for dy in dy_offsets:
    #                 candidate = Pose2D(
    #                     x=seed_pose.x + float(dx),
    #                     y=seed_pose.y + float(dy),
    #                     theta=theta,
    #                 )
    #                 transformed = rotated + np.array([candidate.x, candidate.y], dtype=float)
    #                 quality, rmse = self._score_points_against_reference(
    #                     transformed,
    #                     reference_points,
    #                 )

    #                 if quality > best_quality or (
    #                     math.isclose(quality, best_quality) and rmse < best_rmse
    #                 ):
    #                     best_pose = candidate
    #                     best_quality = quality
    #                     best_rmse = rmse

    #     return best_pose, best_quality, best_rmse

    # def _search_relative_pose_against_reference(
    #     self,
    #     curr_points_local: np.ndarray,
    #     reference_points: np.ndarray,
    #     seed_motion: Pose2D,
    # ) -> tuple[Pose2D | None, float, float]:
    #     """Search around a relative-motion seed in the previous scan frame."""
    #     best_motion: Pose2D | None = None
    #     best_quality = 0.0
    #     best_rmse = float("inf")

    #     dx_offsets = np.arange(
    #         -self.search_translation_window_m,
    #         self.search_translation_window_m + 0.5 * self.search_translation_step_m,
    #         self.search_translation_step_m,
    #     )
    #     dy_offsets = np.arange(
    #         -self.search_translation_window_m,
    #         self.search_translation_window_m + 0.5 * self.search_translation_step_m,
    #         self.search_translation_step_m,
    #     )
    #     dtheta_offsets = np.arange(
    #         -self.search_rotation_window_rad,
    #         self.search_rotation_window_rad + 0.5 * self.search_rotation_step_rad,
    #         self.search_rotation_step_rad,
    #     )

    #     for dtheta in dtheta_offsets:
    #         theta = normalize_angle(seed_motion.theta + dtheta)
    #         cos_t = math.cos(theta)
    #         sin_t = math.sin(theta)
    #         rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=float)
    #         rotated = curr_points_local @ rot.T

    #         for dx in dx_offsets:
    #             for dy in dy_offsets:
    #                 candidate = Pose2D(
    #                     x=seed_motion.x + float(dx),
    #                     y=seed_motion.y + float(dy),
    #                     theta=theta,
    #                 )
    #                 transformed = rotated + np.array([candidate.x, candidate.y], dtype=float)
    #                 quality, rmse = self._score_points_against_reference(
    #                     transformed,
    #                     reference_points,
    #                 )

    #                 if quality > best_quality or (
    #                     math.isclose(quality, best_quality) and rmse < best_rmse
    #                 ):
    #                     best_motion = candidate
    #                     best_quality = quality
    #                     best_rmse = rmse

    #     return best_motion, best_quality, best_rmse

    def _search_absolute_pose_against_reference(
            self,
            curr_points_local: np.ndarray,
            reference_points: np.ndarray,
            seed_pose: Pose2D,
            translation_window_m: float | None = None,
            translation_step_m: float | None = None,
            rotation_window_rad: float | None = None,
            rotation_step_rad: float | None = None,
        ) -> tuple[Pose2D | None, float, float]:
        translation_window_m = self.search_translation_window_m if translation_window_m is None else translation_window_m
        translation_step_m = self.search_translation_step_m if translation_step_m is None else translation_step_m
        rotation_window_rad = self.search_rotation_window_rad if rotation_window_rad is None else rotation_window_rad
        rotation_step_rad = self.search_rotation_step_rad if rotation_step_rad is None else rotation_step_rad

        best_pose = None
        best_quality = 0.0
        best_rmse = float("inf")

        dx_offsets = np.arange(-translation_window_m, translation_window_m + 0.5 * translation_step_m, translation_step_m)
        dy_offsets = np.arange(-translation_window_m, translation_window_m + 0.5 * translation_step_m, translation_step_m)
        dtheta_offsets = np.arange(-rotation_window_rad, rotation_window_rad + 0.5 * rotation_step_rad, rotation_step_rad)

        for dtheta in dtheta_offsets:
            theta = normalize_angle(seed_pose.theta + dtheta)
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=float)
            rotated = curr_points_local @ rot.T

            for dx in dx_offsets:
                for dy in dy_offsets:
                    candidate = Pose2D(
                        x=seed_pose.x + float(dx),
                        y=seed_pose.y + float(dy),
                        theta=theta,
                    )
                    transformed = rotated + np.array([candidate.x, candidate.y], dtype=float)
                    quality, rmse = self._score_points_against_reference(transformed, reference_points)

                    if quality > best_quality or (math.isclose(quality, best_quality) and rmse < best_rmse):
                        best_pose = candidate
                        best_quality = quality
                        best_rmse = rmse

        return best_pose, best_quality, best_rmse
    
    def _search_relative_pose_against_reference(
            self,
            curr_points_local: np.ndarray,
            reference_points: np.ndarray,
            seed_motion: Pose2D,
            translation_window_m: float | None = None,
            translation_step_m: float | None = None,
            rotation_window_rad: float | None = None,
            rotation_step_rad: float | None = None,
        ) -> tuple[Pose2D | None, float, float]:
        """Search around a relative-motion seed in the previous scan frame."""
        translation_window_m = (
            self.search_translation_window_m
            if translation_window_m is None
            else translation_window_m
        )
        translation_step_m = (
            self.search_translation_step_m
            if translation_step_m is None
            else translation_step_m
        )
        rotation_window_rad = (
            self.search_rotation_window_rad
            if rotation_window_rad is None
            else rotation_window_rad
        )
        rotation_step_rad = (
            self.search_rotation_step_rad
            if rotation_step_rad is None
            else rotation_step_rad
        )

        best_motion: Pose2D | None = None
        best_quality = 0.0
        best_rmse = float("inf")

        dx_offsets = np.arange(
            -translation_window_m,
            translation_window_m + 0.5 * translation_step_m,
            translation_step_m,
        )
        dy_offsets = np.arange(
            -translation_window_m,
            translation_window_m + 0.5 * translation_step_m,
            translation_step_m,
        )
        dtheta_offsets = np.arange(
            -rotation_window_rad,
            rotation_window_rad + 0.5 * rotation_step_rad,
            rotation_step_rad,
        )

        for dtheta in dtheta_offsets:
            theta = normalize_angle(seed_motion.theta + dtheta)
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            rot = np.array(
                [
                    [cos_t, -sin_t],
                    [sin_t,  cos_t],
                ],
                dtype=float,
            )
            rotated = curr_points_local @ rot.T

            for dx in dx_offsets:
                for dy in dy_offsets:
                    candidate = Pose2D(
                        x=seed_motion.x + float(dx),
                        y=seed_motion.y + float(dy),
                        theta=theta,
                    )
                    transformed = rotated + np.array([candidate.x, candidate.y], dtype=float)
                    quality, rmse = self._score_points_against_reference(
                        transformed,
                        reference_points,
                    )

                    if quality > best_quality or (
                        math.isclose(quality, best_quality) and rmse < best_rmse
                    ):
                        best_motion = candidate
                        best_quality = quality
                        best_rmse = rmse

        return best_motion, best_quality, best_rmse

    def _relative_motion_from_poses(self, prev_pose: Pose2D, curr_pose: Pose2D) -> RelativeMotion:
        """Convert two global poses into a body-frame relative motion."""
        dx_world = curr_pose.x - prev_pose.x
        dy_world = curr_pose.y - prev_pose.y

        cos_t = math.cos(prev_pose.theta)
        sin_t = math.sin(prev_pose.theta)

        dx_local = cos_t * dx_world + sin_t * dy_world
        dy_local = -sin_t * dx_world + cos_t * dy_world
        dtheta = normalize_angle(curr_pose.theta - prev_pose.theta)

        return RelativeMotion(
            dx=float(dx_local),
            dy=float(dy_local),
            dtheta=float(dtheta),
            covariance=None,
            quality=None,
            source="lidar_submap",
        )

    # def _covariance_from_match(
    #     self,
    #     scan: LidarScan,
    #     rmse: float,
    #     quality: float,
    # ) -> list[list[float]]:
    #     """Adaptive measurement covariance from match residual and confidence."""
    #     mean_range_mm = float(np.mean(scan.ranges) * 1000.0) if scan.ranges else 700.0
    #     mean_range_m = max(mean_range_mm / 1000.0, 0.2)

    #     sigma2_range = max(C_LINEAR * mean_range_mm + B_LINEAR, LIDAR_COVARIANCE_FLOOR)
    #     range_sigma_m = max(math.sqrt(float(sigma2_range)) / 1000.0, 0.005)

    #     quality_scale = 1.0 / max(quality, 0.05)
    #     sigma_xy = (max(rmse, 0.01) + range_sigma_m) * quality_scale
    #     sigma_xy = min(max(sigma_xy, 0.01), 0.50)

    #     sigma_theta = max(self.search_rotation_step_rad, sigma_xy / mean_range_m) * quality_scale
    #     sigma_theta = min(max(sigma_theta, math.radians(0.5)), math.radians(20.0))

    #     return [
    #         [sigma_xy**2, 0.0, 0.0],
    #         [0.0, sigma_xy**2, 0.0],
    #         [0.0, 0.0, sigma_theta**2],
    #     ]

    def _covariance_from_match(
            self,
            scan: LidarScan,
            rmse: float,
            quality: float,
        ) -> list[list[float]]:
        mean_range_mm = float(np.mean(scan.ranges) * 1000.0) if scan.ranges else 700.0
        mean_range_m = max(mean_range_mm / 1000.0, 0.2)

        sigma2_range = max(C_LINEAR * mean_range_mm + B_LINEAR, LIDAR_COVARIANCE_FLOOR)
        range_sigma_m = max(math.sqrt(float(sigma2_range)) / 1000.0, 0.005)

        base_sigma_xy = max(rmse, 0.01) + range_sigma_m
        quality_scale = 1.0 / max(quality, 0.05)

        sigma_xy = min(max(base_sigma_xy * quality_scale, 0.01), 0.35)
        sigma_theta = max(self.search_rotation_step_rad, base_sigma_xy / mean_range_m) * math.sqrt(quality_scale)
        sigma_theta = min(max(sigma_theta, math.radians(0.5)), math.radians(12.0))

        return [
            [sigma_xy**2, 0.0, 0.0],
            [0.0, sigma_xy**2, 0.0],
            [0.0, 0.0, sigma_theta**2],
        ]
    
    def _cap_points(self, points: np.ndarray, max_points: int) -> np.ndarray:
        if len(points) <= max_points:
            return points
        idx = np.linspace(0, len(points) - 1, max_points, dtype=int)
        return points[idx]


    def scan_to_world_points(
        self,
        scan: LidarScan,
        pose: Pose2D,
        assume_preprocessed: bool = False,
    ) -> tuple[LidarScan, np.ndarray]:
        working_scan = scan if assume_preprocessed else self.preprocess_scan(scan)
        points_local = self._scan_to_points_2d(working_scan)
        points_world = self._transform_points(points_local, pose)
        return working_scan, points_world