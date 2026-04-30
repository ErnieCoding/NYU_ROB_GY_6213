"""2D LiDAR preprocessing and scan matching placeholders for the front-end."""

from __future__ import annotations

from FinalProject.robot_python.config.config import FrontendConfig, RobotConfig
from FinalProject.robot_python.data_types import LidarScan, Pose2D, RelativeMotion


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
            range_m = scan.ranges[index]
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

    def match_scans(
        self,
        prev_scan: LidarScan,
        curr_scan: LidarScan,
        init_guess: Pose2D | None = None,
    ) -> RelativeMotion:
        """Estimate relative motion between consecutive LiDAR scans."""
        _ = self.preprocess_scan(prev_scan)
        _ = self.preprocess_scan(curr_scan)
        _ = init_guess

        # TODO: Implement ICP or another 2D scan matcher, including
        # correspondence search, robust residuals, match quality metrics, and
        # covariance estimation from match quality.
        return RelativeMotion(dx=0.0, dy=0.0, dtheta=0.0, quality=0.0, source="lidar_scan_matching")
