"""Configuration containers for robot I/O, front-end, back-end, and outputs."""

from __future__ import annotations

from dataclasses import dataclass, field

from FinalProject.robot_python.data_types import Pose2D


@dataclass
class RobotConfig:
    """Robot hardware and UDP stream settings."""

    udp_host: str = "0.0.0.0"
    udp_port: int = 9000
    wheel_base_m: float = 0.23
    meters_per_tick: float = 0.0005
    lidar_min_range_m: float = 0.05
    lidar_max_range_m: float = 12.0


@dataclass
class CameraConfig:
    """ESP32-CAM, ArUco, and camera calibration settings."""

    stream_url: str = "http://192.168.4.1:81/stream"
    aruco_marker_size_m: float = 0.08
    # Camera matrix K:
    # [[348.0321225    0.         188.28539766]
    #  [  0.         355.6030584  105.48395813]
    #  [  0.           0.           1.        ]]

    # Distortion coefficients:
    # [[ 0.08988576 -2.60316437 -0.02748353  0.02115694 13.16262713]]

    camera_matrix: list[list[float]] = field(
        default_factory=lambda: [
            [348.0321225,   0,188.28539766],
            [0,355.6030584,105.48395813],
            [0.0, 0.0, 1.0],
        ]
    )
    distortion_coefficients: list[float] = field(default_factory=lambda: [0.08988576, -2.60316437, -0.02748353, 0.02115694, 13.16262713])
    camera_to_base: Pose2D = field(default_factory=Pose2D)
    marker_world_map: dict[int, Pose2D] = field(
    default_factory=lambda: {
        0: Pose2D(x=0.0, y=0.0, theta=0.0),
        1: Pose2D(x=1.5, y=0.0, theta=0.0),
        2: Pose2D(x=1.5, y=2.0, theta=0.0),
    }
    )   


@dataclass
class FrontendConfig:
    """Local estimation, ICP, and EKF settings."""

    icp_max_iterations: int = 25
    icp_convergence_tolerance: float = 1e-4
    icp_max_correspondence_distance_m: float = 0.35
    lidar_downsample_step: int = 2
    initial_covariance: float = 0.1
    motion_noise_xy: float = 0.02
    motion_noise_theta: float = 0.01
    lidar_match_noise_xy: float = 0.03
    lidar_match_noise_theta: float = 0.02
    keyframe_translation_m: float = 0.25
    keyframe_rotation_rad: float = 0.25
    max_landmark_keyframe_time_diff_s: float = 0.15


@dataclass
class BackendConfig:
    """Pose graph factor noise and optimization cadence settings."""

    odometry_noise_xy: float = 0.05
    odometry_noise_theta: float = 0.03
    lidar_factor_noise_xy: float = 0.04
    lidar_factor_noise_theta: float = 0.02
    landmark_noise_range: float = 0.1
    landmark_noise_bearing: float = 0.05
    loop_closure_noise_xy: float = 0.03
    loop_closure_noise_theta: float = 0.02
    loop_closure_min_keyframe_separation: int = 10
    loop_closure_candidate_radius_m: float = 0.75
    loop_closure_min_match_quality: float = 0.5
    optimize_every_n_keyframes: int = 5
    optimize_every_seconds: float = 2.0


@dataclass
class RuntimeConfig:
    """Runtime paths and loop timing settings."""

    log_dir: str = "logs"
    output_dir: str = "outputs"
    robot_loop_hz: float = 20.0
    camera_loop_hz: float = 10.0
    offline_replay_speed: float = 1.0
    offline_sync_tolerance_s: float = 0.15


@dataclass
class Config:
    """Top-level project configuration."""

    robot: RobotConfig = field(default_factory=RobotConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    frontend: FrontendConfig = field(default_factory=FrontendConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
