"""Offline replay for DataLogger pickle files, mirroring the GUI SLAM loop."""

from __future__ import annotations

import argparse
import math
import sys
import time
import types
from pathlib import Path

import numpy as np


_APP_DIR = Path(__file__).resolve().parent
_PYTHON_DIR = _APP_DIR.parent
_PROJECT_DIR = _PYTHON_DIR.parent
_ROOT_DIR = _PROJECT_DIR.parent

for _p in [str(_ROOT_DIR), str(_PROJECT_DIR), str(_PYTHON_DIR), str(_APP_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

def normalize_angle(angle: float) -> float:
    """Normalize an angle in radians to the interval [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi

def install_optional_slam_stubs() -> None:
    """Keep offline replay usable when optional solver packages are absent."""
    try:
        import gtsam  # noqa: F401
    except ModuleNotFoundError:
        gtsam_stub = types.ModuleType("gtsam")

        class Pose2:
            def __init__(self, x, y, theta):
                self._x = x
                self._y = y
                self._theta = theta

            def x(self):
                return self._x

            def y(self):
                return self._y

            def theta(self):
                return self._theta

        class Values:
            def __init__(self):
                self._poses = {}

            def insert(self, node_id, pose):
                self._poses[node_id] = pose

            def atPose2(self, node_id):
                return self._poses[node_id]

        class Graph:
            def add(self, _factor):
                return None

        class Optimizer:
            def __init__(self, _graph, initial):
                self.initial = initial

            def optimize(self):
                return self.initial

        gtsam_stub.Pose2 = Pose2
        gtsam_stub.Values = Values
        gtsam_stub.NonlinearFactorGraph = Graph
        gtsam_stub.LevenbergMarquardtOptimizer = Optimizer
        gtsam_stub.PriorFactorPose2 = lambda *args, **kwargs: ("prior", args, kwargs)
        gtsam_stub.BetweenFactorPose2 = lambda *args, **kwargs: ("between", args, kwargs)
        gtsam_stub.noiseModel = types.SimpleNamespace(
            Gaussian=types.SimpleNamespace(Covariance=lambda covariance: covariance)
        )
        sys.modules["gtsam"] = gtsam_stub

    try:
        import open3d  # noqa: F401
    except ModuleNotFoundError:
        open3d_stub = types.ModuleType("open3d")

        class PointCloud:
            def __init__(self):
                self.points = None

        class ICPResult:
            fitness = 0.0
            inlier_rmse = 1.0
            transformation = np.eye(4)

        open3d_stub.geometry = types.SimpleNamespace(PointCloud=PointCloud)
        open3d_stub.utility = types.SimpleNamespace(Vector3dVector=lambda points: points)
        open3d_stub.pipelines = types.SimpleNamespace(
            registration=types.SimpleNamespace(
                registration_icp=lambda *args, **kwargs: ICPResult(),
                TransformationEstimationPointToPoint=lambda: object(),
                ICPConvergenceCriteria=lambda **kwargs: kwargs,
            )
        )
        sys.modules["open3d"] = open3d_stub


install_optional_slam_stubs()

import robot_code  # noqa: E402  # Required for unpickling RobotSensorSignal logs.
from FinalProject.robot_python import parameters  # noqa: E402
from FinalProject.robot_python.app.slam import SLAMSystem  # noqa: E402
from FinalProject.robot_python.config.config import Config  # noqa: E402
from FinalProject.robot_python.data_handling import get_file_data  # noqa: E402
from FinalProject.robot_python.data_types import (  # noqa: E402
    EncoderState,
    LandmarkObservation,
    LidarScan,
    Pose2D,
    RobotFrame,
)


DEFAULT_INITIAL_POSE = Pose2D(x=0.20, y=0.20, theta=math.pi / 2)


class OfflineRunner:
    """Replay GUI/DataLogger pickle files through the SLAM pipeline."""

    def __init__(
        self,
        log_pickle_path: str | Path,
        config: Config | None = None,
        initial_pose: Pose2D = DEFAULT_INITIAL_POSE,
        run_backend: bool = True,
        replay_delay_s: float = 0.0,
    ) -> None:
        self.log_pickle_path = Path(log_pickle_path)
        self.config = config or Config()
        self.slam = SLAMSystem(self.config)
        self.slam.ekf.reset(initial_pose)
        self.run_backend_enabled = run_backend
        self.replay_delay_s = replay_delay_s

    def run(self) -> None:
        """Replay logged robot and camera samples in file order."""
        (
            time_list,
            robot_signal_list,
            _control_signal_list,
            camera_detection_list,
            _encoder_left_count_list,
            _encoder_right_count_list,
        ) = get_file_data(str(self.log_pickle_path))

        if not time_list or not robot_signal_list:
            raise ValueError(f"No replayable robot samples found in {self.log_pickle_path}")

        n = min(len(time_list), len(robot_signal_list))

        for index in range(n):
            timestamp = float(time_list[index])
            robot_signal = robot_signal_list[index]
            robot_frame = self._robot_frame_from_logged_signal(robot_signal, timestamp)
            front_output = self.slam.run_frontend(robot_frame)

            if index < len(camera_detection_list):
                for obs in self._landmark_observations_from_camera_rows(
                    camera_detection_list[index],
                    timestamp,
                ):
                    self.slam.add_landmark_observation(obs)

            if self.run_backend_enabled and front_output.created_keyframe:
                self.slam.run_backend()

            if self.replay_delay_s > 0.0:
                time.sleep(self.replay_delay_s)

        if self.run_backend_enabled:
            self._force_backend_once()
        
        summary = self.slam.pose_graph.get_graph_summary()
        print("\n========== OFFLINE RUN SUMMARY ==========")
        print(f"log: {self.log_pickle_path}")
        print(f"graph summary: {summary}")
        print(f"keyframes: {self.slam._debug_keyframe_count}")
        print(
            "landmarks: "
            f"accepted={self.slam._debug_landmark_accept_count}, "
            f"rejected={self.slam._debug_landmark_reject_count}"
        )
        print(
            "loop closures: "
            f"attempted={self.slam._debug_loop_attempt_count}, "
            f"accepted={self.slam._debug_loop_accept_count}, "
            f"rejected={self.slam._debug_loop_reject_count}"
        )
        print("========================================\n")

    def plot(
        self,
        output_path: str | Path | None = None,
        show: bool = True,
        ground_truth_final_pos: tuple[float, float] | None = None,
        ground_truth_trajectory: list[tuple[float, float]] | None = None,
    ) -> None:
        """Plot room geometry, SLAM trajectory, and live/optimized map points.

        ground_truth_final_pos: (x, y) in metres. When provided, the true final
        position is marked and final-position errors for EKF and optimised
        trajectory are printed beneath the plot.

        ground_truth_trajectory: manually specified trajectory points in cm.
        """
        import matplotlib.pyplot as plt

        local_map = self.slam.get_current_local_map()
        optimized = self.slam.get_current_global_trajectory()

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.set_facecolor("white")
        fig.patch.set_facecolor("white")

        for wall in parameters.walls:
            x1, y1 = parameters.corners[wall[0]]
            x2, y2 = parameters.corners[wall[1]]
            ax.plot([x1, x2], [y1, y2], color="black", linewidth=2.0, label="_nolegend_")

        for tag_id, (tx, ty) in parameters.tags.items():
            ax.scatter(tx, ty, color="black", s=40, marker="D", zorder=5)
            ax.text(tx + 3, ty + 3, f"T{tag_id}", fontsize=7, color="black")

        map_points = (
            local_map.optimized_map_points
            if local_map.optimized_map_points
            else local_map.map_points
        )
        if map_points:
            xs = [p[0] * 100.0 for p in map_points]
            ys = [p[1] * 100.0 for p in map_points]
            color = "#cc0000" if local_map.optimized_map_points else "#00b4cc"
            label = "optimized map" if local_map.optimized_map_points else "frontend map"
            ax.scatter(xs, ys, s=6.0, c=color, alpha=0.6, linewidths=0, label=label)

        ekf_final_cm: tuple[float, float] | None = None
        if local_map.trajectory:
            tx = [estimate.pose.x * 100.0 for estimate in local_map.trajectory]
            ty = [estimate.pose.y * 100.0 for estimate in local_map.trajectory]
            ax.plot(tx, ty, color="#0055cc", linewidth=1.4, label="EKF trajectory")
            ax.scatter(tx[0], ty[0], color="#333333", s=50, zorder=6, label="start")
            ax.scatter(tx[-1], ty[-1], color="#cc3300", s=50, zorder=6, label="end")
            ekf_final_cm = (tx[-1], ty[-1])

        opt_final_cm: tuple[float, float] | None = None
        if optimized:
            ids = sorted(optimized)
            ox = [optimized[node_id].x * 100.0 for node_id in ids]
            oy = [optimized[node_id].y * 100.0 for node_id in ids]
            ax.plot(ox, oy, color="#22bb00", linewidth=1.8, label="optimized trajectory")
            opt_final_cm = (ox[-1], oy[-1])

        if local_map.landmark_observations:
            lx = [obs.robot_pose_meas.x * 100.0 for obs in local_map.landmark_observations]
            ly = [obs.robot_pose_meas.y * 100.0 for obs in local_map.landmark_observations]
            ax.scatter(lx, ly, color="#7700cc", s=18, alpha=0.6, label="camera pose measurements")

        gt_traj_x: list[float] = []
        gt_traj_y: list[float] = []
        if ground_truth_trajectory:
            gt_traj_x = [p[0] for p in ground_truth_trajectory]
            gt_traj_y = [p[1] for p in ground_truth_trajectory]
            ax.plot(
                gt_traj_x,
                gt_traj_y,
                color="purple",
                linestyle="--",
                linewidth=2.0,
                label="estimated ground truth trajectory",
            )
            ax.scatter(gt_traj_x[0], gt_traj_y[0], color="green", s=65, zorder=7, label="gt start")
            ax.scatter(gt_traj_x[-1], gt_traj_y[-1], color="purple", s=65, zorder=7, label="gt end")

        error_lines: list[str] = []
        gt_final_cm: tuple[float, float] | None = None
        if ground_truth_final_pos is not None:
            gt_final_cm = (ground_truth_final_pos[0] * 100.0, ground_truth_final_pos[1] * 100.0)
        elif ground_truth_trajectory:
            gt_final_cm = ground_truth_trajectory[-1]

        if gt_final_cm is not None:
            gt_x_cm, gt_y_cm = gt_final_cm
            ax.scatter(gt_x_cm, gt_y_cm, color="black", s=120, marker="*", zorder=7, label="ground truth final")

            if ekf_final_cm is not None:
                ekf_err = math.hypot(ekf_final_cm[0] - gt_x_cm, ekf_final_cm[1] - gt_y_cm)
                error_lines.append(f"EKF final pos. error:       {ekf_err:.1f} cm  ({ekf_err / 100:.3f} m)")

            if opt_final_cm is not None:
                opt_err = math.hypot(opt_final_cm[0] - gt_x_cm, opt_final_cm[1] - gt_y_cm)
                error_lines.append(f"Optimized final pos. error: {opt_err:.1f} cm  ({opt_err / 100:.3f} m)")

        all_x = [p[0] for p in parameters.corners.values()] + [p[0] for p in parameters.tags.values()]
        all_y = [p[1] for p in parameters.corners.values()] + [p[1] for p in parameters.tags.values()]
        all_x += gt_traj_x
        all_y += gt_traj_y
        ax.set_xlim(min(all_x) - 20, max(all_x) + 20)
        ax.set_ylim(min(all_y) - 20, max(all_y) + 20)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x (cm)", color="black")
        ax.set_ylabel("y (cm)", color="black")
        ax.tick_params(colors="black")
        ax.legend(facecolor="white", edgecolor="#aaaaaa", labelcolor="black")
        ax.set_title(self.log_pickle_path.name, color="black")

        if error_lines:
            ax.text(
                0.02, 0.98,
                "\n".join(error_lines),
                transform=ax.transAxes,
                ha="left", va="top",
                fontsize=9, family="monospace", color="black",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f0f0", edgecolor="#888888", alpha=0.9),
                zorder=10,
            )

        fig.tight_layout()

        if output_path is not None:
            fig.savefig(output_path, dpi=160, bbox_inches="tight")
            print(f"Saved offline SLAM plot to {output_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

    def _robot_frame_from_logged_signal(self, sensor, timestamp: float) -> RobotFrame:
        encoder = EncoderState(
            left_ticks=int(sensor.encoder_left_counts),
            right_ticks=int(sensor.encoder_right_counts),
            timestamp=timestamp,
        )

        lidar_scan = None
        count = min(
            int(getattr(sensor, "num_lidar_rays", 0)),
            len(getattr(sensor, "angles", [])),
            len(getattr(sensor, "distances", [])),
        )
        if count > 0:
            ranges = [sensor.convert_hardware_distance(sensor.distances[i]) for i in range(count)]
            angles = [sensor.convert_hardware_angle(sensor.angles[i]) for i in range(count)]
            lidar_scan = LidarScan(
                ranges=ranges,
                angles=angles,
                timestamp=timestamp,
                frame_id="offline_hardware_lidar",
            )

        return RobotFrame(timestamp=timestamp, encoder=encoder, lidar_scan=lidar_scan)

    def _landmark_observations_from_camera_rows(
        self,
        camera_rows,
        timestamp: float,
    ) -> list[LandmarkObservation]:
        observations: list[LandmarkObservation] = []
        for pose_row in self._iter_camera_pose_rows(camera_rows):
            obs = self._landmark_observation_from_pose_row(pose_row, timestamp)
            if obs is not None:
                observations.append(obs)
        return observations

    def _iter_camera_pose_rows(self, camera_rows):
        if camera_rows is None:
            return
        if isinstance(camera_rows, (list, tuple)) and camera_rows:
            first = camera_rows[0]
            if isinstance(first, (list, tuple)):
                for pose_row in camera_rows:
                    if pose_row is not None and len(pose_row) >= 7:
                        yield pose_row
                return
            if len(camera_rows) >= 7:
                yield camera_rows

    def _landmark_observation_from_pose_row(
        self,
        pose_row,
        timestamp: float,
    ) -> LandmarkObservation | None:
        marker_id = int(pose_row[0])
        if marker_id not in parameters.tags:
            return None

        x_cm = float(pose_row[1])
        z_cm = float(pose_row[3])
        yaw_rad = math.radians(float(pose_row[4]))

        marker_x_m = parameters.tags[marker_id][0] / 100.0
        marker_y_m = parameters.tags[marker_id][1] / 100.0
        dx_robot_m = z_cm / 100.0
        dy_robot_m = -x_cm / 100.0

        print(
            f"[LANDMARK_RAW] marker={marker_id} "
            f"tag_map=({marker_x_m:.3f}, {marker_y_m:.3f}) "
            f"cam_row=(x_cm={x_cm:.3f}, z_cm={z_cm:.3f}, yaw_deg={float(pose_row[4]):.3f}) "
            f"robot_rel=(dx={dx_robot_m:.3f}, dy={dy_robot_m:.3f}) "
            f"yaw_raw_rad={yaw_rad:.3f}"
        )

        yaw_a = normalize_angle(-yaw_rad)
        yaw_b = normalize_angle(yaw_rad)
        yaw_c = normalize_angle(math.pi - yaw_rad)

        print(
            f"[LANDMARK_YAW_TEST] marker={marker_id} "
            f"yaw_a_neg={yaw_a:.3f} "
            f"yaw_b_pos={yaw_b:.3f} "
            f"yaw_c_pi_minus={yaw_c:.3f}"
        )

        cand1_x = marker_x_m - dx_robot_m
        cand1_y = marker_y_m - dy_robot_m
        cand2_x = marker_x_m + dx_robot_m
        cand2_y = marker_y_m + dy_robot_m

        print(
            f"[LANDMARK_POS_TEST] marker={marker_id} "
            f"cand1_minus=({cand1_x:.3f}, {cand1_y:.3f}) "
            f"cand2_plus=({cand2_x:.3f}, {cand2_y:.3f})"
        )

        robot_pose = Pose2D(
            x=marker_x_m + dx_robot_m,
            y=marker_y_m - dy_robot_m,
            theta=normalize_angle(math.pi - yaw_rad),
        )

        print(
            f"[LANDMARK_POSE_TEST] marker={marker_id} "
            f"robot_pose_test=({robot_pose.x:.3f}, {robot_pose.y:.3f}, {robot_pose.theta:.3f})"
        )

        covariance = self._camera_covariance_from_pose_row(pose_row)
        return LandmarkObservation(
            timestamp=timestamp,
            marker_id=marker_id,
            robot_pose_meas=robot_pose,
            covariance=covariance,
            quality={"source": "offline_camera_log"},
        )

    def _camera_covariance_from_pose_row(self, pose_row, rms_pixels: float = 0.6862) -> np.ndarray:
        fx = parameters.camera_matrix[0, 0]
        fy = parameters.camera_matrix[1, 1]
        f_mean = (fx + fy) / 2.0
        z_m = abs(float(pose_row[3]) / 100.0)
        sigma_xy = max((rms_pixels * z_m) / f_mean, 0.02)
        return np.diag([sigma_xy**2, sigma_xy**2, 0.05**2]).astype(np.float64)

    def _force_backend_once(self) -> dict[int, Pose2D] | None:
        if not self.slam.pose_graph.nodes:
            return None
        result = self.slam.optimizer.optimize(self.slam.pose_graph)
        self.slam._last_optimized_trajectory = result
        self.slam.latest_optimization_result = result
        self.slam._rebuild_optimized_map(result)
        return result


def _latest_pickle() -> Path:
    candidates = sorted(_PYTHON_DIR.joinpath("data").rglob("*.pkl"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError("No .pkl files found under robot_python/data")
    return candidates[-1]


def _parse_map_waypoint(text: str) -> tuple[float, float]:
    """Parse a waypoint in cm from a corner label, tag label, or x,y string."""
    value = text.strip()
    if value in parameters.corners:
        x, y = parameters.corners[value]
        return float(x), float(y)

    if value.upper().startswith("T") and value[1:].isdigit():
        tag_id = int(value[1:])
        if tag_id in parameters.tags:
            x, y = parameters.tags[tag_id]
            return float(x), float(y)

    parts = value.split(",")
    if len(parts) == 2:
        return float(parts[0]), float(parts[1])

    raise ValueError(f"Unknown ground-truth trajectory waypoint: {text}")


def _make_robot_like_polyline(
    waypoints: list[tuple[float, float]],
    step_cm: float,
    wobble_cm: float,
) -> list[tuple[float, float]]:
    """Create a segmented non-smoothed trajectory through manual waypoints."""
    if len(waypoints) < 2:
        return waypoints[:]

    path: list[tuple[float, float]] = [waypoints[0]]
    for segment_index, (start, end) in enumerate(zip(waypoints, waypoints[1:])):
        x0, y0 = start
        x1, y1 = end
        dx = x1 - x0
        dy = y1 - y0
        length = math.hypot(dx, dy)
        if length <= 1e-9:
            continue

        nx = -dy / length
        ny = dx / length
        steps = max(1, int(math.ceil(length / max(step_cm, 1e-6))))

        for i in range(1, steps + 1):
            t = i / steps
            base_x = x0 + t * dx
            base_y = y0 + t * dy
            if i == steps:
                path.append((x1, y1))
                continue

            zigzag = -1.0 if (i + segment_index) % 2 == 0 else 1.0
            taper = math.sin(math.pi * t)
            offset = wobble_cm * zigzag * taper
            path.append((base_x + offset * nx, base_y + offset * ny))

    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a DataLogger pickle through SLAM offline.")
    parser.add_argument("--log_pickle", nargs="?", type=Path, default="C:\\Users\\lukelo\\Desktop\\Spring 2026\\Robots\\NYU_ROB_GY_6213\\FinalProject\\robot_python\\data\\final_trials\\robot_data_0_0_05_05_26_05_29_29.pkl")
    parser.add_argument("--output", type=Path, default=_PYTHON_DIR / "offline_slam_plot.png")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--no-backend", action="store_true")
    parser.add_argument("--delay", type=float, default=0.0, help="Optional sleep between replayed samples.")
    parser.add_argument(
        "--gt-final", nargs=2, type=float, metavar=("X", "Y"), default=None,
        help="Ground truth final position in metres, e.g. --gt-final 1.20 0.85",
    )
    parser.add_argument(
        "--gt-traj",
        nargs="+",
        default=None,
        help="Manual ground-truth trajectory waypoints in cm, e.g. --gt-traj 20,20 22,170 T7.",
    )
    parser.add_argument("--gt-traj-step-cm", type=float, default=12.0)
    parser.add_argument("--gt-traj-wobble-cm", type=float, default=2.5)
    args = parser.parse_args()

    gt_final = tuple(args.gt_final) if args.gt_final is not None else None
    gt_trajectory = None
    if args.gt_traj is not None:
        gt_waypoints = [_parse_map_waypoint(point) for point in args.gt_traj]
        gt_trajectory = _make_robot_like_polyline(
            gt_waypoints,
            step_cm=args.gt_traj_step_cm,
            wobble_cm=args.gt_traj_wobble_cm,
        )

    log_pickle = args.log_pickle or _latest_pickle()
    runner = OfflineRunner(
        log_pickle_path=log_pickle,
        run_backend=not args.no_backend,
        replay_delay_s=args.delay,
    )
    runner.run()
    runner.plot(
        output_path=args.output,
        show=not args.no_show,
        ground_truth_final_pos=gt_final,
        ground_truth_trajectory=gt_trajectory,
    )


if __name__ == "__main__":
    main()
