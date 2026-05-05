"""Top-level SLAM system wiring for live and offline runners."""

from __future__ import annotations

import math
import time

from FinalProject.robot_python.backend.optimizer import GraphOptimizer
from FinalProject.robot_python.backend.pose_graph import PoseGraphManager
from FinalProject.robot_python.config.config import Config
from FinalProject.robot_python.data_types import (
    FrontendOutput,
    LandmarkObservation,
    LocalMapState,
    Pose2D,
    PoseEstimate,
    RobotFrame,
)
from FinalProject.robot_python.frontend.ekf import EKFLocalizer
from FinalProject.robot_python.frontend.lidar_matching import LidarMatcher
from FinalProject.robot_python.frontend.motion_model import DifferentialDriveMotionModel


class SLAMSystem:
    """Coordinate front-end local estimation and back-end pose graph updates."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()
        self.motion_model = DifferentialDriveMotionModel(self.config.robot)
        self.lidar_matcher = LidarMatcher(self.config.robot, self.config.frontend)
        self.ekf = EKFLocalizer(self.config.frontend, self.motion_model)
        self.pose_graph = PoseGraphManager()
        self.optimizer = GraphOptimizer(self.config.backend)
        self.local_map = LocalMapState()

        self._prev_robot_frame: RobotFrame | None = None
        self._prev_keyframe_robot_frame: RobotFrame | None = None
        self._latest_keyframe_id: int | None = None
        self._last_optimized_trajectory: dict[int, Pose2D] | None = None
        self._new_keyframes_since_optimization = 0
        self._last_optimization_time = time.time()

        self.latest_frontend_output: FrontendOutput | None = None
        self.latest_optimization_result: dict[int, Pose2D] | None = None

    def run_frontend(self, robot_frame: RobotFrame) -> FrontendOutput:
        """Run front-end prediction, LiDAR correction, and local map update."""
        odom_motion = None
        lidar_motion = None

        # EKF prediction from wheel odometry, replacing the paper's IMU model.
        if self._prev_robot_frame is not None:
            pose_estimate = self.ekf.predict(self._prev_robot_frame.encoder, robot_frame.encoder)
            odom_motion = self.ekf.last_odom_motion
        else:
            pose_estimate = self.ekf.get_state()

        # EKF correction from LiDAR scan-to-scan registration, matching the
        # paper's LiDAR relative-pose observation model.
        if self._prev_robot_frame and self._prev_robot_frame.lidar_scan and robot_frame.lidar_scan:
            lidar_motion = self.lidar_matcher.match_scans(
                self._prev_robot_frame.lidar_scan,
                robot_frame.lidar_scan,
                init_guess=pose_estimate.pose,
            )
            pose_estimate = self.ekf.correct_with_lidar(lidar_motion)

        pose_estimate.timestamp = robot_frame.timestamp
        created_keyframe = self._maybe_add_keyframe(pose_estimate, odom_motion, lidar_motion, robot_frame)
        self._update_local_map(pose_estimate, robot_frame, self._latest_keyframe_id if created_keyframe else None)
        self._prev_robot_frame = robot_frame

        output = FrontendOutput(
            timestamp=robot_frame.timestamp,
            pose_estimate=pose_estimate,
            odom_motion=odom_motion,
            lidar_motion=lidar_motion,
            created_keyframe=created_keyframe,
            keyframe_id=self._latest_keyframe_id if created_keyframe else None,
        )
        self.latest_frontend_output = output
        return output

    def add_landmark_observation(self, obs: LandmarkObservation) -> bool:
        """Attach one processed landmark observation to the nearest keyframe."""
        node_id = self.pose_graph.find_closest_node_by_time(obs.timestamp)
        if node_id is None or not self._is_close_landmark_keyframe_match(node_id, obs.timestamp):
            return False

        self.pose_graph.add_landmark_factor(node_id, obs)
        self.local_map.landmark_observations.append(obs)
        return True

    def run_backend(self) -> dict[int, Pose2D] | None:
        """Run back-end optimization occasionally and cache the latest result."""
        elapsed_time = time.time() - self._last_optimization_time
        if not self.optimizer.should_optimize(self._new_keyframes_since_optimization, elapsed_time):
            return None

        self._last_optimized_trajectory = self.optimizer.optimize(self.pose_graph)
        self.latest_optimization_result = self._last_optimized_trajectory
        self._rebuild_optimized_map(self._last_optimized_trajectory)
        self._new_keyframes_since_optimization = 0
        self._last_optimization_time = time.time()
        return self._last_optimized_trajectory

    def get_current_local_pose(self) -> PoseEstimate:
        """Return the latest EKF local pose estimate."""
        return self.ekf.get_state()

    def get_current_global_trajectory(self) -> dict[int, Pose2D] | None:
        """Return the latest optimized back-end trajectory, if available."""
        return self._last_optimized_trajectory

    def get_current_local_map(self) -> LocalMapState:
        """Return the current front-end local map state."""
        return self.local_map

    def _maybe_add_keyframe(
        self,
        pose_estimate: PoseEstimate,
        odom_motion,
        lidar_motion,
        robot_frame: RobotFrame,
    ) -> bool:
        """Add a keyframe and factors to graph when thresholds indicate enough motion.
                1. Check: has the robot moved enough since the last keyframe?
                2. If yes:
                - add a new keyframe node to the pose graph
                - connect it to the previous keyframe with odometry and LiDAR factors
                - remember that a new keyframe was created
                3. If no:
                - do nothing and wait for the next frame to check again
        """
        should_add = self._latest_keyframe_id is None or self._passes_keyframe_threshold(pose_estimate.pose)
        if not should_add:
            return False

        previous_keyframe_id = self._latest_keyframe_id
        new_keyframe_id = self.pose_graph.add_keyframe(pose_estimate)
        self._latest_keyframe_id = new_keyframe_id
        self._new_keyframes_since_optimization += 1

        if previous_keyframe_id is not None and self._prev_keyframe_robot_frame is not None:
            previous_keyframe_pose = self.pose_graph.nodes[previous_keyframe_id].pose.pose
            keyframe_odom_motion, _ = self.motion_model.predict_motion(
                previous_keyframe_pose,
                self._prev_keyframe_robot_frame.encoder,
                robot_frame.encoder,
            )
            self.pose_graph.add_odometry_factor(previous_keyframe_id, new_keyframe_id, keyframe_odom_motion)

            if self._prev_keyframe_robot_frame.lidar_scan is not None and robot_frame.lidar_scan is not None:
                keyframe_lidar_motion = self.lidar_matcher.match_scans(
                    self._prev_keyframe_robot_frame.lidar_scan,
                    robot_frame.lidar_scan,
                    init_guess=Pose2D(
                        keyframe_odom_motion.dx,
                        keyframe_odom_motion.dy,
                        keyframe_odom_motion.dtheta,
                    ),
                )
                self.pose_graph.add_lidar_factor(previous_keyframe_id, new_keyframe_id, keyframe_lidar_motion)
        ##not sure yet about loop closure factors::
        #self._maybe_add_loop_closure_factor(new_keyframe_id, robot_frame)
        self._prev_keyframe_robot_frame = robot_frame
        return True

    def _passes_keyframe_threshold(self, pose: Pose2D) -> bool:
        """Check translation and rotation thresholds against the latest keyframe."""
        if self._latest_keyframe_id is None:
            return True
        latest_node = self.pose_graph.nodes[self._latest_keyframe_id]
        latest_pose = latest_node.pose.pose
        translation = pose.distance_to(latest_pose)
        rotation = abs(pose.theta - latest_pose.theta)
        return (
            translation >= self.config.frontend.keyframe_translation_m
            or rotation >= self.config.frontend.keyframe_rotation_rad
        )

    def _update_local_map(
        self,
        pose_estimate: PoseEstimate,
        robot_frame: RobotFrame,
        keyframe_id: int | None,
    ) -> None:
        """Update the local map from the corrected global pose estimate."""
        pose = pose_estimate.pose

        # Store the corrected pose history.
        self.local_map.trajectory.append(pose_estimate)

        # Convert local LiDAR rays into global/map-frame points using the
        # current global robot pose.
        if robot_frame.lidar_scan is not None:
            scan_points = []

            for r, angle in zip(robot_frame.lidar_scan.ranges, robot_frame.lidar_scan.angles):
                if r <= 0:
                    continue

                global_angle = pose.theta + angle
                x_map = pose.x + r * math.cos(global_angle)
                y_map = pose.y + r * math.sin(global_angle)

                scan_points.append((x_map, y_map))

            self.local_map.map_points.extend(scan_points)

        # Keep raw keyframe scans so the optimized/global map can be rebuilt
        # later from backend-corrected keyframe poses.
        if keyframe_id is not None and robot_frame.lidar_scan is not None:
            self.local_map.keyframe_scans[keyframe_id] = robot_frame.lidar_scan

    def _rebuild_optimized_map(self, optimized_poses: dict[int, Pose2D]) -> None:
        """Rebuild backend-corrected map points from raw keyframe scans."""
        optimized_points = []

        for keyframe_id, scan in self.local_map.keyframe_scans.items():
            pose = optimized_poses.get(keyframe_id)
            if pose is None:
                continue

            for r, angle in zip(scan.ranges, scan.angles):
                if r <= 0:
                    continue

                global_angle = pose.theta + angle
                x_map = pose.x + r * math.cos(global_angle)
                y_map = pose.y + r * math.sin(global_angle)

                optimized_points.append((x_map, y_map))

        self.local_map.optimized_map_points = optimized_points

    def _is_close_landmark_keyframe_match(self, node_id: int, timestamp: float) -> bool:
        """Check whether a landmark observation is close enough to a keyframe time."""
        node_timestamp = self.pose_graph.timestamp_by_node[node_id]
        return abs(node_timestamp - timestamp) <= self.config.frontend.max_landmark_keyframe_time_diff_s

    def _maybe_add_loop_closure_factor(self, new_keyframe_id: int, robot_frame: RobotFrame) -> None:
        """Try to add an ICP-based loop-closure factor for a new keyframe."""
        if robot_frame.lidar_scan is None:
            return

        candidate_id = self._find_loop_closure_candidate(new_keyframe_id)
        if candidate_id is None:
            return

        candidate_scan = self.local_map.keyframe_scans.get(candidate_id)
        if candidate_scan is None:
            return

        # TODO: Replace this with a dedicated loop-closure verification step:
        # global candidate search, ICP with a better initial guess, residual
        # gating, covariance estimation, and false-positive rejection.
        loop_motion = self.lidar_matcher.match_scans(
            candidate_scan,
            robot_frame.lidar_scan,
            init_guess=None,
        )
        loop_motion.source = "lidar_icp_loop_closure"

        quality = loop_motion.quality if loop_motion.quality is not None else 0.0
        if quality >= self.config.backend.loop_closure_min_match_quality:
            self.pose_graph.add_loop_closure_factor(candidate_id, new_keyframe_id, loop_motion)

    def _find_loop_closure_candidate(self, new_keyframe_id: int) -> int | None:
        """Find an older nearby keyframe to test as a loop-closure candidate."""
        new_pose = self.pose_graph.nodes[new_keyframe_id].pose.pose
        best_candidate_id: int | None = None
        best_distance = float("inf")

        for candidate_id, node in self.pose_graph.nodes.items():
            if candidate_id == new_keyframe_id:
                continue
            if abs(new_keyframe_id - candidate_id) < self.config.backend.loop_closure_min_keyframe_separation:
                continue
            if candidate_id not in self.local_map.keyframe_scans:
                continue

            distance = new_pose.distance_to(node.pose.pose)
            if distance <= self.config.backend.loop_closure_candidate_radius_m and distance < best_distance:
                best_candidate_id = candidate_id
                best_distance = distance

        return best_candidate_id
