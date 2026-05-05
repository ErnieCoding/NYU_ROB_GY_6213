"""Top-level SLAM system wiring for live and offline runners."""

from __future__ import annotations

import math
import time
import numpy as np

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
    normalize_angle
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

        self._frontend_scan_counter = 0
        self._last_backend_request_time = time.time()

        self._debug_keyframe_count = 0

        self._debug_landmark_accept_count = 0
        self._debug_landmark_reject_count = 0

        self._debug_loop_attempt_count = 0
        self._debug_loop_accept_count = 0
        self._debug_loop_reject_count = 0

    def run_frontend(self, robot_frame: RobotFrame) -> FrontendOutput:
        odom_motion = None
        lidar_motion = None

        if robot_frame.lidar_scan is not None:
            self._frontend_scan_counter += 1

        prev_corrected_pose = self.local_map.trajectory[-1].pose if self.local_map.trajectory else None

        if self._prev_robot_frame is not None:
            pose_estimate = self.ekf.predict(self._prev_robot_frame.encoder, robot_frame.encoder)
            odom_motion = self.ekf.last_odom_motion
        else:
            pose_estimate = self.ekf.get_state()

        if self._should_run_lidar_frontend_update(robot_frame, prev_corrected_pose, odom_motion):
            lidar_motion = self.lidar_matcher.match_scan_to_submap(
                curr_scan=robot_frame.lidar_scan,
                predicted_pose=pose_estimate.pose,
                prev_pose=prev_corrected_pose,
            )

            min_quality = float(getattr(self.config.frontend, "lidar_frontend_accept_quality", 0.35))
            if lidar_motion is not None and (lidar_motion.quality or 0.0) >= min_quality:
                pose_estimate = self.ekf.correct_with_lidar(lidar_motion)
            else:
                lidar_motion = None

        pose_estimate.timestamp = robot_frame.timestamp

        created_keyframe = self._maybe_add_keyframe(
            pose_estimate, odom_motion, lidar_motion, robot_frame
        )

        self._update_local_map(
            pose_estimate,
            robot_frame,
            self._latest_keyframe_id if created_keyframe else None,
        )

        if robot_frame.lidar_scan is not None and self._should_update_submap(created_keyframe, lidar_motion):
            self.lidar_matcher.update_submap(robot_frame.lidar_scan, pose_estimate.pose)

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

        if node_id is None:
            self._debug_landmark_reject_count += 1
            print(
                f"[LANDMARK][REJECT] marker={obs.marker_id} "
                f"t={obs.timestamp:.3f} reason=no_keyframe"
            )
            return False

        dt = abs(self.pose_graph.timestamp_by_node[node_id] - obs.timestamp)
        if not self._is_close_landmark_keyframe_match(node_id, obs.timestamp):
            self._debug_landmark_reject_count += 1
            print(
                f"[LANDMARK][REJECT] marker={obs.marker_id} "
                f"t={obs.timestamp:.3f} node={node_id} dt={dt:.3f}s "
                f"reason=time_mismatch"
            )
            return False

        node_pose = self.pose_graph.nodes[node_id].pose.pose

        dx = obs.robot_pose_meas.x - node_pose.x
        dy = obs.robot_pose_meas.y - node_pose.y
        dtheta = normalize_angle(obs.robot_pose_meas.theta - node_pose.theta)

        pos_err = math.hypot(dx, dy)
        ang_err = abs(dtheta)

        max_pos_err = float(
            getattr(self.config.frontend, "landmark_max_position_error_m", 0.50)
        )
        max_ang_err = float(
            getattr(self.config.frontend, "landmark_max_angle_error_rad", math.radians(35.0))
        )

        if pos_err > max_pos_err or ang_err > max_ang_err:
            self._debug_landmark_reject_count += 1
            print(
                f"[LANDMARK][REJECT] marker={obs.marker_id} "
                f"t={obs.timestamp:.3f} node={node_id} dt={dt:.3f}s "
                f"reason=innovation_gate pos_err={pos_err:.3f} ang_err={ang_err:.3f} "
                f"graph_pose=({node_pose.x:.3f}, {node_pose.y:.3f}, {node_pose.theta:.3f}) "
                f"meas_pose=({obs.robot_pose_meas.x:.3f}, {obs.robot_pose_meas.y:.3f}, {obs.robot_pose_meas.theta:.3f})"
            )
            return False


        if obs.marker_id != 2:
            self._debug_landmark_reject_count += 1
            print(
                f"[LANDMARK][REJECT] marker={obs.marker_id} "
                f"t={obs.timestamp:.3f} node={node_id} "
                f"reason=marker_disabled_for_debug"
            )
            return False
        
        landmark_cov = np.array(obs.covariance, dtype=float, copy=True)
        landmark_cov[0, 0] = max(landmark_cov[0, 0], 0.25**2)
        landmark_cov[1, 1] = max(landmark_cov[1, 1], 0.25**2)
        landmark_cov[2, 2] = max(landmark_cov[2, 2], 0.60**2)

        inflated_obs = LandmarkObservation(
            timestamp=obs.timestamp,
            marker_id=obs.marker_id,
            robot_pose_meas=obs.robot_pose_meas,
            covariance=landmark_cov,
            quality=obs.quality,
        )

        print(
            f"[LANDMARK][ADD_FACTOR] marker={inflated_obs.marker_id} node={node_id} "
            f"pos_err={pos_err:.3f} ang_err={ang_err:.3f} "
            f"cov_diag=({inflated_obs.covariance[0,0]:.6f}, "
            f"{inflated_obs.covariance[1,1]:.6f}, "
            f"{inflated_obs.covariance[2,2]:.6f})"
        )

        self.pose_graph.add_landmark_factor(node_id, inflated_obs)
        self.local_map.landmark_observations.append(obs)
        self._debug_landmark_accept_count += 1

        print(
            f"[LANDMARK][ACCEPT] marker={obs.marker_id} "
            f"t={obs.timestamp:.3f} node={node_id} dt={dt:.3f}s "
            f"pos_err={pos_err:.3f} ang_err={ang_err:.3f} "
            f"pose=({obs.robot_pose_meas.x:.3f}, {obs.robot_pose_meas.y:.3f}, {obs.robot_pose_meas.theta:.3f})"
        )
        return True

    def run_backend(self) -> dict[int, Pose2D] | None:
        """Run back-end optimization occasionally and cache the latest result."""
        elapsed_time = time.time() - self._last_optimization_time
        if not self.optimizer.should_optimize(self._new_keyframes_since_optimization, elapsed_time):
            return None

        summary_before = self.pose_graph.get_graph_summary()
        print(
            "[BACKEND][START] "
            f"new_keyframes={self._new_keyframes_since_optimization} "
            f"elapsed={elapsed_time:.2f}s "
            f"summary={summary_before} "
            f"landmarks(acc/rej)=({self._debug_landmark_accept_count}/{self._debug_landmark_reject_count}) "
            f"loops(attempt/acc/rej)=({self._debug_loop_attempt_count}/"
            f"{self._debug_loop_accept_count}/{self._debug_loop_reject_count})"
        )

        self._last_optimized_trajectory = self.optimizer.optimize(self.pose_graph)
        self.latest_optimization_result = self._last_optimized_trajectory
        self._rebuild_optimized_map(self._last_optimized_trajectory)

        summary_after = self.pose_graph.get_graph_summary()
        print(f"[BACKEND][DONE] summary={summary_after}")

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
        self._debug_keyframe_count += 1
        print(
            f"[KEYFRAME] id={new_keyframe_id} total={self._debug_keyframe_count} "
            f"pose=({pose_estimate.pose.x:.3f}, {pose_estimate.pose.y:.3f}, {pose_estimate.pose.theta:.3f})"
        )

        if previous_keyframe_id is not None and self._prev_keyframe_robot_frame is not None:
            previous_keyframe_pose = self.pose_graph.nodes[previous_keyframe_id].pose.pose
            keyframe_odom_motion, _ = self.motion_model.predict_motion(
                previous_keyframe_pose,
                self._prev_keyframe_robot_frame.encoder,
                robot_frame.encoder,
            )
            self.pose_graph.add_odometry_factor(previous_keyframe_id, new_keyframe_id, keyframe_odom_motion)
            print(
                f"[GRAPH][ODOM] {previous_keyframe_id}->{new_keyframe_id} "
                f"motion=({keyframe_odom_motion.dx:.3f}, {keyframe_odom_motion.dy:.3f}, {keyframe_odom_motion.dtheta:.3f})"
            )

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

                if keyframe_lidar_motion is None:
                    print(
                        f"[GRAPH][LIDAR][REJECT] {previous_keyframe_id}->{new_keyframe_id} "
                        f"reason=no_match"
                    )
                else:
                    print(
                        f"[GRAPH][LIDAR][MATCH] {previous_keyframe_id}->{new_keyframe_id} "
                        f"quality={(keyframe_lidar_motion.quality or 0.0):.3f} "
                        f"motion=({keyframe_lidar_motion.dx:.3f}, "
                        f"{keyframe_lidar_motion.dy:.3f}, "
                        f"{keyframe_lidar_motion.dtheta:.3f})"
                    )

                    min_quality = float(
                        getattr(self.config.frontend, "lidar_keyframe_factor_min_quality", 0.45)
                    )

                    if (keyframe_lidar_motion.quality or 0.0) >= min_quality:
                        self.pose_graph.add_lidar_factor(
                            previous_keyframe_id,
                            new_keyframe_id,
                            keyframe_lidar_motion,
                        )
                        print(
                            f"[GRAPH][LIDAR][ACCEPT] {previous_keyframe_id}->{new_keyframe_id} "
                            f"quality={(keyframe_lidar_motion.quality or 0.0):.3f}"
                        )
                    else:
                        print(
                            f"[GRAPH][LIDAR][REJECT] {previous_keyframe_id}->{new_keyframe_id} "
                            f"quality={(keyframe_lidar_motion.quality or 0.0):.3f} "
                            f"threshold={min_quality:.3f}"
                        )
        self._maybe_add_loop_closure_factor(new_keyframe_id, robot_frame)
        self._prev_keyframe_robot_frame = robot_frame
        return True

    def _passes_keyframe_threshold(self, pose: Pose2D) -> bool:
        """Check translation and rotation thresholds against the latest keyframe."""
        if self._latest_keyframe_id is None:
            return True
        latest_node = self.pose_graph.nodes[self._latest_keyframe_id]
        latest_pose = latest_node.pose.pose
        translation = pose.distance_to(latest_pose)
        rotation = abs(normalize_angle(pose.theta - latest_pose.theta))
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
        pose = pose_estimate.pose
        self.local_map.trajectory.append(pose_estimate)

        if robot_frame.lidar_scan is not None:
            processed_scan, world_points = self.lidar_matcher.scan_to_world_points(
                robot_frame.lidar_scan,
                pose,
                assume_preprocessed=False,
            )
            self.local_map.map_points.extend([tuple(p) for p in world_points])

            if keyframe_id is not None:
                self.local_map.keyframe_scans[keyframe_id] = processed_scan


    def _rebuild_optimized_map(self, optimized_poses: dict[int, Pose2D]) -> None:
        optimized_points = []

        for keyframe_id, scan in self.local_map.keyframe_scans.items():
            pose = optimized_poses.get(keyframe_id)
            if pose is None:
                continue

            _, world_points = self.lidar_matcher.scan_to_world_points(
                scan,
                pose,
                assume_preprocessed=True,
            )
            optimized_points.extend([tuple(p) for p in world_points])

        self.local_map.optimized_map_points = optimized_points

    def _is_close_landmark_keyframe_match(self, node_id: int, timestamp: float) -> bool:
        """Check whether a landmark observation is close enough to a keyframe time."""
        node_timestamp = self.pose_graph.timestamp_by_node[node_id]
        return abs(node_timestamp - timestamp) <= self.config.frontend.max_landmark_keyframe_time_diff_s

    def _maybe_add_loop_closure_factor(self, new_keyframe_id: int, robot_frame: RobotFrame) -> None:
        """Try to add one robust loop-closure factor for a new keyframe."""
        print(f"[LOOP] checking new_keyframe={new_keyframe_id}")
        if robot_frame.lidar_scan is None:
            return

        candidates = self._find_loop_closure_candidates(new_keyframe_id)
        if not candidates:
            print(f"[LOOP] new_keyframe={new_keyframe_id} candidates=0")
            return

        print(f"[LOOP] new_keyframe={new_keyframe_id} candidates={candidates}")

        new_pose = self.pose_graph.nodes[new_keyframe_id].pose.pose
        best_motion = None
        best_candidate_id = None
        best_quality = 0.0

        for candidate_id in candidates:
            candidate_scan = self.local_map.keyframe_scans.get(candidate_id)
            if candidate_scan is None:
                continue

            candidate_pose = self.pose_graph.nodes[candidate_id].pose.pose

            dx_world = new_pose.x - candidate_pose.x
            dy_world = new_pose.y - candidate_pose.y
            cos_t = math.cos(candidate_pose.theta)
            sin_t = math.sin(candidate_pose.theta)

            init_guess = Pose2D(
                x=cos_t * dx_world + sin_t * dy_world,
                y=-sin_t * dx_world + cos_t * dy_world,
                theta=normalize_angle(new_pose.theta - candidate_pose.theta),
            )

            self._debug_loop_attempt_count += 1
            print(
                f"[LOOP][TRY] new={new_keyframe_id} candidate={candidate_id} "
                f"init_guess=({init_guess.x:.3f}, {init_guess.y:.3f}, {init_guess.theta:.3f})"
            )

            loop_motion = self.lidar_matcher.match_scans(
                candidate_scan,
                robot_frame.lidar_scan,
                init_guess=init_guess,
            )

            if loop_motion is None:
                self._debug_loop_reject_count += 1
                print(f"[LOOP][REJECT] new={new_keyframe_id} candidate={candidate_id} reason=no_match")
                continue

            quality = loop_motion.quality if loop_motion.quality is not None else 0.0
            min_quality = float(
                getattr(self.config.backend, "loop_closure_min_match_quality", 0.60)
            )

            print(
                f"[LOOP][MATCH] new={new_keyframe_id} candidate={candidate_id} "
                f"quality={quality:.3f} "
                f"motion=({loop_motion.dx:.3f}, {loop_motion.dy:.3f}, {loop_motion.dtheta:.3f})"
            )

            if quality >= min_quality and quality > best_quality:
                loop_motion.source = "lidar_loop_closure"
                best_motion = loop_motion
                best_candidate_id = candidate_id
                best_quality = quality

        if best_candidate_id is not None and best_motion is not None:
            self._debug_loop_accept_count += 1
            print(
                f"[LOOP][ACCEPT] new={new_keyframe_id} candidate={best_candidate_id} "
                f"quality={best_quality:.3f}"
            )
            self.pose_graph.add_loop_closure_factor(
                best_candidate_id,
                new_keyframe_id,
                best_motion,
            )

    def _find_loop_closure_candidates(self, new_keyframe_id: int) -> list[int]:
        """Return older nearby keyframes to test as loop-closure candidates."""
        new_pose = self.pose_graph.nodes[new_keyframe_id].pose.pose
        candidates: list[tuple[float, int]] = []

        max_candidates = int(getattr(self.config.backend, "loop_closure_max_candidates", 5))

        for candidate_id, node in self.pose_graph.nodes.items():
            if candidate_id == new_keyframe_id:
                continue
            if abs(new_keyframe_id - candidate_id) < self.config.backend.loop_closure_min_keyframe_separation:
                continue
            if candidate_id not in self.local_map.keyframe_scans:
                continue

            distance = new_pose.distance_to(node.pose.pose)
            if distance <= self.config.backend.loop_closure_candidate_radius_m:
                candidates.append((distance, candidate_id))

        candidates.sort(key=lambda item: item[0])
        return [candidate_id for _, candidate_id in candidates[:max_candidates]]
    
    def _should_run_lidar_frontend_update(
            self,
            robot_frame: RobotFrame,
            prev_corrected_pose: Pose2D | None,
            odom_motion,
        ) -> bool:
        if prev_corrected_pose is None:
            return False
        if robot_frame.lidar_scan is None:
            return False
        if odom_motion is None:
            return False
        if not self.lidar_matcher.has_submap():
            return False

        decimation = int(getattr(self.config.frontend, "lidar_frontend_decimation", 3))
        if decimation > 1 and (self._frontend_scan_counter % decimation) != 0:
            return False

        translation = math.hypot(odom_motion.dx, odom_motion.dy)
        rotation = abs(odom_motion.dtheta)
        min_translation = float(
            getattr(self.config.frontend, "lidar_min_translation_for_match_m", 0.03)
        )
        min_rotation = float(
            getattr(self.config.frontend, "lidar_min_rotation_for_match_rad", math.radians(2.0))
        )

        return translation >= min_translation or rotation >= min_rotation


    def _should_update_submap(
            self,
            created_keyframe: bool,
            lidar_motion,
        ) -> bool:
        if created_keyframe:
            return True
        if lidar_motion is not None:
            return True

        decimation = int(getattr(self.config.frontend, "submap_update_decimation", 2))
        return decimation <= 1 or (self._frontend_scan_counter % decimation) == 0