# Indoor SLAM Project Scaffold

This repository contains a Python scaffold for an indoor SLAM system for a differential-drive robot. The project is intentionally not a complete SLAM implementation yet. It defines the structure, data flow, placeholder classes, method signatures, docstrings, and TODOs needed for a teammate to fill in the real algorithms later.

The architecture follows a front-end/back-end SLAM design:

- The **front-end** performs local state estimation.
- The **back-end** manages a pose graph and later performs global optimization.
- Wheel odometry replaces the IMU motion model from the original paper/proposal idea.
- 2D LiDAR scan matching provides relative pose measurements.
- ArUco marker observations act as global landmark anchors in the back-end.
- Loop closures are planned as ICP scan-to-scan constraints between non-consecutive keyframes.

## High-Level Data Flow

The system has two main input streams:

1. **Robot stream**
   - Wheel encoder data
   - 2D LiDAR scan data
   - Comes from Arduino/robot UDP in live mode
   - Comes from a pickle file in offline mode

2. **Landmark observation stream**
   - ESP32-CAM image frames
   - Used for ArUco marker detection in live mode before entering SLAM
   - Comes from an OpenCV camera stream in live mode
   - Comes from saved processed `LandmarkObservation`s in offline mode

The robot stream drives the front-end:

```text
RobotFrame
  -> wheel odometry prediction
  -> LiDAR scan-to-scan matching
  -> EKF correction
  -> corrected local pose
  -> local map update
  -> possible pose graph keyframe
  -> odometry / LiDAR / loop-closure factors
```

The landmark observation stream is asynchronous:

```text
CameraFrame  (live runner only)
  -> ArUco marker detection
  -> marker pose estimation
  -> LandmarkObservation
  -> SLAMSystem
  -> attach to closest timestamped keyframe
  -> landmark factor in pose graph
```

The back-end periodically optimizes the pose graph:

```text
Pose graph nodes + factors
  -> GraphOptimizer
  -> optimized global keyframe trajectory
```

## Project Tree

```text
slam_project/
├── app/
│   ├── live.py
│   ├── offline.py
│   └── slam.py
├── config/
│   └── config.py
├── frontend/
│   ├── motion_model.py
│   ├── lidar_matching.py
│   ├── landmark_observation.py
│   └── ekf.py
├── backend/
│   ├── pose_graph.py
│   └── optimizer.py
├── evaluation/
│   ├── metrics.py
│   └── plots.py
└── data_types.py
```

## Main Orchestrator

### `slam_project/app/slam.py`

This file contains `SLAMSystem`, the main class that wires the project together.

It owns:

- `DifferentialDriveMotionModel`
- `LidarMatcher`
- `EKFLocalizer`
- `PoseGraphManager`
- `GraphOptimizer`
- `LocalMapState`

Important methods:

- `run_frontend(robot_frame)`
  - Runs wheel odometry prediction.
  - Runs LiDAR scan matching if two scans are available.
  - Runs EKF prediction/correction.
  - Updates the local map.
  - Adds pose graph keyframes and factors when needed.

- `add_landmark_observation(obs)`
  - Attaches each observation to the closest keyframe by timestamp.
  - Adds landmark factors to the pose graph.

- `run_backend()`
  - Periodically calls the graph optimizer.
  - Stores the latest optimized trajectory.

- `get_current_local_pose()`
  - Returns the current EKF pose estimate.

- `get_current_local_map()`
  - Returns the current front-end local map state.

- `get_current_global_trajectory()`
  - Returns the latest optimized global trajectory, if optimization has run.

This is the file to read first if you want to understand how all components connect.

## Shared Data Types

### `slam_project/data_types.py`

This file defines shared dataclasses used across the whole project.

Important types:

- `Pose2D`
  - Basic 2D pose: `x`, `y`, `theta`
  - Includes small helpers like `moved_by(...)` and `distance_to(...)`

- `PoseEstimate`
  - A pose plus timestamp, covariance placeholder, and source label

- `LidarScan`
  - LiDAR ranges, angles, timestamp, frame id, and optional intensities

- `EncoderState`
  - Left/right wheel encoder ticks and timestamp

- `RobotFrame`
  - One robot-side frame containing encoder data and optional LiDAR scan

- `CameraFrame`
  - One camera frame with timestamp and image, used only before live observation extraction

- `RelativeMotion`
  - Relative pose measurement: `dx`, `dy`, `dtheta`

- `LandmarkObservation`
  - Shared processed SLAM input with timestamp, marker id, robot pose measurement, covariance, and optional quality metadata

- `FrontendOutput`
  - Return value from a front-end update

- `LocalMapState`
  - Lightweight Version 1 local map container
  - Stores corrected trajectory, keyframe scans, and landmark observations

- `OptimizationResult`
  - Placeholder container for future optimizer outputs

## Configuration

### `slam_project/config/config.py`

This file contains project settings grouped into dataclasses.

Major config groups:

- `RobotConfig`
  - UDP host/port
  - wheel base
  - meters per encoder tick
  - LiDAR min/max range

- `CameraConfig`
  - ESP32-CAM stream URL
  - ArUco marker size
  - camera intrinsics placeholder
  - distortion coefficients placeholder
  - camera-to-base transform
  - known marker world map

- `FrontendConfig`
  - ICP tuning placeholders
  - EKF noise placeholders
  - keyframe thresholds
  - max camera/keyframe time association gap

- `BackendConfig`
  - odometry, LiDAR, landmark, and loop-closure noise placeholders
  - loop-closure candidate settings
  - optimization frequency

- `RuntimeConfig`
  - log/output directories
  - live loop rates
  - offline replay speed

Use this file when tuning the system instead of scattering constants around the codebase.

## Front-End Components

### `slam_project/frontend/motion_model.py`

Contains `DifferentialDriveMotionModel`.

Purpose:

- Convert wheel encoder tick differences into a relative robot motion.
- Provide a small pose prediction helper.

Current behavior:

- Uses a simple midpoint-style differential-drive approximation.

TODOs:

- Finalize wheel travel conversion.
- Add proper differential-drive Jacobians.
- Add covariance propagation support for the EKF.

### `slam_project/frontend/ekf.py`

Contains `EKFLocalizer`.

Purpose:

- Maintain the local robot pose estimate.
- Use wheel odometry for prediction.
- Use LiDAR scan matching for correction.

Important detail:

- The EKF owns/calls the motion model during `predict(...)`.
- This keeps the front-end closer to the paper-style structure:

```text
motion model prediction -> LiDAR observation correction -> corrected local pose
```

Current behavior:

- Applies relative odometry motion to the pose.
- Keeps covariance as a placeholder.
- LiDAR correction currently does not modify the pose yet.

TODOs:

- Define the EKF state vector.
- Implement motion Jacobian.
- Propagate covariance.
- Implement LiDAR measurement model.
- Compute innovation, Kalman gain, and correction update.

### `slam_project/frontend/lidar_matching.py`

Contains `LidarMatcher`.

Purpose:

- Preprocess LiDAR scans.
- Estimate relative motion between scans using future ICP logic.

Current behavior:

- Filters ranges outside configured min/max range.
- Downsamples by a configurable step.
- Returns a dummy zero relative motion for scan matching.

TODOs:

- Convert scans to point clouds.
- Add robot body/self-hit filtering.
- Implement ICP.
- Return meaningful match quality.
- Estimate covariance from ICP quality.

### `slam_project/frontend/landmark_observation.py`

Contains `LandmarkObserver`.

Purpose:

- Detect ArUco markers in camera frames.
- Estimate marker poses with camera calibration.
- Convert marker poses into `LandmarkObservation`s.

Current behavior:

- Detection and PnP are placeholders.
- `observe(...)` already defines the intended pipeline.

TODOs:

- Use `cv2.aruco` for marker detection.
- Use `solvePnP` for marker pose estimation.
- Apply camera-to-base transform.
- Use known marker world map from config.
- Add confidence/quality filtering.

## Back-End Components

### `slam_project/backend/pose_graph.py`

Contains `PoseGraphManager`, `PoseGraphNode`, and `PoseGraphFactor`.

Purpose:

- Store keyframe nodes.
- Store graph factors.
- Associate timestamps with keyframes.

Supported factor placeholders:

- `odometry`
- `lidar`
- `landmark`
- `loop_closure`

Important methods:

- `add_keyframe(pose)`
- `add_odometry_factor(i, j, motion)`
- `add_lidar_factor(i, j, motion)`
- `add_landmark_factor(node_id, obs)`
- `add_loop_closure_factor(i, j, motion)`
- `find_closest_node_by_time(timestamp)`
- `get_graph_summary()`

Current behavior:

- Stores everything in simple Python dictionaries/lists.
- Does not depend on GTSAM or any solver yet.

TODOs:

- Convert placeholder factor records into real SE(2) factor definitions.
- Add solver-specific variable keys.
- Add robust loop-closure handling.
- Add better data association strategies.

### `slam_project/backend/optimizer.py`

Contains `GraphOptimizer`.

Purpose:

- Decide when optimization should run.
- Eventually convert `PoseGraphManager` data into a real factor graph solver.

Current behavior:

- Returns initial keyframe poses unchanged.

TODOs:

- Integrate GTSAM or another nonlinear optimizer.
- Add odometry, LiDAR, landmark, and loop-closure residuals.
- Return optimized poses.
- Optionally compute marginal covariances.

## App Runners

### `slam_project/app/live.py`

Contains `LiveRunner`.

Purpose:

- Run the system against live robot and camera inputs.
- Convert raw camera frames into `LandmarkObservation`s before calling `SLAMSystem`.

Current behavior:

- Provides placeholder generators for:
  - Arduino UDP robot frames
  - ESP32-CAM frames
- Feeds data into `SLAMSystem`.

TODOs:

- Implement UDP socket receive loop.
- Decode Arduino packets into `RobotFrame`.
- Use OpenCV to read the ESP32-CAM stream.
- Add asynchronous buffering and timestamp handling.

### `slam_project/app/offline.py`

Contains `OfflineRunner`.

Purpose:

- Replay saved data through the same SLAM pipeline without needing the robot.

Expected robot data:

- One pickle file, default:

```text
logs/robot_frames.pkl
```

The pickle should contain either:

- `list[RobotFrame]`
- `list[dict]` that can be converted to `RobotFrame`

Expected landmark data:

- One pickle file, default:

```text
logs/landmark_observations.pkl
```

The offline replay flow is:

```text
load robot frames
load landmark observations
merge by timestamp
for each item:
  if RobotFrame -> slam.run_frontend(...)
  if LandmarkObservation -> slam.add_landmark_observation(...)
  slam.run_backend()
```

This file is intentionally project-specific and simple. It is not a generic log ingestion framework.

## Evaluation

### `slam_project/evaluation/metrics.py`

Contains reusable metrics.

Current functions:

- `compute_ate(...)`
  - Computes trajectory Absolute Trajectory Error RMSE over matching ids.

- `compute_map_rmse(...)`
  - Computes RMSE between pairwise landmark distances and ground truth distances.

- `summarize_metrics(...)`
  - Returns a small metrics dictionary.

TODOs:

- Add trajectory alignment if estimated and ground-truth trajectories are in different frames.
- Add richer landmark diagnostics.

### `slam_project/evaluation/plots.py`

Contains placeholder plotting functions.

Current functions:

- `plot_trajectory(...)`
- `plot_landmark_map(...)`
- `plot_error_curves(...)`

TODOs:

- Add matplotlib plots for trajectories.
- Add landmark map plots.
- Add error curves over time.

## Key Concepts

### RobotFrame

A `RobotFrame` is the main front-end input. It should contain synchronized or near-synchronized wheel encoder and LiDAR data:

```text
RobotFrame(timestamp, encoder, lidar_scan)
```

This drives:

- wheel odometry prediction
- LiDAR scan matching
- EKF local pose update
- local map update
- keyframe creation
- odometry/LiDAR factors

### CameraFrame

A `CameraFrame` is a live-runner input only. It is converted into one or more `LandmarkObservation`s before data enters `SLAMSystem`.

Landmark observations are attached to the closest keyframe only if the time gap is small enough:

```text
abs(observation_time - keyframe_time) <= max_landmark_keyframe_time_diff_s
```

### Keyframes

The system does not add every robot pose to the back-end graph. Instead, it creates keyframes only when the robot has moved or rotated enough.

This keeps the graph smaller and more meaningful:

```text
every frame -> EKF/local pose
selected frames -> pose graph keyframes
```

### Factors

Factors are constraints between graph nodes or between a node and a landmark.

Current placeholder factor types:

- **Odometry factor**
  - From wheel odometry between consecutive keyframes.

- **LiDAR factor**
  - From ICP scan-to-scan matching between consecutive keyframes.

- **Landmark factor**
  - From ArUco marker observations attached to nearby keyframes.

- **Loop-closure factor**
  - From ICP scan matching between non-consecutive keyframes that appear spatially close.

### Local Map vs Global Map

The current scaffold separates local and global mapping ideas:

- `LocalMapState`
  - Updated by the front-end from corrected EKF poses.
  - Stores corrected trajectory, keyframe scans, and landmark observations.

- Optimized global trajectory
  - Produced by the back-end optimizer from the pose graph.
  - Currently returns initial poses unchanged until a real optimizer is added.

Eventually, a global map can be rebuilt by transforming stored keyframe scans using optimized keyframe poses.

## Current Placeholder Status

Implemented lightly:

- Dataclasses and shared types
- Configuration containers
- Front-end/back-end wiring
- Keyframe creation
- Pose graph factor storage
- Offline pickle/image replay
- Simple ATE and map RMSE metrics

Still placeholders:

- UDP packet decoding
- ESP32-CAM live reading
- ArUco detection
- `solvePnP`
- ICP scan matching
- EKF covariance math
- Real LiDAR correction
- Real pose graph optimization
- Loop-closure verification
- Occupancy grid or point-cloud map building
- Matplotlib plotting

## Suggested Implementation Order

1. Finalize `RobotFrame` logging format and make offline replay work with real saved data.
2. Implement live UDP decoding in `app/live.py`.
3. Implement ESP32-CAM image reading in `app/live.py`.
4. Implement ArUco detection and PnP in `frontend/landmark_observation.py`.
5. Implement LiDAR scan preprocessing and ICP in `frontend/lidar_matching.py`.
6. Complete EKF prediction and correction math in `frontend/ekf.py`.
7. Strengthen keyframe and timestamp association logic in `app/slam.py`.
8. Convert pose graph placeholder factors into solver factors in `backend/optimizer.py`.
9. Add loop-closure candidate verification and robust rejection.
10. Build global map reconstruction from optimized poses and keyframe scans.
11. Add plotting and evaluation scripts for ATE and landmark map RMSE.

## Quick Smoke Test

From the project root, this should import and run the main wiring with synthetic data:

```bash
python -B -c "from slam_project.app.slam import SLAMSystem; from slam_project.data_types import EncoderState, RobotFrame; s=SLAMSystem(); s.run_frontend(RobotFrame(1.0, EncoderState(0,0,1.0))); print(s.pose_graph.get_graph_summary())"
```

Expected result:

```text
{'num_nodes': 1, 'num_factors': 0, 'factor_counts': {}}
```

That means the first robot frame created the first keyframe node.
