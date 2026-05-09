# Indoor Robot SLAM Project

This folder contains the Python side of the indoor SLAM project: live robot control/data collection, motion-model calibration, offline replay, and plotting/evaluation tools.

## System Organization

```text
robot_python/
  robot_gui.py                  live GUI for robot control, logging, plots, camera view
  robot_code.py                 UDP robot I/O, camera processing, data logging
  parameters.py                 robot, camera, maze, tag, and calibration parameters

  app/
    slam.py                     frontend/backend SLAM system wiring
    offline.py                  replay logged pickle files and plot/evaluate results

  frontend/
    motion_model.py             differential-drive encoder motion model
    ekf.py                      EKF prediction/correction
    lidar_matching.py           LiDAR scan matching and submap logic
    landmark_observation.py     ArUco observation helpers

  backend/
    pose_graph.py               keyframes and factor records
    optimizer.py                GTSAM pose-graph optimization

  data/
    motion_model_data/          calibration trials for encoder-to-distance fitting
    final_trials/               final maze evaluation logs

  util/
    motion_model_fitting/       scripts for fitting and plotting the wheel motion model
    camera_calibration/         ESP32 camera calibration and ArUco utilities
    map/                        measured maze map and manual trajectory plotting
```

## Live Robot Runs and Data Collection

Use the GUI for realtime operation, camera view, robot control, SLAM visualization, and logging:

```powershell
python robot_gui.py
```

Run this from inside `robot_python` or adjust the path from the repo root:

```powershell
python robot_python\robot_gui.py
```

Logged trials are saved as pickle files under `robot_python\data`.

## Motion Model Fitting

The wheel encoder motion model is fit from measured straight, arc, and in-place rotation trials.

Run the fitting script:

```powershell
python robot_python\util\motion_model_fitting\fitting_motion_model.py
```

Plot predicted wheel distances and trajectory for a selected file:

```powershell
python robot_python\util\motion_model_fitting\plot_predicted_trajectory.py
```

The trial metadata and manually measured ground-truth displacement/heading values are stored in:

```text
robot_python\util\motion_model_fitting\fitting_data.py
```

## Offline Evaluation

Use `app/offline.py` to replay a logged pickle file through the SLAM system and generate the evaluation plot.

From `robot_python\app`:

```powershell
python offline.py "C:\Users\lukelo\Desktop\Spring 2026\Robots\NYU_ROB_GY_6213\FinalProject\robot_python\data\final_trials\robot_data_0_0_05_05_26_05_35_09.pkl"
```

Useful plotting/evaluation options:

```powershell
--no-show                 save plot without opening the Matplotlib window
--no-backend              plot frontend/EKF result without GTSAM backend optimization
--output PATH             choose output image path
--gt-final X Y            mark ground-truth final position in metres
--gt-traj ...             draw manual ground-truth trajectory waypoints in cm
--gt-traj-wobble-cm 0     draw clean straight waypoint segments
--gt-traj-step-cm 12      spacing for generated trajectory points
```

Example with manually specified ground-truth trajectory:

```powershell
python offline.py "C:\Users\lukelo\Desktop\Spring 2026\Robots\NYU_ROB_GY_6213\FinalProject\robot_python\data\final_trials\robot_data_0_0_05_05_26_05_29_29.pkl" --gt-traj 20,20 22,170 245,168 250,80 65,75 --gt-traj-wobble-cm 0
```

Another final trial:

```powershell
python offline.py "C:\Users\lukelo\Desktop\Spring 2026\Robots\NYU_ROB_GY_6213\FinalProject\robot_python\data\final_trials\robot_data_0_0_05_05_26_05_35_09.pkl" --gt-traj 20,20 22,170 125,168 127,85 250,85 254,170 185,174 --gt-traj-wobble-cm 0
```

If `--gt-traj` is provided, the last trajectory waypoint is also used as the ground-truth final position for the final-position error box.

## Manual Map/Trajectory Plot Only

To draw only the measured maze map and a manually specified trajectory:

```powershell
python robot_python\util\map\draw_trajectory.py --points 20,20 22,170 245,168 250,80 65,75 --output robot_python\util\map\trajectory.png
```

Waypoints can be raw coordinates in cm or named map/tag points such as `O`, `A`, `T0`, or `T7`.

