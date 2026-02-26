# import os
# import math
# import argparse
# import numpy as np
# import matplotlib
# # matplotlib.use("Agg")  # uncomment to save plots without opening windows
# import matplotlib.pyplot as plt
# from matplotlib.patches import Ellipse

# # Your local modules
# import parameters
# import data_handling
# import cv2


# # --------------------------
# # Helpers
# # --------------------------
# def wrap_to_pi(a):
#     return (a + np.pi) % (2*np.pi) - np.pi


# # --------------------------
# # True trajectories (cm -> m)
# # --------------------------
# TRUE_TRAJS = {
#     "video_16_08": {
#         "t": np.array([0, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=float),
#         "x_cm": np.array([-1, 20, 30, 60, 70, 80, 100, 120, 140, 160], dtype=float),
#         "y_cm": np.array([-1, 15, 25, 45, 55, 65, 80, 95, 108, 125], dtype=float),
#     },
#     "video_16_13": {
#         "t": np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=float),
#         "x_cm": np.array([0, 10, 20, 40, 50, 60, 70, 80, 90, 100, 100], dtype=float),
#         "y_cm": np.array([0, 10, 20, 30, 40, 50, 60, 80, 100, 110, 120], dtype=float),
#     },
#     "video_16_16": {
#         "t": np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], dtype=float),
#         "x_cm": np.array([0, 20, 40, 0, 70, 80, 100, 110, 110, 110, 100, 85, 70, 70], dtype=float),
#         "y_cm": np.array([0, 0, 0, 0, 5, 10, 20, 40, 50, 50, 80, 90, 100, 110], dtype=float),
#     },
# }


# def get_truth_xy_m(traj_key: str):
#     if traj_key not in TRUE_TRAJS:
#         raise ValueError(f"Unknown traj key '{traj_key}'. Choose from: {list(TRUE_TRAJS.keys())}")
#     d = TRUE_TRAJS[traj_key]
#     t = d["t"].copy()
#     x = d["x_cm"] / 100.0
#     y = d["y_cm"] / 100.0
#     return t, x, y


# # --------------------------
# # EKF (minimally cleaned)
# # --------------------------
# class ExtendedKalmanFilter:
#     def __init__(self, x_0, Sigma_0, encoder_counts_0, only_prediction: bool):
#         self.state_mean = x_0.astype(float)
#         self.state_covariance = Sigma_0.astype(float)
#         self.predicted_state_mean = np.zeros(3, dtype=float)
#         self.predicted_state_covariance = np.eye(3, dtype=float)
#         self.last_encoder_counts = encoder_counts_0
#         self.only_prediction = only_prediction

#     def update(self, u_t, z_t, delta_t):
#         if self.only_prediction or z_t is None:
#             self.state_covariance, self.state_mean = self.prediction_step(u_t, delta_t)
#         else:
#             self.predicted_state_covariance, self.predicted_state_mean = self.prediction_step(u_t, delta_t)
#             self.correction_step(z_t)

#     def prediction_step(self, u_t, delta_t):
#         mu_pred, s = self.g_function(self.state_mean, u_t, delta_t)
#         G_x_t = self.get_G_x(self.state_mean, s)
#         G_u_t = self.get_G_u(self.state_mean, delta_t)
#         R_t = self.get_R(s)
#         sigma_t_pred = G_x_t @ self.state_covariance @ G_x_t.T + G_u_t @ R_t @ G_u_t.T
#         return sigma_t_pred, mu_pred

#     def correction_step(self, z_t):
#         H_t = self.get_H()
#         Q_t = self.get_Q()
#         S = H_t @ self.predicted_state_covariance @ H_t.T + Q_t
#         K = self.predicted_state_covariance @ H_t.T @ np.linalg.inv(S)

#         residual = z_t - self.get_h_function(self.predicted_state_mean)
#         residual[2] = wrap_to_pi(residual[2])

#         self.state_mean = self.predicted_state_mean + K @ residual
#         self.state_mean[2] = wrap_to_pi(self.state_mean[2])

#         self.state_covariance = (np.eye(3) - K @ H_t) @ self.predicted_state_covariance

#     def distance_travelled_s(self, encoder_counts_delta):
#         return 0.0003063 * encoder_counts_delta

#     def rotational_velocity_w(self, steering_angle_command):
#         return -0.0128 * steering_angle_command + 0.0016

#     def g_function(self, x_tm1, u_t, delta_t):
#         s = self.distance_travelled_s(u_t[0] - self.last_encoder_counts)
#         self.last_encoder_counts = u_t[0]
#         w = self.rotational_velocity_w(u_t[1])

#         x = x_tm1[0] + s * math.cos(x_tm1[2])
#         y = x_tm1[1] + s * math.sin(x_tm1[2])
#         theta = wrap_to_pi(x_tm1[2] + w * delta_t)

#         x_t = np.array([x, y, theta], dtype=float)
#         return x_t, s

#     def get_h_function(self, x_t):
#         return x_t

#     def get_G_x(self, x_tm1, s):
#         G_02 = -s * math.sin(x_tm1[2])
#         G_12 =  s * math.cos(x_tm1[2])
#         return np.array([[1, 0, G_02],
#                          [0, 1, G_12],
#                          [0, 0, 1]], dtype=float)

#     def get_G_u(self, x_tm1, delta_t):
#         return np.array([[math.cos(x_tm1[2]), 0],
#                          [math.sin(x_tm1[2]), 0],
#                          [0, delta_t]], dtype=float)

#     def get_H(self):
#         return np.eye(3, dtype=float)

#     def get_R(self, s):
#         var_s = 0.000000585994 * (s / 0.0003063)
#         var_s = max(var_s, 1e-6)
#         var_w = 0.0009
#         return np.array([[var_s, 0],
#                          [0, var_w]], dtype=float)

#     def get_Q(self):
#         return parameters.Q_t


# # --------------------------
# # Camera -> world transform
# # --------------------------
# def transform_camera_to_world(tvec, rvec):
#     # hardcoded pose at origin (update per camera placement)
#     tvec_init = [-0.21500975, 0.56345664, 0.87650994]
#     rvec_init = [2.72888006, 0.45845892, 0.90720317]

#     tvec = np.array(tvec, dtype=np.float32)
#     rvec = np.array(rvec, dtype=np.float32)
#     t_init = np.array(tvec_init, dtype=np.float32)
#     r_init = np.array(rvec_init, dtype=np.float32)

#     R_init, _ = cv2.Rodrigues(r_init)

#     t_diff = tvec - t_init
#     p_world = R_init.T @ t_diff  # meters

#     x_world = float(p_world[0])
#     y_world = float(p_world[1])

#     R_curr, _ = cv2.Rodrigues(rvec)
#     R_rel = R_init.T @ R_curr
#     yaw = float(np.arctan2(R_rel[1, 0], R_rel[0, 0]))
#     theta_world = wrap_to_pi(yaw)

#     return np.array([x_world, y_world, theta_world], dtype=float)


# # --------------------------
# # Run EKF offline and log arrays
# # (SHIFT IS DONE HERE, BEFORE EKF USES z_t)
# # --------------------------
# def run_ekf_and_log(pkl_path, only_prediction=False, start_row=0, Sigma_0=None):
#     ekf_data = data_handling.get_file_data_for_kf(pkl_path)

#     # Initial camera->world at start_row
#     tvec0 = ekf_data[start_row][3][0:3]
#     rvec0 = ekf_data[start_row][3][3:6]
#     z0_world = transform_camera_to_world(tvec0, rvec0)  # [x0,y0,theta0]

#     # Offset we subtract from ALL subsequent measurements
#     x_off, y_off = float(z0_world[0]), float(z0_world[1])

#     # Initialize EKF at origin, keep theta0
#     x_0 = np.array([0.0, 0.0, float(z0_world[2])], dtype=float)

#     if Sigma_0 is None:
#         Sigma_0 = np.diag([1.0, 1.0, 1.0])

#     encoder_counts_0 = ekf_data[start_row][2].encoder_counts
#     ekf = ExtendedKalmanFilter(x_0, Sigma_0, encoder_counts_0, only_prediction)

#     times, mu_hist, P_hist, cam_ok = [], [], [], []

#     for k in range(start_row + 1, len(ekf_data)):
#         row = ekf_data[k]
#         prev = ekf_data[k - 1]

#         delta_t = row[0] - prev[0]
#         u_t = np.array([row[2].encoder_counts, row[2].steering], dtype=float)

#         tvec = row[3][0:3]
#         rvec = row[3][3:6]

#         if list(tvec) == [0, 0, 0] or list(rvec) == [0, 0, 0]:
#             z_t = None
#             cam_ok.append(False)
#         else:
#             z_world = transform_camera_to_world(tvec, rvec)
#             # SHIFT BEFORE EKF USES IT:
#             z_world[0] -= x_off
#             z_world[1] -= y_off
#             z_t = z_world
#             cam_ok.append(True)

#         ekf.update(u_t, z_t, delta_t)

#         times.append(row[0])
#         mu_hist.append(ekf.state_mean.copy())
#         P_hist.append(ekf.state_covariance.copy())

#     return np.array(times), np.array(mu_hist), np.array(P_hist), np.array(cam_ok, dtype=bool)


# # --------------------------
# # Plotting
# # --------------------------
# def plot_ekf_vs_truth(times, mu_hist, P_hist, truth_t, truth_x, truth_y,
#                      out_path="plots/ekf_vs_truth.png",
#                      ellipse_step=10,
#                      time_offset=0.0):
#     os.makedirs(os.path.dirname(out_path), exist_ok=True)

#     # time shift for truth (kept for consistency)
#     truth_t_shift = truth_t + time_offset

#     fig, ax = plt.subplots()

#     ax.plot(mu_hist[:, 0], mu_hist[:, 1], label="EKF estimate", linewidth=2)
#     ax.plot(truth_x, truth_y, "k--", label="True trajectory", linewidth=2)
#     ax.scatter(truth_x, truth_y, s=20, label="Truth samples")

#     # 95% confidence ellipses
#     chi2_95 = 5.991
#     for i in range(0, len(mu_hist), max(1, ellipse_step)):
#         Pxy = P_hist[i, 0:2, 0:2]
#         vals, vecs = np.linalg.eig(Pxy)
#         vals = np.maximum(vals, 1e-12)
#         order = np.argsort(vals)[::-1]
#         vals = vals[order]
#         vecs = vecs[:, order]

#         angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
#         width  = 2 * np.sqrt(chi2_95 * vals[0])
#         height = 2 * np.sqrt(chi2_95 * vals[1])

#         e = Ellipse((mu_hist[i, 0], mu_hist[i, 1]),
#                     width=width, height=height, angle=angle,
#                     fill=False, linewidth=1, alpha=0.7)
#         ax.add_patch(e)

#     ax.set_xlabel("X (m)")
#     ax.set_ylabel("Y (m)")
#     ax.set_title("EKF vs True Trajectory (confidence ellipses)")
#     ax.grid(True)
#     ax.set_aspect("equal", adjustable="box")
#     ax.legend()

#     fig.tight_layout()
#     plt.show()
#     fig.savefig(out_path, dpi=250)
#     # plt.close(fig)


# def plot_xy_error(times, mu_hist, truth_t, truth_x, truth_y,
#                   out_path="plots/ekf_xy_error.png",
#                   time_offset=0.0):
#     os.makedirs(os.path.dirname(out_path), exist_ok=True)

#     truth_t_shift = truth_t + time_offset

#     # Interpolate truth to EKF timestamps
#     x_true = np.interp(times, truth_t_shift, truth_x)
#     y_true = np.interp(times, truth_t_shift, truth_y)

#     ex = np.abs(mu_hist[:, 0] - x_true)
#     ey = np.abs(mu_hist[:, 1] - y_true)
#     e_norm = np.sqrt(ex**2 + ey**2)

#     fig, ax = plt.subplots()
#     ax.plot(times, ex, label="x error (m)")
#     ax.plot(times, ey, label="y error (m)")
#     # ax.plot(times, e_norm, label="position error norm (m)", linewidth=2)

#     ax.set_xlabel("Time (s)")
#     ax.set_ylabel("Error (m)")
#     ax.axis([-0.5, 14, -0.1, 0.8])
#     ax.set_title("Absolute Error: EKF Position vs Truth Position")
#     ax.grid(True)
#     ax.legend()

#     fig.tight_layout()
#     plt.show()
#     fig.savefig(out_path, dpi=250)
#     plt.close(fig)


# # --------------------------
# # Main
# # --------------------------
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--pkl", type=str, required=True, help="Path to robot .pkl log")
#     parser.add_argument("--traj", type=str, required=True, choices=list(TRUE_TRAJS.keys()),
#                         help="Which truth trajectory to overlay")
#     parser.add_argument("--only_prediction", action="store_true", help="Run EKF with prediction only")
#     parser.add_argument("--time_offset", type=float, default=0.0,
#                         help="Shift truth times: t_truth := t_truth + time_offset")
#     parser.add_argument("--out_dir", type=str, default="plots", help="Where to save plots")
#     parser.add_argument("--start_row", type=int, default=0, help="Row index in pickle to start EKF")
#     parser.add_argument("--normalize_time", action="store_true",
#                         help="Shift EKF timestamps so they start at 0 (useful if truth starts at 0/1)")
#     args = parser.parse_args()

#     Sigma_0 = np.diag([1.0, 1.0, 1.0])

#     times, mu_hist, P_hist, cam_ok = run_ekf_and_log(
#         pkl_path=args.pkl,
#         only_prediction=args.only_prediction,
#         start_row=args.start_row,
#         Sigma_0=Sigma_0
#     )

#     if args.normalize_time and len(times) > 0:
#         times = times - times[0]

#     truth_t, truth_x, truth_y = get_truth_xy_m(args.traj)

#     tag = "pred_only" if args.only_prediction else "pred_corr"
#     out1 = os.path.join(args.out_dir, f"ekf_vs_truth_{args.traj}_{tag}.png")
#     out2 = os.path.join(args.out_dir, f"ekf_xy_error_{args.traj}_{tag}.png")

#     plot_ekf_vs_truth(times, mu_hist, P_hist, truth_t, truth_x, truth_y,
#                       out_path=out1,
#                       ellipse_step=10,
#                       time_offset=args.time_offset)

#     plot_xy_error(times, mu_hist, truth_t, truth_x, truth_y,
#                   out_path=out2,
#                   time_offset=args.time_offset)

#     print(f"Saved:\n  {out1}\n  {out2}")


# if __name__ == "__main__":
#     main()



# ##python plot_ekf.py --pkl "C:\Users\lukelo\Desktop\Spring 2026\Robots\NYU_ROB_GY_6213\base_code_lab_03\robot_python_code\data_trajectory_simple\robot_data_0_0_24_02_26_16_08_22.pkl" --traj "video_16_08"
# ##python plot_ekf.py --pkl "C:\Users\lukelo\Desktop\Spring 2026\Robots\NYU_ROB_GY_6213\base_code_lab_03\robot_python_code\data_trajectory_simple\robot_data_0_-5_24_02_26_16_13_54.pkl" --traj "video_16_13"
# ##python plot_ekf.py --pkl "C:\Users\lukelo\Desktop\Spring 2026\Robots\NYU_ROB_GY_6213\base_code_lab_03\robot_python_code\data_trajectory_complex\robot_data_0_0_24_02_26_16_16_58.pkl" --traj "video_16_16"
"""
plot_out_of_frame.py

Offline EKF plotting for the "In-Frame vs Out-of-Frame" case (NO truth required).

What it does:
1) Runs EKF offline over a .pkl log.
2) Detects when camera pose is unusable (zero pose OR stale repeated frames).
3) Logs cam_ok per timestep (True = correction used, False = prediction-only).
4) Plots:
   - XY EKF trajectory + 95% confidence ellipses
   - Dropout points marked
   - Uncertainty vs time (sigma_x, sigma_y, trace(P_xy)) with shaded dropout intervals

Usage:
python plot_out_of_frame.py --pkl "path/to/robot_data.pkl" --out_dir plots --normalize_time

Notes:
- Update tvec_init / rvec_init in transform_camera_to_world() for EACH camera placement.
- This script assumes your data_handling.get_file_data_for_kf(pkl) returns rows where:
  row[0]=time, row[2]=sensor obj with encoder_counts & steering, row[3]=[tvec(3), rvec(3)].
"""

import os
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import cv2

import parameters
import data_handling


# --------------------------
# Helpers
# --------------------------
def wrap_to_pi(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


# --------------------------
# EKF
# --------------------------
class ExtendedKalmanFilter:
    def __init__(self, x_0, Sigma_0, encoder_counts_0, only_prediction: bool):
        self.state_mean = x_0.astype(float)
        self.state_covariance = Sigma_0.astype(float)

        self.predicted_state_mean = np.zeros(3, dtype=float)
        self.predicted_state_covariance = np.eye(3, dtype=float)

        self.last_encoder_counts = float(encoder_counts_0)
        self.only_prediction = only_prediction

    def update(self, u_t, z_t, delta_t):
        if self.only_prediction or z_t is None:
            self.state_covariance, self.state_mean = self.prediction_step(u_t, delta_t)
        else:
            self.predicted_state_covariance, self.predicted_state_mean = self.prediction_step(u_t, delta_t)
            self.correction_step(z_t)

    def prediction_step(self, u_t, delta_t):
        mu_pred, s = self.g_function(self.state_mean, u_t, delta_t)
        G_x_t = self.get_G_x(self.state_mean, s)
        G_u_t = self.get_G_u(self.state_mean, delta_t)
        R_t = self.get_R(s)

        sigma_pred = G_x_t @ self.state_covariance @ G_x_t.T + G_u_t @ R_t @ G_u_t.T
        return sigma_pred, mu_pred

    def correction_step(self, z_t):
        H_t = self.get_H()
        Q_t = self.get_Q()

        S = H_t @ self.predicted_state_covariance @ H_t.T + Q_t
        K = self.predicted_state_covariance @ H_t.T @ np.linalg.inv(S)

        residual = z_t - self.h(self.predicted_state_mean)
        residual[2] = wrap_to_pi(residual[2])

        self.state_mean = self.predicted_state_mean + K @ residual
        self.state_mean[2] = wrap_to_pi(self.state_mean[2])

        self.state_covariance = (np.eye(3) - K @ H_t) @ self.predicted_state_covariance

    # --- motion model pieces (same as your current setup) ---
    def distance_travelled_s(self, encoder_counts_delta):
        return 0.0003063 * encoder_counts_delta

    def rotational_velocity_w(self, steering_angle_command):
        return -0.0128 * steering_angle_command + 0.0016

    def g_function(self, x_tm1, u_t, delta_t):
        # u_t = [encoder_counts, steering]
        s = self.distance_travelled_s(u_t[0] - self.last_encoder_counts)
        self.last_encoder_counts = float(u_t[0])
        w = self.rotational_velocity_w(u_t[1])

        x = x_tm1[0] + s * math.cos(x_tm1[2])
        y = x_tm1[1] + s * math.sin(x_tm1[2])
        theta = wrap_to_pi(x_tm1[2] + w * delta_t)

        return np.array([x, y, theta], dtype=float), s

    def h(self, x_t):
        # Measurement is directly [x, y, theta]
        return x_t

    def get_G_x(self, x_tm1, s):
        G_02 = -s * math.sin(x_tm1[2])
        G_12 =  s * math.cos(x_tm1[2])
        return np.array([[1, 0, G_02],
                         [0, 1, G_12],
                         [0, 0, 1]], dtype=float)

    def get_G_u(self, x_tm1, delta_t):
        return np.array([[math.cos(x_tm1[2]), 0],
                         [math.sin(x_tm1[2]), 0],
                         [0, delta_t]], dtype=float)

    def get_H(self):
        return np.eye(3, dtype=float)

    def get_R(self, s):
        # encoder variance model + constant steering-to-w variance
        var_s = 0.000000585994 * (s / 0.0003063)
        var_s = max(var_s, 1e-6)
        var_w = 0.0009
        return np.array([[var_s, 0],
                         [0, var_w]], dtype=float)

    def get_Q(self):
        return parameters.Q_t


# --------------------------
# Camera -> world transform
# --------------------------
def transform_camera_to_world(tvec, rvec):
    """
    IMPORTANT:
    Update these per camera placement.
    You already do this by capturing (tvec_init, rvec_init) at the origin facing +X.
    """
    # tvec_init = [-0.21500975, 0.56345664, 0.87650994]
    # rvec_init = [2.72888006, 0.45845892, 0.90720317]
    tvec_init = [-0.49274365, 0.05354998, 1.29572172]
    rvec_init = [ 2.17807045, -0.29410592, 0.29292633]

    tvec = np.array(tvec, dtype=np.float32)
    rvec = np.array(rvec, dtype=np.float32)
    t_init = np.array(tvec_init, dtype=np.float32)
    r_init = np.array(rvec_init, dtype=np.float32)

    R_init, _ = cv2.Rodrigues(r_init)

    # position
    p_world = R_init.T @ (tvec - t_init)
    x_world = float(p_world[0])
    y_world = float(p_world[1])

    # yaw (relative)
    R_curr, _ = cv2.Rodrigues(rvec)
    R_rel = R_init.T @ R_curr
    yaw = float(np.arctan2(R_rel[1, 0], R_rel[0, 0]))
    theta_world = wrap_to_pi(yaw)

    return np.array([x_world, y_world, theta_world], dtype=float)


# --------------------------
# Run EKF with stale/zero detection + initial offset correction
# --------------------------
def run_ekf_out_of_frame(pkl_path, only_prediction=False, start_row=0, Sigma_0=None,
                         STALE_N=5, zero_atol=1e-6):
    ekf_data = data_handling.get_file_data_for_kf(pkl_path)

    # --- initial measurement at start_row (used for offset + theta init) ---
    tvec0 = ekf_data[start_row][3][0:3]
    rvec0 = ekf_data[start_row][3][3:6]
    z0_world = transform_camera_to_world(tvec0, rvec0)

    # offset correction: force starting position to (0,0)
    x_off, y_off = float(z0_world[0]), float(z0_world[1])

    # init EKF: (0,0, theta0)
    x_0 = np.array([0.0, 0.0, float(z0_world[2])], dtype=float)

    if Sigma_0 is None:
        Sigma_0 = np.diag([1.0, 1.0, 1.0])

    encoder_counts_0 = ekf_data[start_row][2].encoder_counts
    ekf = ExtendedKalmanFilter(x_0, Sigma_0, encoder_counts_0, only_prediction)

    times, mu_hist, P_hist, cam_ok = [], [], [], []

    
    #for keeping track if the camera measurement is stale (i.e. out-of-frame)
    last_pose = None
    stale = 0
    STALE_N = 15        # after 10 repeated frames => out-of-frame
    EPS = 1e-9         # small threshold
    last_encoder_counts = 0
    encoder_change=0

    for t in range(start_row + 1, len(ekf_data)):
        row = ekf_data[t]
        prev = ekf_data[t - 1]

        delta_t = row[0] - prev[0]
        u_t = np.array([row[2].encoder_counts, row[2].steering], dtype=float)

        tvec = row[3][0:3]
        rvec = row[3][3:6]
        pose = np.hstack([tvec, rvec])
         # 2. Check if the camera is stuck on the exact same frame
        # If the robot is moving (u_t[0] is changing), the pose SHOULD change.
        # If it doesn't change at all, it's stale.
        is_stale = False
        if last_pose is not None:
            if np.array_equal(pose, last_pose): # Perfect match usually means frozen buffer
                stale += 1
            else:
                stale = 0 # Reset immediately if even one bit changes
        
        if stale >= STALE_N:
            is_stale = True

        last_pose = pose.copy()

        encoder_change = abs(u_t[0] - last_encoder_counts)
        last_encoder_counts = u_t[0]
        # 3. Final decision
        # Reject if the data is frozen AND we know the robot is actually moving
        if is_stale and encoder_change > 0:
            print(f"DEBUG: Prediction Only - Camera Stale (Frozen at {stale} frames)")
            z_t = None
            cam_ok.append(False)

        # Otherwise, the data is likely valid (either moving and changing, or stopped and same)
        else:
            z_world = np.array(transform_camera_to_world(tvec, rvec))
            # apply initial offset so start is (0,0)
            z_world[0] -= x_off
            z_world[1] -= y_off
            z_t = z_world
            cam_ok.append(True)

        ekf.update(u_t, z_t, delta_t)

        times.append(row[0])
        mu_hist.append(ekf.state_mean.copy())
        P_hist.append(ekf.state_covariance.copy())

    return np.array(times), np.array(mu_hist), np.array(P_hist), np.array(cam_ok, dtype=bool)


# --------------------------
# Plotting (no truth)
# --------------------------
def plot_out_of_frame(times, mu_hist, P_hist, cam_ok,
                      out_path="plots/ekf_out_of_frame.png",
                      ellipse_step=10):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fig,ax1,= plt.subplots(1, 1, figsize=(8, 6))

    # ---- Left: trajectory + ellipses + dropout markers ----
    ax1.plot(mu_hist[:, 0], mu_hist[:, 1], label="EKF estimate", linewidth=2)

    if cam_ok is not None and len(cam_ok) == len(times):
        ax1.scatter(mu_hist[~cam_ok, 0], mu_hist[~cam_ok, 1],
                    s=25, color="Red", marker="x", label="Prediction-only (out-frame)", alpha=0.9)

    # 95% ellipses
    chi2_95 = 5.991
    for i in range(0, len(mu_hist), max(1, ellipse_step)):
        Pxy = P_hist[i, 0:2, 0:2]
        vals, vecs = np.linalg.eig(Pxy)
        vals = np.maximum(vals, 1e-12)
        order = np.argsort(vals)[::-1]
        vals = vals[order]
        vecs = vecs[:, order]

        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        width = 2 * np.sqrt(chi2_95 * vals[0])
        height = 2 * np.sqrt(chi2_95 * vals[1])

        e = Ellipse((mu_hist[i, 0], mu_hist[i, 1]),
                    width=width, height=height, angle=angle,
                    fill=False, linewidth=1, alpha=0.7)
        ax1.add_patch(e)

    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_title("EKF trajectory with Confidence ellipses")
    ax1.grid(True)
    ax1.set_aspect("equal", adjustable="box")
    ax1.legend()

    # # ---- Right: uncertainty over time + shaded dropout intervals ----
    # sig_x = np.sqrt(np.maximum(P_hist[:, 0, 0], 0))
    # sig_y = np.sqrt(np.maximum(P_hist[:, 1, 1], 0))
    # tr_xy = P_hist[:, 0, 0] + P_hist[:, 1, 1]

    # ax2.plot(times, sig_x, label=r"$\sigma_x$ (m)")
    # ax2.plot(times, sig_y, label=r"$\sigma_y$ (m)")
    # ax2.plot(times, tr_xy, label=r"$\mathrm{trace}(P_{xy})$", linewidth=2)

    # # shade dropout spans
    # if cam_ok is not None and len(cam_ok) == len(times):
    #     in_dropout = False
    #     start_t = None
    #     for k in range(len(times)):
    #         if (not cam_ok[k]) and (not in_dropout):
    #             in_dropout = True
    #             start_t = times[k]
    #         if in_dropout and (cam_ok[k] or k == len(times) - 1):
    #             end_t = times[k]
    #             ax2.axvspan(start_t, end_t, alpha=0.15)
    #             in_dropout = False

    # ax2.set_xlabel("Time (s)")
    # ax2.set_ylabel("Uncertainty")
    # ax2.set_title("Uncertainty vs time (shaded = camera missing)")
    # ax2.grid(True)
    # ax2.legend()

    fig.tight_layout()
    plt.show()
    fig.savefig(out_path, dpi=250)
    plt.close(fig)


# --------------------------
# Main
# --------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", type=str, required=True, help="Path to robot .pkl log")
    parser.add_argument("--out_dir", type=str, default="plots", help="Where to save plots")
    parser.add_argument("--only_prediction", action="store_true", help="Run prediction only (ignore camera always)")
    parser.add_argument("--start_row", type=int, default=0, help="Row index in pickle to start EKF")
    parser.add_argument("--normalize_time", action="store_true", help="Shift EKF timestamps so they start at 0")
    parser.add_argument("--ellipse_step", type=int, default=10, help="Draw ellipse every N points")
    parser.add_argument("--stale_n", type=int, default=15, help="Stale threshold: repeated pose frames")
    parser.add_argument("--zero_atol", type=float, default=1e-6, help="Allclose threshold for zero-pose detection")
    args = parser.parse_args()

    Sigma_0 = np.diag([1.0, 1.0, 1.0])

    times, mu_hist, P_hist, cam_ok = run_ekf_out_of_frame(
        pkl_path=args.pkl,
        only_prediction=args.only_prediction,
        start_row=args.start_row,
        Sigma_0=Sigma_0,
        STALE_N=args.stale_n,
        zero_atol=args.zero_atol
    )

    if args.normalize_time and len(times) > 0:
        times = times - times[0]

    tag = "pred_only" if args.only_prediction else "pred_corr"
    out_path = os.path.join(args.out_dir, f"ekf_out_of_frame_{tag}.png")

    plot_out_of_frame(times, mu_hist, P_hist, cam_ok, out_path=out_path, ellipse_step=args.ellipse_step)
    print(f"Saved:\n  {out_path}")


if __name__ == "__main__":
    main()


# ##python plot_ekf.py --pkl "C:\Users\lukelo\Desktop\Spring 2026\Robots\NYU_ROB_GY_6213\base_code_lab_03\robot_python_code\data_in-out_frame\robot_data_0_0_24_02_26_23_09_17.pkl" 

##python plot_ekf.py --pkl "C:\Users\lukelo\Desktop\Spring 2026\Robots\NYU_ROB_GY_6213\base_code_lab_03\robot_python_code\data_in-out_frame\robot_data_0_0_24_02_26_23_06_34.pkl" 

##python plot_ekf.py --pkl "C:\Users\lukelo\Desktop\Spring 2026\Robots\NYU_ROB_GY_6213\base_code_lab_03\robot_python_code\data_in-out_frame\robot_data_0_0_24_02_26_23_06_34.pkl" 