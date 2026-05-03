
###CODE 2:Plotting and seeing some predicted on a single file: 
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))    # FinalProject/ → allows `robot_python.*`
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))) # repo root  → allows `FinalProject.*`
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))       # robot_python/ → allows bare `robot_code`
from robot_python.data_handling import get_file_data
from fitting_data import b, files_and_data

# -----------------------------
# FITTED ENCODER FUNCTIONS
# -----------------------------
k_L=0.00035210144602487893
k_R=0.0003565082979875711

cm_to_m = 0.01

ROBOT_X_AXIS_COMPASS_DEG = 102.0

ROBOT_X_AXIS_COMPASS_DEG = 102.0

def compass_to_robot_global_theta(compass_deg):
    theta_deg = ROBOT_X_AXIS_COMPASS_DEG - compass_deg
    theta_rad = np.deg2rad(theta_deg)

    # wrap to [-pi, pi]
    theta_rad = np.arctan2(np.sin(theta_rad), np.cos(theta_rad))

    return theta_rad

def find_trial_info(filename):
    """Find the manually measured trial info from files_and_data."""
    base_filename = os.path.basename(filename)

    for d in files_and_data:
        if d["filename"] == base_filename:  # rremove data/ from filename if present
            return d
    return None


def predict_wheel_distance_over_time(filename):
    """
    Open robot data file, read encoder counts over time,
    and convert encoder change to wheel distance.
    """
    (
        time_list,
        encoder_left_count_list,
        encoder_right_count_list,
        velocity_left_list,
        velocity_right_list,
        steering_angle_list,
        x_camera_list,
        y_camera_list,
        z_camera_list,
        yaw_camera_list,
    ) = get_file_data(filename)

    if len(encoder_left_count_list) < 2 or len(encoder_right_count_list) < 2:
        raise ValueError("Not enough encoder data in this file.")

    # Initial encoder readings
    eL0 = encoder_left_count_list[0]
    eR0 = encoder_right_count_list[0]

    # Encoder change from start at every timestamp
    d_eL_list = np.array(encoder_left_count_list) - eL0
    d_eR_list = np.array(encoder_right_count_list) - eR0

    # Apply fitted functions
    sL_pred_list = k_L * d_eL_list
    sR_pred_list = k_R * d_eR_list

    return (
        np.array(time_list),
        d_eL_list,
        d_eR_list,
        sL_pred_list,
        sR_pred_list,
    )

def predict_trajectory_from_wheels(sL_pred_list, sR_pred_list, theta0_deg=0.0):
    """
    Convert predicted left/right wheel distances over time into x, y, theta trajectory.
    """
    x_list = [0.0]
    y_list = [0.0]
    theta_list = [theta0_deg]

    for i in range(1, len(sL_pred_list)):
        # incremental wheel distances
        dsL = sL_pred_list[i] - sL_pred_list[i - 1]
        dsR = sR_pred_list[i] - sR_pred_list[i - 1]

        ds = (dsR + dsL) / 2.0
        dtheta = (dsR - dsL) / b

        theta_prev = theta_list[-1]
        theta_mid = theta_prev + dtheta / 2.0

        x_new = x_list[-1] + ds * np.cos(theta_mid)
        y_new = y_list[-1] + ds * np.sin(theta_mid)
        theta_new = theta_prev + dtheta

        x_list.append(x_new)
        y_list.append(y_new)
        theta_list.append(theta_new)

    return np.array(x_list), np.array(y_list), np.array(theta_list)



def run_one_file(filename):
    """Run prediction and plotting for one selected file."""

    (
        time_list,
        d_eL_list,
        d_eR_list,
        sL_pred_list,
        sR_pred_list,
    ) = predict_wheel_distance_over_time(filename)

    trial_info = find_trial_info(filename)

    print("\nSelected file:", filename)
    print("Left encoder change final:", d_eL_list[-1])
    print("Right encoder change final:", d_eR_list[-1])
    print(f"Predicted left wheel distance final:  {sL_pred_list[-1]:.4f} m")
    print(f"Predicted right wheel distance final: {sR_pred_list[-1]:.4f} m")

    if trial_info is not None:
        x_true = trial_info["x_finish"] * cm_to_m
        y_true = trial_info["y_finish"] * cm_to_m
        s_center_true = np.sqrt(x_true**2 + y_true**2)

        print(f"Measured final x: {x_true:.4f} m")
        print(f"Measured final y: {y_true:.4f} m")
        # print(f"Measured center displacement: {s_center_true:.4f} m")
        # theta0_deg = trial_info["theta_start"]

    theta0_deg = trial_info["theta_start"]
    theta0_rad = compass_to_robot_global_theta(trial_info["theta_start"])

    print(f"Compass theta_start: {trial_info['theta_start']} deg")
    print(f"Robot global theta_start: {np.rad2deg(theta0_rad):.2f} deg")
    x_pred, y_pred, theta_pred = predict_trajectory_from_wheels(
        sL_pred_list,
        sR_pred_list,
        theta0_deg=theta0_rad
    )


    #final displacement in x and y from tue final position
    x_final_pred = x_pred[-1]
    y_final_pred = y_pred[-1]
    print(f"Predicted final x: {x_final_pred:.4f} m")
    print(f"Predicted final y: {y_final_pred:.4f} m")
    

    # -----------------------------
    # Plot 1: predicted wheel distance over time
    # -----------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(time_list, sL_pred_list, label="Predicted left wheel distance")
    plt.plot(time_list, sR_pred_list, label="Predicted right wheel distance")
    plt.xlabel("Time")
    plt.ylabel("Distance from start (m)")
    plt.title("Predicted Wheel Distance Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    ## PLOT: TRAJECTORY
    plt.figure(figsize=(7, 7))
    plt.plot(x_pred, y_pred, label="Predicted trajectory")
    plt.scatter(x_pred[0], y_pred[0], label="Start")
    plt.scatter(x_pred[-1], y_pred[-1], label="Predicted final")

    if trial_info is not None:
        x_true = trial_info["x_finish"] * cm_to_m
        y_true = trial_info["y_finish"] * cm_to_m
        plt.scatter(x_true, y_true, marker="x", s=100, label="Measured final")

    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Predicted Robot Trajectory from Encoders")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# -----------------------------
# CHOOSE FILE HERE
# -----------------------------
filename_to_run = "../../data/robot_data_0_0_03_05_26_13_21_47.pkl"
run_one_file(filename_to_run)