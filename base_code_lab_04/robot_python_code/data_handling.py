# External Libraries
import matplotlib.pyplot as plt
from pathlib import Path
import math
import numpy as np

# Internal Libraries
import parameters
import robot_python_code
import motion_models


# -----------------------------
# Helpers
# -----------------------------
def _normalize_time_list(time_list):
    """Shift time so trial starts at t = 0."""
    if len(time_list) == 0:
        return time_list
    t0 = time_list[0]
    return [t - t0 for t in time_list]


def _unpack_traj_output(traj_output):
    """
    Handle motion model outputs robustly.
    Some code paths assume:
        x_list, y_list, theta_list
    Others assume:
        x_list, y_list, theta_list, distance_list
    """
    if len(traj_output) == 3:
        x_list, y_list, theta_list = traj_output
        distance_list = x_list  # fallback for straight-line distance usage in older code
        return x_list, y_list, theta_list, distance_list

    if len(traj_output) == 4:
        x_list, y_list, theta_list, distance_list = traj_output
        return x_list, y_list, theta_list, distance_list

    raise ValueError(
        "Unexpected number of outputs from motion model traj_propagation(). "
        f"Expected 3 or 4, got {len(traj_output)}"
    )


# -----------------------------
# File loading
# -----------------------------
def get_file_data(filename):
    """
    Open a file and return data in a form ready to plot.

    Expected dictionary keys in file:
    ['time', 'control_signal', 'robot_sensor_signal', ...]
    Camera data may be absent in some files, so we handle that safely.
    """
    data_loader = robot_python_code.DataLoader(filename)
    data_dict = data_loader.load()

    time_list = data_dict.get('time', [])
    control_signal_list = data_dict.get('control_signal', [])
    robot_sensor_signal_list = data_dict.get('robot_sensor_signal', [])
    camera_sensor_signal_list = data_dict.get('camera_sensor_signal', [])

    encoder_count_list = []
    velocity_list = []
    steering_angle_list = []
    x_camera_list = []
    y_camera_list = []
    z_camera_list = []
    yaw_camera_list = []

    for row in robot_sensor_signal_list:
        encoder_count_list.append(row.encoder_counts)

    for row in control_signal_list:
        # Assumes control_signal is something like [velocity, steering_angle]
        velocity_list.append(row[0])
        steering_angle_list.append(row[1])

    for row in camera_sensor_signal_list:
        # Assumes pose_estimate = [x, y, z, rvecx, rvecy, rvecz]
        x_camera_list.append(row[0])
        y_camera_list.append(row[1])
        z_camera_list.append(row[2])
        yaw_camera_list.append(row[5])

    time_list = _normalize_time_list(list(time_list))

    return (
        time_list,
        encoder_count_list,
        velocity_list,
        steering_angle_list,
        x_camera_list,
        y_camera_list,
        z_camera_list,
        yaw_camera_list,
    )


def get_file_data_for_kf(filename):
    """
    Open a file and return data packed for KF use:
    [time_from_zero, control_signal, robot_sensor_signal, camera_sensor_signal]
    """
    data_loader = robot_python_code.DataLoader(filename)
    data_dict = data_loader.load()

    time_list = data_dict.get('time', [])
    control_signal_list = data_dict.get('control_signal', [])
    robot_sensor_signal_list = data_dict.get('robot_sensor_signal', [])
    camera_sensor_signal_list = data_dict.get('camera_sensor_signal', [])

    time_list = _normalize_time_list(list(time_list))

    ekf_data = []
    n = min(
        len(time_list),
        len(control_signal_list),
        len(robot_sensor_signal_list),
        len(camera_sensor_signal_list),
    )

    for i in range(n):
        row = [
            time_list[i],
            control_signal_list[i],
            robot_sensor_signal_list[i],
            camera_sensor_signal_list[i],
        ]
        ekf_data.append(row)

    return ekf_data


def get_file_data_for_pf(filename):
    """
    Open a file and return data packed for PF use:
    [time_from_zero, control_signal, robot_sensor_signal]
    """
    data_loader = robot_python_code.DataLoader(filename)
    data_dict = data_loader.load()

    time_list = data_dict.get('time', [])
    control_signal_list = data_dict.get('control_signal', [])
    robot_sensor_signal_list = data_dict.get('robot_sensor_signal', [])

    time_list = _normalize_time_list(list(time_list))

    pf_data = []
    n = min(len(time_list), len(control_signal_list), len(robot_sensor_signal_list))

    for i in range(n):
        row = [time_list[i], control_signal_list[i], robot_sensor_signal_list[i]]
        pf_data.append(row)

    return pf_data


# -----------------------------
# Plotting basics
# -----------------------------
def plot_trial_basics(filename):
    (
        time_list,
        encoder_count_list,
        velocity_list,
        steering_angle_list,
        x_camera_list,
        y_camera_list,
        z_camera_list,
        yaw_camera_list,
    ) = get_file_data(filename)

    plt.plot(time_list, encoder_count_list)
    plt.title('Encoder Values')
    plt.xlabel('Time (s)')
    plt.ylabel('Encoder counts')
    plt.show()

    plt.plot(time_list, velocity_list)
    plt.title('Speed')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity command')
    plt.show()

    plt.plot(time_list, steering_angle_list)
    plt.title('Steering')
    plt.xlabel('Time (s)')
    plt.ylabel('Steering command')
    plt.show()


# -----------------------------
# Motion model trial execution
# -----------------------------
def run_my_model_on_trial(filename, show_plot=True, plot_color='ko'):
    """
    Plot a trajectory using the motion model on one saved trial.
    Uses PF-style packed data because that contains time, control, and robot sensor data.
    """
    pf_data = get_file_data_for_pf(filename)

    time_list = []
    encoder_count_list = []
    steering_angle_list = []

    for row in pf_data:
        t, control_signal, robot_sensor_signal = row

        time_list.append(t)
        encoder_count_list.append(robot_sensor_signal.encoder_counts)

        # Assumes control_signal = [velocity, steering_angle]
        steering_angle_list.append(control_signal[1])

    motion_model = motion_models.MyMotionModel([0, 0, 0], 0)
    traj_output = motion_model.traj_propagation(
        time_list,
        encoder_count_list,
        steering_angle_list,
    )
    x_list, y_list, theta_list, distance_list = _unpack_traj_output(traj_output)

    plt.plot(x_list, y_list, plot_color)
    plt.title('Motion Model Predicted XY Traj (m)')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.axis([-0.5, 1.5, -1, 1])

    if show_plot:
        plt.show()


def plot_many_trial_predictions(directory):
    directory_path = Path(directory)
    plot_color_list = [
        'r.', 'k.', 'g.', 'c.', 'b.',
        'r.', 'k.', 'g.', 'c.', 'b.',
        'r.', 'k.', 'g.', 'c.', 'b.',
        'r.', 'k.', 'g.', 'c.', 'b.',
    ]

    count = 0
    for item in directory_path.iterdir():
        if item.is_file():
            plot_color = plot_color_list[count % len(plot_color_list)]
            run_my_model_on_trial(str(item), False, plot_color)
            count += 1

    plt.show()


# -----------------------------
# Prediction utilities
# -----------------------------
def run_my_model_to_predict_distance(filename):
    (
        time_list,
        encoder_count_list,
        velocity_list,
        steering_angle_list,
        x_camera_list,
        y_camera_list,
        z_camera_list,
        yaw_camera_list,
    ) = get_file_data(filename)

    motion_model = motion_models.MyMotionModel([0, 0, 0], 0)
    traj_output = motion_model.traj_propagation(
        time_list,
        encoder_count_list,
        steering_angle_list,
    )
    x_list, y_list, theta_list, distance_list = _unpack_traj_output(traj_output)

    index_of_end = -30 if len(distance_list) >= 30 else -1
    distance = distance_list[index_of_end]
    return distance


def run_my_model_to_predict_state(filename):
    (
        time_list,
        encoder_count_list,
        velocity_list,
        steering_angle_list,
        x_camera_list,
        y_camera_list,
        z_camera_list,
        yaw_camera_list,
    ) = get_file_data(filename)

    motion_model = motion_models.MyMotionModel([0, 0, 0], 0)
    traj_output = motion_model.traj_propagation(
        time_list,
        encoder_count_list,
        steering_angle_list,
    )
    x_list, y_list, theta_list, distance_list = _unpack_traj_output(traj_output)

    index_of_end = -30 if len(x_list) >= 30 else -1

    x = x_list[index_of_end]
    y = y_list[index_of_end]
    theta = theta_list[index_of_end]
    distance = distance_list[index_of_end]
    time_stamp = time_list[index_of_end]

    return time_stamp, x, y, theta, distance


# -----------------------------
# Error analysis
# -----------------------------
def get_diff_squared(m_list, p_list):
    diff_squared_list = []
    for i in range(len(m_list)):
        diff_squared = math.pow(m_list[i] - p_list[i], 2)
        diff_squared_list.append(diff_squared)

    coefficients = np.polyfit(m_list, diff_squared_list, 2)
    poly = np.poly1d(coefficients)

    plt.plot(m_list, diff_squared_list, 'ko')
    plt.plot(m_list, poly(m_list), 'ro')
    plt.title("Error Squared (m^2)")
    plt.xlabel('Measured distance travelled (m)')
    plt.ylabel('(Actual - Predicted)^2 (m^2)')
    plt.show()

    return diff_squared_list


def get_diff_w_squared(dist_list, m_w_list, p_w_list):
    diff_squared_list = []
    for i in range(len(m_w_list)):
        diff_squared = math.pow(m_w_list[i] - p_w_list[i], 2)
        diff_squared_list.append(diff_squared)

    plt.plot(dist_list, diff_squared_list, 'ko')
    plt.plot([0], [0], 'ko')
    plt.title("Error Squared (rad^2/s^2)")
    plt.xlabel('Measured distance travelled (m)')
    plt.ylabel('(Actual w - Predicted w)^2')
    plt.show()

    return diff_squared_list


# -----------------------------
# Batch processing
# -----------------------------
def process_files_and_plot(files_and_data, directory):
    predicted_distance_list = []
    measured_distance_list = []

    for row in files_and_data:
        filename = row[0]
        measured_distance = row[1]
        measured_distance_list.append(measured_distance)

        full_path = str(Path(directory) / filename)
        predicted_distance = run_my_model_to_predict_distance(full_path)
        predicted_distance_list.append(predicted_distance)

    plt.plot(measured_distance_list + [0], predicted_distance_list + [0], 'ko')
    plt.plot([0, 1.7], [0, 1.7])
    plt.title('Distance Trials')
    plt.xlabel('Measured Distance (m)')
    plt.ylabel('Predicted Distance (m)')
    plt.legend(['Measured vs Predicted', 'Slope 1 Line'])
    plt.show()

    get_diff_squared(measured_distance_list, predicted_distance_list)


def process_files_and_plot_curve(files_and_data, directory):
    x_measured_list = []
    y_measured_list = []
    theta_measured_list = []

    x_predicted_list = []
    y_predicted_list = []
    theta_predicted_list = []

    w_measured_list = []
    w_predicted_list = []
    distance_predicted_list = []

    for row in files_and_data:
        filename = row[0]
        x_measured_distance = row[1]
        y_measured_distance = row[2]

        x_measured_list.append(x_measured_distance)
        y_measured_list.append(y_measured_distance)

        theta_measured = 2 * math.atan2(y_measured_distance, x_measured_distance)
        theta_measured_list.append(theta_measured)

        full_path = str(Path(directory) / filename)
        time_stamp, x_predicted, y_predicted, theta_predicted, distance_predicted = (
            run_my_model_to_predict_state(full_path)
        )

        x_predicted_list.append(x_predicted)
        y_predicted_list.append(y_predicted)
        theta_predicted_list.append(theta_predicted)

        if time_stamp != 0:
            w_measured_list.append(theta_measured / time_stamp)
            w_predicted_list.append(theta_predicted / time_stamp)
        else:
            w_measured_list.append(0.0)
            w_predicted_list.append(0.0)

        print("W:", w_measured_list[-1], w_predicted_list[-1])
        distance_predicted_list.append(distance_predicted)

    plt.plot(theta_measured_list + [0], theta_predicted_list + [0], 'ko')
    plt.plot([0, 2.1], [0, 2.1], 'r')
    plt.title('Rotation Trials')
    plt.xlabel('Measured Theta (rad)')
    plt.ylabel('Predicted Theta (rad)')
    plt.legend(['Measured vs Predicted', 'Slope 1 Line'])
    plt.show()

    get_diff_w_squared(distance_predicted_list, w_measured_list, w_predicted_list)


# -----------------------------
# Simulation sampling
# -----------------------------
def sample_model(num_samples):
    traj_duration = 10

    for _ in range(num_samples):
        model = motion_models.MyMotionModel([0, 0, 0], 0)
        traj_x, traj_y, traj_theta = model.generate_simulated_traj(traj_duration)
        plt.plot(traj_x, traj_y, 'k.')

    plt.title('Sampling the model')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.show()


def run_my_model_on_trial_with_title(filename, show_plot=True, plot_color='ko'):
    """
    Same as run_my_model_on_trial, but writes the filename in the title.
    """
    pf_data = get_file_data_for_pf(filename)

    time_list = []
    encoder_count_list = []
    steering_angle_list = []

    for row in pf_data:
        t, control_signal, robot_sensor_signal = row

        time_list.append(t)
        encoder_count_list.append(robot_sensor_signal.encoder_counts)
        steering_angle_list.append(control_signal[1])  # [velocity, steering_angle]

    motion_model = motion_models.MyMotionModel([0, 0, 0], 0)
    traj_output = motion_model.traj_propagation(
        time_list,
        encoder_count_list,
        steering_angle_list,
    )
    x_list, y_list, theta_list, distance_list = _unpack_traj_output(traj_output)

    plt.figure()
    plt.plot(x_list, y_list, plot_color)
    plt.title(f'Motion Model Predicted XY Traj\n{Path(filename).name}')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.axis([-0.5, 1.5, -1, 1])

    if show_plot:
        plt.show()


def plot_many_trial_predictions_one_by_one(directory):
    """
    Plot each trial in its own figure, one at a time, with the filename on top.
    """
    directory_path = Path(directory)
    plot_color_list = [
        'r.', 'k.', 'g.', 'c.', 'b.',
        'r.', 'k.', 'g.', 'c.', 'b.',
        'r.', 'k.', 'g.', 'c.', 'b.',
        'r.', 'k.', 'g.', 'c.', 'b.',
    ]

    count = 0
    for item in sorted(directory_path.iterdir()):
        if item.is_file() and item.suffix == '.pkl':
            plot_color = plot_color_list[count % len(plot_color_list)]
            run_my_model_on_trial_with_title(str(item), show_plot=True, plot_color=plot_color)
            count += 1

# -----------------------------
# Sample data
# -----------------------------
files_and_data = [
    ['robot_data_60_0_28_01_26_13_41_44.pkl', 67 / 100],
    ['robot_data_60_0_28_01_26_13_43_41.pkl', 68 / 100],
    ['robot_data_60_0_28_01_26_13_37_15.pkl', 113 / 100],
    ['robot_data_60_0_28_01_26_13_35_18.pkl', 107 / 100],
    ['robot_data_60_0_28_01_26_13_41_10.pkl', 65 / 100],
    ['robot_data_60_0_28_01_26_13_42_55.pkl', 70 / 100],
    ['robot_data_60_0_28_01_26_13_39_36.pkl', 138 / 100],
    ['robot_data_60_0_28_01_26_13_42_19.pkl', 69 / 100],
    ['robot_data_60_0_28_01_26_13_36_10.pkl', 109 / 100],
    ['robot_data_60_0_28_01_26_13_33_20.pkl', 100 / 100],
    ['robot_data_60_0_28_01_26_13_34_28.pkl', 103 / 100],
]

files_and_data_curve = [
    ['robot_data_60_10_28_01_26_13_44_28.pkl', 61 / 100, 31 / 100],
    ['robot_data_60_10_28_01_26_13_45_14.pkl', 61 / 100, 32 / 100],
    ['robot_data_60_10_28_01_26_13_45_56.pkl', 61 / 100, 30 / 100],
    ['robot_data_60_10_28_01_26_13_46_26.pkl', 61 / 100, 31 / 100],
    ['robot_data_60_10_28_01_26_13_47_10.pkl', 62 / 100, 29 / 100],
    ['robot_data_60_10_28_01_26_13_48_25.pkl', 70 / 100, 106 / 100],
    ['robot_data_60_10_28_01_26_13_49_08.pkl', 73 / 100, 106 / 100],
    ['robot_data_60_10_28_01_26_13_50_55.pkl', 73 / 100, 71 / 100],
    ['robot_data_60_10_28_01_26_13_51_34.pkl', 76 / 100, 69 / 100],
    ['robot_data_60_10_28_01_26_13_52_07.pkl', 78 / 100, 71 / 100],
    ['robot_data_60_10_28_01_26_13_52_35.pkl', 76 / 100, 70 / 100],
    ['robot_data_60_10_28_01_26_13_53_08.pkl', 76 / 100, 71 / 100],
]


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    # Plot the motion model predictions for a single trial
    if True:
        filename = './data/robot_data_0_0_10_03_26_23_56_30.pkl'
        run_my_model_on_trial(filename)

    # Plot the motion model predictions for each trial in a folder
    if False:
        directory = './data/'
        plot_many_trial_predictions(directory)

    if False:
        directory = './data/'
        plot_many_trial_predictions_one_by_one(directory)

    # Compare predicted with actual distances
    if False:
        directory = './data_straight/'
        process_files_and_plot(files_and_data, directory)

    if False:
        directory = './data_curve/'
        process_files_and_plot_curve(files_and_data_curve, directory)

    # Sample with the motion model
    if False:
        sample_model(200)

    # Try to load some camera data from a single trial
    if False:
        filename = './data/robot_data_68_0_06_02_26_17_12_19.pkl'
        (
            time_list,
            encoder_count_list,
            velocity_list,
            steering_angle_list,
            x_camera_list,
            y_camera_list,
            z_camera_list,
            yaw_camera_list,
        ) = get_file_data(filename)

        wheel_radius = 0.034
        encoder_counts_per_revolution = 152
        encoder_counts_to_distance = -2 * math.pi * wheel_radius / encoder_counts_per_revolution

        plt.plot(
            time_list,
            (np.array(encoder_count_list) - encoder_count_list[0]) * encoder_counts_to_distance,
            'k',
        )
        plt.plot(time_list[:len(x_camera_list)], x_camera_list, 'g')
        plt.plot(time_list[:len(y_camera_list)], y_camera_list, 'b')
        plt.plot(time_list[:len(z_camera_list)], z_camera_list, 'c')
        plt.legend(['Encoder s', 'x', 'y', 'z'])
        plt.show()