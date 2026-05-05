# External Libraries
import matplotlib.pyplot as plt
from pathlib import Path
import math
import numpy as np
import sys

# Internal Libraries
# import FinalProject.robot_python.parameters as parameters
import robot_code

_APP_DIR     = Path(__file__).resolve().parent
_PYTHON_DIR  = _APP_DIR.parent
_PROJECT_DIR = _PYTHON_DIR.parent
_ROOT_DIR    = _PROJECT_DIR.parent

for _p in [str(_ROOT_DIR), str(_PROJECT_DIR), str(_PYTHON_DIR), str(_APP_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


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
    data_loader = robot_code.DataLoader(filename)
    data_dict = data_loader.load()

    time_list = data_dict.get('time', [])
    control_signal_list = data_dict.get('control_signal', [])
    robot_sensor_signal_list = data_dict.get('robot_sensor_signal', [])
    camera_sensor_signal_list = data_dict.get('camera_sensor_signal', [])

    encoder_left_count_list = []
    encoder_right_count_list = []
    x_camera_list = []
    y_camera_list = []
    z_camera_list = []
    yaw_camera_list = []

    for row in robot_sensor_signal_list:
        encoder_left_count_list.append(row.encoder_left_counts)
        encoder_right_count_list.append(row.encoder_right_counts)

    for row in camera_sensor_signal_list:
        x_camera_list.append(row[0])
        y_camera_list.append(row[1])
        z_camera_list.append(row[2])
        yaw_camera_list.append(row[5])

    time_list = _normalize_time_list(list(time_list))

    return (
        time_list,
        encoder_left_count_list,
        encoder_right_count_list,
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
# Motion data (fitting encoder counts - distance)
# -----------------------------
files_and_data = [
    # filename, encoder_left_start, encoder_right_start, encoder_left_finish, encoder_right_finish, x_finish, y_finish, theta_start, theta_finish
   {"filename": "robot_data_0_0_02_05_26_23_59_25.pkl",
    "encoder_left_start": 47944,
    "encoder_right_start": 48572,
    "encoder_left_finish": 49341,
    "encoder_right_finish": 49956,
    "x_finish": 54,
    "y_finish": 0,
    "theta_start": 102,
    "theta_finish": 100
   },
   {"filename": "robot_data_0_0_03_05_26_00_00_53.pkl",
    "encoder_left_start": 49341,
    "encoder_right_start": 49956,
    "encoder_left_finish": 52639,
    "encoder_right_finish": 53248,
    "x_finish": 125,
    "y_finish": 0,
    "theta_start": 102,
    "theta_finish": 99
   },
   {"filename": "robot_data_0_0_03_05_26_00_04_04.pkl",
    "encoder_left_start": 57586,
    "encoder_right_start": 58048,
    "encoder_left_finish": 62392,
    "encoder_right_finish": 62697,
    "x_finish": 178,
    "y_finish": -18,
    "theta_start": 103,
    "theta_finish": 111
   },
   {"filename": "robot_data_0_0_03_05_26_00_06_49.pkl",
    "encoder_left_start": 62397,
    "encoder_right_start": 62703,
    "encoder_left_finish": 69092,
    "encoder_right_finish": 69245,
    "x_finish": 244,
    "y_finish": -5,
    "theta_start": 104,
    "theta_finish": 110,

   },
   {"filename": "robot_data_0_0_03_05_26_00_09_46.pkl",
    "encoder_left_start": 69128,
    "encoder_right_start": 69260,
    "encoder_left_finish": 77339,
    "encoder_right_finish": 77266,
    "x_finish": 301,
    "y_finish": -7,
    "theta_start": 103,
    "theta_finish": 114
   },
   {"filename": "robot_data_0_0_03_05_26_00_12_09.pkl",
    "encoder_left_start": 77339,
    "encoder_right_start": 77269,
    "encoder_left_finish": 88563,
    "encoder_right_finish": 88004,
    "x_finish": 378,
    "y_finish": 112,
    "theta_start": 105,
    "theta_finish": 134
   },
   {"filename": "robot_data_0_0_03_05_26_00_15_48.pkl",
    "encoder_left_start": 88565,
    "encoder_right_start": 88010,
    "encoder_left_finish": 91788,
    "encoder_right_finish": 91185,
    "x_finish": 95,
    "y_finish": 74,
    "theta_start": 68,
    "theta_finish": 70
   },
   {"filename": "robot_data_0_0_03_05_26_00_16_58.pkl", # starting angle unsure
    "encoder_left_start": 91788,
    "encoder_right_start": 91185,
    "encoder_left_finish": 96945,
    "encoder_right_finish": 96280,
    "x_finish": 150,
    "y_finish": 116,
    "theta_start": 68,
    "theta_finish": 69
   },
   {"filename": "robot_data_0_0_03_05_26_00_18_46.pkl",
    "encoder_left_start": 97433,
    "encoder_right_start": 96409,
    "encoder_left_finish": 103146,
    "encoder_right_finish": 101933,
    "x_finish": 145,
    "y_finish": 151,
    "theta_start": 61,
    "theta_finish": 72
   },
   {"filename": "robot_data_0_0_03_05_26_00_20_15.pkl",
    "encoder_left_start": 103157,
    "encoder_right_start": 101970,
    "encoder_left_finish": 112723,
    "encoder_right_finish": 111245,
    "x_finish": 298,
    "y_finish": 164,
    "theta_start": 73,
    "theta_finish": 82
   },
   {"filename": "robot_data_0_0_03_05_26_00_23_04.pkl",
    "encoder_left_start": 113203,
    "encoder_right_start": 111446,
    "encoder_left_finish": 119954,
    "encoder_right_finish": 117893,
    "x_finish": 127,
    "y_finish": 201,
    "theta_start": 41,
    "theta_finish": 62
   },
   {"filename": "robot_data_0_0_03_05_26_00_27_17.pkl",
    "encoder_left_start": 120065,
    "encoder_right_start": 118062,
    "encoder_left_finish": 130905,
    "encoder_right_finish": 128392,
    "x_finish": 370,
    "y_finish": 94,
    "theta_start": 80,
    "theta_finish": 113
   },
   # Tests with turns
   {"filename": "robot_data_0_0_03_05_26_00_46_02.pkl",
    "encoder_left_start": 241357,
    "encoder_right_start": 237367,
    "encoder_left_finish": 247655,
    "encoder_right_finish": 244888,
    "x_finish": 54,
    "y_finish": 138,
    "theta_start": 116,
    "theta_finish": 10
   },
   {"filename": "robot_data_0_0_03_05_26_00_50_46.pkl",
    "encoder_left_start": 247760,
    "encoder_right_start": 244937,
    "encoder_left_finish": 256471,
    "encoder_right_finish": 252057,
    "x_finish": 165,
    "y_finish": 130,
    "theta_start": 26,
    "theta_finish": 159
   },
   {"filename": "robot_data_0_0_03_05_26_00_52_07.pkl", # starting angle unsure
    "encoder_left_start": 247760,
    "encoder_right_start": 244937,
    "encoder_left_finish": 266596,
    "encoder_right_finish": 261528,
    "x_finish": 325,
    "y_finish": 95,
    "theta_start": 75,
    "theta_finish": 127
   },
   {"filename": "robot_data_0_0_03_05_26_00_54_33.pkl",
    "encoder_left_start": 266598,
    "encoder_right_start": 261554,
    "encoder_left_finish": 277133,
    "encoder_right_finish": 271365,
    "x_finish": 318,
    "y_finish": 102,
    "theta_start": 75,
    "theta_finish": 134 
   },
   {"filename": "robot_data_0_0_03_05_26_01_04_09.pkl",
    "encoder_left_start": 277147,
    "encoder_right_start": 271376,
    "encoder_left_finish": 286109,
    "encoder_right_finish": 281384,
    "x_finish": 60,
    "y_finish": 254,
    "theta_start": 357,
    "theta_finish": 88 
   },
   {"filename": "robot_data_0_0_03_05_26_01_06_58.pkl",
    "encoder_left_start": 286889,
    "encoder_right_start": 283268,
    "encoder_left_finish": 288261,
    "encoder_right_finish": 286278,
    "x_finish": -5,
    "y_finish": 66,
    "theta_start": 107,
    "theta_finish": 327
   },
   {"filename": "robot_data_0_0_03_05_26_01_08_41.pkl",
    "encoder_left_start": 288318,
    "encoder_right_start": 286281,
    "encoder_left_finish": 291519,
    "encoder_right_finish": 291632,
    "x_finish": -9,
    "y_finish": 93,
    "theta_start": 110,
    "theta_finish": 286
   },
   {"filename": "robot_data_0_0_03_05_26_01_10_17.pkl",
    "encoder_left_start": 291531,
    "encoder_right_start": 291635,
    "encoder_left_finish": 295132,
    "encoder_right_finish": 293626,
    "x_finish": 82,
    "y_finish": 5,
    "theta_start": 51,
    "theta_finish": 187
   },
   {"filename": "robot_data_0_0_03_05_26_01_12_44.pkl",
    "encoder_left_start": 299149,
    "encoder_right_start": 294882,
    "encoder_left_finish": 301506,
    "encoder_right_finish": 295753,
    "x_finish": 48,
    "y_finish": -5,
    "theta_start": 58,
    "theta_finish": 185
   },
   {"filename": "robot_data_0_0_03_05_26_01_13_51.pkl",
    "encoder_left_start": 301631,
    "encoder_right_start": 295821,
    "encoder_left_finish": 305567,
    "encoder_right_finish": 297609,
    "x_finish": 64,
    "y_finish": 7,
    "theta_start": 30,
    "theta_finish": 212
   },
]

files_and_data_spinning = [
    # going clockwise
    {
        "filename": "robot_data_0_0_03_05_26_13_20_10.pkl",
        "encoder_left_start": 3017,
        "encoder_right_pass": -1753,
        "encoder_left_finish": 3762,
        "encoder_right_finish": -2084,
        "theta_start": 17,
        "theta_finish": 136,
    },
    {
        "filename": "robot_data_0_0_03_05_26_13_21_47.pkl",
        "encoder_left_start": 3766,
        "encoder_right_pass": -2071,
        "encoder_left_finish": 5011,
        "encoder_right_finish": -2831,
        "theta_start": 104,
        "theta_finish": 301,
    },
    {
        "filename": "robot_data_0_0_03_05_26_13_23_37.pkl",
        "encoder_left_start": 5011,
        "encoder_right_pass": -2831,
        "encoder_left_finish": 7042,
        "encoder_right_finish": -4143,
        "theta_start": 300,
        "theta_finish": 252,
    },
    {
        "filename": "robot_data_0_0_03_05_26_13_34_11.pkl",
        "encoder_left_start": 4451,
        "encoder_right_pass": 6728,
        "encoder_left_finish": 6532,
        "encoder_right_finish": 5374,
        "theta_start": 269,
        "theta_finish": 228,
    },
    {
        "filename": "robot_data_0_0_03_05_26_13_35_47.pkl",
        "encoder_left_start": 6532,
        "encoder_right_pass": 5374,
        "encoder_left_finish": 9526,
        "encoder_right_finish": 3366,
        "theta_start": 227,
        "theta_finish": 317,
    },
    {
        "filename": "robot_data_0_0_03_05_26_13_38_35.pkl", # spinning clockwise 3 times
        "encoder_left_start": 7315,
        "encoder_right_pass": 6623,
        "encoder_left_finish": 14585,
        "encoder_right_finish": 1486,
        "theta_start": 174,
        "theta_finish": 174,
    },

    # going counter clockwise
    {
        "filename": "robot_data_0_0_03_05_26_13_26_38.pkl",
        "encoder_left_start": 7042,
        "encoder_right_pass": -4143,
        "encoder_left_finish": 6887,
        "encoder_right_finish": -3624,
        "theta_start": 251,
        "theta_finish": 161,
    },
    {
        "filename": "robot_data_0_0_03_05_26_13_27_55.pkl",
        "encoder_left_start": 6887,
        "encoder_right_pass": -3624,
        "encoder_left_finish": 6664,
        "encoder_right_finish": -3046,
        "theta_start": 161,
        "theta_finish": 55,
    },
    {
        "filename": "robot_data_0_0_03_05_26_13_28_52.pkl",
        "encoder_left_start": 6664,
        "encoder_right_pass": -3046,
        "encoder_left_finish": 6017,
        "encoder_right_finish": -1978,
        "theta_start": 55,
        "theta_finish": 226,
    },
    {
        "filename": "robot_data_0_0_03_05_26_13_30_05.pkl",
        "encoder_left_start": 6026,
        "encoder_right_pass": -1836,
        "encoder_left_finish": 4803,
        "encoder_right_finish": -83,
        "theta_start": 200,
        "theta_finish": 267,
    },
    {
        "filename": "robot_data_0_0_03_05_26_13_32_47.pkl",
        "encoder_left_start": 5186,
        "encoder_right_pass": 5419,
        "encoder_left_finish": 4451,
        "encoder_right_finish": 6728,
        "theta_start": 124,
        "theta_finish": 269,
    },
    {
        "filename": "robot_data_0_0_03_05_26_13_36_46.pkl",
        "encoder_left_start": 9526,
        "encoder_right_pass": 3366,
        "encoder_left_finish": 7315,
        "encoder_right_finish": 6623,
        "theta_start": 317,
        "theta_finish": 175,
    },
    {
        "filename": "robot_data_0_0_03_05_26_13_40_13.pkl", # spinning counter clockwise 3 times
        "encoder_left_start": 14585,
        "encoder_right_pass": 1486,
        "encoder_left_finish": 9322,
        "encoder_right_finish": 8245,
        "theta_start": 173,
        "theta_finish": 190,
    },
]


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    # Plot the motion model predictions for a single trial
    if True:
        filename = '/home/ernest/Desktop/NYU_ROB_GY_6213/FinalProject/robot_python/data/trial runs/robot_data_0_0_03_05_26_19_01_49.pkl'
        data = get_file_data(filename)
        encoder_left, encoder_right = data[1], data[2]
        camera_x, camera_y, camera_z = data[6], data[7], data[8]

        for i in range(len(camera_x)):
            print(f"[CAMERA] X: {camera_x} || Y: {camera_y} || Z: {camera_z}\n")

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