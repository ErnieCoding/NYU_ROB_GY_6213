# External libraries
import math
import numpy as np

# UDP parameters
# localIP = "192.168.4.2" # LAPTOP'S IP ON ESP32-CAM
localIP = "192.168.0.196" # LAPTOP'S IP on wifi router Tenda_7F76C0
# localIP = "192.168.0.199" # LAPTOP'S IP ON WIFI ROUTER Tenda_9C9E80

# arduinoIP = "192.168.4.3" # ARDUINO'S IP ON ESP32-CAM
arduinoIP = "192.168.0.197" # ARDUINO'S IP ON WIFI ROUTER Tenda_7F76C0
# arduinoIP = "192.168.0.198" # ARDUINO'S IP ON WIFI ROUTER Tenda_9C9E80
localPort = 4010
arduinoPort = 4010
bufferSize = 1024

# Camera parameters
camera_url = "http://192.168.0.195:81/stream" # CAMERA'S IP ON THE WIFI ROUTER
# camera_url = "http://192.168.0.195:81/stream" # CAMERA'S IP ON ESP32-CAM
marker_length = 0.095
camera_matrix = np.array([[ 348.0321225, 0, 188.28539766 ],
                          [ 0, 355.6030584, 105.48395813 ],
                          [ 0, 0, 1 ]], dtype=np.float32)
dist_coeffs = np.array([ 0.08988576, -2.60316437, -0.02748353, 0.02115694, 13.16262713], dtype=np.float32)

# Robot parameters
num_robot_sensors = 2 # 2 encoders
num_robot_control_signals = 2 # 2 encoders
wheel_base = 0.215

# Logging parameters
max_num_lines_before_write = 1
filename_start = './data/robot_data'
data_name_list = ['time', 'control_signal', 'robot_sensor_signal', 'camera_sensor_signal', 'state_mean', 'state_covariance']

# Experiment trial parameters
trial_time = 20000 # milliseconds
extra_trial_log_time = 20000 # milliseconds

# KF parameters
I3 = np.array([[1, 0, 0],[0, 1, 0], [0, 0, 1]])
covariance_plot_scale = 100
 

# TODO: Update variances for our Lidar, Camera, and encoders

# Parameters for adaptive measurement variance calculations - used during EKF correction
LIDAR_CALIB_DIST = np.array([300, 400, 500, 600, 700, 800, 900, 1000, 1200])
LIDAR_CALIB_BIAS = np.array([5.9245, 13.2960, 9.7610, 6.0795, 9.1545,
                        9.1160,  4.0555,  6.7945,  3.6305])
LIDAR_COVARIANCE_FLOOR = 0.02
C_LINEAR = 6.433919e-04
B_LINEAR = -0.195505


encoder_left_variance = ...
encoder_right_variance = ...
camera_variance = ...


# Map parameters
corners = {
    "O": (0, 0),
    "P": (48, 0),
    "A": (0, 203),
    "B": (79, 203),
    "C": (79, 263),
    "D": (162, 263),
    "E": (162, 203),
    "F": (245, 203),
    "G": (245, 285),
    "H": (308, 285),
    "I": (308, 122),
    "Y": (274, 122),
    "J": (274, 54),
    "Z": (193, 54),
    "K": (193, 13),
    "L": (130, 13),
    "M": (130, 54),
    "N": (48, 54),
    
}

# LEFT TABLE
corners["Q"] = (48, 105)
corners["S"] = (47, 136)
corners["R"] = (88, 105)
corners["T"] = (87, 136)

# RIGHT TABLE
corners["U"] = (167, 144)
corners["V"] = (167, 108)
corners["X"] = (208, 144)
corners["W"] = (208, 108)


walls = [
    ("O", "P"),
    ("P", "N"),
    ("O", "A"),

    ("A", "B"),
    ("B", "C"),
    ("C", "D"),
    ("D", "E"),
    ("E", "F"),
    ("F", "G"),
    ("G", "H"),
    ("H", "I"),

    ("I", "Y"),
    ("Y", "J"),
    ("J", "Z"),
    ("Z", "K"),
    ("K", "L"),
    ("L", "M"),
    ("M", "N"),

    #first inner table:
    ("Q", "S"),
    ("S", "T"),
    ("T", "R"),
    ("R", "Q"),


    #second  inner table:
    ("U", "V"),
    ("V", "W"),
    ("W", "X"),
    ("X", "U"),
]


tags = {
    # From O
    0: (corners["O"][0] + 20, corners["O"][1]),
    1: (corners["O"][0], corners["O"][1] + 94),

    # From A
    3: (corners["A"][0] + 12, corners["A"][1]),
    2: (corners["A"][0], corners["A"][1] - 30),

    # From T
    4: (corners["T"][0], corners["T"][1] - 17),

    # From C
    5: (corners["C"][0] + 48, corners["C"][1]),

    # From D
    6: (corners["D"][0], corners["D"][1] - 31),

    # From U
    7: (corners["U"][0], corners["U"][1] - 18),

    # From G
    8: (corners["G"][0] + 30, corners["G"][1]),

    # From H
    9: (corners["H"][0], corners["H"][1] - 91),

    # From J
    10: (corners["J"][0], corners["J"][1] + 54),
    11: (corners["J"][0] - 32, corners["J"][1]),

    # From K
    12: (corners["K"][0] - 30, corners["K"][1]),
    14: (corners["K"][0], corners["K"][1] + 32),

    # From L
    13: (corners["L"][0], corners["L"][1] + 30),

    # From Q 
    15: (corners["Q"][0] + 20, corners["Q"][1]),
}
