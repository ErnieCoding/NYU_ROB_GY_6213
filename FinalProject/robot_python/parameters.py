# External libraries
import math
import numpy as np

# UDP parameters
# localIP = "192.168.4.2" # LAPTOP'S IP ON ESP32-CAM
localIP = "192.168.0.196" # LAPTOP'S IP on wifi router

# arduinoIP = "192.168.4.3" # ARDUINO'S IP ON ESP32-CAM
arduinoIP = "192.168.0.197" # ARDUINO'S IP ON WIFI ROUTER
localPort = 4010
arduinoPort = 4010
bufferSize = 1024

# Camera parameters
camera_url = "" # CAMERA'S IP ON THE WIFI ROUTER
# camera_url = "http://192.168.0.195:81/stream" # CAMERA'S IP ON ESP32-CAM
marker_length = 0.067
camera_matrix = np.array([[ 348.0321225, 0, 188.28539766 ],
                          [ 0, 355.6030584, 105.48395813 ],
                          [ 0, 0, 1 ]], dtype=np.float32)
dist_coeffs = np.array([ 0.08988576, -2.60316437, -0.02748353, 0.02115694, 13.16262713], dtype=np.float32)

# Robot parameters
num_robot_sensors = 2 # 2 encoders
num_robot_control_signals = 2 # 2 encoders

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
 
# OLD ROOM WALLS
wall_corner_list = [
    [0, 0, 390/100, 0], 
    [390/100, 0, 390/100, 110/100], 
    [390/100, 110/100,0, 200/100],
    [ 0, 200/100,0, 0],
]


# NEW ROOM WALLS
# wall_corner_list = [
#     [0, 0, 390/100, 0],
#     [390/100, 0, 390/100, 200/100],
#     [390/100, 200/100, 257/100, 200/100],
#     [257/100, 200/100, 257/100, 246/100],
#     [257/100, 246/100, 130/100, 246/100],
#     [130/100, 246/100, 130/100, 200/100],
#     [130/100, 200/100, 0, 200/100],
#     [0, 200/100, 0, 0]
# ]

# PROFESSOR'S ROOM
# wall_corner_list = [
#     [0, 0, 2.74, 0], 
#     [0, 0, 0, 3.78], 
#     [0, 3.78, 1.92, 3.78],
#     [1.03, 1.61, 1.03, 2.19],
#     [1.03, 2.19, 1.41, 2.19],
#     [1.92, 3.78, 1.92, 3.32],
#     [1.92, 3.32, 2.74, 3.32],
#     [2.74, 3.32, 2.74, 0]
# ]

# TODO: Update variances for our Lidar, Camera, and encoders
lidar_variance = 0.0225
encoder_variance = ...
camera_variance = ...
