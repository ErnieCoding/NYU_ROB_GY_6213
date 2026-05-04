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
encoder-right_variance = ...
camera_variance = ...
