# External libraries
import math
import numpy as np

# UDP parameters
localIP = "192.168.0.199" # Put your laptop computer's IP here 199
arduinoIP = "192.168.0.200" # Put your arduino's IP here 200
localPort = 4010
arduinoPort = 4010
bufferSize = 1024

# Camera parameters
camera_id = 0
marker_length = 0.067
camera_matrix = np.array([[1.03843829e+03, 0.00000000e+00, 5.70058553e+02],
  [0.00000000e+00, 1.06325837e+03, 3.09600558e+02],
  [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=np.float32)
dist_coeffs = np.array([-0.41925883,  0.32857265,  0.00174434, -0.00148671, -0.21424311], dtype=np.float32)


# Robot parameters
num_robot_sensors = 2 # encoder, steering
num_robot_control_signals = 2 # speed, steering

# Logging parameters
max_num_lines_before_write = 1
filename_start = './data/robot_data'
data_name_list = ['time', 'control_signal', 'robot_sensor_signal', 'camera_sensor_signal', 'state_mean', 'state_covariance']

# Experiment trial parameters
trial_time = 10000 # milliseconds
extra_trial_log_time = 2000 # milliseconds

# KF parameters
# Var(err_x)     = 0.0029696401 m^2
# Var(err_y)     = 0.0262374521 m^2
# Var(err_theta) = 0.0104829944 rad^2
Q_t = np.array([[0.0029696401, 0, 0],
                [0, 0.0262374521, 0], 
                [0, 0, 0.0104829944]])
I3 = np.array([[1,0,0], [0,1,0], [0,0,1]])
covariance_plot_scale = 100