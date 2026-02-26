# External libraries
import numpy as np
import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import argparse

# Local libraries
import parameters
import data_handling
import cv2
# Main class
class ExtendedKalmanFilter:
    def __init__(self, x_0, Sigma_0, encoder_counts_0, only_prediction:bool = False):
        self.state_mean = x_0
        self.state_covariance = Sigma_0
        self.predicted_state_mean = [0,0,0]
        self.predicted_state_covariance = np.array([[1,0,0], [0,1,0], [0,0,1]]) * 1.0
        self.last_encoder_counts = encoder_counts_0
        self.only_prediction = only_prediction

    # Call the prediction and correction steps
    def update(self, u_t, z_t, delta_t):
        if self.only_prediction or z_t is None:
            print("ONLY PREDICTION STEP\n")
            self.state_covariance, self.state_mean = self.prediction_step(u_t, delta_t)
        else:
            print("BOTH STEP\n")
            self.predicted_state_covariance, self.predicted_state_mean = self.prediction_step(u_t, delta_t)
            self.correction_step(z_t)

    # Set the EKF's predicted state mean and covariance matrix
    def prediction_step(self, u_t, delta_t):
        mu_pred, s = self.g_function(self.state_mean, u_t, delta_t)

        G_x_t = self.get_G_x(self.state_mean, s)
        G_u_t = self.get_G_u(self.state_mean, delta_t)
        R_t = self.get_R(s)
        
        sigma_t_pred = G_x_t @ self.state_covariance @ np.transpose(G_x_t) + G_u_t @ R_t @ np.transpose(G_u_t)
        
        return sigma_t_pred, mu_pred

    # Set the EKF's corrected state mean and covariance matrix
    def correction_step(self, z_t):
        H_t = self.get_H()
        Q_t = self.get_Q()

        inverse = np.linalg.inv(H_t @ self.predicted_state_covariance @ np.transpose(H_t) + Q_t)

        Kalman_gain = self.predicted_state_covariance @ np.transpose(H_t) @ inverse

        # print(f"\nKALMAN GAIN AT STEP: \n{Kalman_gain}\nDIAGONAL:\n {np.diag(Kalman_gain)}")


        residual = z_t - self.get_h_function(self.predicted_state_mean)
        residual[2] = (residual[2] + np.pi) % (2 * np.pi) - np.pi
        self.state_mean = self.predicted_state_mean + Kalman_gain @ residual

        self.state_covariance = (np.eye(3) - Kalman_gain @ H_t) @ self.predicted_state_covariance

    # Function to calculate distance from encoder counts
    def distance_travelled_s(self, encoder_counts):
        s = 0.0003063 * encoder_counts
        return s    
            
    # Function to calculate rotational velocity from steering and dist travelled or speed
    def rotational_velocity_w(self, steering_angle_command): 
        w = -0.0128 * steering_angle_command + 0.0016       
        return w

    # The nonlinear transition equation that provides new states from past states
    def g_function(self, x_tm1, u_t, delta_t):
        s = self.distance_travelled_s(u_t[0]-self.last_encoder_counts)
        self.last_encoder_counts = u_t[0]

        w = self.rotational_velocity_w(u_t[1])

        x = x_tm1[0] + s * math.cos(x_tm1[2])
        y = x_tm1[1] + s * math.sin(x_tm1[2])
        theta = x_tm1[2] + w * delta_t

        x_t = np.array([x, y, theta])
        return x_t, s
    
    #TODO: CHANGE IF INCORRECT
    # The nonlinear measurement function
    def get_h_function(self, x_t:list) -> list:
        """
        Converts predicted model state to camera space measurements z_t = [tvec_x, tvec_y, rvec_z]
        """
        return x_t
    
    # This function returns a matrix with the partial derivatives dg/dx
    # g outputs x_t, y_t, theta_t, and we take derivatives wrt inputs x_tm1, y_tm1, theta_tm1
    def get_G_x(self, x_tm1, s):
        G_02= -s*math.sin(x_tm1[2])
        G_12 = s * math.cos(x_tm1[2])
        return np.array([[1, 0, G_02], [0, 1, G_12], [0, 0, 1]])

    # This function returns a matrix with the partial derivatives dg/du
    def get_G_u(self, x_tm1, delta_t):     
        m = -0.0128           
        return np.array([[math.cos(x_tm1[2]), 0], [math.sin(x_tm1[2]), 0], [0, m * delta_t]])

    #TODO: CHANGE IF INCORRECT 
    # This function returns a matrix with the partial derivatives dh_t/dx_t
    def get_H(self):
        return np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    
    # This function returns the R_t matrix which contains transition function covariance terms.
    def get_R(self, s):
        var_s = 0.000000585994 * (s/0.0003063)

        var_w = 0.0009

        return np.array([[var_s, 0], [0, var_w]])

    # This function returns the Q_t matrix which contains measurement covariance terms.
    def get_Q(self):
        return parameters.Q_t

class KalmanFilterPlot:

    def __init__(self):
        self.dir_length = 0.1
        fig, ax = plt.subplots()
        self.ax = ax
        self.fig = fig
        plt.ion()                 # interactive mode ON
        plt.show(block=False)     # show ONCE, non-blocking


    def update(self, state_mean, state_covaraiance, t):
        plt.clf()

        # Plot covariance ellipse
        lambda_, v = np.linalg.eig(state_covaraiance)
        lambda_ = np.sqrt(lambda_)
        xy = (state_mean[0], state_mean[1])
        angle=np.rad2deg(np.arctan2(*v[:,0][::-1]))
        ell = Ellipse(xy, alpha=0.5, facecolor='red',width=lambda_[0], height=lambda_[1], angle = angle)
        ax = self.fig.gca()
        ax.add_artist(ell)
        
        # Plot state estimate
        plt.plot(state_mean[0], state_mean[1],'ro')
        plt.plot([state_mean[0], state_mean[0]+ self.dir_length*math.cos(state_mean[2]) ], [state_mean[1], state_mean[1]+ self.dir_length*math.sin(state_mean[2]) ],'r')
        plt.xlabel('X(m)')
        plt.ylabel('Y(m)')
        plt.axis([-2, 2, -2, 3])
        plt.grid()
        # plt.savefig(f"kalman_filter_plots/plot_{t}")

        # redraw the existing window instead of reopening/locking
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)
    
def transform_camera_to_world(tvec, rvec):
    # Ensure inputs are float32 numpy arrays
    ##THIS IS HARCODED AND CHANGES EVERY TIME YOU CHANGE CAMERA POSITION, SO UPDATE BEFORE RUNNING EKF
    # tvec_init = [-0.21500975, 0.56345664, 0.87650994]
    # rvec_init = [2.72888006, 0.45845892, 0.90720317]

    tvec_init = [-0.49274365, 0.05354998, 1.29572172]
    rvec_init = [ 2.17807045, -0.29410592, 0.29292633]

    tvec = np.array(tvec, dtype=np.float32)
    rvec = np.array(rvec, dtype=np.float32)
    t_init = np.array(tvec_init, dtype=np.float32)
    r_init = np.array(rvec_init, dtype=np.float32)

    # 1. Rotation Matrix for Camera Tilt at Origin
    R_init, _ = cv2.Rodrigues(r_init)

    # 2. Position Transformation (Un-tilting)
    t_diff = tvec - t_init
    # Project back to World Frame using the Transpose
    p_world_meters = np.dot(R_init.T, t_diff)

    # 3. Unit Conversion (meters to cm)
    x_world = p_world_meters[0] 
    y_world = p_world_meters[1] 

    # 4. Heading Transformation
    R_curr, _ = cv2.Rodrigues(rvec)
    R_rel = np.dot(R_init.T, R_curr)
    yaw_rad = np.arctan2(R_rel[1, 0], R_rel[0, 0])
    theta_world = np.degrees(yaw_rad)
    # Wrap to [-180, 180]
    theta_world = (theta_world + 180) % 360 - 180

    return x_world, y_world, math.radians(theta_world)


# Code to run your EKF offline with a data file.
def offline_efk(only_prediction = False):

    # Get data to filter
    filename = './data_in-out_frame/robot_data_0_0_24_02_26_23_06_34.pkl'
    ekf_data = data_handling.get_file_data_for_kf(filename)

    # Instantiate PF with no initial guess
    row_num = 0

    tvec = [ekf_data[row_num][3][0], ekf_data[row_num][3][1], ekf_data[row_num][3][2]]
    rvec = [ekf_data[row_num][3][3], ekf_data[row_num][3][4], ekf_data[row_num][3][5]]
    x_0 = np.array(transform_camera_to_world(tvec, rvec))


    Sigma_0 = np.array([[1,0,0], [0,1,0], [0,0,1]])
    encoder_counts_0 = ekf_data[row_num][2].encoder_counts
    extended_kalman_filter = ExtendedKalmanFilter(x_0, Sigma_0, encoder_counts_0, only_prediction)

    # Create plotting tool for ekf
    kalman_filter_plot = KalmanFilterPlot()

    #for keeping track if the camera measurement is stale (i.e. out-of-frame)
    last_pose = None
    stale = 0
    STALE_N = 10        # after 10 repeated frames => out-of-frame
    EPS = 1e-9         # small threshold
    last_encoder_counts = 0
    # Loop over sim data
    for t in range(1, len(ekf_data)):
        row = ekf_data[t] # Current row of data

        delta_t = ekf_data[t][0] - ekf_data[t-1][0] # time step size
        u_t = np.array([row[2].encoder_counts, row[2].steering]) # robot_sensor_signal
        z_t = np.array([row[3][0],row[3][1],row[3][2],row[3][3],row[3][4],row[3][5]])
        tvec = row[3][0:3]
        rvec = row[3][3:6]
        print(f"tvec: {tvec}\nrvec: {rvec}\n")

        pose = np.hstack([tvec, rvec])

        # Check if the camera is stuck on the exact same frame
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

        # Reject if the data is frozen AND we know the robot is actually moving
        if is_stale and encoder_change > 0:
            print(f"DEBUG: Prediction Only - Camera Stale (Frozen at {stale} frames)")
            z_t = None

        # Otherwise, the data is likely valid (either moving and changing, or stopped and same)
        else:
            z_t = np.array(transform_camera_to_world(tvec, rvec))


        # print(f"Time: {row[0]}, u_t: {u_t}, z_t: {z_t}, delta_t: {delta_t}")
        # Run the EKF for a time step
        extended_kalman_filter.update(u_t, z_t, delta_t)
    
        kalman_filter_plot.update(extended_kalman_filter.state_mean, extended_kalman_filter.state_covariance[0:2,0:2],t)
    

####### MAIN #######
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--only_prediction", action="store_true", help="Run EKF with only the prediction step")

    args = parser.parse_args()
    if True:
        offline_efk(args.only_prediction)
