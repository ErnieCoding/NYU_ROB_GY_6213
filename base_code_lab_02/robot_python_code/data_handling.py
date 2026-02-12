# TODO: Save all plots as png or jpg, especially for trajectories
import matplotlib.pyplot as plt
from pathlib import Path
import math
import numpy as np

# Internal Libraries
import parameters
import robot_python_code
import motion_models

# Data import from trials - measured_data.py
from measured_data import encoder_data, rotational_velocity_data

def get_file_data(filename:str) -> list:
    """
    Opens a file and returns data in a form ready to plot

    Arguments:
        - filename: path to file to open
    
    Returns:
        - list: the list with time, encoder count, velocity, and steering anglee lists
    """
    data_loader = robot_python_code.DataLoader(filename)
    data_dict = data_loader.load()

    # The dictionary should have keys ['time', 'control_signal', 'robot_sensor_signal', 'camera_sensor_signal']
    time_list = data_dict['time']
    control_signal_list = data_dict['control_signal']
    robot_sensor_signal_list = data_dict['robot_sensor_signal']
    encoder_count_list = []
    velocity_list = []
    steering_angle_list = []
    for row in robot_sensor_signal_list:
        encoder_count_list.append(row.encoder_counts)
    for row in control_signal_list:
        velocity_list.append(row[0])
        steering_angle_list.append(row[1])
    
    return time_list, encoder_count_list, velocity_list, steering_angle_list

def run_my_model_on_trial(filename: str, show_plot: bool = True, plot_color: str = 'r') -> None:
    """
    Plots a trajectory of a single file using the motion model

    Arguments:
        - filename: path to a file to be processed
        - show_plot: show plot
        - plot_color: select color of the plot
    """
    time_list, encoder_count_list, velocity_list, steering_angle_list = get_file_data(filename)

    motion_model = motion_models.MyMotionModel([0,0,0], 0)
    x_list, y_list, theta_list = motion_model.traj_propagation(time_list, encoder_count_list, steering_angle_list)

    plt.plot(x_list, y_list, plot_color)
    plt.title('Motion Model Predicted XY Traj (m)')
    plt.axis([-0.5, 1.5, -1, 1])
    if show_plot:

        plt.show()


# Iterate through many trials and plot them as trajectories with motion model
def plot_many_trial_predictions(directory:str) -> None:
    """
    Iterates through directory for each trial and plots them as trajectories using the motion model

    Argument: 
        - directory: path to directory where files are located
    """
    directory_path = Path(directory)
    plot_color_list = ['r.','k.','g.','c.', 'b.', 'r.','k.','g.','c.', 'b.','r.','k.','g.','c.', 'b.', 'r.','k.','g.','c.', 'b.','r.','k.','g.','c.', 'b.', 'r.','k.','g.','c.', 'b.','r.','k.','g.','c.', 'b.', 'r.','k.','g.','c.', 'b.']
    count = 0
    for item in directory_path.iterdir():
        filename = item.name
        plot_color = plot_color_list[count]
        run_my_model_on_trial(directory + filename, False, plot_color)
        count += 1
    plt.show()

# Calculate the predicted distance from single trial for a motion model
def run_my_model_to_predict_distance(filename):
    """
    Calculates the predicted distance from a signle trial file using the motion model.

    Arguments:
        - filename: path to a file to be processed
    """
    time_list, encoder_count_list, velocity_list, steering_angle_list = get_file_data(filename)
    motion_model = motion_models.MyMotionModel([0,0,0], 0)
    x_list, _, _ = motion_model.traj_propagation(time_list, encoder_count_list, steering_angle_list)
    distance = x_list[-30]
    
    return distance

# Calculate the differences between two lists, and square them.
def get_diff_squared(m_list,p_list):
    diff_squared_list = []
    for i in range(len(m_list)):
        diff_squared = math.pow(m_list[i]-p_list[i],2)
        diff_squared_list.append(diff_squared)

    coefficients = np.polyfit(m_list, diff_squared_list, 2)
    p=np.poly1d(coefficients)

    plt.plot(m_list, diff_squared_list,'ko')
    plt.plot(m_list, p(m_list),'ro')
    plt.title("Error Squared (m^2)")
    plt.xlabel('Measured distance travelled (m)')
    plt.ylabel('(Actual - Predicted)^2 (m^2)')
    plt.show()

    return diff_squared_list


def process_files_and_plot(files_and_data: list, directory: str) -> None:
    """
    Processes trial files with actual measured data and plot against predicted distances from our model

    Arguments:
        - files_and_data: a list of filenames and measured distances in the form of (filename, distance)
        - directory: the name of the directory where the trial files are located
    """
    predicted_distance_list = []
    measured_distance_list = []
    for row in files_and_data:
        filename = row[0]
        measured_distance = row[1]
        measured_distance_list.append(measured_distance)
        predicted_distance = run_my_model_to_predict_distance(directory + filename)
        predicted_distance_list.append(predicted_distance)
        print(f"Processing file: {filename} with measured distance: {measured_distance:.2f} m predicted distance: {predicted_distance:.2f} m")
    # Plot predicted and measured distance travelled.
    plt.plot(measured_distance_list+[0], predicted_distance_list+[0], 'ko')
    plt.plot([0,1.7],[0,1.7])
    plt.title('Distance Trials')
    plt.xlabel('Measured Distance (m)')
    plt.ylabel('Predicted Distance (m)')
    plt.legend(['Measured vs Predicted', 'Slope 1 Line'])
    plt.show()

    # Plot the associated variance
    get_diff_squared(measured_distance_list, predicted_distance_list)


# Sample and plot some simulated trials
def sample_model(num_samples:int, iteration:int):
    traj_duration = 10
    for i in range(num_samples):
        model = motion_models.MyMotionModel([0,0,0], 0)
        traj_x, traj_y, traj_theta = model.generate_simulated_traj(traj_duration)
        plt.plot(traj_x, traj_y, 'k.')

    plt.title('Sampling the model')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.savefig(f"./plotted_data/simulated_trials_100/simulation.png")
    plt.show()

def plot_trial_basics(encoder_data: list, rotational_velocity_data: list) -> None:
    """
    Fits functions and plots data measured during trials for step 4 & 5 of Lab 2. 

    Arguments:
        - encoder_data: encoder vs distance data with tuples of (time_s, encoder_start, encoder_end, x_g, y_g, distance)
        - rotational_velocity_data: rotational velocity vs steering angle/speed data with tuples of (time_s, speed, steering angle, X_g, Y_g, distance)
    """

    # STEP 4
    distances_traveled = []
    encoder_counts = []
    endpoint_coordinates = []

    for trial in encoder_data:
        time_in_seconds, encoder_start, encoder_end, x_g, y_g = trial[0], trial[1], trial[2], trial[3], trial[4]

        endpoint_coordinates.append((x_g, y_g))

        distance_trial = (math.sqrt(x_g**2 + y_g**2))/100
        encoder_count_trial = encoder_end - encoder_start

        distances_traveled.append(distance_trial)
        encoder_counts.append(encoder_count_trial)
    
    # Converted to numpy for math operations
    x_val = np.array(encoder_counts) # X-value: encoder counts
    y_val = np.array(distances_traveled) # Y-value: distance traveled calculated from x_g, y_g
    endpoint_coordinates = np.array(endpoint_coordinates)
    measured_x = endpoint_coordinates[:, 0] / 100
    measured_y = endpoint_coordinates[:, 1] / 100  

    # Predict distance vs encoder counts
    m = np.sum(x_val * y_val) / np.sum(x_val**2)
    b = 0
    distance_predicted = m * x_val + b
    
    # Variance function f_ss(e)
    residuals = y_val - distance_predicted
    sigma_sq_actual = residuals ** 2


    # polynomial - apparently wrong
    # p_a, p_b, p_c = np.polyfit(x_val, sigma_sq_actual, deg=2)
    # sigma_sq_predicted = p_a * (x_val**2) + p_b * x_val + p_c

    k = np.sum(x_val * sigma_sq_actual) / np.sum(x_val**2)
    sigma_sq_predicted = k * x_val
    
    print("-" * 30)
    print("STEP 4 FUNCTIONS\n")
    print("Distance Model f_se(e):")
    print(f"s = {m:.5f} * e")

    print("\nVariance Model f_ss(e):")
    print(f"sigma^2 = {k:.8f} * e")
    print("-" * 30)


    # PLOTTING
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))
    sort_idx = np.argsort(x_val)

    # STEP 4 PLOTS
    # Encoder vs Distance
    ax1.scatter(x_val, y_val, label='Measured Data')
    ax1.plot(x_val[sort_idx], distance_predicted[sort_idx], color='red', label='Prediction f_se(e)')
    ax1.set_title('Distance Calibration: f_se(e)')
    ax1.set_xlabel('Encoder Counts')
    ax1.set_ylabel('Distance (m)')
    ax1.legend()
    ax1.grid(True)

    # Variance
    ax2.scatter(x_val, sigma_sq_actual, label='Squared Residuals')
    ax2.plot(x_val[sort_idx], sigma_sq_predicted[sort_idx], color='green', label='Variance Fit f_ss(e)')
    ax2.set_title('Error Model: f_ss(e)')
    ax2.set_xlabel('Encoder Counts')
    ax2.set_ylabel('Squared Error (sigma^2)')
    ax2.legend()
    ax2.grid(True)

    # Endpoints x_g, y_g measured during trials
    predicted_x = distance_predicted
    predicted_y = np.zeros_like(predicted_x)
    ax3.scatter(measured_x, measured_y, label="Measured Endpoints", color = "green")
    ax3.set_title('Measured Endpoints X_g and Y_g')
    ax3.set_xlabel("X_g (m)")
    ax3.set_ylabel("Y_g (m)")
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()


    # STEP 5
    # Calculating rotational velocity
    rotationa_vel_data_updated = []
    theta_start = math.radians(45)
    for rotationa_vel_trial in rotational_velocity_data:
        time_s, speed, steering_angle, X_g, Y_g, distance = rotationa_vel_trial
        #calcualte the final angle
        final_angle=math.atan(Y_g/X_g)
        #calculate the angle in degrees
        # print(f"Final angle in degrees: {math.degrees(final_angle)} degrees")
        # print(f"Angle of rotation: {math.degrees(final_angle)-45} degrees")

        time_s, speed, steering_angle, X_g, Y_g, distance = rotationa_vel_trial

        # --- chord angle from origin to final position ---
        phi_atan2 = math.atan2(Y_g, X_g)  # robust, handles quadrants and X_g=0 safely

        # --- circular-arc correction: chord angle = theta_start + DeltaTheta/2 ---
        delta_theta = 2.0 * (phi_atan2 - theta_start)

        # --- yaw rate ---
        w = delta_theta / time_s

        rotationa_vel_data_updated.append((time_s, speed, steering_angle, w))
        
        # print(f"Time: {time_s}s, Speed: {speed}km/h, Steering Angle: {steering_angle}°, X_g: {X_g}, Y_g: {Y_g}, W: {w}") # printing trial data
    
    # fitting the steering angle and angular velocity data to a linear model
    steering_angles = np.array([d[2] for d in rotationa_vel_data_updated])
    angular_velocities = np.array([d[3] for d in rotationa_vel_data_updated])
    coefficients_angle = np.polyfit(steering_angles, angular_velocities, 1)
    linear_fit_rotation_velocity_steering = np.poly1d(coefficients_angle)

    #fitting the speed and angular velocity data to a linear model
    speed_values = np.array([d[1] for d in rotationa_vel_data_updated])
    coefficients_speed = np.polyfit(speed_values, angular_velocities, 1)
    # linear_fit_rotation_velocity_speed = np.poly1d(coefficients_speed)
    

    # Calculating variance
    variance_data_steering_angle = []
    variance_data_speed_angle = []


    for data in rotationa_vel_data_updated:
        steering_angles = data[2]
        
        predicted_anglular_velocity = linear_fit_rotation_velocity_steering(steering_angles)
        
        error_steering = predicted_anglular_velocity - data[3]
        
        error_steering_squared = error_steering**2
        
        # print(f"Steering Angle: {steering_angles} degrees, Predicted w: {predicted_anglular_velocity:.4f} radians/s, Actual w: {data[3]:.4f} radians/s, Error^2: {error_steering_squared:.4f} radians/s") # printing variance data
        
        variance_data_speed_angle.append((steering_angles, data[3], 
        predicted_anglular_velocity,error_steering_squared))
    
    # Fitting function for the error vs steering angle - linear
    steering_angles = [d[0] for d in variance_data_speed_angle]
    errors_steering_angle = [d[3] for d in variance_data_speed_angle]

    error_coefficients_steering = np.polyfit(steering_angles, errors_steering_angle, 1)
    error_linear_fit_steering = np.poly1d(error_coefficients_steering)

    # printing the functions:
    print("STEP 5 FUNCTIONS\n")
    linear_fit_rotation_velocity_steering_func = f"w = {coefficients_angle[0]:.4f} * steering_angle + {coefficients_angle[1]:.4f}"
    print(f"Linear fit function: w = {coefficients_angle[0]:.4f} * steering_angle + {coefficients_angle[1]:.4f}")
    
    linear_fit_rotation_velocity_speed_func = f"w = {coefficients_speed[0]:.4f} * speed + {coefficients_speed[1]:.4f}"
    print(f"Linear fit function: w = {coefficients_speed[0]:.4f} * speed + {coefficients_speed[1]:.4f}")
    
    print(f"Error Linear fit function: sigma_^2 = {error_coefficients_steering[0]:.4f} * steering_angle + {error_coefficients_steering[1]:.4f}")

    # STEP 5 PLOTS
    # Rotational Velocity vs Steering Angle
    fig, (ax3, ax4, ax5) = plt.subplots(3, 1, figsize=(10, 10))
    ax3.scatter(steering_angles, angular_velocities, c='blue', label='Steering Angle vs Rotational Velocity')
    ax3.plot(steering_angles, linear_fit_rotation_velocity_steering(steering_angles), c='red', label='Prediction w')
    ax3.set_title('Rotational Velocity vs Steering Angle:' + linear_fit_rotation_velocity_steering_func)   
    ax3.set_xlabel('Steering Angle(Degrees)')
    ax3.set_ylabel('Rotational Velocity (w) (radians/s)')
    ax3.legend()
    ax3.grid(True)

    # Rotational Velocity vs Speed - not applicable
    ax4.scatter(speed_values, angular_velocities, c='blue', label='Data Points')
    # plt.plot(speed_values, linear_fit_rotation_velocity_speed(speed_values), c='red', label='Linear Fit')
    ax4.set_title('Rotational Velocity vs Speed: Fit Function: ' + linear_fit_rotation_velocity_speed_func)   
    ax4.set_xlabel('Speed (m/s)')
    ax4.set_ylabel('Angular Velocity (w) (radians/s)')
    ax4.legend()
    ax4.grid(True)

    # Squared Error vs Steering Angle + Error
    ax5.scatter(steering_angles, errors_steering_angle, c='blue', label='Error Data Points')
    ax5.plot(steering_angles, error_linear_fit_steering(steering_angles), c='red', label='Error Linear Fit')
    ax5.set_title(f"Squared Error vs Steering Angle: sigma_^2 = {error_coefficients_steering[0]:.4f} * steering_angle + {error_coefficients_steering[1]:.4f}")
    ax5.set_xlabel('Steering Angle (Degrees)')
    ax5.set_ylabel('Squared Error (radians/s)^2')
    ax5.legend()
    ax5.grid(True)

    plt.tight_layout()
    plt.show()



######### MAIN ########
if __name__ == "__main__":
    # print("Printing time list from pickle file:\n\n")
    # get_file_data("data/robot_data_70_-5_07_02_26_00_02_47.pkl")

    # Some sample data to test with
    # files_and_data = [
    #     ['robot_data_60_0_28_01_26_13_41_44.pkl', 67/100], # filename, measured distance in meters
    #     ['robot_data_60_0_28_01_26_13_43_41.pkl', 68/100],
    #     ['robot_data_60_0_28_01_26_13_37_15.pkl', 113/100],
    #     ['robot_data_60_0_28_01_26_13_35_18.pkl', 107/100],
    #     ['robot_data_60_0_28_01_26_13_41_10.pkl', 65/100],
    #     ['robot_data_60_0_28_01_26_13_42_55.pkl', 70/100],
    #     ['robot_data_60_0_28_01_26_13_39_36.pkl', 138/100],
    #     ['robot_data_60_0_28_01_26_13_42_19.pkl', 69/100],
    #     ['robot_data_60_0_28_01_26_13_36_10.pkl', 109/100],
    #     ['robot_data_60_0_28_01_26_13_33_20.pkl', 100/100],
    #     ['robot_data_60_0_28_01_26_13_34_28.pkl', 103/100],
    # ]

    # Step 4 trial files and data
    files_and_data_step_4 = [
    ['robot_data_70_0_06_02_26_21_54_37.pkl'], # filename, measured distance in meters
    ['robot_data_70_0_06_02_26_21_56_10.pkl'],
    ['robot_data_70_0_06_02_26_21_57_17.pkl'],
    ['robot_data_70_0_06_02_26_21_58_39.pkl'], 
    ['robot_data_70_0_06_02_26_22_00_18.pkl'],
    ['robot_data_70_0_06_02_26_22_01_56.pkl'],
    ['robot_data_70_0_06_02_26_22_02_51.pkl'],
    ['robot_data_70_0_06_02_26_22_04_10.pkl'],
    ['robot_data_70_0_06_02_26_22_06_05.pkl'],
    ['robot_data_70_0_06_02_26_22_08_03.pkl'],
    ['robot_data_70_0_06_02_26_22_09_53.pkl'],
    ['robot_data_70_0_06_02_26_22_11_49.pkl'],
    ]

    for i in range(len(encoder_data)):
        x_g, y_g = encoder_data[i][3], encoder_data[i][4]
        measured_distance = math.sqrt(x_g ** 2 + y_g ** 2) / 100 # getting measured distances in meters

        files_and_data_step_4[i].append(measured_distance)
    
    # Step 5 trial files and data
    files_and_data_step_5 = [
        "robot_data_70_15_06_02_26_23_03_33.pkl",
        "robot_data_70_10_06_02_26_23_07_22.pkl", 
        "robot_data_70_15_06_02_26_23_08_15.pkl",
        "robot_data_70_-10_06_02_26_23_10_19.pkl",
        "robot_data_70_-15_06_02_26_23_13_00.pkl", 
        "robot_data_85_10_06_02_26_23_14_17.pkl", 
        "robot_data_85_-15_06_02_26_23_16_26.pkl",
        "robot_data_70_10_06_02_26_23_20_45.pkl", 
        "robot_data_70_7_06_02_26_23_23_48.pkl", 
        "robot_data_70_-10_06_02_26_23_26_13.pkl", 
        "robot_data_70_-10_06_02_26_23_28_44.pkl", 
        "robot_data_70_-7_06_02_26_23_31_53.pkl", 
        "robot_data_85_10_06_02_26_23_33_59.pkl", 
        "robot_data_85_-7_06_02_26_23_37_38.pkl", 
        "robot_data_85_-7_06_02_26_23_54_19.pkl", 
        "robot_data_70_7_06_02_26_23_57_07.pkl", 
        "robot_data_70_3_07_02_26_00_00_25.pkl", 
        "robot_data_70_-5_07_02_26_00_02_47.pkl", 
        "robot_data_70_-7_07_02_26_00_04_41.pkl", 
        "robot_data_85_3_07_02_26_00_07_25.pkl", 
        "robot_data_85_5_07_02_26_00_11_31.pkl", 
        "robot_data_85_-4_07_02_26_00_13_23.pkl", 
        "robot_data_85_-2_07_02_26_00_15_25.pkl"
    ]

    # Plot data from trials + fitted functions and variances
    # plot_trial_basics(encoder_data, rotational_velocity_data)
    

    # Plot the motion model predictions for a single trial
    if False:
        filename = './data/come_back_to_origin_diff_speed.pkl'
        run_my_model_on_trial(filename)

    # Plot the motion model predictions for each trial in a folder
    if False:
        directory = ('./data_stage5/')
        plot_many_trial_predictions(directory)

    # A list of files to open, process, and plot - for comparing predicted with actual distances
    if False:
        directory = ('./data_stage4/')    
        process_files_and_plot(files_and_data, directory)

    # Try to sample with the motion model
    if False:
        sample_model(200)




# DATA FROM TRIALS - ALSO IN measured_data.py
### task 4: starting timestamp 21:54:37 to 22:11:49
# encoder_data = [ # data from trials: (time_s, encoder_start, encoder_end, X_g, Y_g, distance)
#     (5, 22308, 24760, 66, 60, 73),
#     (5, 24760, 27193, 60, 62, 73),
#     (7, 27202, 30673, 75, 81, 103),
#     (7, 30673, 34114, 76, 81, 104),
#     (10, 34105, 39038, 114, 110, 146),
#     (10, 39038, 43962, 99, 120, 147),
#     (10, 47398, 52294, 106, 115, 147),
#     (15, 52294, 59316, 130, 167, 208),
#     (15, 59312, 66462, 160, 144, 223),
#     (15, 66462, 73668, 153, 151, 219),
#     (20, 73534, 82615, 190, 195, 270),
#     (20, 82615, 91672, 196, 190, 272)
# ]

# rotational_velocity_data = [ # data from rotational velocity trials: (time_s, speed, steering angle, X_g, Y_g, distance)
#     (5, 70, 10, 70, 36, 69),
#     (5, 70, 15, 70, 24, 66),
#     (5, 70, -10, 39, 56, 66),
#     (5, 70, -15, 23, 56, 57),
#     (5, 85, 10, 107, 40, 105),
#     (5, 85, -15, 10, 103, 104),
#     (10, 70, 10, 130, 33, 130),
#     (10, 70, 7, 132, 48, 137),
#     (10, 70, -10, 31, 100, 106),
#     (10, 70, -7, 55, 110, 123),
#     (10, 85, 10, 185, 17, 185),
#     (10, 85, -7, 30, 168, 170),
#     (15, 70, 7, 197, 41, 196),
#     (15, 70, 3, 170, 133, 196),
#     (15, 70, -5, 16, 134, 146),
#     (15, 70, -7, 36, 157, 167),
#     (15, 85, 3, 244, 174, 297),
#     (15, 85, 5, 290, 89, 306),
#     (15, 85, -4, 76, 267, 271),
#     (15, 85, -2, 157, 252, 286),
# ]

