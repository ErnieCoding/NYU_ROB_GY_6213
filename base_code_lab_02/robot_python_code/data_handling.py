# External Libraries
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

# Open a file and return data in a form ready to plot
def get_file_data(filename):
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


# For a given trial, plot the encoder counts, velocities, steering angles
# data from trials format: 
# encoder_data (time_s, encoder_start, encoder_end, X_g, Y_g, distance)
# rotational_velocity_data (time_s, speed, steering angle, X_g, Y_g, distance)
def plot_trial_basics():
    distances_traveled = []
    encoder_counts = []

    for trial in encoder_data:
        time_in_seconds, encoder_start, encoder_end, x_g, y_g = trial[0], trial[1], trial[2], trial[3], trial[4]

        distance_trial = math.sqrt(x_g**2 + y_g**2)
        encoder_count_trial = encoder_end - encoder_start

        distances_traveled.append(distance_trial)
        encoder_counts.append(encoder_count_trial)
    
    # Converted to numpy for math operations
    x_val = np.array(encoder_counts)
    y_val = np.array(distances_traveled)

    # Predict distance vs encoder counts
    m, b = np.polyfit(x_val, y_val, deg=1)
    distance_predicted = m * x_val + b
    
    # Variance function f_ss(e)
    residuals = y_val - distance_predicted
    sigma_sq_actual = residuals ** 2

    p_a, p_b, p_c = np.polyfit(x_val, sigma_sq_actual, deg=2)
    sigma_sq_predicted = p_a * (x_val**2) + p_b * x_val + p_c
    
    print("-" * 30)
    print("Distance Model f_se(e):")
    print(f"s = {m:.5f} * e + {b:.5f}")
    print("\nVariance Model f_ss(e):")
    print(f"sigma^2 = {p_a:.8f} * e^2 + {p_b:.5f} * e + {p_c:.5f}")
    print("-" * 30)


    # PLOTTING

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    sort_idx = np.argsort(x_val)

    # Distance
    ax1.scatter(x_val, y_val, label='Measured Data')
    ax1.plot(x_val[sort_idx], distance_predicted[sort_idx], color='red', label='Prediction f_se(e)')
    ax1.set_title('Distance Calibration: f_se(e)')
    ax1.set_xlabel('Encoder Counts')
    ax1.set_ylabel('Distance (cm)')
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

    plt.tight_layout()
    plt.show()

    # Rest of the plots
    # plt.plot(time_list, velocity_list)
    # plt.title('Speed')
    # plt.show()
    # plt.plot(time_list, steering_angle_list)
    # plt.title('Steering')
    # plt.show()


# Plot a trajectory using the motion model, input data ste from a single trial.
def run_my_model_on_trial(filename, show_plot = True, plot_color = 'ko'):
    time_list, encoder_count_list, velocity_list, steering_angle_list = get_file_data(filename)
    
    motion_model = motion_models.MyMotionModel([0,0,0], 0)
    x_list, y_list, theta_list = motion_model.traj_propagation(time_list, encoder_count_list, steering_angle_list)

    plt.plot(x_list, y_list,plot_color)
    plt.title('Motion Model Predicted XY Traj (m)')
    plt.axis([-0.5, 1.5, -1, 1])
    if show_plot:
        plt.show()


# Iterate through many trials and plot them as trajectories with motion model
def plot_many_trial_predictions(directory):
    directory_path = Path(directory)
    plot_color_list = ['r.','k.','g.','c.', 'b.', 'r.','k.','g.','c.', 'b.','r.','k.','g.','c.', 'b.', 'r.','k.','g.','c.', 'b.']
    count = 0
    for item in directory_path.iterdir():
        filename = item.name
        plot_color = plot_color_list[count]
        run_my_model_on_trial(directory + filename, False, plot_color)
        count += 1
    plt.show()

# Calculate the predicted distance from single trial for a motion model
def run_my_model_to_predict_distance(filename):
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


# Open files, plot them to predict with the motion model, and compare with real values
def process_files_and_plot(files_and_data, directory):
    predicted_distance_list = []
    measured_distance_list = []
    for row in files_and_data:
        filename = row[0]
        measured_distance = row[1]
        measured_distance_list.append(measured_distance)
        predicted_distance = run_my_model_to_predict_distance(directory + filename)
        predicted_distance_list.append(predicted_distance)

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
def sample_model(num_samples):
    traj_duration = 10
    for i in range(num_samples):
        model = motion_models.MyMotionModel([0,0,0], 0)
        traj_x, traj_y, traj_theta = model.generate_simulated_traj(traj_duration)
        plt.plot(traj_x, traj_y, 'k.')

    plt.title('Sampling the model')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.show()


######### MAIN ########
if __name__ == "__main__":
    plot_trial_basics() # call plotting function for data

    # Some sample data to test with
    # files_and_data = [
    #     ['robot_data_78_0_05_02_26_18_47_21.pkl', 67/100], # filename, measured distance in meters
    #     ['robot_data_100_0_05_02_26_18_38_56.pkl', 68/100],
    #     ['robot_data_60_0_28_01_26_13_37_15.pkl', 113/100],
    #     ['robot_data_60_0_28_01_26_13_35_18.pkl', 107/100],
    #     ['robot_data_60_0_28_01_26_13_41_10.pkl', 65/100],
    #     ['robot_data_60_0_28_01_26_13_42_55.pkl', 70/100],
    #     ['robot_data_60_0_28_01_26_13_39_36.pkl', 138/100],
    #     ['robot_data_60_0_28_01_26_13_42_19.pkl', 69/100],
    #     ['robot_data_60_0_28_01_26_13_36_10.pkl', 109/100],
    #     ['robot_data_60_0_28_01_26_13_33_20.pkl', 100/100],
    #     ['robot_data_60_0_28_01_26_13_34_28.pkl', 103/100],
    #     ]

    # Plot the motion model predictions for a single trial
    if False:
        filename = './data/robot_data_100_0_05_02_26_18_38_56.pkl'
        run_my_model_on_trial(filename)

    # Plot the motion model predictions for each trial in a folder
    if False:
        directory = ('./data_straight/')
        plot_many_trial_predictions(directory)

    # A list of files to open, process, and plot - for comparing predicted with actual distances
    if False:
        directory = ('./data/')    
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

