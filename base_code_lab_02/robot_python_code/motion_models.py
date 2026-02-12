# External Libraries
import math
import random

# Motion Model constants

#TODO: update these functions to be in meters(right now they were in centimeters)
# A function for obtaining variance in distance travelled as a function of distance travelled
def variance_distance_travelled_s(distance):
    # Add student code here
    var_s = 0.000000585994 * (distance/0.03063)
    return var_s


# Function to calculate distance from encoder counts
def distance_travelled_s(encoder_counts):
    # Add student code here
    s=0.0003063 * encoder_counts
    return s


# A function for obtaining variance in distance travelled as a function of distance travelled
def variance_rotational_velocity_w(distance):
    # Add student code here
    SIGMA_W2_CONST = 0.0009 
    return SIGMA_W2_CONST

def rotational_velocity_w(steering_angle_command):
    # Add student code here
    w = -0.0128 * steering_angle_command + 0.0016
    return w

# This class is an example structure for implementing your motion model.
class MyMotionModel:

    # Constructor, change as you see fit.
    def __init__(self, initial_state, last_encoder_count):
        self.state = initial_state
        self.last_encoder_count = last_encoder_count

    # This is the key step of your motion model, which implements x_t = f(x_{t-1}, u_t)
    def step_update(self, encoder_counts, steering_angle_command, delta_t):
        # Guard against bad timestamps
        if delta_t <= 0:
            return self.state

        # 1) Encoder increment for this step
        delta_e = encoder_counts - self.last_encoder_count
        self.last_encoder_count = encoder_counts

        # 2) Mean controls
        s_mean = distance_travelled_s(delta_e)  # meters
        w_mean = rotational_velocity_w(steering_angle_command)  # rad/s

        # 3) Variances
        var_s = variance_distance_travelled_s(abs(s_mean))
        var_w = variance_rotational_velocity_w(abs(s_mean))  # input ignored if constant

        # Safety clamp
        var_s = max(0.0, var_s)
        var_w = max(0.0, var_w)

        # 4) Sample noisy controls
        s_noisy = s_mean + random.gauss(0.0, math.sqrt(var_s))
        w_noisy = w_mean + random.gauss(0.0, math.sqrt(var_w))

        # 5) State update
        x, y, theta = self.state

        x_new = x + s_noisy * math.cos(theta)
        y_new = y + s_noisy * math.sin(theta)
        theta_new = theta + w_noisy * delta_t

        self.state = [x_new, y_new, theta_new]
        return self.state
    
    # This is a great tool to take in data from a trial and iterate over the data to create 
    # a robot trajectory in the global frame, using your motion model.
    def traj_propagation(self, time_list, encoder_count_list, steering_angle_list):
        x_list = [self.state[0]]
        y_list = [self.state[1]]
        theta_list = [self.state[2]]
        self.last_encoder_count = encoder_count_list[0]
        for i in range(1, len(encoder_count_list)):
            delta_t = time_list[i] - time_list[i-1]
            new_state = self.step_update(encoder_count_list[i], steering_angle_list[i], delta_t)
            x_list.append(new_state[0])
            y_list.append(new_state[1])
            theta_list.append(new_state[2])

        return x_list, y_list, theta_list
    

    # Coming soon
    def generate_simulated_traj(self, duration, steering_angle):
        delta_t = 0.1
        t_list = [0]
        x_list = [self.state[0]]
        y_list = [self.state[1]]
        theta_list = [self.state[2]]
        t = 0
        encoder_counts = 0
        while t < duration:
            new_state = self.step_update(
                encoder_counts,
                steering_angle,
                delta_t
            )
            
            x_list.append(new_state[0])
            y_list.append(new_state[1])
            theta_list.append(new_state[2])

            encoder_counts += 20
            t += delta_t
            t_list.append(t) 
        
        return t_list, x_list, y_list, theta_list
            