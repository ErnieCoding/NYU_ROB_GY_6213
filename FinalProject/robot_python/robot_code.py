# External libraries
import serial
import time
import pickle
import cv2
import cv2.aruco as aruco
import math
import numpy as np
import matplotlib.pyplot as plt
import socket
from time import strftime
import threading
from pathlib import Path
import sys

import data_types
import parameters



# ---- MAIN ROBOT CLASS ---- 
class Robot:

    def __init__(self):
        self.connected_to_hardware = False
        self.running_trial = False
        self.extra_logging = False
        self.trial_start_time = 0
        self.msg_sender = None
        self.msg_receiver = None
        self.camera_sensor = CameraSensor(parameters.camera_url)
        self.data_logger = DataLogger(parameters.filename_start, parameters.data_name_list)
        self.robot_sensor_signal = RobotSensorSignal([0, 0, 0])
        self.camera_sensor_signal = [0,0,0,0,0,0]
        # TODO: ADD FILTER
        
        
    # Create udp senders and receiver instances with the udp communication
    def setup_udp_connection(self, udp_communication):
        self.msg_sender = MsgSender(time.perf_counter(), parameters.num_robot_control_signals, udp_communication)
        self.msg_receiver = MsgReceiver(time.perf_counter(), parameters.num_robot_sensors, udp_communication)
        print("Reset msg_senders and receivers!")

    # Stop udp senders and receiver instances with the udp communication
    def eliminate_udp_connection(self):
        self.msg_sender = None
        self.msg_receiver = None
        print("Eliminate UDP !!!")

    # TODO: UPDATE TO OUR STATE ESTIMATION
    # def update_state_estimate(self):
    #     u_t = np.array([self.robot_sensor_signal.encoder_counts, self.robot_sensor_signal.steering]) # robot_sensor_signal
    #     z_t = self.robot_sensor_signal
    #     delta_t = 0.1
    #     self.particle_filter.update(u_t, z_t, delta_t)

    # One iteration of the control loop to be called repeatedly
    def control_loop(self, cmd_speed_left = 0, cmd_speed_right = 0, logging_switch_on = False):        
        # Receive msg
        if self.msg_sender is not None:
            self.robot_sensor_signal = self.msg_receiver.receive_robot_sensor_signal(self.robot_sensor_signal)
        
        self.camera_sensor_signal = self.camera_sensor.get_signal(self.camera_sensor_signal)
        print(f"[CAMERA] Received sensor signal: {self.camera_sensor_signal}")

        # TODO: UPDATE TO OUR STATE ESTIMATION
        # Update the state estimates
        # self.update_state_estimate()

        # Update control signals
        control_signal = [cmd_speed_left, cmd_speed_right]
                
        # Send msg
        if self.msg_receiver is not None:
            self.msg_sender.send_control_signal(control_signal)
            
        # Log the data
        self.data_logger.log(logging_switch_on, time.perf_counter(), control_signal, self.robot_sensor_signal, self.camera_sensor_signal)




# Function to try to connect to the robot via udp over wifi
def create_udp_communication(arduinoIP, localIP, arduinoPort, localPort, bufferSize):
    try:
        udp = UDPCommunication(arduinoIP, localIP, arduinoPort, localPort, bufferSize)
        print("Success in creating udp communication")
        return udp, True
    except Exception as e:
        print(f"Failed to create udp communication: {e}")
        return e.__repr__(), False
        
        
# Class to hold the UPD over wifi connection setup
class UDPCommunication:
    def __init__(self, arduinoIP, localIP, arduinoPort, localPort, bufferSize):
        self.arduinoIP = arduinoIP
        self.arduinoPort = arduinoPort
        self.localIP = localIP
        self.localPort = localPort
        self.bufferSize = bufferSize
        self.UDPServerSocket = socket.socket(family = socket.AF_INET, type = socket.SOCK_DGRAM)
        self.UDPServerSocket.settimeout(0.05)
        self.UDPServerSocket.bind((localIP, localPort))
        
    # Receive a message from the robot
    def receive_msg(self):
        try:
            bytesAddressPair = self.UDPServerSocket.recvfrom(self.bufferSize)
            message = bytesAddressPair[0]
            address = bytesAddressPair[1]
            clientMsg = "{}".format(message.decode())

            # print(f"[DEBUG] Received msg from the robot: {clientMsg}")
            return clientMsg
        except (socket.timeout, TimeoutError):
            return ""
        except Exception as e:
            print(f"[UDP] receive_msg error: {e}")
            return ""

       
    # Send a message to the robot
    def send_msg(self, msg):
        bytesToSend = str.encode(msg)
        self.UDPServerSocket.sendto(bytesToSend, (self.arduinoIP, self.arduinoPort))


# ----------------------- DATA LOGGING -----------------------
# Class to hold the data logger that records data when needed
class DataLogger:

    # Constructor
    def __init__(self, filename_start, data_name_list):
        self.filename_start = filename_start
        self.filename = filename_start
        self.line_count = 0
        self.dictionary = {}
        self.data_name_list = data_name_list
        for name in data_name_list:
            self.dictionary[name] = []
        self.currently_logging = False

    # Open the log file
    def reset_logfile(self, control_signal):
        self.filename = self.filename_start + "_"+str(control_signal[0])+"_"+str(control_signal[1]) + strftime("_%d_%m_%y_%H_%M_%S.pkl")
        self.dictionary = {}
        for name in self.data_name_list:
            self.dictionary[name] = []

        
    # Log one time step of data
    def log(self, logging_switch_on, time, control_signal, robot_sensor_signal, camera_sensor_signal, state_mean = [0, 0, 0]):
        if not logging_switch_on:
            if self.currently_logging:
                self.currently_logging = False
        else:
            if not self.currently_logging:
                self.currently_logging = True
                self.reset_logfile(control_signal)

        if self.currently_logging:
            self.dictionary['time'].append(time)
            self.dictionary['control_signal'].append(control_signal)
            self.dictionary['robot_sensor_signal'].append(robot_sensor_signal)
            self.dictionary['camera_sensor_signal'].append(camera_sensor_signal)
            self.dictionary['state_mean'].append(state_mean)
            

            self.line_count += 1
            if self.line_count > parameters.max_num_lines_before_write:
                self.line_count = 0
                with open(self.filename, 'wb') as file_handle:
                    pickle.dump(self.dictionary, file_handle)

# Utility for loading saved data
class DataLoader:

    # Constructor
    def __init__(self, filename):
        self.filename = filename
        
    # Load a dictionary from file.
    def load(self):
        with open(self.filename, 'rb') as file_handle:
            loaded_dict = pickle.load(file_handle)
        return loaded_dict
    
# ----------------------- END DATA LOGGING -----------------------

# Class to hold a message sender
class MsgSender:

    # Time step size between message to robot sends, in seconds
    delta_send_time = 0.05

    # Constructor
    def __init__(self, last_send_time, msg_size, udp_communication):
        self.last_send_time = last_send_time
        self.msg_size = msg_size
        self.udp_communication = udp_communication
        
    # Pack and send a control signal to the robot.
    def send_control_signal(self, control_signal):
        packed_send_msg = self.pack_msg(control_signal)
        self.send(packed_send_msg)
    
    # If its time, send the control signal to the robot.
    def send(self, msg):
        new_send_time = time.perf_counter()
        if new_send_time - self.last_send_time > self.delta_send_time:
            message = ""
            for data in msg:
                message = message + str(data)
            self.udp_communication.send_msg(message)
            self.last_send_time = new_send_time
      
    # Pack a message so it is in the correct format for the robot to receive it.
    def pack_msg(self, msg):
        packed_msg = ""
        for data in msg:
            if packed_msg == "":
                packed_msg = packed_msg + str(data)
            else:
                packed_msg = packed_msg + ", "+ str(data)
        packed_msg = packed_msg + "\n"
        return packed_msg
        
# The robot's message receiver
class MsgReceiver:

    # Determines how often to look for incoming data from the robot.
    delta_receive_time = 0.05

    # Constructor
    def __init__(self, last_receive_time, msg_size, udp_communication):
        self.last_receive_time = last_receive_time
        self.msg_size = msg_size
        self.udp_communication = udp_communication
      
    # Check if its time to look for a new message from the robot.
    def receive(self):
        new_receive_time = time.perf_counter()
        if new_receive_time - self.last_receive_time > self.delta_receive_time:
            received_msg = self.udp_communication.receive_msg()
            self.last_receive_time = new_receive_time
            return True, received_msg
            
        return False, ""
    
    # Given a new message, put it in a digestable format
    def unpack_msg(self, packed_msg):
        unpacked_msg = []
        msg_list = packed_msg.split(',')
        if len(msg_list) >= self.msg_size:
            for data in msg_list:
                unpacked_msg.append(float(data))
            return True, unpacked_msg

        return False, unpacked_msg
        
    # Check for new message and unpack it if there is one.
    def receive_robot_sensor_signal(self, last_robot_sensor_signal):
        robot_sensor_signal = last_robot_sensor_signal
        receive_ret, packed_receive_msg = self.receive()
        if receive_ret:
            unpack_ret, unpacked_receive_msg = self.unpack_msg(packed_receive_msg)
            if unpack_ret:
                robot_sensor_signal = RobotSensorSignal(unpacked_receive_msg)
            
        return robot_sensor_signal

# Class to hold a camera sensor data. Not needed for lab 1.
class CameraSensor:

    # Constructor
    def __init__(self, camera_url):
        self.camera_url = camera_url
        self.cap = None
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_250)
        self.parameters = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(self.aruco_dict, self.parameters)
        self.landmark_observations = []

        self._lock = threading.Lock()
        self._latest_raw_frame = None
        self._latest_annotated_frame = None
        self._latest_pose = [None, None, None, None, None, None, None]
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        self._latest_pose_by_id = {}
    
    def _capture_loop(self):
        while self._running:
            if self.cap is None or not self.cap.isOpened():
                print(f"[CAMERA] Attempting to connect to {self.camera_url}...")
                if self.cap is not None:
                    self.cap.release()
                
                cap = cv2.VideoCapture(self.camera_url)
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000)
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000)

                if cap.isOpened():
                    self.cap = cap
                    print("[CAMERA Connected...]")
                else:
                    cap.release()
                    time.sleep(2.0)
                    continue

            ret, raw_frame = self.cap.read()
            if not ret or raw_frame is None:
                print("[CAMERA] Stream lost. Reconnecting...")
                self.cap.release()
                self.cap = None
                time.sleep(1.0)
                continue

            annotated_frame = raw_frame.copy()
            pose_result = None

            gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = self.detector.detectMarkers(gray)

            if ids is not None:
                cv2.aruco.drawDetectedMarkers(annotated_frame, corners, ids)

                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners,
                    parameters.marker_length,
                    parameters.camera_matrix,
                    parameters.dist_coeffs
                )

                for i in range(len(ids)):
                    marker_id = ids[i][0]
                    rvec = rvecs[i]
                    tvec = tvecs[i][0]

                    x_cm = tvec[0] * 100
                    y_cm = tvec[1] * 100
                    z_cm = tvec[2] * 100

                    R, _ = cv2.Rodrigues(rvec)
                    yaw   = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
                    pitch = np.degrees(np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2)))
                    roll  = np.degrees(np.arctan2(R[2, 1], R[2, 2]))

                    print(
                        f"[CAMERA] ID {marker_id} | "
                        f"X: {x_cm:.2f} cm, Y: {y_cm:.2f} cm, Z: {z_cm:.2f} cm | "
                        f"Yaw: {yaw:.2f}°, Pitch: {pitch:.2f}°, Roll: {roll:.2f}°"
                    )

                    cv2.drawFrameAxes(
                        annotated_frame,
                        parameters.camera_matrix,
                        parameters.dist_coeffs,
                        rvec,
                        tvecs[i],
                        0.05
                    )

                    cv2.putText(
                        annotated_frame,
                        f"ID {marker_id}: X={x_cm:.1f} Y={y_cm:.1f} Z={z_cm:.1f} cm",
                        (5, 40 + 30 * i),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        2
                    )

                    
                    pose_result = [marker_id, x_cm, y_cm, z_cm, yaw, pitch, roll]

                    with self._lock:
                        last = self._latest_pose_by_id.get(marker_id)
                    

                    # TODO: Fill in camera variance
                    tvec_m = tvec / 100.0
                    cov = self.get_camera_covariance(tvec_m)
                    if last is None or x_cm != self._latest_pose[1] or z_cm != self._latest_pose[3]:
                        robot_pose = self.marker_to_robot_pose(pose_result)
                        obs = data_types.LandmarkObservation(
                                timestamp=time.perf_counter(),
                                marker_id=marker_id, 
                                robot_pose_meas=robot_pose, 
                                covariance=cov
                            )

                        with self._lock:
                            self.landmark_observations.append(obs) 

            # --- Atomically store results ---
            with self._lock:
                self._latest_raw_frame = raw_frame
                self._latest_annotated_frame = annotated_frame
                self._latest_pose = pose_result

            time.sleep(0.05)

    def marker_to_robot_pose(self, pose):
        marker_id = pose[0]
        camera_x = pose[1]
        camera_z = pose[3]
        yaw_rad = math.radians(pose[4])

        dx_robot = camera_z
        dy_robot = -camera_x

        x_robot = parameters.tags[marker_id][0] - dx_robot
        y_robot = parameters.tags[marker_id][1] - dy_robot
        theta_robot = -yaw_rad

        return data_types.Pose2D(x_robot, y_robot, theta_robot)
  
    def get_camera_covariance(self, tvec, rms_pixels=0.6862):
        fx = parameters.camera_matrix[0, 0]  # 348.03
        fy = parameters.camera_matrix[1, 1]  # 355.60
        f_mean = (fx + fy) / 2.0             # ~351.8 px

        z_cam = abs(tvec[2])                 # depth in meters
        sigma_xy = (rms_pixels * z_cam) / f_mean

        # Clamp to a minimum floor (e.g., 2 cm) — calibration RMS is optimistic
        sigma_xy = max(sigma_xy, 0.02)

        return np.diag([sigma_xy**2, sigma_xy**2, 0.05**2]).astype(np.float64)

    
    def get_latest_frame(self):
        with self._lock:
            frame = self._latest_annotated_frame
            if frame is None:
                frame = self._latest_raw_frame
            
            return frame.copy() if frame is not None else None
    
    def get_pose_estimate(self):
        with self._lock:
            pose = self._latest_pose
        if pose is not None:
            return True, pose
        return False, []
        
    # Get a new pose estimate from a camera image
    def get_signal(self, last_camera_signal):
        ret, pose_estimate = self.get_pose_estimate()

        if ret:
            return pose_estimate
        
        return last_camera_signal
        
    
    # Close the camera stream
    def close(self):
        self._running = False
        self._thread.join(timeout=2.0)
        if self.cap is not None:
            self.cap.release()


# A storage vessel for an instance of a robot signal
class RobotSensorSignal:

    # Constructor
    def __init__(self, unpacked_msg):
        self.encoder_left_counts = int(unpacked_msg[0])
        self.encoder_right_counts = int(unpacked_msg[1])
        self.num_lidar_rays = int(unpacked_msg[2])
        self.angles = []
        self.distances = []
        for i in range(self.num_lidar_rays):
            index = 3 + i*2
            self.angles.append(unpacked_msg[index])
            self.distances.append(unpacked_msg[index+1])
    
    # Print the robot sensor signal contents.
    def print(self):
        print("Robot Sensor Signal")
        print(" encoder LEFT: ", self.encoder_left_counts)
        print(" encoder RIGHT:" , self.encoder_right_counts)
        print(" num_lidar_rays: ", self.num_lidar_rays)
        print(" angles: ",self.angles)
        print(" distances: ", self.distances)
    
    # Convert the sensor signal to a list of ints and floats.
    def to_list(self):
        sensor_data_list = []
        sensor_data_list.append(self.encoder_left_counts)
        sensor_data_list.append(self.encoder_right_counts)
        sensor_data_list.append(self.num_lidar_rays)
        for i in range(self.num_lidar_rays):
            sensor_data_list.append(self.angles[i])
            sensor_data_list.append(self.distances[i])
        
        return sensor_data_list
    
    # Put lidar angles in the correct units and correct direction.
    def convert_hardware_angle(self, angle):
        return -angle * math.pi / 180 # degrees to rad
    
    # Put lidar distances in the correct units.
    def convert_hardware_distance(self, distance):
        return distance / 1000 # mm to m
    

