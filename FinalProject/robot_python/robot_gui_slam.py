# External libraries
import asyncio
import cv2
import math
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib
from nicegui import ui, app, run, events
import numpy as np
import sys
import time
import time as time_module
import types
from fastapi import Response
from time import time
import pygame

# Local libraries
import robot_code
import parameters

# Global variables
logging = False
stream_video = False

matplotlib.use('Agg')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))


def create_slam_system():
    """Create the SLAM system used by this GUI copy."""
    install_optional_slam_stubs()

    from FinalProject.robot_python.app.slam import SLAMSystem
    from FinalProject.robot_python.data_types import EncoderState, LidarScan, RelativeMotion, RobotFrame
    from FinalProject.robot_python.frontend.motion_model import DifferentialDriveMotionModel

    patch_motion_model_delta_distance(DifferentialDriveMotionModel, EncoderState, RelativeMotion)
    return SLAMSystem(), EncoderState, LidarScan, RobotFrame


def make_robot_frame(sensor, EncoderState, LidarScan, RobotFrame):
    """Convert robot_code.RobotSensorSignal into the SLAM RobotFrame type."""
    timestamp = time_module.perf_counter()
    encoder = EncoderState(
        left_ticks=int(sensor.encoder_left_counts),
        right_ticks=int(sensor.encoder_right_counts),
        timestamp=timestamp,
    )

    lidar_scan = None
    count = min(
        int(getattr(sensor, 'num_lidar_rays', 0)),
        len(getattr(sensor, 'angles', [])),
        len(getattr(sensor, 'distances', [])),
    )
    if count > 0:
        ranges = [sensor.convert_hardware_distance(sensor.distances[i]) for i in range(count)]
        angles = [sensor.convert_hardware_angle(sensor.angles[i]) for i in range(count)]
        lidar_scan = LidarScan(
            ranges=ranges,
            angles=angles,
            timestamp=timestamp,
            frame_id='hardware_lidar',
        )

    return RobotFrame(timestamp=timestamp, encoder=encoder, lidar_scan=lidar_scan)


def install_optional_slam_stubs():
    """Keep the GUI usable if optional SLAM solver packages are not installed."""
    try:
        import gtsam  # noqa: F401
    except ModuleNotFoundError:
        gtsam_stub = types.ModuleType('gtsam')

        class Pose2:
            def __init__(self, x, y, theta):
                self._x = x
                self._y = y
                self._theta = theta

            def x(self):
                return self._x

            def y(self):
                return self._y

            def theta(self):
                return self._theta

        class Values:
            def __init__(self):
                self._poses = {}

            def insert(self, node_id, pose):
                self._poses[node_id] = pose

            def atPose2(self, node_id):
                return self._poses[node_id]

        class Graph:
            def add(self, _factor):
                return None

        class Optimizer:
            def __init__(self, _graph, initial):
                self.initial = initial

            def optimize(self):
                return self.initial

        gtsam_stub.Pose2 = Pose2
        gtsam_stub.Values = Values
        gtsam_stub.NonlinearFactorGraph = Graph
        gtsam_stub.LevenbergMarquardtOptimizer = Optimizer
        gtsam_stub.PriorFactorPose2 = lambda *args, **kwargs: ('prior', args, kwargs)
        gtsam_stub.BetweenFactorPose2 = lambda *args, **kwargs: ('between', args, kwargs)
        gtsam_stub.noiseModel = types.SimpleNamespace(
            Gaussian=types.SimpleNamespace(Covariance=lambda covariance: covariance)
        )
        sys.modules['gtsam'] = gtsam_stub

    try:
        import open3d  # noqa: F401
    except ModuleNotFoundError:
        open3d_stub = types.ModuleType('open3d')

        class PointCloud:
            def __init__(self):
                self.points = None

        class ICPResult:
            fitness = 0.0
            inlier_rmse = 1.0
            transformation = np.eye(4)

        open3d_stub.geometry = types.SimpleNamespace(PointCloud=PointCloud)
        open3d_stub.utility = types.SimpleNamespace(Vector3dVector=lambda points: points)
        open3d_stub.pipelines = types.SimpleNamespace(
            registration=types.SimpleNamespace(
                registration_icp=lambda *args, **kwargs: ICPResult(),
                TransformationEstimationPointToPoint=lambda: object(),
                ICPConvergenceCriteria=lambda **kwargs: kwargs,
            )
        )
        sys.modules['open3d'] = open3d_stub


def patch_motion_model_delta_distance(DifferentialDriveMotionModel, EncoderState, RelativeMotion):
    """Patch only this process so the current EKF can read motion.delta_distance."""
    if getattr(DifferentialDriveMotionModel.RelativeMotion_change, '_gui_slam_patch', False):
        return

    def relative_motion_change(self, prev_encoder, curr_encoder):
        left_delta_ticks = curr_encoder.left_ticks - prev_encoder.left_ticks
        right_delta_ticks = curr_encoder.right_ticks - prev_encoder.right_ticks
        left_distance = self.left_encoder_to_distance(left_delta_ticks)
        right_distance = self.right_encoder_to_distance(right_delta_ticks)
        distance = 0.5 * (left_distance + right_distance)
        dtheta = (right_distance - left_distance) / self.config.wheel_base_m
        motion = RelativeMotion(
            dx=distance * math.cos(0.5 * dtheta),
            dy=distance * math.sin(0.5 * dtheta),
            dtheta=(dtheta + math.pi) % (2.0 * math.pi) - math.pi,
            covariance=self._estimate_motion_covariance(distance, dtheta),
            source='wheel_odometry',
        )
        motion.delta_distance = distance
        return motion

    relative_motion_change._gui_slam_patch = True
    DifferentialDriveMotionModel.RelativeMotion_change = relative_motion_change


# Frame converter for the video stream, from OpenCV to a JPEG image
def convert(frame: np.ndarray) -> bytes:
    """Converts a frame from OpenCV to a JPEG image.
    This is a free function (not in a class or inner-function),
    to allow run.cpu_bound to pickle it and send it to a separate process.
    """
    _, imencode_image = cv2.imencode('.jpg', frame)
    return imencode_image.tobytes()


# Create the connection with a real camera.
def connect_with_camera():
    video_capture = cv2.VideoCapture(1)
    return video_capture


def update_video(video_image):
    if stream_video:
        video_image.force_reload()


def get_time_in_ms():
    return int(time() * 1000)


# Create the gui page
@ui.page('/')
def main():

    # Custom UI elements
    ui.add_head_html("""
    <style>
    .arrow-keys-widget { display:flex; flex-direction:column; align-items:center; gap:5px; padding:10px; }
    .key-row           { display:flex; gap:5px; }
    .key {
        width:46px; height:46px;
        display:flex; align-items:center; justify-content:center;
        background:#1e1e1e; border:1px solid #3a3a3a; border-radius:7px;
        font-size:20px; color:#555;
        transition: background 110ms ease, color 110ms ease,
                    transform 110ms ease, box-shadow 110ms ease;
        box-shadow: 0 3px 0 #0a0a0a;
        user-select:none;
    }
    .key.key-active {
        background:#0f3028; border-color:#2a7a5e; color:#3ddba0;
        transform:translateY(2px); box-shadow:0 1px 0 #061a14;
    }
    .key.key-inactive { opacity:0.25; pointer-events:none; }
    </style>
    """)

    # Robot variables
    robot = robot_code.Robot()
    try:
        slam, EncoderState, LidarScan, RobotFrame = create_slam_system()
        slam_error = None
    except Exception as exc:
        slam = None
        EncoderState = None
        LidarScan = None
        RobotFrame = None
        slam_error = exc
        print(f"[SLAM] Failed to initialize: {exc}")

    # Lidar data
    max_lidar_range = 12
    lidar_angle_res = 2
    num_angles = int(360 / lidar_angle_res)
    lidar_distance_list = []
    lidar_cos_angle_list = []
    lidar_sin_angle_list = []
    for i in range(num_angles):
        lidar_distance_list.append(max_lidar_range)
        lidar_cos_angle_list.append(math.cos(i * lidar_angle_res / 180 * math.pi))
        lidar_sin_angle_list.append(math.sin(i * lidar_angle_res / 180 * math.pi))

    # Set dark mode for gui
    dark = ui.dark_mode()
    dark.value = True


    # Set up the video stream, not needed for lab 1
    if stream_video:
        # video_capture = cv2.VideoCapture(parameters.camera_id)
        video_capture = robot.camera_sensor.cap

    # Enable frame grabs from the video stream.
    @app.get('/video/frame')
    async def grab_video_frame() -> Response:
        if not video_capture.isOpened():
            return "[ERROR] ERROR CAPTURING VIDEO"
        _, frame = await run.io_bound(video_capture.read)
        if frame is None:
            return "[ERROR] EMPTY FRAME" 
        jpeg = await run.cpu_bound(convert, frame)
        return Response(content=jpeg, media_type='image/jpeg')

    # Convert lidar data to something visible in correct units.
    def update_lidar_data():
        for i in range(robot.robot_sensor_signal.num_lidar_rays):
            distance_in_mm = robot.robot_sensor_signal.distances[i]
            angle = 360 - robot.robot_sensor_signal.angles[i]
            if distance_in_mm > 0 and abs(angle) < 360:
                index = max(
                    0,
                    min(
                        int(360 / lidar_angle_res - 1),
                        int((angle - (lidar_angle_res / 2)) / lidar_angle_res),
                    ),
                )
                lidar_distance_list[index] = distance_in_mm / 1000


    cmd_state = {'left': 0, 'right': 0} # state dictionary for speeds
    key_state = {'up': False, 'left': False, 'right': False}
    ctrl = {'joystick': None}

    BASE_SPEED = 85
    SPIN_SPEED = 70
    TURN_OFFSET = 40

    # Determine what speed to send to each motor
    def update_commands(e: events.KeyEventArguments):
        pressed = not e.action.keyup   # True on keydown/repeat, False on keyup

        shift_pressed = e.modifiers.shift
        if shift_pressed and e.key.arrow_up:
            key_state['up'] = pressed
        elif shift_pressed and e.key.arrow_left:
            key_state['left'] = pressed
        elif shift_pressed and e.key.arrow_right:
            key_state['right'] = pressed
        else:
            key_state['left'] = False
            key_state['right'] = False
            key_state['up'] = False
            #return  # ignore all non-arrow keys

        if key_state['up']:
            if key_state['left']:
                # Driving forward + curving left: slow down left wheel
                cmd_state['left']  = BASE_SPEED - TURN_OFFSET
                cmd_state['right'] = BASE_SPEED
            elif key_state['right']:
                # Driving forward + curving right: slow down right wheel
                cmd_state['left']  = BASE_SPEED
                cmd_state['right'] = BASE_SPEED - TURN_OFFSET
            else:
                # Straight forward
                cmd_state['left']  = BASE_SPEED
                cmd_state['right'] = BASE_SPEED
        elif key_state['left']:
            # Pivot left in place: right wheel drives, left wheel stopped
            cmd_state['left']  = 0
            cmd_state['right'] = SPIN_SPEED
        elif key_state['right']:
            # Pivot right in place: left wheel drives, right wheel stopped
            cmd_state['left']  = SPIN_SPEED
            cmd_state['right'] = 0
        else:
            # No arrow keys held → stop
            cmd_state['left']  = 0
            cmd_state['right'] = 0
        
        # Animate when keys are pressed
        ui.run_javascript(
            f"document.getElementById('key-up').classList.toggle('key-active',{str(key_state['up']).lower()});"
            f"document.getElementById('key-left').classList.toggle('key-active',{str(key_state['left']).lower()});"
            f"document.getElementById('key-right').classList.toggle('key-active',{str(key_state['right']).lower()});"
        )
    
    DEADZONE = 0.05 # controller deadzone
    
    def on_controller_switch():
        if controller_switch.value:
            try:
                pygame.init()
                pygame.joystick.init()
                if pygame.joystick.get_count() == 0:
                    print("[Controller] No controller found.")
                    controller_switch.value = False
                    return
                ctrl['joystick'] = pygame.joystick.Joystick(0)
                ctrl['joystick'].init()
                print(f"[Controller] Connected: {ctrl['joystick'].get_name()}")
            except Exception as e:
                print(f"[Controller] Init error: {e}")
                controller_switch.value = False
        else:
            if ctrl['joystick']:
                ctrl['joystick'].quit()
            ctrl['joystick'] = None
            pygame.joystick.quit()
            cmd_state['left'] = cmd_state['right'] = 0  # stop robot on disconnect


    # Update connection
    def update_connection_to_robot():
        if udp_switch.value:
            if not robot.connected_to_hardware:
                udp, udp_success = robot_code.create_udp_communication(
                    parameters.arduinoIP,
                    parameters.localIP,
                    parameters.arduinoPort,
                    parameters.localPort,
                    parameters.bufferSize,
                )
                if udp_success:
                    robot.setup_udp_connection(udp)
                    robot.connected_to_hardware = True
                    print("Should be set for UDP!")
                else:
                    udp_switch.value = False
                    robot.connected_to_hardware = False
        else:
            if robot.connected_to_hardware:
                robot.eliminate_udp_connection()
                robot.connected_to_hardware = False

    # TODO: Visualize LiDAR with a different library specifically for LiDAR
    # Visualize the lidar scans
    # def show_lidar_plot():
    #     with main_plot:
    #         fig = main_plot.fig
    #         fig.patch.set_facecolor('black')
    #         plt.clf()
    #         plt.style.use('dark_background')
    #         plt.tick_params(axis='x', colors='lightgray')
    #         plt.tick_params(axis='y', colors='lightgray')

    #         for i in range(num_angles):
    #             distance = lidar_distance_list[i]
    #             cos_ang = lidar_cos_angle_list[i]
    #             sin_ang = lidar_sin_angle_list[i]
    #             x = [distance * cos_ang, max_lidar_range * cos_ang]
    #             y = [distance * sin_ang, max_lidar_range * sin_ang]
    #             plt.plot(x, y, 'r')
    #         plt.grid(True)
    #         plt.xlim(-2, 2)
    #         plt.ylim(-2, 2)

    # # Visualize localization
    # def show_localization_plot():
    #     with main_plot:
    #         fig = main_plot.fig
    #         fig.patch.set_facecolor('black')
    #         plt.clf()
    #         plt.style.use('dark_background')
    #         plt.tick_params(axis='x', colors='lightgray')
    #         plt.tick_params(axis='y', colors='lightgray')

    #         pf = robot.particle_filter
    #         state_mean = pf.particle_set.mean_state
    #         particle_set = pf.particle_set
    #         map_obj = pf.map

    #         # Plot map walls
    #         for wall in map_obj.wall_list:
    #             plt.plot(
    #                 [wall.corner1.x, wall.corner2.x],
    #                 [wall.corner1.y, wall.corner2.y],
    #                 'w',
    #                 linewidth=2
    #             )

    #         # Plot particles
    #         # x_particles = [p.state.x for p in particle_set.particle_list]
    #         # y_particles = [p.state.y for p in particle_set.particle_list]
    #         # plt.plot(x_particles, y_particles, 'g.', markersize=4)

    #         # Plot estimated state
    #         plt.plot(state_mean.x, state_mean.y, 'ro', markersize=8)

    #         # Plot heading arrow
    #         dir_length = 0.15
    #         plt.plot(
    #             [state_mean.x, state_mean.x + dir_length * math.cos(state_mean.theta)],
    #             [state_mean.y, state_mean.y + dir_length * math.sin(state_mean.theta)],
    #             'r',
    #             linewidth=2
    #         )

    #         # Confidence ellipse from all particles
    #         x_particles_np = np.array([p.state.x for p in particle_set.particle_list])
    #         y_particles_np = np.array([p.state.y for p in particle_set.particle_list])

    #         if len(x_particles_np) > 1:
    #             cov = np.cov(np.vstack((x_particles_np, y_particles_np)))

    #             if np.all(np.isfinite(cov)):
    #                 eigvals, eigvecs = np.linalg.eig(cov)
    #                 eigvals = np.maximum(eigvals, 1e-9)

    #                 order = np.argsort(eigvals)[::-1]
    #                 eigvals = eigvals[order]
    #                 eigvecs = eigvecs[:, order]

    #                 angle = math.degrees(math.atan2(eigvecs[1, 0], eigvecs[0, 0]))

    #                 width = 4 * math.sqrt(eigvals[0])   # 2-sigma
    #                 height = 4 * math.sqrt(eigvals[1])  # 2-sigma

    #                 ellipse = Ellipse(
    #                     (state_mean.x, state_mean.y),
    #                     width=width,
    #                     height=height,
    #                     angle=angle,
    #                     edgecolor='cyan',
    #                     facecolor='none',
    #                     linewidth=2
    #                 )
    #                 plt.gca().add_patch(ellipse)

    #         plt.xlabel('X (m)')
    #         plt.ylabel('Y (m)')
    #         plt.axis(map_obj.plot_range)
    #         plt.grid(True)
    #         plt.gca().set_aspect('equal', adjustable='box')

    #         # # Plot heading arrow
    #         # dir_length = 0.15
    #         # plt.plot(
    #         #     [state_mean.x, state_mean.x + dir_length * math.cos(state_mean.theta)],
    #         #     [state_mean.y, state_mean.y + dir_length * math.sin(state_mean.theta)],
    #         #     'r',
    #         #     linewidth=2
    #         # )

    #         # plt.xlabel('X (m)')
    #         # plt.ylabel('Y (m)')
    #         # plt.axis(map_obj.plot_range)
    #         # plt.grid(True)
    #         # plt.gca().set_aspect('equal', adjustable='box')

    # ---- LEGACY CODE FOR TRIALS ----
    # # Run an experiment trial from a button push
    # def run_trial():
    #     if not udp_switch.value:
    #         print("Please connect to robot first.")
    #         return

    #     robot.trial_start_time = get_time_in_ms()
    #     robot.running_trial = True
    #     robot.extra_logging = False
    #     logging_switch.value = True

    #     # Optional: manual switches off during scripted run
    #     speed_switch.value = False
    #     steering_switch.value = False

    #     print("Start time:", robot.trial_start_time)

    
    with ui.row().classes('w-full h-full gap-2 no-wrap'):

        # ── LEFT COLUMN ─────────────────────────────────────────────
        with ui.column().classes('w-64 gap-2 flex-shrink-0'):

            # Arrow key widget
            with ui.card().classes('w-full items-center'):
                ui.label('KEYBOARD').style('font-size:11px; font-weight:600; letter-spacing:0.1em; color:#666;')
                ui.html("""
                <div class="arrow-keys-widget">
                    <div class="key-row">
                        <div id="key-up" class="key">↑</div>
                    </div>
                    <div class="key-row">
                        <div id="key-left"  class="key">←</div>
                        <div                class="key key-inactive">↓</div>
                        <div id="key-right" class="key">→</div>
                    </div>
                </div>
                """, sanitize=False)

            # Switches
            with ui.card().classes('w-full'):
                controller_switch = ui.switch('Controller', on_change=on_controller_switch)
                logging_switch    = ui.switch('Data Logging')
                udp_switch        = ui.switch('Robot Connect')

            # Encoder counts
            with ui.card().classes('w-full'):
                ui.label('ENCODERS').style('font-size:11px; font-weight:600; letter-spacing:0.1em; color:#666;')
                with ui.row().classes('w-full items-center justify-between'):
                    ui.label('Left')
                    encoder_left_count_label = ui.label('0').style('font-variant-numeric:tabular-nums;')
                with ui.row().classes('w-full items-center justify-between'):
                    ui.label('Right')
                    encoder_right_count_label = ui.label('0').style('font-variant-numeric:tabular-nums;')

            # SLAM state
            with ui.card().classes('w-full'):
                ui.label('SLAM').style('font-size:11px; font-weight:600; letter-spacing:0.1em; color:#666;')
                slam_status_label = ui.label('Ready' if slam else f'Offline: {slam_error}')
                slam_pose_label = ui.label('x=0.000  y=0.000  theta=0.000').style('font-variant-numeric:tabular-nums;')

        # ── RIGHT COLUMN (plots + camera) ───────────────────────────
        with ui.column().classes('flex-1 gap-2 overflow-hidden'):

            # Plot — top half
            with ui.card().classes('w-full items-center'):
                ui.label('LiDAR / Localization').style('font-weight:600;')
                main_plot = ui.pyplot(figsize=(10, 4))  # wider, shorter

            # Camera — bottom half
            with ui.card().classes('w-full overflow-hidden items-center'):
                ui.label('Camera').style('font-weight:600;')
                if stream_video:
                    video_image = ui.interactive_image('/video/frame').classes('w-full')
                else:
                    ui.image('./robot_image.jpg').style(
                        'width:100%; max-height: 285px; object-fit:contain;')
                    video_image = None    
    
    
    # ----------------- LEGACY CODE FOR LIDAR, CAMERA, AND ENCODER UI -----------------
    # # Create the video camera, lidar, and encoder sensor visualizations.
    # with ui.card().classes('w-full'):
    #     with ui.grid(columns=3).classes('w-full items-center'):
    #         with ui.card().classes('w-full items-center h-60'):
    #             if stream_video:
    #                 video_image = ui.interactive_image('/video/frame').classes('w-full h-full')
    #             else:
    #                 ui.image('./a_robot_image.jpg').props('height=2')
    #                 video_image = None

    #         with ui.card().classes('w-full items-center h-60'):
    #             main_plot = ui.pyplot(figsize=(3, 3))

    #         with ui.card().classes('items-center h-60'):
    #             ui.label('Encoder LEFT:').style('text-align: center;')
    #             encoder_left_count_label = ui.label('0')
    #             ui.label('Encoder RIGHT:').style('text-align: center;')
    #             encoder_right_count_label = ui.label('0')
    #             logging_switch = ui.switch('Data Logging ')
    #             udp_switch = ui.switch('Robot Connect')

    # Update slider values, plots, etc. and run robot control loop
    ui.keyboard(on_key=update_commands)
    async def control_loop():
        update_connection_to_robot()
        
        if controller_switch.value and ctrl['joystick']:
            pygame.event.pump()                    # refresh joystick state
            js = ctrl['joystick']

            r2    = js.get_axis(5)                 # R2 trigger:   [-1, 1]
            steer = js.get_axis(0)                 # Left stick X: [-1, 1]
            speed = (r2 + 1) / 2                   # normalize to  [ 0, 1]

            if speed > DEADZONE:
                # Forward + steer: mix speed and steering into L/R
                # steer > 0 → turn right (slow right, fast left)
                left_val  = speed * (1 + steer)
                right_val = speed * (1 - steer)
                cmd_state['left']  = int(min(1.0, max(0.0, left_val))  * BASE_SPEED)
                cmd_state['right'] = int(min(1.0, max(0.0, right_val)) * BASE_SPEED)
            elif abs(steer) > DEADZONE:
                # No throttle → spin in place
                if steer > 0:   # stick right → spin right (left wheel drives)
                    cmd_state['left']  = int(steer  * SPIN_SPEED)
                    cmd_state['right'] = 0
                else:            # stick left  → spin left  (right wheel drives)
                    cmd_state['left']  = 0
                    cmd_state['right'] = int(-steer * SPIN_SPEED)
            else:
                cmd_state['left'] = cmd_state['right'] = 0
        
        
        robot.control_loop(cmd_state['left'], cmd_state['right'], logging_switch.value)
        encoder_left_count_label.set_text(str(robot.robot_sensor_signal.encoder_left_counts))
        encoder_right_count_label.set_text(str(robot.robot_sensor_signal.encoder_right_counts))
        update_lidar_data()

        if slam is not None:
            try:
                frame = make_robot_frame(robot.robot_sensor_signal, EncoderState, LidarScan, RobotFrame)
                frontend_output = slam.run_frontend(frame)
                slam.run_backend()
                pose = frontend_output.pose_estimate.pose
                slam_pose_label.set_text(f'x={pose.x:.3f}  y={pose.y:.3f}  theta={pose.theta:.3f}')
                if frame.lidar_scan is None:
                    slam_status_label.set_text('Running: encoder only')
                else:
                    slam_status_label.set_text(f'Running: {len(frame.lidar_scan.ranges)} LiDAR rays')
            except Exception as exc:
                slam_status_label.set_text(f'Error: {exc}')
                print(f"[SLAM] Step failed: {exc}")

        # TODO: Update plots
        # show_localization_plot()
        # show_lidar_plot()
        
        update_video(video_image)

    ui.timer(0.1, control_loop)


# Run the gui
ui.run(native=False)
