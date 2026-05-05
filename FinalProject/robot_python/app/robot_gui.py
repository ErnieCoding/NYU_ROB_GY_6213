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

_APP_DIR     = Path(__file__).resolve().parent          # .../app/
_PYTHON_DIR  = _APP_DIR.parent                          # .../robot_python/
_PROJECT_DIR = _PYTHON_DIR.parent                       # .../FinalProject/
_ROOT_DIR    = _PROJECT_DIR.parent                      # contains FinalProject/

for _p in [str(_ROOT_DIR), str(_PROJECT_DIR), str(_PYTHON_DIR), str(_APP_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Local libraries
# import FinalProject.robot_python.robot_code as robot_code
import robot_code
from FinalProject.robot_python import parameters

# Global variables
logging = False
stream_video = True

matplotlib.use('Agg')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))


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


def create_slam_system():
    """Create the SLAM system used by this GUI copy."""
    install_optional_slam_stubs()

    from FinalProject.robot_python.app.slam import SLAMSystem
    from FinalProject.robot_python.data_types import EncoderState, LidarScan, RobotFrame

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


# Frame converter for the video stream, from OpenCV to a JPEG image
def convert(frame: np.ndarray) -> bytes:
    """Converts a frame from OpenCV to a JPEG image.
    This is a free function (not in a class or inner-function),
    to allow run.cpu_bound to pickle it and send it to a separate process.
    """
    _, imencode_image = cv2.imencode('.jpg', frame)
    return imencode_image.tobytes()


# Create the connection with a real camera.
# def connect_with_camera():
#     video_capture = cv2.VideoCapture(1)
#     return video_capture


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
        EncoderState = LidarScan = RobotFrame = None
        slam_error = exc
        print(f"[SLAM] Failed to initialize: {exc}")

    # Accumulated SLAM visualization state
    slam_state = {
        'cloud_x':       [],
        'cloud_y':       [],
        'traj_x':        [],
        'traj_y':        [],
        'slam_traj_x':   [],
        'slam_traj_y':   [],
        'pose_x':        0.0,
        'pose_y':        0.0,
        'pose_theta':    0.0,
        'covariance':    None,
        'max_cloud_pts': 6000,
        '_plot_tick': 0,
    }

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
        frame = robot.camera_sensor.get_latest_frame()  # Non-blocking, always fresh
        if frame is None:
            return Response(status_code=503)
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
    key_state = {'up': False, 'left': False, 'right': False, 'down': False}
    ctrl = {'joystick': None}

    BASE_SPEED = 65
    SPIN_SPEED = 55
    TURN_OFFSET = 30

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
        elif shift_pressed and e.key.arrow_down:
            key_state['down'] = pressed
        else:
            key_state['left'] = False
            key_state['right'] = False
            key_state['up'] = False
            key_state['down'] = False
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
            # Spin in place
            cmd_state['left']  = (-1) * SPIN_SPEED
            cmd_state['right'] = SPIN_SPEED
        elif key_state['right']:
            # Pivot right in place: left wheel drives, right wheel stopped
            cmd_state['left']  = SPIN_SPEED
            cmd_state['right'] = (-1) * SPIN_SPEED
        elif key_state['down']:
            cmd_state['left'] = (-1) * BASE_SPEED
            cmd_state['right'] = (-1) * BASE_SPEED
        else:
            # No arrow keys held → stop
            cmd_state['left']  = 0
            cmd_state['right'] = 0
        
        # Animate when keys are pressed
        ui.run_javascript(
            f"document.getElementById('key-up').classList.toggle('key-active',{str(key_state['up']).lower()});"
            f"document.getElementById('key-left').classList.toggle('key-active',{str(key_state['left']).lower()});"
            f"document.getElementById('key-right').classList.toggle('key-active',{str(key_state['right']).lower()});"
            f"document.getElementById('key-down').classList.toggle('key-active',{str(key_state['down']).lower()});"
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
                        <div id="key-down"  class="key">↓</div>
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
            
            # After the encoder card, inside the left column:

            # ── PIPELINE MODE SWITCH ──────────────────────────────────
            with ui.card().classes('w-full'):
                ui.label('PIPELINE').style(
                    'font-size:11px; font-weight:600; letter-spacing:0.1em; color:#666;'
                )
                backend_switch = ui.switch(
                    'Frontend + Backend',
                    value=False,
                ).tooltip('OFF = EKF frontend only  |  ON = EKF + graph optimizer')

            # ── POSE ESTIMATES ────────────────────────────────────────
            with ui.card().classes('w-full'):
                ui.label('POSE ESTIMATE').style(
                    'font-size:11px; font-weight:600; letter-spacing:0.1em; color:#666;'
                )

            # --- Global frame (EKF frontend) ---
            ui.label('EKF Frontend (global)').style('font-size:10px; color:#888; margin-top:4px;')
            with ui.row().classes('w-full items-center justify-between'):
                ui.label('x (m)')
                ekf_x_label = ui.label('—').style(
                    'font-variant-numeric:tabular-nums; font-family:monospace;'
                )
            with ui.row().classes('w-full items-center justify-between'):
                ui.label('y (m)')
                ekf_y_label = ui.label('—').style(
                    'font-variant-numeric:tabular-nums; font-family:monospace;'
                )
            with ui.row().classes('w-full items-center justify-between'):
                ui.label('θ (°)')
                ekf_theta_label = ui.label('—').style(
                    'font-variant-numeric:tabular-nums; font-family:monospace;'
                )
            with ui.row().classes('w-full items-center justify-between'):
                ui.label('σ_x (m)')
                ekf_sx_label = ui.label('—').style(
                    'font-variant-numeric:tabular-nums; font-family:monospace; color:#888;'
                )
            with ui.row().classes('w-full items-center justify-between'):
                ui.label('σ_y (m)')
                ekf_sy_label = ui.label('—').style(
                    'font-variant-numeric:tabular-nums; font-family:monospace; color:#888;'
                )

            # --- Backend (graph-optimized) ---
            ui.label('Graph Backend (global)').style('font-size:10px; color:#888; margin-top:8px;')
            with ui.row().classes('w-full items-center justify-between'):
                ui.label('x (m)')
                opt_x_label = ui.label('—').style(
                    'font-variant-numeric:tabular-nums; font-family:monospace; color:#69ff47;'
                )
            with ui.row().classes('w-full items-center justify-between'):
                ui.label('y (m)')
                opt_y_label = ui.label('—').style(
                    'font-variant-numeric:tabular-nums; font-family:monospace; color:#69ff47;'
                )
            with ui.row().classes('w-full items-center justify-between'):
                ui.label('θ (°)')
                opt_theta_label = ui.label('—').style(
                    'font-variant-numeric:tabular-nums; font-family:monospace; color:#69ff47;'
                )

            # --- Robot frame (velocity-domain, EKF-derived) ---
            ui.label('Robot frame (local velocity)').style('font-size:10px; color:#888; margin-top:8px;')
            with ui.row().classes('w-full items-center justify-between'):
                ui.label('v_x (m/s)')
                rf_vx_label = ui.label('—').style(
                    'font-variant-numeric:tabular-nums; font-family:monospace; color:#ffd600;'
                )
            with ui.row().classes('w-full items-center justify-between'):
                ui.label('v_y (m/s)')
                rf_vy_label = ui.label('—').style(
                    'font-variant-numeric:tabular-nums; font-family:monospace; color:#ffd600;'
                )
            with ui.row().classes('w-full items-center justify-between'):
                ui.label('ω (°/s)')
                rf_omega_label = ui.label('—').style(
                    'font-variant-numeric:tabular-nums; font-family:monospace; color:#ffd600;'
                )

        CAMERA_W = 400
        CAMERA_H = 296
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
                    video_image = ui.interactive_image('/video/frame').style(
                        f"width:{CAMERA_W}px;"
                        f"height:{CAMERA_H}px;"
                        f"object-fit:contain;"
                        f"display:block;"
                    )
                else:
                    ui.image('./robot_image.jpg').style(
                        'width:100%; max-height: 285px; object-fit:contain;')
                    video_image = None    
    

    # TODO: Visualize LiDAR with a different library specifically for LiDAR
    # Visualize the lidar scans
    ROBOT_HALF_L = 13.5
    ROBOT_HALF_W = 11.0

    # Precompute map bounds
    _all_map_x = ([p[0] for p in parameters.corners.values()]
                  + [p[0] for p in parameters.tags.values()])
    _all_map_y = ([p[1] for p in parameters.corners.values()]
                  + [p[1] for p in parameters.tags.values()])
    MAP_XLIM = (min(_all_map_x) - 20, max(_all_map_x) + 20)
    MAP_YLIM = (min(_all_map_y) - 20, max(_all_map_y) + 20)

    def _robot_rect_world(rx, ry, theta):
        c, s = math.cos(theta), math.sin(theta)
        local = [
            ( ROBOT_HALF_L,  ROBOT_HALF_W),
            ( ROBOT_HALF_L, -ROBOT_HALF_W),
            (-ROBOT_HALF_L, -ROBOT_HALF_W),
            (-ROBOT_HALF_L,  ROBOT_HALF_W),
            ( ROBOT_HALF_L,  ROBOT_HALF_W),
        ]
        xs = [rx + c * lx - s * ly for lx, ly in local]
        ys = [ry + s * lx + c * ly for lx, ly in local]
        return xs, ys

    def draw_room():
        """Draw static room layer once at startup: walls + ArUco tags."""
        with main_plot:
            fig = main_plot.fig
            fig.patch.set_facecolor('#0d0d0d')
            ax = fig.gca()
            ax.set_facecolor('#0d0d0d')

            for w in parameters.walls:
                x1, y1 = parameters.corners[w[0]]
                x2, y2 = parameters.corners[w[1]]
                ax.plot([x1, x2], [y1, y2],
                        color='#e0e0e0', linewidth=1.8, zorder=2,
                        solid_capstyle='round')

            for tag_id, (tx, ty) in parameters.tags.items():
                ax.scatter(tx, ty, color='#ff4444', s=55, marker='D', zorder=4)
                ax.text(tx + 3, ty + 3, f"T{tag_id}",
                        fontsize=6, color='#ff9999', zorder=4)

            ax.set_xlim(*MAP_XLIM)
            ax.set_ylim(*MAP_YLIM)
            ax.set_aspect('equal', adjustable='box')
            ax.axis('off')
            ax.grid(False)
            fig.tight_layout(pad=0)
            main_plot.update()
                        
    # TODO: Use Open3D for LiDAR visualization
    def show_lidar_plot():
        with main_plot:
            fig = main_plot.fig
            fig.patch.set_facecolor('black')
            plt.clf()
            plt.style.use('dark_background')
            plt.tick_params(axis='x', colors='lightgray')
            plt.tick_params(axis='y', colors='lightgray')

            for i in range(num_angles):
                distance = lidar_distance_list[i]
                cos_ang = lidar_cos_angle_list[i]
                sin_ang = lidar_sin_angle_list[i]
                x = [distance * cos_ang, max_lidar_range * cos_ang]
                y = [distance * sin_ang, max_lidar_range * sin_ang]
                plt.plot(x, y, 'r')
            plt.grid(True)
            plt.xlim(-2, 2)
            plt.ylim(-2, 2)

    
    def update_pose_labels():
        """Refresh the live pose readout from cached SLAM outputs."""
        if slam is None:
            return

        # EKF frontend pose
        fe = slam.latest_frontend_output
        if fe is not None and fe.pose_estimate is not None:
            p = fe.pose_estimate.pose
            ekf_x_label.set_text(f"{p.x:+.3f}")
            ekf_y_label.set_text(f"{p.y:+.3f}")
            ekf_theta_label.set_text(f"{math.degrees(p.theta):+.1f}")

            cov = fe.pose_estimate.covariance
            if cov is not None:
                c = np.array(cov)
                ekf_sx_label.set_text(f"{math.sqrt(max(c[0,0], 0)):.4f}")
                ekf_sy_label.set_text(f"{math.sqrt(max(c[1,1], 0)):.4f}")

            # Robot-frame velocity: derive from last odom motion and loop dt
            om = fe.odom_motion
            if om is not None:
                dt = 0.1  # matches ui.timer interval
                rf_vx_label.set_text(f"{om.dx / dt:+.3f}")
                rf_vy_label.set_text(f"{om.dy / dt:+.3f}")
                rf_omega_label.set_text(f"{math.degrees(om.dtheta / dt):+.1f}")

        # Backend optimized pose (latest keyframe)
        opt = slam.latest_optimization_result
        if opt:
            last_id = max(opt.keys())
            op = opt[last_id]
            opt_x_label.set_text(f"{op.x:+.3f}")
            opt_y_label.set_text(f"{op.y:+.3f}")
            opt_theta_label.set_text(f"{math.degrees(op.theta):+.1f}")
    
    # # Visualize localization
    def update_slam_plot():
        """
        Project current LiDAR scan into world frame, accumulate the point cloud,
        and redraw all dynamic layers on top of the static room base.
        Called at the end of control_loop() every tick.
        """
        pose_x_cm  = slam_state['pose_x']
        pose_y_cm  = slam_state['pose_y']
        pose_theta = slam_state['pose_theta']
        cov        = slam_state['covariance']

        # A. Get pose from SLAM frontend
        if slam is not None:
            try:
                fe_out = slam.latest_frontend_output
                if fe_out is not None and fe_out.pose_estimate is not None:
                    p = fe_out.pose_estimate.pose
                    pose_x_cm  = p.x * 100.0
                    pose_y_cm  = p.y * 100.0
                    pose_theta = p.theta
                    cov_raw    = fe_out.pose_estimate.covariance
                    cov = np.array(cov_raw) if cov_raw is not None else None
                    slam_state.update({
                        'pose_x': pose_x_cm, 'pose_y': pose_y_cm,
                        'pose_theta': pose_theta, 'covariance': cov,
                    })
            except Exception:
                pass

        # B. Append to odometry trajectory
        slam_state['traj_x'].append(pose_x_cm)
        slam_state['traj_y'].append(pose_y_cm)

        # C. Project LiDAR scan into world frame (mm → cm, hardware angle convention)
        sig      = robot.robot_sensor_signal
        num_rays = int(getattr(sig, 'num_lidar_rays', 0))
        new_cx, new_cy = [], []
        for i in range(num_rays):
            dist_mm = sig.distances[i] if i < len(sig.distances) else 0
            ang_deg = sig.angles[i]    if i < len(sig.angles)    else 0
            if dist_mm <= 0:
                continue
            dist_cm = dist_mm * 0.1
            ray_rad = math.radians(360.0 - ang_deg) + pose_theta
            new_cx.append(pose_x_cm + dist_cm * math.cos(ray_rad))
            new_cy.append(pose_y_cm + dist_cm * math.sin(ray_rad))

        slam_state['cloud_x'].extend(new_cx)
        slam_state['cloud_y'].extend(new_cy)
        cap = slam_state['max_cloud_pts']
        if len(slam_state['cloud_x']) > cap:
            slam_state['cloud_x'] = slam_state['cloud_x'][-cap:]
            slam_state['cloud_y'] = slam_state['cloud_y'][-cap:]

        # D. Pull SLAM-optimised trajectory from backend
        if slam is not None:
            try:
                opt = slam.latest_optimization_result  # dict[int, Pose2D] | None
                if opt:
                    ids = sorted(opt.keys())
                    slam_state['slam_traj_x'] = [opt[k].x * 100 for k in ids]
                    slam_state['slam_traj_y'] = [opt[k].y * 100 for k in ids]
            except Exception:
                pass

        # E. Redraw dynamic layer
        with main_plot:
            fig = main_plot.fig
            ax  = fig.gca()

            # Remove only previously drawn dynamic artists
            for artist in list(ax.get_children()):
                if getattr(artist, '_slam_dyn', False):
                    try:
                        artist.remove()
                    except Exception:
                        pass

            def _tag(a):
                a._slam_dyn = True
                return a

            # 1. Accumulated world-frame point cloud
            if slam_state['cloud_x']:
                _tag(ax.scatter(
                    slam_state['cloud_x'], slam_state['cloud_y'],
                    s=1.5, c='#00e5ff', alpha=0.28, zorder=3, linewidths=0,
                ))

            # 2. Current scan rays (subsampled to ≤60 lines)
            ray_step = max(1, len(new_cx) // 60)
            for j in range(0, len(new_cx), ray_step):
                _tag(ax.plot(
                    [pose_x_cm, new_cx[j]], [pose_y_cm, new_cy[j]],
                    color='#00e5ff', alpha=0.07, linewidth=0.5, zorder=3,
                )[0])

            # 3. Odometry / dead-reckoning trajectory
            if len(slam_state['traj_x']) > 1:
                _tag(ax.plot(
                    slam_state['traj_x'], slam_state['traj_y'],
                    color='#ffd600', linewidth=1.2, alpha=0.65, zorder=5,
                )[0])

            # 4. SLAM graph-optimised trajectory
            if len(slam_state['slam_traj_x']) > 1:
                _tag(ax.plot(
                    slam_state['slam_traj_x'], slam_state['slam_traj_y'],
                    color='#69ff47', linewidth=1.6, alpha=0.85, zorder=6,
                )[0])

            # 5. Covariance ellipse (2-sigma, XY only)
            if cov is not None:
                try:
                    cov2     = np.array(cov)[:2, :2] * (100.0 ** 2)  # m² → cm²
                    eigvals, eigvecs = np.linalg.eigh(cov2)
                    eigvals  = np.maximum(eigvals, 1e-12)
                    ang_deg  = math.degrees(math.atan2(eigvecs[1, 0], eigvecs[0, 0]))
                    ell      = Ellipse(
                        xy=(pose_x_cm, pose_y_cm),
                        width=4.0 * math.sqrt(eigvals[1]),
                        height=4.0 * math.sqrt(eigvals[0]),
                        angle=ang_deg,
                        edgecolor='#aa00ff', facecolor='none',
                        linewidth=1.3, alpha=0.75, zorder=7,
                    )
                    ell._slam_dyn = True
                    ax.add_patch(ell)
                except Exception:
                    pass

            # 6. Robot body rectangle
            rx_pts, ry_pts = _robot_rect_world(pose_x_cm, pose_y_cm, pose_theta)
            _tag(ax.fill(rx_pts, ry_pts, color='#ff6d00', alpha=0.28, zorder=8)[0])
            _tag(ax.plot(rx_pts, ry_pts,  color='#ff6d00', linewidth=2.0, zorder=8)[0])

            # 7. Heading arrow
            arrow_len = ROBOT_HALF_L * 1.5
            ann = ax.annotate(
                '',
                xy=(pose_x_cm + arrow_len * math.cos(pose_theta),
                    pose_y_cm + arrow_len * math.sin(pose_theta)),
                xytext=(pose_x_cm, pose_y_cm),
                arrowprops=dict(arrowstyle='->', color='#ffffff', lw=2.0),
                zorder=9,
            )
            ann._slam_dyn = True

            ax.set_xlim(*MAP_XLIM)
            ax.set_ylim(*MAP_YLIM)
            main_plot.update()
    
    
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
    # Main control loop
    async def control_loop():
        try:
            update_connection_to_robot()

            if controller_switch.value and ctrl['joystick']:
                pygame.event.pump()
                js = ctrl['joystick']
                r2    = js.get_axis(5)
                steer = js.get_axis(0)
                speed = (r2 + 1) / 2
                if speed > DEADZONE:
                    left_val  = speed * (1 + steer)
                    right_val = speed * (1 - steer)
                    cmd_state['left']  = int(min(1.0, max(0.0, left_val))  * BASE_SPEED)
                    cmd_state['right'] = int(min(1.0, max(0.0, right_val)) * BASE_SPEED)
                elif abs(steer) > DEADZONE:
                    if steer > 0:
                        cmd_state['left']  = int(steer  * SPIN_SPEED)
                        cmd_state['right'] = 0
                    else:
                        cmd_state['left']  = 0
                        cmd_state['right'] = int(-steer * SPIN_SPEED)
                else:
                    cmd_state['left'] = cmd_state['right'] = 0

            await run.io_bound(
                robot.control_loop,
                cmd_state['left'],
                cmd_state['right'],
                logging_switch.value,
            )

            encoder_left_count_label.set_text(str(robot.robot_sensor_signal.encoder_left_counts))
            encoder_right_count_label.set_text(str(robot.robot_sensor_signal.encoder_right_counts))
            update_lidar_data()

            if slam is not None and EncoderState is not None:
                try:
                    robot_frame = make_robot_frame(
                        robot.robot_sensor_signal, EncoderState, LidarScan, RobotFrame
                    )
                    slam.run_frontend(robot_frame)

                    with robot.camera_sensor._lock:
                        pending_obs = robot.camera_sensor.landmark_observations[:]
                        robot.camera_sensor.landmark_observations.clear()
                    for obs in pending_obs:
                        slam.add_landmark_observation(obs)

                    if backend_switch.value:
                        slam.run_backend()

                except Exception as exc:
                    print(f"[SLAM] pipeline error: {exc}")

            update_pose_labels()

            slam_state['_plot_tick'] += 1
            if slam_state['_plot_tick'] % 5 == 0:
                update_slam_plot()

            update_video(video_image)

        except Exception as exc:
            import traceback
            print(f"[TIMER FATAL] {exc}")
            traceback.print_exc()
    
    ui.keyboard(on_key=update_commands)
    draw_room()
    ui.timer(0.1, control_loop)


# Run the gui
ui.run(native=False)