import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import parameters
import data_handling
import particle_filter


# ============================================================
# User settings
# ============================================================


# If True, initialize PF around given initial_state
KNOWN_START = True
INITIAL_STATE = particle_filter.State(0.2, 0.3, 0.65398)
INITIAL_STDEV = particle_filter.State(0.1, 0.1, 0.1)

# Plot settings
SHOW_PARTICLE_SNAPSHOTS = True
NUM_SNAPSHOTS = 10              # number of later particle snapshots along trajectory
MAX_PARTICLES_TO_DRAW = 200    # downsample particles in static figure

# Output behavior
SHOW_FIGURES = True
SAVE_FIGURES = False
SAVE_DIR = './pf_report_plots_known_start' if KNOWN_START else './pf_report_plots_unknown_start'


# True trajectory as a hardcoded XY list.
# Put None if not available for this file.
# TRUE_XY = None
# DATA_FILE = './data/robot_data_0_0_10_03_26_22_59_41.pkl'
# SIMPLE TRAJECTORY: 
DATA_FILE = './data/robot_data_0_0_10_03_26_23_56_30.pkl'  # simple
TRUE_XY = [
    (0.00, 0.00),
    (0.25, 0.15),
    (0.50, 0.30),
    (0.75, 0.47),
    (1.00, 0.63),
    (1.25, 0.80),
    (1.50, 0.97),
    (1.75, 1.13),
    (2.00, 1.30),
]

# # BIT COMPLEX TRAJECTORY:
# DATA_FILE = './data/robot_data_0_0_10_03_26_23_53_22.pkl'    # complex
# TRUE_XY = [
#     (0.00, 0.00),
#     (0.40, 0.22),
#     (0.80, 0.40),
#     (1.10, 0.52),
#     (1.35, 0.64),
#     (1.55, 0.74),
#     (1.70, 0.82),
#     (1.82, 0.85),
#     (1.95, 0.85),
#     (2.08, 0.84),
#     (2.22, 0.81),
#     (2.36, 0.74),
#     (2.50, 0.64),
#     (2.63, 0.51),
#     (2.75, 0.35),
#     (2.83, 0.18),
#     (2.90, 0.00),
# ]


# ============================================================
# Helpers
# ============================================================

def angle_wrap(angle):
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def position_error(est_x, est_y, true_x, true_y):
    return np.sqrt((est_x - true_x) ** 2 + (est_y - true_y) ** 2)


def resample_truth_to_length(true_xy, target_len):
    """
    Resample a hardcoded truth list to match the number of PF timesteps.
    Assumes truth points are ordered in time.
    """
    if true_xy is None:
        return None, None

    true_xy = np.array(true_xy, dtype=float)
    if len(true_xy) == 0:
        return None, None

    if len(true_xy) == target_len:
        return true_xy[:, 0], true_xy[:, 1]

    idx_src = np.linspace(0, len(true_xy) - 1, len(true_xy))
    idx_tgt = np.linspace(0, len(true_xy) - 1, target_len)

    tx = np.interp(idx_tgt, idx_src, true_xy[:, 0])
    ty = np.interp(idx_tgt, idx_src, true_xy[:, 1])
    return tx, ty


def maybe_output_figure(fig, out_path):
    if SAVE_FIGURES:
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f'Saved: {out_path}')
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)


# ============================================================
# Offline run + collect data
# ============================================================

def run_pf_and_collect(data_file):
    map_obj = particle_filter.Map(parameters.wall_corner_list)
    pf_data = data_handling.get_file_data_for_pf(data_file)

    pf = particle_filter.ParticleFilter(
        parameters.num_particles,
        map_obj,
        initial_state=INITIAL_STATE,
        state_stdev=INITIAL_STDEV,
        known_start_state=KNOWN_START,
        encoder_counts_0=pf_data[0][2].encoder_counts
    )

    # Initial particles before motion
    step_init = max(1, len(pf.particle_set.particle_list) // MAX_PARTICLES_TO_DRAW)
    initial_x = [p.state.x for p in pf.particle_set.particle_list[::step_init]]
    initial_y = [p.state.y for p in pf.particle_set.particle_list[::step_init]]

    times = []
    est_x = []
    est_y = []
    est_theta = []

    particle_snapshots = []

    snapshot_indices = set()
    if len(pf_data) > 1 and NUM_SNAPSHOTS > 0:
        snapshot_indices = set(
            np.linspace(1, len(pf_data) - 1, NUM_SNAPSHOTS, dtype=int).tolist()
        )

    for t in range(1, len(pf_data)):
        row = pf_data[t]
        dt = pf_data[t][0] - pf_data[t - 1][0]
        u_t = np.array([row[2].encoder_counts, row[2].steering])
        z_t = row[2]

        pf.update(u_t, z_t, dt)

        mean_state = pf.particle_set.mean_state

        times.append(pf_data[t][0] - pf_data[0][0])
        est_x.append(mean_state.x)
        est_y.append(mean_state.y)
        est_theta.append(mean_state.theta)

        if SHOW_PARTICLE_SNAPSHOTS and t in snapshot_indices:
            step = max(1, len(pf.particle_set.particle_list) // MAX_PARTICLES_TO_DRAW)
            xs = [p.state.x for p in pf.particle_set.particle_list[::step]]
            ys = [p.state.y for p in pf.particle_set.particle_list[::step]]
            particle_snapshots.append({
                't': times[-1],
                'x': xs,
                'y': ys,
            })

    est_x = np.array(est_x, dtype=float)
    est_y = np.array(est_y, dtype=float)
    est_theta = np.array(est_theta, dtype=float)
    times = np.array(times, dtype=float)

    true_x, true_y = resample_truth_to_length(TRUE_XY, len(est_x))
    truth_available = (true_x is not None and true_y is not None)
    final_position_error = None
    if truth_available:
        final_position_error = float(
            position_error(est_x[-1], est_y[-1], true_x[-1], true_y[-1])
        )

    return {
        'map': map_obj,
        'times': times,
        'est_x': est_x,
        'est_y': est_y,
        'est_theta': est_theta,
        'true_x': true_x,
        'true_y': true_y,
        'truth_available': truth_available,
        'initial_particles_x': np.array(initial_x, dtype=float),
        'initial_particles_y': np.array(initial_y, dtype=float),
        'particle_snapshots': particle_snapshots,
        'final_position_error': final_position_error,
    }


# ============================================================
# Plotting
# ============================================================

def plot_xy_trajectory(results, save_dir):
    fig, ax = plt.subplots(figsize=(10, 8))
    map_obj = results['map']

    # Walls
    for i, wall in enumerate(map_obj.wall_list):
        label = 'Map walls' if i == 0 else None
        ax.plot(
            [wall.corner1.x, wall.corner2.x],
            [wall.corner1.y, wall.corner2.y],
            'k',
            linewidth=2,
            label=label
        )

    # Estimated trajectory
    ax.plot(
        results['est_x'],
        results['est_y'],
        'r-',
        linewidth=2,
        label='Estimated trajectory'
    )

    # Estimated start and end
    ax.plot(
        results['est_x'][0],
        results['est_y'][0],
        'o',
        color='orange',
        markersize=8,
        label='Estimated start'
    )
    ax.plot(
        results['est_x'][-1],
        results['est_y'][-1],
        's',
        color='blue',
        markersize=8,
        label='Estimated end'
    )

    # True trajectory if available
    if results['truth_available']:
        ax.plot(
            results['true_x'],
            results['true_y'],
            linestyle='--',
            color='purple',
            linewidth=2,
            label='True trajectory'
        )
        ax.plot(
            results['true_x'][0],
            results['true_y'][0],
            'o',
            color='purple',
            markersize=7,
            label='True start'
        )
        ax.plot(
            results['true_x'][-1],
            results['true_y'][-1],
            's',
            color='purple',
            markersize=7,
            label='True end'
        )

    # Initial particles
    if SHOW_PARTICLE_SNAPSHOTS:
        ax.plot(
            results['initial_particles_x'],
            results['initial_particles_y'],
            '.',
            color='peru',
            markersize=4,
            alpha=0.55,
            label='Initial particles'
        )

        # Later particle snapshots
        for idx, snap in enumerate(results['particle_snapshots']):
            label = 'Later particle snapshots' if idx == 0 else None
            ax.plot(
                snap['x'],
                snap['y'],
                '.',
                color='green',
                markersize=3,
                alpha=0.35,
                label=label
            )

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('PF XY Trajectory')
    ax.axis(map_obj.plot_range)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)
    ax.legend(loc='best')

    out_path = os.path.join(save_dir, 'pf_xy_trajectory.png')
    maybe_output_figure(fig, out_path)


def plot_state_vs_time(results, save_dir):
    times = results['times']

    # X vs time
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times, results['est_x'], 'r-', linewidth=2, label='Estimated x')
    if results['truth_available']:
        ax.plot(times, results['true_x'], linestyle='--', color='purple', linewidth=2, label='True x')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('X (m)')
    ax.set_title('X vs Time')
    ax.grid(True)
    ax.legend()
    out_path = os.path.join(save_dir, 'pf_x_vs_time.png')
    maybe_output_figure(fig, out_path)

    # Y vs time
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times, results['est_y'], 'r-', linewidth=2, label='Estimated y')
    if results['truth_available']:
        ax.plot(times, results['true_y'], linestyle='--', color='purple', linewidth=2, label='True y')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Y vs Time')
    ax.grid(True)
    ax.legend()
    out_path = os.path.join(save_dir, 'pf_y_vs_time.png')
    maybe_output_figure(fig, out_path)

    # Theta vs time
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times, results['est_theta'], 'r-', linewidth=2, label='Estimated theta')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Theta (rad)')
    ax.set_title('Theta vs Time')
    ax.grid(True)
    ax.legend()
    out_path = os.path.join(save_dir, 'pf_theta_vs_time.png')
    maybe_output_figure(fig, out_path)


def plot_error_vs_time(results, save_dir):
    if not results['truth_available']:
        print('Truth not available. Skipping error plot.')
        return

    times = results['times']
    pos_err = position_error(
        results['est_x'],
        results['est_y'],
        results['true_x'],
        results['true_y'],
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times, pos_err, color='magenta', linewidth=2, label='Position error')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position error (m)')
    ax.set_title('Position Error vs Time')
    ax.grid(True)
    ax.legend()

    out_path = os.path.join(save_dir, 'pf_position_error_vs_time.png')
    maybe_output_figure(fig, out_path)


def plot_report_figures(data_file, save_dir):
    ensure_dir(save_dir)
    results = run_pf_and_collect(data_file)
    plot_xy_trajectory(results, save_dir)
    # plot_state_vs_time(results, save_dir)
    plot_error_vs_time(results, save_dir)
    if results['truth_available']:
        print(f"Final position error: {results['final_position_error']:.4f} m")
    else:
        print("Final position error: truth not available")


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    plot_report_figures(DATA_FILE, SAVE_DIR)