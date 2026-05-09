"""Draw a simple hand-specified robot trajectory on the measured maze map.

Examples:
    python draw_trajectory.py --points 20,20 80,110 190,120 270,185
    python draw_trajectory.py --points T0 80,110 T7 T9 --output trajectory.png

Points are in centimetres. Named points can be corner labels such as O, A, T,
or tag labels such as T0, T7, T15.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt


corners = {
    "O": (0, 0),
    "P": (48, 0),
    "A": (0, 203),
    "B": (79, 203),
    "C": (79, 263),
    "D": (162, 263),
    "E": (162, 203),
    "F": (245, 203),
    "G": (245, 285),
    "H": (308, 285),
    "I": (308, 122),
    "Y": (274, 122),
    "J": (274, 54),
    "Z": (193, 54),
    "K": (193, 13),
    "L": (130, 13),
    "M": (130, 54),
    "N": (48, 54),
    "Q": (48, 105),
    "S": (47, 136),
    "R": (88, 105),
    "T": (87, 136),
    "U": (167, 144),
    "V": (167, 108),
    "X": (208, 144),
    "W": (208, 108),
}

walls = [
    ("O", "P"),
    ("P", "N"),
    ("O", "A"),
    ("A", "B"),
    ("B", "C"),
    ("C", "D"),
    ("D", "E"),
    ("E", "F"),
    ("F", "G"),
    ("G", "H"),
    ("H", "I"),
    ("I", "Y"),
    ("Y", "J"),
    ("J", "Z"),
    ("Z", "K"),
    ("K", "L"),
    ("L", "M"),
    ("M", "N"),
    ("Q", "S"),
    ("S", "T"),
    ("T", "R"),
    ("R", "Q"),
    ("U", "V"),
    ("V", "W"),
    ("W", "X"),
    ("X", "U"),
]

tags = {
    0: (corners["O"][0] + 20, corners["O"][1]),
    1: (corners["O"][0], corners["O"][1] + 94),
    3: (corners["A"][0] + 12, corners["A"][1]),
    2: (corners["A"][0], corners["A"][1] - 30),
    4: (corners["T"][0], corners["T"][1] - 17),
    5: (corners["C"][0] + 48, corners["C"][1]),
    6: (corners["D"][0], corners["D"][1] - 31),
    7: (corners["U"][0], corners["U"][1] - 18),
    8: (corners["G"][0] + 30, corners["G"][1]),
    9: (corners["H"][0], corners["H"][1] - 91),
    10: (corners["J"][0], corners["J"][1] + 54),
    11: (corners["J"][0] - 32, corners["J"][1]),
    12: (corners["K"][0] - 30, corners["K"][1]),
    14: (corners["K"][0], corners["K"][1] + 32),
    13: (corners["L"][0], corners["L"][1] + 30),
    15: (corners["Q"][0] + 20, corners["Q"][1]),
}


def parse_point(text: str) -> tuple[float, float]:
    """Parse a waypoint from either a map label or an x,y coordinate."""
    value = text.strip()
    if value in corners:
        x, y = corners[value]
        return float(x), float(y)

    if value.upper().startswith("T") and value[1:].isdigit():
        tag_id = int(value[1:])
        if tag_id in tags:
            x, y = tags[tag_id]
            return float(x), float(y)

    parts = value.split(",")
    if len(parts) == 2:
        return float(parts[0]), float(parts[1])

    raise ValueError(f"Unknown waypoint '{text}'. Use a corner name, tag label, or x,y.")


def make_robot_like_polyline(
    waypoints: list[tuple[float, float]],
    step_cm: float = 12.0,
    wobble_cm: float = 2.5,
) -> list[tuple[float, float]]:
    """Create a deterministic, piecewise-linear path through waypoints.

    The output is not smoothed. It inserts small lateral offsets along each
    segment so the plotted line resembles a hand-driven robot trajectory rather
    than a perfect straight ruler line.
    """
    if len(waypoints) < 2:
        return waypoints[:]

    path: list[tuple[float, float]] = [waypoints[0]]
    for segment_index, (start, end) in enumerate(zip(waypoints, waypoints[1:])):
        x0, y0 = start
        x1, y1 = end
        dx = x1 - x0
        dy = y1 - y0
        length = math.hypot(dx, dy)
        if length <= 1e-9:
            continue

        ux = dx / length
        uy = dy / length
        nx = -uy
        ny = ux
        steps = max(1, int(math.ceil(length / step_cm)))

        for i in range(1, steps + 1):
            t = i / steps
            base_x = x0 + t * dx
            base_y = y0 + t * dy

            if i == steps:
                path.append((x1, y1))
                continue

            zigzag = -1.0 if (i + segment_index) % 2 == 0 else 1.0
            taper = math.sin(math.pi * t)
            offset = wobble_cm * zigzag * taper
            path.append((base_x + offset * nx, base_y + offset * ny))

    return path


def draw_map(ax) -> None:
    for wall in walls:
        x0, y0 = corners[wall[0]]
        x1, y1 = corners[wall[1]]
        ax.plot([x0, x1], [y0, y1], color="black", linewidth=2)

    for name, (x, y) in corners.items():
        ax.scatter(x, y, color="blue", s=18)
        ax.text(x + 2, y + 2, name, fontsize=7, color="blue")

    for tag_id, (x, y) in tags.items():
        ax.scatter(x, y, color="red", s=45, marker="x")
        ax.text(x + 2, y + 2, f"T{tag_id}", fontsize=8, color="red")


def draw_trajectory(
    waypoints: list[tuple[float, float]],
    output_path: Path | None,
    show: bool,
    step_cm: float,
    wobble_cm: float,
) -> None:
    path = make_robot_like_polyline(waypoints, step_cm=step_cm, wobble_cm=wobble_cm)

    fig, ax = plt.subplots(figsize=(8, 10))
    draw_map(ax)

    xs = [p[0] for p in path]
    ys = [p[1] for p in path]
    ax.plot(xs, ys, color="#ff9900", linewidth=2.4, label="hand-specified trajectory")
    ax.scatter(xs[0], ys[0], color="green", s=70, zorder=5, label="start")
    ax.scatter(xs[-1], ys[-1], color="purple", s=70, zorder=5, label="end")

    corner_x = [p[0] for p in waypoints[1:-1]]
    corner_y = [p[1] for p in waypoints[1:-1]]
    if corner_x:
        ax.scatter(corner_x, corner_y, color="#ff9900", edgecolors="black", s=55, zorder=5, label="waypoints")

    all_x = [p[0] for p in corners.values()] + [p[0] for p in tags.values()] + xs
    all_y = [p[1] for p in corners.values()] + [p[1] for p in tags.values()] + ys
    ax.set_xlim(min(all_x) - 15, max(all_x) + 15)
    ax.set_ylim(min(all_y) - 15, max(all_y) + 15)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (cm)")
    ax.set_ylabel("Y (cm)")
    ax.set_title("Ground Truth Maze Map with Hand-Specified Trajectory")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=160)
        print(f"Saved trajectory plot to {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw a hand-specified trajectory on the maze map.")
    parser.add_argument(
        "--points",
        nargs="+",
        required=True,
        help="Waypoints as x,y coordinates or labels such as O, A, T0, T7.",
    )
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("trajectory.png"))
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--step-cm", type=float, default=12.0)
    parser.add_argument("--wobble-cm", type=float, default=2.5)
    args = parser.parse_args()

    waypoints = [parse_point(point) for point in args.points]
    draw_trajectory(
        waypoints=waypoints,
        output_path=args.output,
        show=not args.no_show,
        step_cm=args.step_cm,
        wobble_cm=args.wobble_cm,
    )


if __name__ == "__main__":
    main()
