"""Generate random rectangles, mark their orientation, and correct them.

This script implements the workflow described in the user instructions:

1. Randomly generate several rectangles whose aspect ratio is greater than
   1.5. Each rectangle is drawn with four differently coloured edges.
2. For each rectangle, pick one of the long edges and label its endpoints A
   (smaller X coordinate) and B (greater X coordinate).
3. Compute the vector **C = B - A**.
4. Compute the centre points of the two long edges and label them G (smaller X
   coordinate) and H (greater X coordinate).
5. Decide the rectangle's *top* vertex and direction vector **V** using the
   provided heuristics.
6. Attach an isosceles triangle with a 120° apex angle to the top vertex, with
   the triangle pointing along **V**.
7. Rotate the rectangle and triangle so that the triangle points towards the
   positive Y direction.
8. Draw three panels: the original rectangles, the rectangles with the attached
   triangles, and the rotated, corrected results.

Running the script will save an ``output.png`` image in the current working
directory.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


Point = np.ndarray


@dataclass
class RectangleArtifact:
    """Container for the geometry created for a single rectangle."""

    corners: np.ndarray
    triangle: np.ndarray
    rotated_corners: np.ndarray
    rotated_triangle: np.ndarray


def generate_rectangle() -> np.ndarray:
    """Generate the four ordered corner points of a random rectangle."""

    # Random centre position keeps the rectangle within a roughly centred area.
    centre = np.array([random.uniform(-5.0, 5.0), random.uniform(-5.0, 5.0)])

    # Ensure the rectangle has a long side that is at least 1.5 times longer
    # than the short side.
    long_side = random.uniform(3.0, 6.0)
    aspect_ratio = random.uniform(1.6, 2.8)
    short_side = long_side / aspect_ratio

    angle_deg = random.uniform(0.0, 360.0)
    angle_rad = math.radians(angle_deg)

    long_dir = np.array([math.cos(angle_rad), math.sin(angle_rad)])
    short_dir = np.array([-math.sin(angle_rad), math.cos(angle_rad)])

    half_long = long_dir * (long_side / 2.0)
    half_short = short_dir * (short_side / 2.0)

    # Order the corners so consecutive points form the rectangle edges.
    corners = np.array(
        [
            centre - half_long - half_short,
            centre + half_long - half_short,
            centre + half_long + half_short,
            centre - half_long + half_short,
        ]
    )
    return corners


def edge_lengths(points: np.ndarray) -> List[float]:
    lengths: List[float] = []
    for idx in range(len(points)):
        start = points[idx]
        end = points[(idx + 1) % len(points)]
        lengths.append(float(np.linalg.norm(end - start)))
    return lengths


def pick_long_edge(points: np.ndarray) -> Tuple[int, Tuple[Point, Point]]:
    lengths = edge_lengths(points)
    long_length = max(lengths)
    long_indices = [i for i, length in enumerate(lengths) if math.isclose(length, long_length, rel_tol=1e-7)]
    idx = long_indices[0]
    start = points[idx]
    end = points[(idx + 1) % len(points)]
    return idx, (start, end)


def other_long_edge_index(idx: int) -> int:
    # In an ordered rectangle, the opposite edge is two steps away.
    return (idx + 2) % 4


def classify_vertices(
    points: np.ndarray,
    long_edge_idx: int,
    long_edge: Tuple[Point, Point],
) -> Tuple[Point, Point, Point, Point, np.ndarray]:
    """Return (A, B, G, H, vector_c) as described in the specification."""

    start, end = long_edge

    if start[0] <= end[0]:
        a, b = start, end
    else:
        a, b = end, start

    vector_c = b - a

    other_idx = other_long_edge_index(long_edge_idx)
    other_edge = (points[other_idx], points[(other_idx + 1) % len(points)])

    centre_1 = (start + end) / 2.0
    centre_2 = (other_edge[0] + other_edge[1]) / 2.0

    if centre_1[0] <= centre_2[0]:
        g, h = centre_1, centre_2
    else:
        g, h = centre_2, centre_1

    return a, b, g, h, vector_c


def determine_top_vertex_and_direction(
    vector_c: np.ndarray,
    g: Point,
    h: Point,
) -> Tuple[Point, np.ndarray]:
    """Determine the top vertex and direction vector V."""

    angle_to_y = math.degrees(math.atan2(vector_c[0], vector_c[1]))

    if -5.0 <= angle_to_y <= 5.0:
        top_vertex = g
        direction = h - g
    elif vector_c[1] < 0:
        top_vertex = h
        direction = g - h
    else:
        top_vertex = g
        direction = h - g

    return top_vertex, direction


def rotation_matrix(degrees_value: float) -> np.ndarray:
    radians_value = math.radians(degrees_value)
    cos_v = math.cos(radians_value)
    sin_v = math.sin(radians_value)
    return np.array([[cos_v, -sin_v], [sin_v, cos_v]])


def rotate_vector(vec: np.ndarray, degrees_value: float) -> np.ndarray:
    return rotation_matrix(degrees_value) @ vec


def build_triangle(top_vertex: Point, direction: np.ndarray, base_length: float) -> np.ndarray:
    """Construct an isosceles triangle with a 120° apex angle."""

    direction_norm = np.linalg.norm(direction)
    if direction_norm == 0.0:
        raise ValueError("Direction vector V cannot be zero length.")

    axis = direction / direction_norm

    # Rotate the axis by ±60° to create the two other sides.
    side_vec_1 = rotate_vector(axis, 60.0)
    side_vec_2 = rotate_vector(axis, -60.0)

    point_1 = top_vertex + side_vec_1 * base_length
    point_2 = top_vertex + side_vec_2 * base_length

    triangle = np.vstack([top_vertex, point_1, point_2])
    return triangle


def rotate_shape(points: np.ndarray, centre: np.ndarray, angle_deg: float) -> np.ndarray:
    rot_mat = rotation_matrix(angle_deg)
    translated = points - centre
    rotated = translated @ rot_mat.T
    return rotated + centre


def process_rectangle(points: np.ndarray) -> RectangleArtifact:
    long_idx, long_edge = pick_long_edge(points)
    a, b, g, h, vector_c = classify_vertices(points, long_idx, long_edge)
    top_vertex, direction = determine_top_vertex_and_direction(vector_c, g, h)

    long_length = np.linalg.norm(long_edge[1] - long_edge[0])
    base_length = long_length * 0.35
    triangle = build_triangle(top_vertex, direction, base_length)

    # Rotate so the triangle points to the positive Y direction.
    angle_to_y = math.degrees(math.atan2(direction[0], direction[1]))
    rotation_angle = -angle_to_y

    centre = points.mean(axis=0)

    rotated_corners = rotate_shape(points, centre, rotation_angle)
    rotated_triangle = rotate_shape(triangle, centre, rotation_angle)

    return RectangleArtifact(
        corners=points,
        triangle=triangle,
        rotated_corners=rotated_corners,
        rotated_triangle=rotated_triangle,
    )


def draw_rectangle(ax: plt.Axes, corners: np.ndarray, colors: Sequence[str]) -> None:
    for idx in range(4):
        start = corners[idx]
        end = corners[(idx + 1) % 4]
        ax.plot([start[0], end[0]], [start[1], end[1]], color=colors[idx], linewidth=2)


def draw_triangle(ax: plt.Axes, triangle: np.ndarray, color: str = "#ff7f0e") -> None:
    loop = np.vstack([triangle, triangle[0]])
    ax.plot(loop[:, 0], loop[:, 1], color=color, linewidth=2)
    ax.fill(triangle[:, 0], triangle[:, 1], color=color, alpha=0.2)


def prepare_axes(ax: plt.Axes, points: Iterable[np.ndarray]) -> None:
    combined = np.vstack(list(points))
    min_x, min_y = combined.min(axis=0)
    max_x, max_y = combined.max(axis=0)

    padding_x = (max_x - min_x) * 0.2 + 1.0
    padding_y = (max_y - min_y) * 0.2 + 1.0

    ax.set_xlim(min_x - padding_x, max_x + padding_x)
    ax.set_ylim(min_y - padding_y, max_y + padding_y)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")


def main() -> None:
    random.seed(42)
    np.random.seed(42)

    rectangles = [generate_rectangle() for _ in range(4)]
    artifacts = [process_rectangle(rect) for rect in rectangles]

    colours_cycle = ["tab:red", "tab:green", "tab:blue", "tab:purple"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: original rectangles.
    for artifact in artifacts:
        draw_rectangle(axes[0], artifact.corners, colours_cycle)
    prepare_axes(axes[0], (artifact.corners for artifact in artifacts))
    axes[0].set_title("原始矩形")

    # Panel 2: rectangles with triangles.
    for artifact in artifacts:
        draw_rectangle(axes[1], artifact.corners, colours_cycle)
        draw_triangle(axes[1], artifact.triangle)
    prepare_axes(
        axes[1],
        (
            np.vstack([artifact.corners, artifact.triangle])
            for artifact in artifacts
        ),
    )
    axes[1].set_title("原始矩形 + 三角形")

    # Panel 3: rotated rectangles with triangles.
    for artifact in artifacts:
        draw_rectangle(axes[2], artifact.rotated_corners, colours_cycle)
        draw_triangle(axes[2], artifact.rotated_triangle)
    prepare_axes(
        axes[2],
        (
            np.vstack([artifact.rotated_corners, artifact.rotated_triangle])
            for artifact in artifacts
        ),
    )
    axes[2].set_title("纠正后矩形 + 三角形")

    plt.tight_layout()
    fig.savefig("output.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
