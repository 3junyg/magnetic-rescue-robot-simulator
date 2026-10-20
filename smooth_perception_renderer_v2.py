import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

from config_v2 import BOARD_HEIGHT, BOARD_WIDTH, SENSOR_RANGE
from simulation.board_v2 import Board
from ui.renderer_v2 import draw_board as draw_standard_board
from ui.renderer_v2 import draw_perceived_walls


def _resize_linear(values: np.ndarray, scale: int) -> np.ndarray:
    rows, columns = values.shape
    source_x = np.arange(columns, dtype=float)
    source_y = np.arange(rows, dtype=float)
    target_x = np.linspace(0.0, columns - 1.0, columns * scale)
    target_y = np.linspace(0.0, rows - 1.0, rows * scale)
    horizontal = np.vstack([np.interp(target_x, source_x, row) for row in values])
    return np.vstack([np.interp(target_y, source_y, horizontal[:, column]) for column in range(horizontal.shape[1])]).T


def _continuous_anomaly(board: Board, scale: int = 6) -> tuple[np.ndarray, np.ndarray]:
    anomaly = board.perception.magnetic_anomaly()
    valid = np.isfinite(anomaly).astype(float)
    weighted = _resize_linear(np.nan_to_num(anomaly) * valid, scale)
    weight = _resize_linear(valid, scale)
    continuous = np.full_like(weighted, np.nan)
    visible = weight >= 0.35
    continuous[visible] = weighted[visible] / np.maximum(weight[visible], 1e-9)
    return continuous, weight


def draw_smooth_perception(axis, board: Board, robot_perception: bool = True) -> None:
    if not robot_perception:
        draw_standard_board(axis, board, False)
        return
    axis.set_facecolor("#dbeaf2")
    anomaly, visibility = _continuous_anomaly(board)
    axis.imshow(
        visibility,
        origin="lower",
        extent=[0, BOARD_WIDTH, 0, BOARD_HEIGHT],
        cmap="Blues",
        alpha=np.clip(visibility * 0.22, 0.0, 0.22),
        vmin=0.0,
        vmax=1.0,
        interpolation="bicubic",
        resample=True,
        zorder=0,
    )
    axis.imshow(
        anomaly,
        origin="lower",
        extent=[0, BOARD_WIDTH, 0, BOARD_HEIGHT],
        cmap="YlOrBr",
        alpha=0.62,
        vmin=0.0,
        vmax=25.0,
        interpolation="bicubic",
        resample=True,
        zorder=1,
    )
    draw_perceived_walls(axis, board.perception)
    candidates = board.perception.metal_candidates()
    if len(candidates):
        axis.scatter(candidates[:, 0], candidates[:, 1], marker="s", s=80, facecolors="none", edgecolors="#ef6c00", linewidths=1.8, label="Metal candidate", zorder=5)
    for endpoint in board.range_scan.endpoints[::4]:
        axis.plot([board.robot.position[0], endpoint[0]], [board.robot.position[1], endpoint[1]], color="#78909c", linewidth=0.5, alpha=0.3, antialiased=True, zorder=2)
    if board.detection_tracker.markers:
        positions = np.asarray([marker.position for marker in board.detection_tracker.markers])
        confidences = np.asarray([marker.confidence for marker in board.detection_tracker.markers])
        axis.scatter(positions[:, 0], positions[:, 1], marker="P", s=110 + confidences * 90, color="#d500f9", edgecolors="white", linewidths=1.0, label="Robot human estimate", zorder=9)
    robot = board.robot
    if len(robot.path) > 1:
        path = np.asarray(robot.path)
        axis.plot(path[:, 0], path[:, 1], color="#42a5f5", linewidth=1.4, alpha=0.8, antialiased=True, zorder=6)
    axis.scatter([robot.position[0]], [robot.position[1]], marker="o", s=130, color="#1565c0", edgecolors="white", label="Robot", zorder=7)
    axis.arrow(robot.position[0], robot.position[1], 3.5 * np.cos(robot.heading), 3.5 * np.sin(robot.heading), width=0.25, color="#0d47a1", zorder=8)
    axis.add_patch(Circle(robot.position, SENSOR_RANGE, fill=False, linestyle="--", linewidth=2.0, edgecolor="#1565c0", antialiased=True, zorder=5))
    axis.set_xlim(0, BOARD_WIDTH)
    axis.set_ylim(0, BOARD_HEIGHT)
    axis.set_aspect("equal")
    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    axis.set_title(f"{board.environment} | t={board.time:.1f}s | SMOOTH ROBOT PERCEPTION")
    axis.legend(loc="upper right")


def render_smooth_perception(board: Board):
    figure, axis = plt.subplots(figsize=(12, 8))
    draw_smooth_perception(axis, board)
    figure.tight_layout()
    return figure
