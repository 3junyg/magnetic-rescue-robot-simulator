import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, Rectangle
import numpy as np

from config import BOARD_HEIGHT, BOARD_WIDTH, SENSOR_RANGE
from evaluation.localization_score import evaluate_localization
from simulation.board import Board


if any(font.name == "Malgun Gothic" for font in font_manager.fontManager.ttflist):
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False


def draw_board(axis, board: Board, robot_perception: bool) -> None:
    axis.set_facecolor("#dbeaf2")
    if robot_perception:
        observed = np.where(board.perception.observed, 1.0, np.nan)
        axis.imshow(observed, origin="lower", extent=[0, BOARD_WIDTH, 0, BOARD_HEIGHT], cmap="Blues", alpha=0.2, vmin=0.0, vmax=1.0, zorder=0)
        anomaly = board.perception.magnetic_anomaly()
        axis.imshow(anomaly, origin="lower", extent=[0, BOARD_WIDTH, 0, BOARD_HEIGHT], cmap="YlOrBr", alpha=0.55, vmin=0.0, vmax=25.0, zorder=1)
        if board.perception.obstacle_points:
            points = np.asarray(board.perception.obstacle_points)
            axis.scatter(points[:, 0], points[:, 1], marker=".", s=13, color="#263238", label="Perceived obstacle", zorder=3)
        candidates = board.perception.metal_candidates()
        if len(candidates):
            axis.scatter(candidates[:, 0], candidates[:, 1], marker="s", s=80, facecolors="none", edgecolors="#ef6c00", linewidths=1.8, label="Metal candidate", zorder=5)
        for endpoint in board.range_scan.endpoints[::4]:
            axis.plot([board.robot.position[0], endpoint[0]], [board.robot.position[1], endpoint[1]], color="#78909c", linewidth=0.5, alpha=0.35, zorder=2)
    else:
        for obstacle in board.obstacles:
            axis.add_patch(Rectangle((obstacle.x, obstacle.y), obstacle.width, obstacle.height, facecolor="#37474f", edgecolor="#1c2529", zorder=3))
        if board.metals:
            positions = np.asarray([metal.position for metal in board.metals])
            sizes = np.asarray([metal.size for metal in board.metals])
            axis.scatter(positions[:, 0], positions[:, 1], marker="s", s=18.0 + sizes * 16.0, color="#a66b35", label="Actual metal", zorder=4)
    if not robot_perception and board.people:
        positions = np.asarray([person.position for person in board.people])
        axis.scatter(positions[:, 0], positions[:, 1], marker="*", s=145, color="#e53935", label="Actual person", zorder=6)
    if board.detection_tracker.markers:
        positions = np.asarray([marker.position for marker in board.detection_tracker.markers])
        confidences = np.asarray([marker.confidence for marker in board.detection_tracker.markers])
        axis.scatter(positions[:, 0], positions[:, 1], marker="P", s=110 + confidences * 90, color="#d500f9", edgecolors="white", linewidths=1.0, label="Robot human estimate", zorder=9)
        if not robot_perception:
            evaluation = evaluate_localization(board.detection_tracker.markers, board.people)
            for match in evaluation.matches:
                marker = board.detection_tracker.markers[match.marker_index]
                person = board.people[match.person_index]
                axis.plot([marker.position[0], person.position[0]], [marker.position[1], person.position[1]], linestyle="--", color="#d500f9", linewidth=1.2, alpha=0.8, zorder=8)
                axis.text(marker.position[0] + 0.8, marker.position[1] + 0.8, f"{match.score:.0f}", color="#8e24aa", fontsize=9, zorder=10)
    robot = board.robot
    if len(robot.path) > 1:
        path = np.asarray(robot.path)
        axis.plot(path[:, 0], path[:, 1], color="#42a5f5", linewidth=1.4, alpha=0.8)
    axis.scatter([robot.position[0]], [robot.position[1]], marker="o", s=130, color="#1565c0", edgecolors="white", label="Robot", zorder=7)
    axis.arrow(robot.position[0], robot.position[1], 3.5 * np.cos(robot.heading), 3.5 * np.sin(robot.heading), width=0.25, color="#0d47a1", zorder=8)
    axis.add_patch(Circle(robot.position, SENSOR_RANGE, fill=False, linestyle="--", linewidth=2.0, edgecolor="#1565c0", zorder=5))
    axis.set_xlim(0, BOARD_WIDTH)
    axis.set_ylim(0, BOARD_HEIGHT)
    axis.set_aspect("equal")
    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    mode = "ROBOT PERCEPTION" if robot_perception else "ACTUAL ENVIRONMENT"
    axis.set_title(f"{board.environment} | t={board.time:.1f}s | {mode}")
    axis.legend(loc="upper right")


def render_board(board: Board, robot_perception: bool):
    figure, axis = plt.subplots(figsize=(12, 8))
    draw_board(axis, board, robot_perception)
    figure.tight_layout()
    return figure


def render_sensor_history(board: Board):
    measurements = board.measurements
    times = [item.timestamp for item in measurements]
    figure, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    axes[0].plot(times, [item.bx for item in measurements], label="Bx")
    axes[0].plot(times, [item.by for item in measurements], label="By")
    axes[0].plot(times, [item.bz for item in measurements], label="Bz")
    axes[0].set_ylabel("Field")
    axes[0].legend(loc="upper right", ncol=3)
    axes[0].grid(alpha=0.25)
    axes[1].plot(times, [item.magnitude for item in measurements], color="#7b1fa2")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("|B|")
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    return figure
