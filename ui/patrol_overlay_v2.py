import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from config_v2 import PATROL_REGION_SIZE


def draw_patrol_overlay(axis, runtime, position: np.ndarray) -> None:
    if runtime is None or not runtime.patrol_active:
        return
    scheduler = runtime.patrol_scheduler
    patches, colors = [], []
    for region in scheduler.goals:
        if region in scheduler.unreachable:
            continue
        row, col = region
        patches.append(Rectangle((col * PATROL_REGION_SIZE, row * PATROL_REGION_SIZE), PATROL_REGION_SIZE, PATROL_REGION_SIZE))
        colors.append('#43a047' if scheduler.visited_round[region] >= scheduler.round else '#ffb300')
    axis.add_collection(PatchCollection(patches, facecolors=colors, edgecolors='#607d8b', linewidths=0.4, alpha=0.18, zorder=2))
    if scheduler.target is not None:
        target = scheduler.target
        axis.scatter([target[0]], [target[1]], marker='*', s=220, color='#00acc1', edgecolors='#004d40', zorder=11)
        planner = runtime.active_planner
        route = planner._goal_route
        if route:
            centers = (np.asarray(route)[:, ::-1] + 0.5) * planner.resolution
            start = int(np.argmin(np.linalg.norm(centers - position, axis=1)))
            path = np.vstack([position, centers[start:]])
            axis.plot(path[:, 0], path[:, 1], color='#00acc1', linestyle='--', linewidth=1.6, zorder=10)
    state = scheduler.status()
    axis.text(0.015, 0.98, f"순찰 {state['round']}회차 · 방문 {state['visited']}/{state['regions']} 구역", transform=axis.transAxes, va='top', fontsize=10, bbox={'facecolor': 'white', 'alpha': 0.85, 'edgecolor': 'none'}, zorder=12)
    previous_legend = axis.get_legend()
    legend = axis.legend(handles=[Patch(facecolor='#43a047', alpha=0.5, label='이번 순찰 방문'), Patch(facecolor='#ffb300', alpha=0.5, label='이번 순찰 미방문'), Line2D([], [], marker='*', color='#00acc1', linestyle='--', label='순찰 목표 / 계획 경로')], loc='lower left', fontsize=8)
    if previous_legend is not None:
        axis.add_artist(previous_legend)
