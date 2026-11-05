import numpy as np

from config_v2_1 import BOARD_HEIGHT, BOARD_WIDTH, RANGE_SENSOR_RAYS, SENSOR_RANGE
from simulation.entities_v2_1 import RangeScan


class RangeSensor:
    def __init__(self, sensor_range: float = SENSOR_RANGE) -> None:
        self.sensor_range = sensor_range
        self.angles = np.linspace(-np.pi, np.pi, RANGE_SENSOR_RAYS, endpoint=False)
        self.directions = np.column_stack([np.cos(self.angles), np.sin(self.angles)])

    def scan(self, position: np.ndarray, obstacles) -> RangeScan:
        distances = np.full(len(self.angles), self.sensor_range, dtype=float)
        directions = self.directions
        for axis, limit in ((0, BOARD_WIDTH), (1, BOARD_HEIGHT)):
            component = directions[:, axis]
            boundary = np.where(component > 0, limit, 0.0)
            t = np.divide(boundary - position[axis], component, out=np.full_like(component, np.inf), where=np.abs(component) > 1e-12)
            distances = np.minimum(distances, t)
        for obstacle in obstacles:
            lower = np.array([obstacle.x, obstacle.y])
            upper = lower + [obstacle.width, obstacle.height]
            parallel = np.abs(directions) <= 1e-12
            near = np.divide(lower - position, directions, out=np.full_like(directions, -np.inf), where=~parallel)
            far = np.divide(upper - position, directions, out=np.full_like(directions, np.inf), where=~parallel)
            entry = np.minimum(near, far).max(axis=1)
            exit = np.maximum(near, far).min(axis=1)
            outside_parallel = (parallel & ((position < lower) | (position > upper))).any(axis=1)
            valid = (~outside_parallel) & (exit >= np.maximum(entry, 0.0)) & (entry >= 0.0)
            distances = np.minimum(distances, np.where(valid, entry, np.inf))
        hits = distances < self.sensor_range
        endpoints = position[None, :] + self.directions * distances[:, None]
        return RangeScan(self.angles, distances, endpoints, hits, self.sensor_range)
