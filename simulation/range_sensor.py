import numpy as np

from config import BOARD_HEIGHT, BOARD_WIDTH, RANGE_SENSOR_RAYS, SENSOR_RANGE
from simulation.entities import RangeScan


class RangeSensor:
    def __init__(self) -> None:
        self.angles = np.linspace(-np.pi, np.pi, RANGE_SENSOR_RAYS, endpoint=False)
        self.directions = np.column_stack([np.cos(self.angles), np.sin(self.angles)])
        self.sample_distances = np.arange(0.5, SENSOR_RANGE + 0.5, 0.5)

    def scan(self, position: np.ndarray, obstacles) -> RangeScan:
        points = position[None, None, :] + self.directions[:, None, :] * self.sample_distances[None, :, None]
        blocked = (
            (points[:, :, 0] < 0.0)
            | (points[:, :, 0] > BOARD_WIDTH)
            | (points[:, :, 1] < 0.0)
            | (points[:, :, 1] > BOARD_HEIGHT)
        )
        for obstacle in obstacles:
            blocked |= (
                (points[:, :, 0] >= obstacle.x)
                & (points[:, :, 0] <= obstacle.x + obstacle.width)
                & (points[:, :, 1] >= obstacle.y)
                & (points[:, :, 1] <= obstacle.y + obstacle.height)
            )
        hits = blocked.any(axis=1)
        first_indices = blocked.argmax(axis=1)
        distances = np.where(hits, self.sample_distances[first_indices], SENSOR_RANGE).astype(float)
        endpoints = position[None, :] + self.directions * distances[:, None]
        return RangeScan(self.angles, distances, endpoints, hits)
