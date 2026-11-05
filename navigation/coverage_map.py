import numpy as np

from config import BOARD_HEIGHT, BOARD_WIDTH, PERCEPTION_CELL_SIZE, SENSOR_RANGE


class CoverageMap:
    def __init__(self) -> None:
        self.rows = int(np.ceil(BOARD_HEIGHT / PERCEPTION_CELL_SIZE))
        self.columns = int(np.ceil(BOARD_WIDTH / PERCEPTION_CELL_SIZE))
        columns = (np.arange(self.columns) + 0.5) * PERCEPTION_CELL_SIZE
        rows = (np.arange(self.rows) + 0.5) * PERCEPTION_CELL_SIZE
        grid_x, grid_y = np.meshgrid(columns, rows)
        self.centers = np.column_stack([grid_x.ravel(), grid_y.ravel()])
        self.observed = np.zeros((self.rows, self.columns), dtype=bool)
        self.obstacles = np.zeros((self.rows, self.columns), dtype=bool)

    def update(self, position: np.ndarray, range_scan) -> None:
        offsets = self.centers - position
        distances = np.linalg.norm(offsets, axis=1)
        active = distances <= SENSOR_RANGE
        angles = (np.arctan2(offsets[:, 1], offsets[:, 0]) + np.pi) % (2.0 * np.pi) - np.pi
        ray_count = len(range_scan.angles)
        ray_indices = np.round((angles + np.pi) / (2.0 * np.pi) * ray_count).astype(int) % ray_count
        visible = active & (distances <= range_scan.distances[ray_indices] + PERCEPTION_CELL_SIZE * 0.75)
        self.observed |= visible.reshape(self.rows, self.columns)
        for endpoint, hit in zip(range_scan.endpoints, range_scan.hits):
            if not hit:
                continue
            column = min(self.columns - 1, max(0, int(endpoint[0] / PERCEPTION_CELL_SIZE)))
            row = min(self.rows - 1, max(0, int(endpoint[1] / PERCEPTION_CELL_SIZE)))
            self.obstacles[row, column] = True
            if row > 0:
                self.obstacles[row - 1, column] = True
            if row + 1 < self.rows:
                self.obstacles[row + 1, column] = True
            if column > 0:
                self.obstacles[row, column - 1] = True
            if column + 1 < self.columns:
                self.obstacles[row, column + 1] = True

    def frontier(self) -> np.ndarray:
        known_free = self.observed & ~self.obstacles
        adjacent_unknown = np.zeros_like(self.observed)
        adjacent_unknown[1:] |= ~self.observed[:-1]
        adjacent_unknown[:-1] |= ~self.observed[1:]
        adjacent_unknown[:, 1:] |= ~self.observed[:, :-1]
        adjacent_unknown[:, :-1] |= ~self.observed[:, 1:]
        return known_free & adjacent_unknown

    def observation(self, position: np.ndarray, downsample: int = 2) -> np.ndarray:
        frontier = self.frontier()
        robot = np.zeros_like(self.observed)
        column = min(self.columns - 1, max(0, int(position[0] / PERCEPTION_CELL_SIZE)))
        row = min(self.rows - 1, max(0, int(position[1] / PERCEPTION_CELL_SIZE)))
        robot[row, column] = True
        channels = np.stack([self.observed, self.obstacles, frontier, robot]).astype(np.float32)
        if downsample == 1:
            return channels
        rows = self.rows // downsample
        columns = self.columns // downsample
        pooled = channels[:, : rows * downsample, : columns * downsample]
        pooled = pooled.reshape(4, rows, downsample, columns, downsample).max(axis=(2, 4))
        return pooled.astype(np.float32)
