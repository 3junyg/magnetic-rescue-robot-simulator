import numpy as np

from config_v2 import BOARD_HEIGHT, BOARD_WIDTH, METAL_DISPLAY_THRESHOLD, PERCEPTION_CELL_SIZE
from config_v2 import WALL_SURFACE_SAMPLE_SPACING, WALL_SURFACE_CONNECT_DISTANCE


class RobotPerception:
    def __init__(self) -> None:
        self.rows = int(np.ceil(BOARD_HEIGHT / PERCEPTION_CELL_SIZE))
        self.columns = int(np.ceil(BOARD_WIDTH / PERCEPTION_CELL_SIZE))
        self.field_sum = np.zeros((self.rows, self.columns, 3), dtype=float)
        self.field_count = np.zeros((self.rows, self.columns), dtype=int)
        self.observed = np.zeros((self.rows, self.columns), dtype=bool)
        self.obstacle_points: list[np.ndarray] = []
        self.wall_segments: dict[tuple[int, float], list[tuple[float, float]]] = {}
        columns, rows = np.meshgrid(np.arange(self.columns), np.arange(self.rows))
        self.cell_centers = np.column_stack(
            [(columns.ravel() + 0.5) * PERCEPTION_CELL_SIZE, (rows.ravel() + 0.5) * PERCEPTION_CELL_SIZE]
        )

    def update(self, position, magnetic_scan, range_scan) -> None:
        columns = np.clip((magnetic_scan.positions[:, 0] / PERCEPTION_CELL_SIZE).astype(int), 0, self.columns - 1)
        rows = np.clip((magnetic_scan.positions[:, 1] / PERCEPTION_CELL_SIZE).astype(int), 0, self.rows - 1)
        np.add.at(self.field_sum, (rows, columns), magnetic_scan.vectors)
        np.add.at(self.field_count, (rows, columns), 1)
        offset = self.cell_centers - position
        distance = np.linalg.norm(offset, axis=1)
        ray_count = len(range_scan.angles)
        angle = np.arctan2(offset[:, 1], offset[:, 0])
        ray_index = (np.rint((angle + np.pi) / (2.0 * np.pi) * ray_count).astype(int)) % ray_count
        visible = (distance <= float(np.max(range_scan.distances))) & (distance <= range_scan.distances[ray_index])
        self.observed |= visible.reshape(self.rows, self.columns)
        self._update_wall_segments(range_scan)
        for point in range_scan.endpoints[range_scan.hits]:
            recent = np.asarray(self.obstacle_points[-800:], dtype=float)
            if recent.size == 0 or not np.any(np.linalg.norm(recent - point, axis=1) < 0.8):
                self.obstacle_points.append(point.copy())
        if len(self.obstacle_points) > 2500:
            self.obstacle_points = self.obstacle_points[-2500:]
        if self.wall_segments:
            points = [np.asarray(self.obstacle_points).reshape(-1, 2)]
            for (axis, fixed), intervals in self.wall_segments.items():
                for lower, upper in intervals:
                    values = np.linspace(lower, upper, max(2, int(np.ceil((upper - lower) / WALL_SURFACE_SAMPLE_SPACING)) + 1))
                    segment = np.empty((len(values), 2))
                    segment[:, axis] = fixed
                    segment[:, 1 - axis] = values
                    points.append(segment)
            self.obstacle_points = list(np.unique(np.round(np.vstack(points), 6), axis=0))

    def _update_wall_segments(self, scan) -> None:
        for index in range(len(scan.hits)):
            next_index = (index + 1) % len(scan.hits)
            if not (scan.hits[index] and scan.hits[next_index]):
                continue
            a, b = scan.endpoints[index], scan.endpoints[next_index]
            delta = np.abs(a - b)
            if np.linalg.norm(delta) > WALL_SURFACE_CONNECT_DISTANCE:
                continue
            axes = np.flatnonzero(delta < 1e-7)
            if not len(axes):
                continue
            axis = int(axes[0])
            key = (axis, round(float(a[axis]), 6))
            lower, upper = sorted((float(a[1 - axis]), float(b[1 - axis])))
            intervals = sorted([*self.wall_segments.get(key, []), (lower, upper)])
            merged = []
            for start, end in intervals:
                if merged and start <= merged[-1][1] + 1e-6:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], end))
                else:
                    merged.append((start, end))
            self.wall_segments[key] = merged

    def magnetic_anomaly(self) -> np.ndarray:
        mean = np.zeros_like(self.field_sum)
        observed = self.field_count > 0
        mean[observed] = self.field_sum[observed] / self.field_count[observed][:, None]
        anomaly = np.full((self.rows, self.columns), np.nan)
        anomaly[observed] = np.linalg.norm(mean[observed] - np.array([24.0, -7.0, 39.0]), axis=1)
        return anomaly

    def metal_candidates(self) -> np.ndarray:
        anomaly = self.magnetic_anomaly()
        rows, columns = np.where(anomaly >= METAL_DISPLAY_THRESHOLD)
        if len(rows) == 0:
            return np.empty((0, 2))
        points = np.column_stack(
            [
                (columns + 0.5) * PERCEPTION_CELL_SIZE,
                (rows + 0.5) * PERCEPTION_CELL_SIZE,
            ]
        )
        strengths = anomaly[rows, columns]
        selected = []
        for index in np.argsort(strengths)[::-1]:
            point = points[index]
            if all(np.linalg.norm(point - saved) >= 10.0 for saved in selected):
                selected.append(point)
            if len(selected) >= 12:
                break
        return np.asarray(selected)

    def candidate_distances(self, position: np.ndarray) -> np.ndarray:
        candidates = self.metal_candidates()
        if len(candidates) == 0:
            return np.empty(0)
        return np.linalg.norm(candidates - position, axis=1)
