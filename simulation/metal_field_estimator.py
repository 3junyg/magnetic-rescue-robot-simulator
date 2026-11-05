import numpy as np

from config import METAL_CANDIDATE_THRESHOLD, PERCEPTION_CELL_SIZE, SENSOR_RANGE


class MetalFieldEstimator:
    def __init__(self, min_observations: int = 8, max_temporal_std: float = 2.5) -> None:
        self.rows = 40
        self.columns = 60
        self.min_observations = min_observations
        self.max_temporal_std = max_temporal_std
        self.field_sum = np.zeros((self.rows, self.columns, 3), dtype=float)
        self.field_square_sum = np.zeros((self.rows, self.columns, 3), dtype=float)
        self.field_count = np.zeros((self.rows, self.columns), dtype=int)
        self.background = np.array([24.0, -7.0, 39.0], dtype=float)

    def update(self, scan) -> None:
        for point, vector in zip(scan.positions, scan.vectors):
            column = min(self.columns - 1, max(0, int(point[0] / PERCEPTION_CELL_SIZE)))
            row = min(self.rows - 1, max(0, int(point[1] / PERCEPTION_CELL_SIZE)))
            self.field_sum[row, column] += vector
            self.field_square_sum[row, column] += vector * vector
            self.field_count[row, column] += 1

    def _candidates(self) -> list[tuple[np.ndarray, np.ndarray]]:
        observed, mean, temporal_std = self._stable_map()
        anomaly = np.full((self.rows, self.columns), np.nan)
        anomaly[observed] = np.linalg.norm(mean[observed] - self.background, axis=1)
        rows, columns = np.where((anomaly >= METAL_CANDIDATE_THRESHOLD) & (temporal_std <= self.max_temporal_std))
        points = []
        for index in np.argsort(anomaly[rows, columns])[::-1]:
            point = np.array([(columns[index] + 0.5) * PERCEPTION_CELL_SIZE, (rows[index] + 0.5) * PERCEPTION_CELL_SIZE])
            if all(np.linalg.norm(point - saved[0]) >= 10.0 for saved in points):
                points.append((point, mean[rows[index], columns[index]] - self.background))
            if len(points) >= 12:
                break
        return points

    def _stable_map(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        observed = self.field_count >= self.min_observations
        mean = np.zeros_like(self.field_sum)
        mean[observed] = self.field_sum[observed] / self.field_count[observed][:, None]
        variance = np.zeros_like(self.field_sum)
        variance[observed] = np.maximum(
            0.0,
            self.field_square_sum[observed] / self.field_count[observed][:, None] - mean[observed] * mean[observed],
        )
        temporal_std = np.linalg.norm(np.sqrt(variance), axis=2)
        stable = observed & (temporal_std <= self.max_temporal_std)
        return stable, mean, temporal_std

    def predict_at(self, positions: np.ndarray) -> np.ndarray:
        prediction = np.tile(self.background, (len(positions), 1))
        stable, mean, _ = self._stable_map()
        for index, position in enumerate(positions):
            column = min(self.columns - 1, max(0, int(position[0] / PERCEPTION_CELL_SIZE)))
            row = min(self.rows - 1, max(0, int(position[1] / PERCEPTION_CELL_SIZE)))
            if stable[row, column]:
                prediction[index] = mean[row, column]
        for point, source_vector in self._candidates():
            distance = np.linalg.norm(positions - point, axis=1)
            attenuation = 1.0 / (1.0 + (distance / 3.0) ** 3)
            stable_values = np.array([
                stable[min(self.rows - 1, max(0, int(position[1] / PERCEPTION_CELL_SIZE))), min(self.columns - 1, max(0, int(position[0] / PERCEPTION_CELL_SIZE)))]
                for position in positions
            ])
            prediction += (~stable_values * attenuation)[:, None] * source_vector
        return prediction
