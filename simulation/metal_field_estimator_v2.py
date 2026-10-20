import numpy as np

from config_v2 import METAL_CANDIDATE_THRESHOLD, PERCEPTION_CELL_SIZE, SENSOR_RANGE


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
        self._map_cache = None
        self._candidate_cache = None

    def update(self, scan) -> None:
        columns, rows = self._indices(scan.positions)
        np.add.at(self.field_sum, (rows, columns), scan.vectors)
        np.add.at(self.field_square_sum, (rows, columns), scan.vectors * scan.vectors)
        np.add.at(self.field_count, (rows, columns), 1)
        self._map_cache = None
        self._candidate_cache = None

    def _indices(self, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        indices = (np.asarray(positions) / PERCEPTION_CELL_SIZE).astype(int)
        return np.clip(indices[:, 0], 0, self.columns - 1), np.clip(indices[:, 1], 0, self.rows - 1)

    def _candidates(self) -> list[tuple[np.ndarray, np.ndarray]]:
        if self._candidate_cache is not None:
            return self._candidate_cache
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
        self._candidate_cache = points
        return points

    def _stable_map(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._map_cache is not None:
            return self._map_cache
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
        self._map_cache = (stable, mean, temporal_std)
        return self._map_cache

    def predict_at(self, positions: np.ndarray) -> np.ndarray:
        prediction = np.tile(self.background, (len(positions), 1))
        stable, mean, _ = self._stable_map()
        columns, rows = self._indices(positions)
        stable_values = stable[rows, columns]
        prediction[stable_values] = mean[rows[stable_values], columns[stable_values]]
        for point, source_vector in self._candidates():
            distance = np.linalg.norm(positions - point, axis=1)
            attenuation = 1.0 / (1.0 + (distance / 3.0) ** 3)
            prediction += (~stable_values * attenuation)[:, None] * source_vector
        return prediction
