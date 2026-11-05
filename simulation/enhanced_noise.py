import numpy as np

from simulation.entities import MagneticMeasurement, MagneticScan


class EnhancedNoiseSensor:
    def __init__(self, base_sensor, seed: int, level: float = 0.0) -> None:
        self.base_sensor = base_sensor
        self.rng = np.random.default_rng(int(seed) + 918273)
        self.level = max(0.0, float(level))
        self.last_time = 0.0
        self.correlated = np.zeros(3, dtype=float)
        self.transient = np.zeros(3, dtype=float)
        self.bias_jump = np.zeros(3, dtype=float)
        self.low_amplitude = self.rng.uniform(0.18, 0.85, (3, 3))
        self.low_frequency = self.rng.uniform(0.015, 0.18, 3)
        self.low_phase = self.rng.uniform(0.0, 2.0 * np.pi, (3, 3))
        self.wave_vector = self.rng.uniform(-0.10, 0.10, (3, 2))
        self.wave_frequency = self.rng.uniform(0.02, 0.14, 3)
        self.wave_phase = self.rng.uniform(0.0, 2.0 * np.pi, 3)
        self.wave_axis = self.rng.normal(0.0, 1.0, (3, 3))
        self.wave_axis /= np.maximum(np.linalg.norm(self.wave_axis, axis=1, keepdims=True), 1e-9)

    def set_level(self, level: float) -> None:
        self.level = float(np.clip(level, 0.0, 5.0))
        self.correlated.fill(0.0)
        self.transient.fill(0.0)
        self.bias_jump.fill(0.0)

    def _advance(self, board) -> None:
        current_time = float(board.time)
        dt = current_time - self.last_time
        if dt <= 0.0:
            return
        self.last_time = current_time
        self.correlated = np.exp(-dt / 7.0) * self.correlated + self.rng.normal(0.0, 0.18 * self.level * np.sqrt(dt), 3)
        self.transient *= np.exp(-dt / 0.8)
        if self.rng.random() < 0.035 * dt:
            self.transient += self.rng.normal(0.0, 3.2 * self.level, 3)
        self.bias_jump *= np.exp(-dt / 18.0)
        if self.rng.random() < 0.006 * dt:
            self.bias_jump += self.rng.normal(0.0, 1.1 * self.level, 3)

    def _common(self, board) -> np.ndarray:
        angles = 2.0 * np.pi * self.low_frequency[:, None] * board.time + self.low_phase
        low = self.level * np.sum(self.low_amplitude * np.sin(angles), axis=0)
        return self.correlated + self.transient + self.bias_jump + low

    def _spatial(self, positions: np.ndarray, time: float) -> np.ndarray:
        phase = positions @ self.wave_vector.T
        phase += 2.0 * np.pi * self.wave_frequency[None, :] * time + self.wave_phase[None, :]
        return (np.sin(phase) * self.level) @ self.wave_axis

    def _motion(self, board) -> np.ndarray:
        if not board.robot.moving:
            return np.zeros(3, dtype=float)
        heading = float(board.robot.heading)
        strength = 0.16 * float(board.robot.speed) * self.level
        return strength * np.array([np.cos(heading), np.sin(heading), 0.35 * np.sin(2.0 * heading)])

    def measure(self, board) -> MagneticMeasurement:
        self._advance(board)
        measurement = self.base_sensor.measure(board)
        vector = np.array([measurement.bx, measurement.by, measurement.bz], dtype=float)
        vector += self._common(board) + self._spatial(board.robot.position.reshape(1, 2), board.time)[0] + self._motion(board)
        vector += self.rng.normal(0.0, np.array([0.22, 0.28, 0.36]) * self.level)
        return MagneticMeasurement(measurement.timestamp, float(vector[0]), float(vector[1]), float(vector[2]), float(np.linalg.norm(vector)), measurement.robot_moving)

    def scan(self, board) -> MagneticScan:
        self._advance(board)
        scan = self.base_sensor.scan(board)
        vectors = scan.vectors.copy()
        vectors += self._common(board)[None, :] + self._spatial(scan.positions, board.time) + self._motion(board)[None, :]
        vectors += self.rng.normal(0.0, np.array([0.22, 0.28, 0.36]) * self.level, vectors.shape)
        return MagneticScan(scan.positions, vectors, np.linalg.norm(vectors, axis=1))
