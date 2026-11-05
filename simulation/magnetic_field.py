import numpy as np

from config import BOARD_HEIGHT, BOARD_WIDTH, PERCEPTION_CELL_SIZE, PERSON_FAST_MODULATION, PERSON_FIELD_DISTANCE_SCALE, PERSON_SLOW_MODULATION, SENSOR_RANGE
from simulation.entities import MagneticMeasurement, MagneticScan


class MagneticFieldSensor:
    def __init__(self, noise: float, rng: np.random.Generator) -> None:
        self.noise = noise
        self.rng = rng
        self.bias = rng.normal(0.0, 0.25, 3)
        self.drift = np.zeros(3)

    def _field_at(self, position, board) -> np.ndarray:
        total = np.array([24.0, -7.0, 39.0], dtype=float)
        for metal in board.metals:
            offset = position - metal.position
            distance = float(np.linalg.norm(offset))
            if 0.01 < distance <= SENSOR_RANGE:
                direction = np.array([np.cos(metal.orientation), np.sin(metal.orientation), 0.55])
                attenuation = metal.strength / (1.0 + (distance / (metal.size + 1.0)) ** 3)
                total += direction * attenuation
        for person in board.people:
            offset = position - person.position
            distance = float(np.linalg.norm(offset))
            if 0.01 < distance <= SENSOR_RANGE:
                temporal = (
                    1.0
                    + PERSON_SLOW_MODULATION * np.sin(2.0 * np.pi * 0.23 * board.time + person.phase[0])
                    + PERSON_FAST_MODULATION * np.sin(2.0 * np.pi * 1.85 * board.time + person.phase[1])
                    + person.variation
                )
                direction = np.array(
                    [np.cos(person.heading), np.sin(person.heading), 0.3 * np.sin(person.phase[2])]
                )
                attenuation = person.signal_strength * temporal / (1.0 + (distance / PERSON_FIELD_DISTANCE_SCALE) ** 2)
                total += direction * attenuation
        return total

    def measure(self, board) -> MagneticMeasurement:
        self.drift = 0.995 * self.drift + self.rng.normal(0.0, 0.006, 3)
        total = self._field_at(board.robot.position, board)
        total += self.bias + self.drift + self.rng.normal(0.0, self.noise, 3)
        return MagneticMeasurement(
            board.time,
            float(total[0]),
            float(total[1]),
            float(total[2]),
            float(np.linalg.norm(total)),
            board.robot.moving,
        )

    def scan(self, board) -> MagneticScan:
        position = board.robot.position
        minimum_column = max(0, int((position[0] - SENSOR_RANGE) / PERCEPTION_CELL_SIZE))
        maximum_column = min(int(BOARD_WIDTH / PERCEPTION_CELL_SIZE) - 1, int((position[0] + SENSOR_RANGE) / PERCEPTION_CELL_SIZE))
        minimum_row = max(0, int((position[1] - SENSOR_RANGE) / PERCEPTION_CELL_SIZE))
        maximum_row = min(int(BOARD_HEIGHT / PERCEPTION_CELL_SIZE) - 1, int((position[1] + SENSOR_RANGE) / PERCEPTION_CELL_SIZE))
        columns = (np.arange(minimum_column, maximum_column + 1) + 0.5) * PERCEPTION_CELL_SIZE
        rows = (np.arange(minimum_row, maximum_row + 1) + 0.5) * PERCEPTION_CELL_SIZE
        grid_x, grid_y = np.meshgrid(columns, rows)
        points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
        points = points[np.linalg.norm(points - position, axis=1) <= SENSOR_RANGE]
        vectors = np.tile(np.array([24.0, -7.0, 39.0]), (len(points), 1))
        for metal in board.metals:
            if np.linalg.norm(metal.position - position) > SENSOR_RANGE:
                continue
            distances = np.linalg.norm(points - metal.position, axis=1)
            active = (distances > 0.01) & (distances <= SENSOR_RANGE)
            direction = np.array([np.cos(metal.orientation), np.sin(metal.orientation), 0.55])
            attenuation = metal.strength / (1.0 + (distances[active] / (metal.size + 1.0)) ** 3)
            vectors[active] += attenuation[:, None] * direction
        for person in board.people:
            if np.linalg.norm(person.position - position) > SENSOR_RANGE:
                continue
            distances = np.linalg.norm(points - person.position, axis=1)
            active = (distances > 0.01) & (distances <= SENSOR_RANGE)
            temporal = (
                1.0
                + PERSON_SLOW_MODULATION * np.sin(2.0 * np.pi * 0.23 * board.time + person.phase[0])
                + PERSON_FAST_MODULATION * np.sin(2.0 * np.pi * 1.85 * board.time + person.phase[1])
                + person.variation
            )
            direction = np.array([np.cos(person.heading), np.sin(person.heading), 0.3 * np.sin(person.phase[2])])
            attenuation = person.signal_strength * temporal / (1.0 + (distances[active] / PERSON_FIELD_DISTANCE_SCALE) ** 2)
            vectors[active] += attenuation[:, None] * direction
        vectors += self.bias + self.drift
        vectors += self.rng.normal(0.0, self.noise, vectors.shape)
        return MagneticScan(points, vectors, np.linalg.norm(vectors, axis=1))
