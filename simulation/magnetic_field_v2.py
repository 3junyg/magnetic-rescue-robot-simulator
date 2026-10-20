import numpy as np

from config_v2 import BOARD_HEIGHT, BOARD_WIDTH, PERCEPTION_CELL_SIZE, SENSOR_RANGE
from simulation.entities_v2 import MagneticMeasurement, MagneticScan
from simulation.field_model_v2 import LegacyMagneticFieldModel, MagneticFieldModel


class MagneticFieldSensor:
    def __init__(self, noise: float, rng: np.random.Generator, field_model: MagneticFieldModel | None = None) -> None:
        self.noise = noise
        self.rng = rng
        self.field_model = field_model or LegacyMagneticFieldModel()
        self.bias = rng.normal(0.0, 0.25, 3)
        self.drift = np.zeros(3)

    def measure(self, board) -> MagneticMeasurement:
        self.drift = 0.995 * self.drift + self.rng.normal(0.0, 0.006, 3)
        total = self.field_model.field_at(board.robot.position, board)
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
        vectors = self.field_model.scan_at(points, position, board)
        vectors += self.bias + self.drift
        vectors += self.rng.normal(0.0, self.noise, vectors.shape)
        return MagneticScan(points, vectors, np.linalg.norm(vectors, axis=1))
