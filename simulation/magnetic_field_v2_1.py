import numpy as np

from config_v2_1 import BOARD_HEIGHT, BOARD_WIDTH, PERCEPTION_CELL_SIZE, SENSOR_RANGE
from simulation.entities_v2_1 import MagneticMeasurement, MagneticScan
from simulation.field_model_v2_1 import LegacyMagneticFieldModel, MagneticFieldModel


class MagneticFieldSensor:
    def __init__(self, noise: float, rng: np.random.Generator, field_model: MagneticFieldModel | None = None) -> None:
        self.noise = noise
        self.rng = rng
        self.field_model = field_model or LegacyMagneticFieldModel()
        self.bias = rng.normal(0.0, 0.25, 3)
        self.drift = np.zeros(3)
        self.mi_diagnostics = None

    def _active_sample(self, points, board):
        model = self.field_model
        p = model.profile
        total, reference = model.receive(points, board)
        compensated = total - p["primary_cancellation_fraction"] * reference
        input_sigma = p["receiver_output_noise_rms_v"] / p["receiver_gain"]
        noise = self.rng.normal(0.0, input_sigma / np.sqrt(2.0), compensated.shape) + 1j * self.rng.normal(0.0, input_sigma / np.sqrt(2.0), compensated.shape)
        encoded_error = self.bias + self.drift + self.rng.normal(0.0, self.noise, compensated.shape)
        noise += encoded_error * p["detector_input_v_per_unit"] * np.exp(1j * np.deg2rad(p["demodulation_phase_deg"]))
        output = (compensated + noise) * p["receiver_gain"]
        amplitudes = np.abs(output)
        clipped = amplitudes > p["receiver_max_ac_rms_v"]
        output *= np.minimum(1.0, p["receiver_max_ac_rms_v"] / np.maximum(amplitudes, 1e-20))
        vectors = model.encode(output / p["receiver_gain"])
        diagnostics = {"input_unit": "V_RMS_complex", "detector_unit": "normalized_demodulated_synthetic", "i_output_v": output.real[0].tolist(), "q_output_v": output.imag[0].tolist(), "output_rms_v": np.abs(output[0]).tolist(), "primary_input_rms_v": np.abs(reference[0]).tolist(), "clipped": bool(np.any(clipped)), "calibration": model.calibration.status()}
        return vectors, diagnostics

    def measure(self, board) -> MagneticMeasurement:
        self.drift = 0.995 * self.drift + self.rng.normal(0.0, 0.006, 3)
        if hasattr(self.field_model, "receive") and getattr(self.field_model, "uses_physical_mi", True):
            values, self.mi_diagnostics = self._active_sample(board.robot.position.reshape(1, 2), board)
            total = values[0]
        else:
            total = self.field_model.field_at(board.robot.position, board)
            total += self.bias + self.drift + self.rng.normal(0.0, self.noise, 3)
            if hasattr(self.field_model, "calibration"):
                self.mi_diagnostics = {"calibration": self.field_model.calibration.status(), "output_rms_v": [],
                                       "clipped": False, "signal_mode": self.field_model.profile.get("signal_mode", "scaled_v2"),
                                       "scaling": self.field_model.scaling}
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
        radius = board.sensor_range
        minimum_column = max(0, int((position[0] - radius) / PERCEPTION_CELL_SIZE))
        maximum_column = min(int(BOARD_WIDTH / PERCEPTION_CELL_SIZE) - 1, int((position[0] + radius) / PERCEPTION_CELL_SIZE))
        minimum_row = max(0, int((position[1] - radius) / PERCEPTION_CELL_SIZE))
        maximum_row = min(int(BOARD_HEIGHT / PERCEPTION_CELL_SIZE) - 1, int((position[1] + radius) / PERCEPTION_CELL_SIZE))
        columns = (np.arange(minimum_column, maximum_column + 1) + 0.5) * PERCEPTION_CELL_SIZE
        rows = (np.arange(minimum_row, maximum_row + 1) + 0.5) * PERCEPTION_CELL_SIZE
        grid_x, grid_y = np.meshgrid(columns, rows)
        points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
        points = points[np.linalg.norm(points - position, axis=1) <= radius]
        if hasattr(self.field_model, "receive") and getattr(self.field_model, "uses_physical_mi", True):
            vectors, _ = self._active_sample(points, board)
        else:
            vectors = self.field_model.scan_at(points, position, board)
            vectors += self.bias + self.drift
            vectors += self.rng.normal(0.0, self.noise, vectors.shape)
        return MagneticScan(points, vectors, np.linalg.norm(vectors, axis=1), radius)
