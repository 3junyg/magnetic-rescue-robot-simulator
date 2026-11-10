import argparse
from pathlib import Path
import subprocess
import sys

import numpy as np

from simulation.board_v2 import Board
from simulation.entities_v2 import MagneticMeasurement, MagneticScan
import training.generate_hard_human_dataset_v2 as hard_generator


class EnhancedNoiseSensor:
    def __init__(self, base_sensor, board: Board, level: float) -> None:
        self.base_sensor = base_sensor
        self.rng = np.random.default_rng(int(board.seed) + 918273)
        self.level = max(0.0, float(level))
        self.last_time = float(board.time)
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

    def _advance(self, board: Board) -> None:
        current_time = float(board.time)
        dt = current_time - self.last_time
        if dt <= 0.0:
            return
        self.last_time = current_time
        decay = np.exp(-dt / 7.0)
        innovation_scale = 0.18 * self.level * np.sqrt(max(dt, 1e-6))
        self.correlated = decay * self.correlated + self.rng.normal(0.0, innovation_scale, 3)
        self.transient *= np.exp(-dt / 0.8)
        if self.rng.random() < 0.035 * dt:
            self.transient += self.rng.normal(0.0, 3.2 * self.level, 3)
        self.bias_jump *= np.exp(-dt / 18.0)
        if self.rng.random() < 0.006 * dt:
            self.bias_jump += self.rng.normal(0.0, 1.1 * self.level, 3)

    def _low_frequency(self, time: float) -> np.ndarray:
        angles = 2.0 * np.pi * self.low_frequency[:, None] * time + self.low_phase
        return self.level * np.sum(self.low_amplitude * np.sin(angles), axis=0)

    def set_level(self, level: float) -> None:
        self.level = max(0.0, float(level))
        self.correlated.fill(0.0)
        self.transient.fill(0.0)
        self.bias_jump.fill(0.0)

    def _spatial_field(self, positions: np.ndarray, time: float) -> np.ndarray:
        phase = positions @ self.wave_vector.T
        phase += 2.0 * np.pi * self.wave_frequency[None, :] * time + self.wave_phase[None, :]
        weights = np.sin(phase) * (1.0 * self.level)
        return weights @ self.wave_axis

    def _motion_field(self, board: Board) -> np.ndarray:
        if not board.robot.moving:
            return np.zeros(3, dtype=float)
        heading = float(board.robot.heading)
        strength = 0.16 * float(board.robot.speed) * self.level
        return strength * np.array([np.cos(heading), np.sin(heading), 0.35 * np.sin(2.0 * heading)])

    def _common(self, board: Board) -> np.ndarray:
        return self.correlated + self.transient + self.bias_jump + self._low_frequency(float(board.time))

    def measure(self, board: Board) -> MagneticMeasurement:
        self._advance(board)
        measurement = self.base_sensor.measure(board)
        vector = np.array([measurement.bx, measurement.by, measurement.bz], dtype=float)
        position = board.robot.position.reshape(1, 2)
        vector += self._common(board)
        vector += self._spatial_field(position, float(board.time))[0]
        vector += self._motion_field(board)
        vector += self.rng.normal(0.0, np.array([0.22, 0.28, 0.36]) * self.level)
        return MagneticMeasurement(
            measurement.timestamp,
            float(vector[0]),
            float(vector[1]),
            float(vector[2]),
            float(np.linalg.norm(vector)),
            measurement.robot_moving,
        )

    def scan(self, board: Board) -> MagneticScan:
        self._advance(board)
        scan = self.base_sensor.scan(board)
        vectors = scan.vectors.copy()
        vectors += self._common(board)[None, :]
        vectors += self._spatial_field(scan.positions, float(board.time))
        vectors += self._motion_field(board)[None, :]
        vectors += self.rng.normal(0.0, np.array([0.22, 0.28, 0.36]) * self.level, vectors.shape)
        return MagneticScan(scan.positions, vectors, np.linalg.norm(vectors, axis=1))


class EnhancedNoiseBoard(Board):
    noise_level_min = 0.7
    noise_level_max = 2.0

    def __init__(self, seed: int, environment: str = "도심 침수 지역") -> None:
        super().__init__(seed, environment)
        level_rng = np.random.default_rng(int(seed) + 417731)
        self.enhanced_noise_level = float(level_rng.uniform(self.noise_level_min, self.noise_level_max))
        self.sensor = EnhancedNoiseSensor(self.sensor, self, self.enhanced_noise_level)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/human_enhanced_noise_v1.npz"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/human_detector_enhanced_noise"))
    parser.add_argument("--episodes", type=int, default=480)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--noise-min", type=float, default=0.7)
    parser.add_argument("--noise-max", type=float, default=2.0)
    parser.add_argument("--noise-level", type=float, default=None)
    parser.add_argument("--predictor-epochs", type=int, default=25)
    parser.add_argument("--classifier-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()
    if args.noise_level is not None:
        EnhancedNoiseBoard.noise_level_min = max(0.0, args.noise_level)
        EnhancedNoiseBoard.noise_level_max = EnhancedNoiseBoard.noise_level_min
    else:
        EnhancedNoiseBoard.noise_level_min = max(0.0, min(args.noise_min, args.noise_max))
        EnhancedNoiseBoard.noise_level_max = max(EnhancedNoiseBoard.noise_level_min, max(args.noise_min, args.noise_max))
    hard_generator.Board = EnhancedNoiseBoard
    hard_generator.generate_dataset(args.data, args.episodes, args.steps, args.seed)
    if args.skip_training:
        return
    command = [
        sys.executable,
        "-m",
        "training.train_human_detector_v2",
        "--data",
        str(args.data),
        "--output-dir",
        str(args.model_dir),
        "--predictor-epochs",
        str(args.predictor_epochs),
        "--classifier-epochs",
        str(args.classifier_epochs),
        "--batch-size",
        str(args.batch_size),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
