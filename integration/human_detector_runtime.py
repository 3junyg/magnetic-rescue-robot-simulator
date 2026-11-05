from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ml.features import extract_scan_features, extract_vector_features, local_motion
from ml.models import MagneticTransitionPredictor, ResidualHumanCNN
from config import BOARD_HEIGHT, BOARD_WIDTH, SENSOR_RANGE
from simulation.metal_field_estimator import MetalFieldEstimator


@dataclass
class HumanDetectorResult:
    person_probability: float
    predicted_label: int
    status: str
    timestamp: float
    estimated_position: np.ndarray | None
    estimated_count: int
    window_size: int
    location_probabilities: np.ndarray | None = None


class HumanDetectorRuntime:
    def __init__(self, model_path: str | Path | None = None, device: str = "cpu", add_markers: bool = True) -> None:
        root = Path(__file__).resolve().parents[1]
        self.model_path = Path(model_path) if model_path else root / "models" / "human_detector" / "best_human_transition_detector.pth"
        if not self.model_path.is_absolute():
            self.model_path = root / self.model_path
        if not self.model_path.exists():
            raise FileNotFoundError(f"Human detector model not found: {self.model_path}")
        self.device = torch.device(device)
        self.add_markers = add_markers
        if self.device.type == "cpu":
            torch.set_num_threads(1)
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        self.window = int(checkpoint["window"])
        self.scan_size = int(checkpoint["scan_size"])
        channels = int(checkpoint["classifier_channels"])
        self.presence_threshold = max(0.5, float(checkpoint.get("presence_threshold", 0.5)))
        self.location_threshold = float(checkpoint.get("location_threshold", 0.375))
        self.predictor = MagneticTransitionPredictor(self.scan_size).to(self.device)
        self.classifier = ResidualHumanCNN(channels, len(checkpoint.get("count_classes", ["0", "1", "2", "3+"])), int(checkpoint.get("location_cells", 32))).to(self.device)
        self.predictor.load_state_dict(checkpoint["predictor_state_dict"])
        self.classifier.load_state_dict(checkpoint["classifier_state_dict"])
        self.predictor.eval()
        self.classifier.eval()
        self.scan_mean = self._array(checkpoint["scan_mean"], self.scan_size)
        self.scan_std = self._array(checkpoint["scan_std"], self.scan_size)
        self.motion_mean = self._array(checkpoint["motion_mean"], 3)
        self.motion_std = self._array(checkpoint["motion_std"], 3)
        self.physics_mean = self._array(checkpoint["physics_mean"], 8)
        self.physics_std = self._array(checkpoint["physics_std"], 8)
        self.center_mean = self._array(checkpoint["center_mean"], 3)
        self.center_std = self._array(checkpoint["center_std"], 3)
        self.estimator = MetalFieldEstimator()
        self.features: deque[np.ndarray] = deque(maxlen=self.window)
        self.previous_scan: np.ndarray | None = None
        self.previous_position: np.ndarray | None = None
        self.previous_heading: float | None = None
        self.marker_probability_threshold = max(0.60, self.presence_threshold)
        self.marker_persistence = 1
        self.marker_cooldown = 5.0
        self.marker_min_distance = 5.0
        self.positive_streak = 0
        self.probability_history: deque[float] = deque(maxlen=5)
        self.last_marker_time = -np.inf
        self.last_marker_position: np.ndarray | None = None
        self.ignored_positions: list[np.ndarray] = []
        self.result = HumanDetectorResult(0.0, 0, "WARMING UP", 0.0, None, 0, 0)

    @staticmethod
    def _array(value, size: int) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32).reshape(size)
        return np.where(np.abs(array) < 1e-6, 1.0, array)

    @staticmethod
    def _normalize(value: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        return ((value - mean) / std).astype(np.float32)

    def reset(self) -> None:
        self.estimator = MetalFieldEstimator()
        self.features.clear()
        self.previous_scan = None
        self.previous_position = None
        self.previous_heading = None
        self.positive_streak = 0
        self.probability_history.clear()
        self.last_marker_time = -np.inf
        self.last_marker_position = None
        self.ignored_positions.clear()
        self.result = HumanDetectorResult(0.0, 0, "WARMING UP", 0.0, None, 0, 0)

    def ignore_position(self, position: np.ndarray, radius: float = 8.0) -> None:
        if not any(float(np.linalg.norm(position - saved)) <= radius for saved in self.ignored_positions):
            self.ignored_positions.append(np.asarray(position, dtype=float).copy())

    def _location_to_world(self, index: int, position: np.ndarray, heading: float) -> np.ndarray:
        radial = index // 8
        angular = index % 8
        distance = (radial + 0.5) / 4.0 * SENSOR_RANGE
        angle = (angular + 0.5) / 8.0 * 2.0 * np.pi
        world_angle = heading + angle
        point = position + distance * np.array([np.cos(world_angle), np.sin(world_angle)])
        return np.clip(point, np.array([0.0, 0.0]), np.array([BOARD_WIDTH, BOARD_HEIGHT]))

    def _location_from_scan(self, index: int, positions: np.ndarray, residual_vectors: np.ndarray, robot_position: np.ndarray, heading: float) -> np.ndarray:
        relative = positions - robot_position
        cosine = np.cos(heading)
        sine = np.sin(heading)
        local_x = cosine * relative[:, 0] + sine * relative[:, 1]
        local_y = -sine * relative[:, 0] + cosine * relative[:, 1]
        distances = np.linalg.norm(relative, axis=1)
        angles = (np.arctan2(local_y, local_x) + 2.0 * np.pi) % (2.0 * np.pi)
        radial = np.minimum(3, (distances / SENSOR_RANGE * 4).astype(int))
        angular = np.minimum(7, (angles / (2.0 * np.pi) * 8).astype(int))
        active = (radial == index // 8) & (angular == index % 8)
        if not np.any(active):
            return self._location_to_world(index, robot_position, heading)
        candidates = np.where(active)[0]
        best = candidates[int(np.argmax(np.linalg.norm(residual_vectors[candidates], axis=1)))]
        return np.asarray(positions[best], dtype=float).copy()

    def process_board(self, board) -> HumanDetectorResult:
        scan = board.magnetic_scan
        position = np.asarray(board.robot.position, dtype=np.float32).copy()
        heading = float(board.robot.heading)
        current = extract_scan_features(scan, position, heading).reshape(-1)
        current = self._normalize(current, self.scan_mean, self.scan_std)
        expected = self.estimator.predict_at(scan.positions)
        physics = extract_vector_features(scan.positions, scan.vectors - expected, position, heading, reference=np.zeros(3))
        physics = self._normalize(physics, self.physics_mean, self.physics_std).reshape(-1)
        measurement = board.measurements[-1]
        center_raw = np.array([measurement.bx, measurement.by, measurement.bz], dtype=np.float32) - self.estimator.predict_at(position.reshape(1, 2))[0]
        center = self._normalize(center_raw, self.center_mean, self.center_std)
        self.estimator.update(scan)
        if self.previous_scan is None:
            self.previous_scan = current
            self.previous_position = position
            self.previous_heading = heading
            self.result = HumanDetectorResult(0.0, 0, "WARMING UP", float(board.time), None, 0, 0)
            return self.result
        motion_raw = local_motion(self.previous_position, position, self.previous_heading, heading)
        motion = self._normalize(motion_raw, self.motion_mean, self.motion_std)
        with torch.inference_mode():
            predicted = self.predictor(torch.from_numpy(self.previous_scan[None]).to(self.device), torch.from_numpy(motion[None]).to(self.device)).cpu().numpy()[0]
        temporal_change = current - self.previous_scan
        transition = (temporal_change - predicted).reshape(32, 8)
        current_grid = current.reshape(32, 8)
        delta_grid = temporal_change.reshape(32, 8)
        physics_grid = physics.reshape(32, 8)
        movement_grid = np.repeat(motion[None, :], 32, axis=0)
        center_grid = np.repeat(center[None, :], 32, axis=0)
        feature = np.concatenate([transition, current_grid, delta_grid, movement_grid, physics_grid, center_grid], axis=1)
        self.features.append(feature.astype(np.float32))
        self.previous_scan = current
        self.previous_position = position
        self.previous_heading = heading
        if len(self.features) < self.window:
            self.result = HumanDetectorResult(0.0, 0, "WARMING UP", float(board.time), None, 0, len(self.features))
            return self.result
        sequence = np.stack(self.features, axis=0).transpose(1, 2, 0)
        with torch.inference_mode():
            presence, count, location = self.classifier(torch.from_numpy(sequence[None]).to(self.device))
            probability = float(torch.softmax(presence, dim=1)[0, 1].cpu())
            predicted_label = int(probability >= self.presence_threshold)
            estimated_count = int(torch.argmax(count, dim=1)[0].cpu())
            location_probability = torch.sigmoid(location[0]).cpu().numpy()
        best_index = int(np.argmax(location_probability))
        estimated_position = None
        self.probability_history.append(probability)
        sustained_signal = len(self.probability_history) >= 3 and float(np.mean(self.probability_history)) >= 0.50 and probability >= 0.45
        if probability >= self.marker_probability_threshold or sustained_signal:
            self.positive_streak += 1
        else:
            self.positive_streak = 0
        should_mark = self.positive_streak >= self.marker_persistence
        if should_mark and float(location_probability[best_index]) >= self.location_threshold:
            estimated_position = self._location_from_scan(best_index, scan.positions, scan.vectors - expected, position, heading)
            if any(float(np.linalg.norm(estimated_position - saved)) <= 8.0 for saved in self.ignored_positions):
                estimated_position = None
        if should_mark and estimated_position is not None:
            position_changed = self.last_marker_position is None or np.linalg.norm(estimated_position - self.last_marker_position) >= self.marker_min_distance
            time_elapsed = float(board.time) - self.last_marker_time >= self.marker_cooldown
            if position_changed and time_elapsed:
                if self.add_markers:
                    board.detection_tracker.add(estimated_position, probability, float(board.time))
                self.last_marker_position = estimated_position.copy()
                self.last_marker_time = float(board.time)
        self.result = HumanDetectorResult(probability, predicted_label, "READY", float(board.time), estimated_position, estimated_count, len(self.features), location_probability)
        return self.result
