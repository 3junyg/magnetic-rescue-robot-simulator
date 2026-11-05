from __future__ import annotations

from pathlib import Path

import numpy as np

from config_v2_1 import PERSON_FAST_MODULATION, PERSON_SLOW_MODULATION, SENSOR_RANGE
from simulation.coil_circuit_v2_1 import MU0, coil_field
from simulation.mi_calibration_v2_1 import MICalibration
from simulation.relay_settings_v2_1 import relay_summary
from simulation.field_model_v2_1 import EnhancedMagneticFieldModel
from simulation.signal_scaling_v2_1 import signal_scaling


class ActiveMIFieldModel:
    def __init__(self, profile_path: str | Path | None = None, profile_overrides: dict | None = None) -> None:
        self.calibration = MICalibration(profile_path, profile_overrides)
        self.profile = self.calibration.profile
        mode = self.profile.get("signal_mode", "v2")
        if mode not in ("v2", "scaled_v2", "calibrated_mi"):
            raise ValueError("signal_mode must be v2, scaled_v2 or calibrated_mi")
        self.uses_physical_mi = mode == "calibrated_mi"
        self.scaling = signal_scaling(self.calibration)
        self.sensor_range = self.scaling["sensor_range"] if mode == "scaled_v2" else SENSOR_RANGE
        self.strength_factor = self.scaling["strength_factor"] if mode == "scaled_v2" else 1.0
        self.v2_model = EnhancedMagneticFieldModel(self.sensor_range)
        self.relay_summary = relay_summary(self.profile)
        self.relay_factor = complex(self.relay_summary["ratio_real"], self.relay_summary["ratio_imag"])
        self.background = np.array([24.0, -7.0, 39.0])
        self._source_key = None
        self._sources = None

    def _transmitter(self, board) -> tuple[np.ndarray, np.ndarray]:
        heading = board.robot.heading
        normal = np.array([np.cos(heading), np.sin(heading), 0.0])
        center = np.r_[board.robot.position * self.profile["world_unit_m"], 0.0] - self.profile["tx_rx_offset_m"] * normal
        return center, normal

    def _primary(self, points_m: np.ndarray, board) -> np.ndarray:
        p = self.profile
        center, normal = self._transmitter(board)
        field = coil_field(points_m, center, normal, p["tx_radius_m"], int(p["tx_turns"]), p["tx_current_rms_a"], int(p["coil_segments"]), p["wire_core_m"])
        return field * self.calibration.factors(np.linalg.norm(points_m - center, axis=1))[:, None]

    def _target_sources(self, board) -> tuple[np.ndarray, np.ndarray]:
        key = (board.time, tuple(board.robot.position), board.robot.heading, tuple((tuple(item.position), item.signal_strength, item.heading, item.variation) for item in board.people), tuple((tuple(item.position), item.strength, item.orientation) for item in board.metals))
        if key == self._source_key:
            return self._sources
        p = self.profile
        positions, moments = [], []
        omega = 2.0 * np.pi * p["frequency_hz"]
        for kind, objects in (("metal", board.metals), ("human", board.people)):
            for item in objects:
                if np.linalg.norm(item.position - board.robot.position) > SENSOR_RANGE:
                    continue
                center = np.r_[item.position * p["world_unit_m"], 0.0]
                primary_h = self._primary(center[None, :], board)[0] / MU0
                if kind == "human":
                    temporal = max(0.05, 1.0 + 0.25 * PERSON_SLOW_MODULATION * np.sin(2.0 * np.pi * 0.23 * board.time + item.phase[0]) + 0.10 * PERSON_FAST_MODULATION * np.sin(2.0 * np.pi * 1.85 * board.time + item.phase[1]) + 0.10 * item.variation)
                    strength = item.signal_strength / 14.0 * temporal
                    orientation = item.heading
                else:
                    strength = item.strength / 28.5 * (item.size / 2.6)**3
                    orientation = item.orientation
                direction = np.array([np.cos(orientation), np.sin(orientation), 0.0])
                response = 0.55 * primary_h + 0.45 * direction * np.dot(primary_h, direction)
                alpha = p[f"{kind}_polarizability_m3"] * strength / (1.0 + 1j * omega * p[f"{kind}_response_time_s"])
                positions.append(center)
                moments.append(alpha * response)
        self._source_key = key
        self._sources = (np.asarray(positions).reshape(-1, 3), np.asarray(moments, dtype=complex).reshape(-1, 3))
        return self._sources

    def receive(self, points: np.ndarray, board) -> tuple[np.ndarray, np.ndarray]:
        p = self.profile
        _, normal = self._transmitter(board)
        receivers = np.column_stack([np.asarray(points) * p["world_unit_m"], np.zeros(len(points))]) + p["tx_rx_offset_m"] * normal
        primary = self._primary(receivers, board).astype(complex)
        total = primary.copy()
        sources, moments = self._target_sources(board)
        if len(sources):
            relative = receivers[:, None, :] - sources[None, :, :]
            squared = np.sum(relative**2, axis=2) + p["target_core_m"]**2
            projection = np.sum(relative * moments[None, :, :], axis=2)
            secondary = MU0 / (4.0 * np.pi) * (3.0 * relative * projection[:, :, None] / squared[:, :, None] - moments[None, :, :]) / squared[:, :, None]**1.5
            secondary *= self.calibration.factors(np.linalg.norm(relative, axis=2))[:, :, None]
            total += np.sum(secondary, axis=1)
        conversion = -1j * 2.0 * np.pi * p["frequency_hz"] * p["rx_turns"] * np.pi * p["rx_radius_m"]**2
        return self.relay_factor * conversion * total, self.relay_factor * conversion * primary

    def encode(self, compensated_input_v: np.ndarray) -> np.ndarray:
        phase = np.exp(-1j * np.deg2rad(self.profile["demodulation_phase_deg"]))
        return self.background + np.real(compensated_input_v * phase) / self.profile["detector_input_v_per_unit"]

    def field_at(self, position: np.ndarray, board) -> np.ndarray:
        if not self.uses_physical_mi:
            field = self.v2_model.field_at(position, board)
            return field if self.strength_factor == 1.0 else self.background + (field - self.background) * self.strength_factor
        total, primary = self.receive(np.asarray(position).reshape(1, 2), board)
        return self.encode(total - self.profile["primary_cancellation_fraction"] * primary)[0]

    def scan_at(self, points: np.ndarray, robot_position: np.ndarray, board) -> np.ndarray:
        if not self.uses_physical_mi:
            field = self.v2_model.scan_at(points, robot_position, board)
            return field if self.strength_factor == 1.0 else self.background + (field - self.background) * self.strength_factor
        total, primary = self.receive(points, board)
        return self.encode(total - self.profile["primary_cancellation_fraction"] * primary)
