from __future__ import annotations

from typing import Protocol

import numpy as np

from config_v2 import PERSON_FAST_MODULATION, PERSON_FIELD_DISTANCE_SCALE, PERSON_SLOW_MODULATION, SENSOR_RANGE


class MagneticFieldModel(Protocol):
    def field_at(self, position: np.ndarray, board) -> np.ndarray: ...

    def scan_at(self, points: np.ndarray, robot_position: np.ndarray, board) -> np.ndarray: ...


class LegacyMagneticFieldModel:
    def field_at(self, position: np.ndarray, board) -> np.ndarray:
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
                direction = np.array([np.cos(person.heading), np.sin(person.heading), 0.3 * np.sin(person.phase[2])])
                attenuation = person.signal_strength * temporal / (1.0 + (distance / PERSON_FIELD_DISTANCE_SCALE) ** 2)
                total += direction * attenuation
        return total

    def scan_at(self, points: np.ndarray, robot_position: np.ndarray, board) -> np.ndarray:
        vectors = np.tile(np.array([24.0, -7.0, 39.0]), (len(points), 1))
        for metal in board.metals:
            if np.linalg.norm(metal.position - robot_position) > SENSOR_RANGE:
                continue
            distances = np.linalg.norm(points - metal.position, axis=1)
            active = (distances > 0.01) & (distances <= SENSOR_RANGE)
            direction = np.array([np.cos(metal.orientation), np.sin(metal.orientation), 0.55])
            attenuation = metal.strength / (1.0 + (distances[active] / (metal.size + 1.0)) ** 3)
            vectors[active] += attenuation[:, None] * direction
        for person in board.people:
            if np.linalg.norm(person.position - robot_position) > SENSOR_RANGE:
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
        return vectors


class EnhancedMagneticFieldModel(LegacyMagneticFieldModel):
    def _dynamic_signal(self, points: np.ndarray, robot_position: np.ndarray, board) -> np.ndarray:
        signal = np.zeros((len(points), 3), dtype=float)
        for person in board.people:
            relative = points - person.position
            distances = np.linalg.norm(relative, axis=1)
            active = (distances > 0.01) & (distances <= SENSOR_RANGE)
            if not np.any(active):
                continue
            radial = relative / np.maximum(distances[:, None], 1e-6)
            velocity = np.asarray(person.velocity, dtype=float)
            speed = float(np.linalg.norm(velocity))
            heading_vector = np.array([np.cos(person.heading), np.sin(person.heading), 0.0])
            motion_projection = np.clip(radial @ velocity, -1.0, 1.0)
            breathing = 0.12 * np.sin(2.0 * np.pi * 0.23 * board.time + person.phase[0])
            micro = 0.035 * np.sin(2.0 * np.pi * 1.85 * board.time + person.phase[1])
            motion = 0.06 * motion_projection + 0.025 * np.tanh(speed)
            envelope = person.signal_strength / (1.0 + (distances / PERSON_FIELD_DISTANCE_SCALE) ** 2)
            direction = np.column_stack([0.75 * heading_vector[0] + 0.25 * radial[:, 0], 0.75 * heading_vector[1] + 0.25 * radial[:, 1], np.zeros(len(points))])
            amplitude = (breathing + micro + motion[active])[:, None] * envelope[active][:, None]
            signal[active] += amplitude * direction[active]
        return signal

    def field_at(self, position: np.ndarray, board) -> np.ndarray:
        base = super().field_at(position, board)
        return base + self._dynamic_signal(np.asarray(position, dtype=float).reshape(1, 2), np.asarray(position, dtype=float), board)[0]

    def scan_at(self, points: np.ndarray, robot_position: np.ndarray, board) -> np.ndarray:
        vectors = super().scan_at(points, robot_position, board)
        return vectors + self._dynamic_signal(points, robot_position, board)
