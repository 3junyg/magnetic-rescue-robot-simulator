from dataclasses import dataclass

import numpy as np


@dataclass
class RescueEvent:
    person_index: int
    position: np.ndarray
    distance: float


class RescueManager:
    def __init__(self, match_distance: float = 5.0, ignore_radius: float = 8.0, pin_activation_radius: float = 4.5) -> None:
        self.match_distance = match_distance
        self.ignore_radius = ignore_radius
        self.pin_activation_radius = pin_activation_radius
        self.rescued_indices: set[int] = set()
        self.ignored_positions: list[np.ndarray] = []
        self.pending_pins: list[np.ndarray] = []

    def reset(self) -> None:
        self.rescued_indices.clear()
        self.ignored_positions.clear()
        self.pending_pins.clear()

    def is_ignored(self, position: np.ndarray) -> bool:
        return any(float(np.linalg.norm(position - saved)) <= self.ignore_radius for saved in self.ignored_positions)

    def all_rescued(self, board) -> bool:
        return bool(board.people) and len(self.rescued_indices) >= len(board.people)

    def _remember_pin(self, board, position: np.ndarray) -> None:
        position = np.asarray(position, dtype=float).copy()
        if any(float(np.linalg.norm(position - saved)) <= self.ignore_radius for saved in self.pending_pins):
            return
        self.pending_pins.append(position)

    def _activate_pin(self, board) -> RescueEvent | None:
        for index, pin in enumerate(self.pending_pins):
            candidates = [
                (person_index, float(np.linalg.norm(person.position - pin)))
                for person_index, person in enumerate(board.people)
                if person_index not in self.rescued_indices and np.linalg.norm(person.position - pin) <= self.pin_activation_radius
            ]
            if not candidates:
                continue
            person_index, distance = min(candidates, key=lambda item: item[1])
            person = board.people[person_index]
            person.velocity = np.zeros(2, dtype=float)
            person.signal_strength = 0.0
            self.rescued_indices.add(person_index)
            saved_position = np.asarray(person.position, dtype=float).copy()
            self.ignored_positions.append(saved_position)
            self.pending_pins.pop(index)
            return RescueEvent(person_index, saved_position, distance)
        return None

    def process(self, board, detector_result) -> RescueEvent | None:
        estimate = detector_result.estimated_position
        if estimate is not None and detector_result.person_probability >= 0.6 and not self.is_ignored(estimate):
            candidates = [
                (index, float(np.linalg.norm(person.position - estimate)))
                for index, person in enumerate(board.people)
                if index not in self.rescued_indices
            ]
            if candidates:
                person_index, distance = min(candidates, key=lambda item: item[1])
                if distance <= self.match_distance:
                    person = board.people[person_index]
                    person.velocity = np.zeros(2, dtype=float)
                    person.signal_strength = 0.0
                    self.rescued_indices.add(person_index)
                    saved_position = np.asarray(person.position, dtype=float).copy()
                    self.ignored_positions.append(saved_position)
                    return RescueEvent(person_index, saved_position, distance)
            self._remember_pin(board, estimate)
        return self._activate_pin(board)
