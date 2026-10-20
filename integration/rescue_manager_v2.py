from dataclasses import dataclass

import numpy as np


@dataclass
class RescueEvent:
    person_index: int
    position: np.ndarray
    distance: float


class RescueManager:
    def __init__(self, match_distance: float = 6.5, ignore_radius: float = 8.0, pin_activation_radius: float = 6.5) -> None:
        self.match_distance = match_distance
        self.ignore_radius = ignore_radius
        self.pin_activation_radius = pin_activation_radius
        self.rescued_indices: set[int] = set()
        self.ignored_positions: list[np.ndarray] = []
        self.active_pins: list[np.ndarray] = []
        self.last_events: list[RescueEvent] = []

    def reset(self) -> None:
        self.rescued_indices.clear()
        self.ignored_positions.clear()
        self.active_pins.clear()
        self.last_events.clear()

    def is_ignored(self, position: np.ndarray) -> bool:
        return any(float(np.linalg.norm(position - saved)) <= self.ignore_radius for saved in self.ignored_positions)

    def all_rescued(self, board) -> bool:
        return bool(board.people) and len(self.rescued_indices) >= len(board.people)

    def _remember_pin(self, position: np.ndarray) -> None:
        position = np.asarray(position, dtype=float).copy()
        if any(float(np.linalg.norm(position - saved)) <= self.ignore_radius for saved in self.active_pins):
            return
        self.active_pins.append(position)

    def _activate_pins(self, board) -> list[RescueEvent]:
        events = []
        for pin in self.active_pins:
            for person_index, person in enumerate(board.people):
                if person_index in self.rescued_indices:
                    continue
                distance = float(np.linalg.norm(person.position - pin))
                if distance > self.pin_activation_radius:
                    continue
                person.velocity = np.zeros(2, dtype=float)
                person.signal_strength = 0.0
                self.rescued_indices.add(person_index)
                saved_position = np.asarray(person.position, dtype=float).copy()
                if not any(np.linalg.norm(saved_position - saved) <= 1.0 for saved in self.ignored_positions):
                    self.ignored_positions.append(saved_position)
                events.append(RescueEvent(person_index, saved_position, distance))
        return events

    def process(self, board, detector_result) -> RescueEvent | None:
        estimate = detector_result.estimated_position
        if estimate is not None and not self.is_ignored(estimate):
            self._remember_pin(estimate)
        self.last_events = self._activate_pins(board)
        return self.last_events[0] if self.last_events else None
