from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class DetectionProposal:
    position: np.ndarray
    location_probability: float
    dynamic_score: float
    cell_index: int


class DetectionCandidateSelector:
    def __init__(
        self,
        max_candidates: int = 2,
        min_probability: float = 0.39,
        secondary_probability: float = 0.40,
        secondary_relative_probability: float = 0.70,
        min_separation: float = 6.0,
    ) -> None:
        self.max_candidates = max_candidates
        self.min_probability = min_probability
        self.secondary_probability = secondary_probability
        self.secondary_relative_probability = secondary_relative_probability
        self.min_separation = min_separation

    @staticmethod
    def _is_local_peak(index: int, probabilities: np.ndarray) -> bool:
        radial, angular = divmod(index, 8)
        value = probabilities[index]
        for radial_offset in (-1, 0, 1):
            neighbor_radial = radial + radial_offset
            if not 0 <= neighbor_radial < 4:
                continue
            for angular_offset in (-1, 0, 1):
                if radial_offset == 0 and angular_offset == 0:
                    continue
                neighbor = neighbor_radial * 8 + (angular + angular_offset) % 8
                if probabilities[neighbor] > value:
                    return False
        return True

    def select(
        self,
        probabilities: np.ndarray,
        transition: np.ndarray,
        locate: Callable[[int], np.ndarray],
    ) -> list[DetectionProposal]:
        values = np.asarray(probabilities, dtype=float).reshape(32)
        proposals: list[DetectionProposal] = []
        best_probability = float(np.max(values))
        for index in np.argsort(values)[::-1]:
            index = int(index)
            value = float(values[index])
            minimum = self.min_probability if not proposals else max(
                self.secondary_probability,
                self.secondary_relative_probability * best_probability,
            )
            if value < minimum:
                break
            if not self._is_local_peak(index, values):
                continue
            position = np.asarray(locate(index), dtype=float)
            if any(float(np.linalg.norm(position - proposal.position)) < self.min_separation for proposal in proposals):
                continue
            proposals.append(
                DetectionProposal(position, value, float(np.linalg.norm(transition[index])), index)
            )
            if len(proposals) >= self.max_candidates:
                break
        return proposals
