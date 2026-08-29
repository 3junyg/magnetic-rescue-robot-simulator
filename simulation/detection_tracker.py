from dataclasses import dataclass

import numpy as np


@dataclass
class HumanDetectionMarker:
    position: np.ndarray
    confidence: float
    timestamp: float
    observations: int = 1


class DetectionTracker:
    def __init__(self, merge_distance: float = 5.0) -> None:
        self.merge_distance = merge_distance
        self.markers: list[HumanDetectionMarker] = []

    def add(self, position: np.ndarray, confidence: float, timestamp: float) -> None:
        if self.markers:
            distances = [np.linalg.norm(marker.position - position) for marker in self.markers]
            index = int(np.argmin(distances))
            if distances[index] <= self.merge_distance:
                marker = self.markers[index]
                weight = marker.observations
                marker.position = (marker.position * weight + position) / (weight + 1)
                marker.confidence = max(marker.confidence, confidence)
                marker.timestamp = timestamp
                marker.observations += 1
                return
        self.markers.append(HumanDetectionMarker(position.copy(), confidence, timestamp))

    def clear(self) -> None:
        self.markers.clear()
