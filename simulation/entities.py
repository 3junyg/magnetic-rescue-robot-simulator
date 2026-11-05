from dataclasses import dataclass, field

import numpy as np


@dataclass
class Person:
    position: np.ndarray
    velocity: np.ndarray
    heading: float
    signal_strength: float
    phase: np.ndarray
    variation: float = 0.0


@dataclass
class Metal:
    position: np.ndarray
    size: float
    strength: float
    orientation: float


@dataclass
class Obstacle:
    x: float
    y: float
    width: float
    height: float

    def contains(self, point: np.ndarray, margin: float = 0.0) -> bool:
        return (
            self.x - margin <= point[0] <= self.x + self.width + margin
            and self.y - margin <= point[1] <= self.y + self.height + margin
        )


@dataclass
class Robot:
    position: np.ndarray
    heading: float
    radius: float
    speed: float
    turn_remaining: float = 0.0
    moving: bool = True
    path: list[np.ndarray] = field(default_factory=list)

#tlqkf#
@dataclass
class MagneticMeasurement:
    timestamp: float
    bx: float
    by: float
    bz: float
    magnitude: float
    robot_moving: bool


@dataclass
class MagneticScan:
    positions: np.ndarray
    vectors: np.ndarray
    magnitudes: np.ndarray


@dataclass
class RangeScan:
    angles: np.ndarray
    distances: np.ndarray
    endpoints: np.ndarray
    hits: np.ndarray
