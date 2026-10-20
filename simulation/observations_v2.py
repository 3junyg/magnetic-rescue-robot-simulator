from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from simulation.entities_v2 import MagneticMeasurement, MagneticScan, RangeScan


@dataclass(frozen=True)
class RobotObservation:
    time: float
    robot_position: np.ndarray
    robot_heading: float
    robot_moving: bool
    measurement: MagneticMeasurement
    magnetic_scan: MagneticScan
    range_scan: RangeScan


def observation_from_board(board) -> RobotObservation:
    return RobotObservation(
        time=float(board.time),
        robot_position=np.asarray(board.robot.position, dtype=np.float32).copy(),
        robot_heading=float(board.robot.heading),
        robot_moving=bool(board.robot.moving),
        measurement=board.measurements[-1],
        magnetic_scan=board.magnetic_scan,
        range_scan=board.range_scan,
    )
