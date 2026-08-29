from pathlib import Path

import numpy as np

from config import BOARD_HEIGHT, BOARD_WIDTH, HISTORY_LENGTH


class ControlledBoardStepper:
    def __init__(self, forward_distance: float = 2.0, turn_angle: float = np.pi / 6.0) -> None:
        self.forward_distance = forward_distance
        self.turn_angle = turn_angle
        self.consecutive_turns = 0

    def reset(self) -> None:
        self.consecutive_turns = 0

    @staticmethod
    def _path_free(board, start: np.ndarray, end: np.ndarray, radius: float) -> bool:
        distance = float(np.linalg.norm(end - start))
        for fraction in np.linspace(0.0, 1.0, max(2, int(np.ceil(distance / 0.25)) + 1))[1:]:
            if not board._is_free(start + fraction * (end - start), radius):
                return False
        return True

    def step(self, board, action: str, dt: float, update_perception: bool = True) -> None:
        board.time += dt
        board._update_people(dt)
        board.robot.moving = False
        if action == "turn_left":
            board.robot.heading += self.turn_angle
            self.consecutive_turns += 1
        elif action == "turn_right":
            board.robot.heading -= self.turn_angle
            self.consecutive_turns += 1
        elif action == "forward":
            self.consecutive_turns = 0
        else:
            self.consecutive_turns = 0
        move_forward = action == "forward" or (action in ("turn_left", "turn_right") and self.consecutive_turns >= 3)
        if move_forward:
            direction = np.array([np.cos(board.robot.heading), np.sin(board.robot.heading)])
            candidate = board.robot.position + direction * self.forward_distance
            if self._path_free(board, board.robot.position, candidate, board.robot.radius):
                board.robot.position = candidate
                board.robot.moving = True
                self.consecutive_turns = 0
        board.robot.heading = (board.robot.heading + np.pi) % (2.0 * np.pi) - np.pi
        board.robot.path.append(board.robot.position.copy())
        if len(board.robot.path) > HISTORY_LENGTH:
            board.robot.path.pop(0)
        measurement = board.sensor.measure(board)
        board.measurements.append(measurement)
        board.magnetic_scan = board.sensor.scan(board)
        if update_perception:
            board.range_scan = board.range_sensor.scan(board.robot.position, board.obstacles)
            board.perception.update(board.robot.position, board.magnetic_scan, board.range_scan)
        if len(board.measurements) > HISTORY_LENGTH:
            board.measurements.pop(0)
