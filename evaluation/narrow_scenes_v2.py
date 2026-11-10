import numpy as np

from simulation.board_v2 import Board
from simulation.entities_v2 import Obstacle
from simulation.perception_v2 import RobotPerception


def create_narrow_scene(seed: int, width: float, kind: str = 'straight') -> Board:
    if kind not in ('straight', 'corner', 'bend', 'room', 'choice') or width < 2.0 or width > 6.0:
        raise ValueError('Unsupported passage configuration')
    board = Board(seed)
    board.people = []
    if kind == 'bend':
        board.obstacles = [Obstacle(0, 0, 120, 40), Obstacle(0, 40 + width, 60 - width, 40 - width), Obstacle(60, 40, 60, 40)]
        board.robot.position = np.array([50.0, 40 + width / 2])
        known_columns = int((60 - width) / 2)
    else:
        center = 40.37
        lower, upper = center - width / 2, center + width / 2
        board.obstacles = [Obstacle(58, 0, 4, lower), Obstacle(58, upper, 4, 80 - upper)]
        if kind == 'corner':
            board.obstacles.append(Obstacle(62, upper, 10, 3))
        if kind == 'choice':
            board.obstacles = [Obstacle(58, 0, 4, 12), Obstacle(58, 24, 4, lower - 24), Obstacle(58, upper, 4, 80 - upper)]
        board.robot.position = np.array([40.0 if kind in ('room', 'choice') else 54.0, center])
        known_columns = 29
    board.robot.heading = float(np.random.default_rng(seed).uniform(-np.pi, np.pi))
    board.robot.path = [board.robot.position.copy()]
    board.perception = RobotPerception()
    board.range_scan = board.range_sensor.scan(board.robot.position, board.obstacles)
    board.magnetic_scan = board.sensor.scan(board)
    board.measurements = [board.sensor.measure(board)]
    board.perception.update(board.robot.position, board.magnetic_scan, board.range_scan)
    board.perception.observed[:, :known_columns] = True
    return board
