import numpy as np

from config_v2 import BOARD_HEIGHT, BOARD_WIDTH


def configure_cluster_scene(board, people_mode: str = 'none') -> None:
    rng = np.random.default_rng(board.seed + 810001)
    centers = []
    for cluster in range(3):
        for _ in range(500):
            angle = board.robot.heading + rng.uniform(-1.4, 1.4) + cluster * 1.8
            radius = rng.uniform(9.0, 18.0)
            point = board.robot.position + radius * np.array([np.cos(angle), np.sin(angle)])
            point = np.clip(point, [4.0, 4.0], [BOARD_WIDTH - 4.0, BOARD_HEIGHT - 4.0])
            if board._is_free(point, 0.8):
                centers.append(point)
                break
    if not centers:
        centers.append(board._free_point(1.0))
    for index, metal in enumerate(board.metals):
        center = centers[index % len(centers)]
        for _ in range(100):
            point = center + rng.normal(0.0, 1.8, 2)
            if board._is_free(point, 0.3):
                metal.position = point
                metal.size = rng.uniform(1.5, 4.5)
                metal.strength = rng.uniform(24.0, 52.0)
                metal.orientation = rng.uniform(-np.pi, np.pi)
                break
    if people_mode == 'none':
        board.people = []
    elif people_mode == 'mixed':
        for index, person in enumerate(board.people[:3]):
            center = centers[index % len(centers)]
            for _ in range(100):
                point = center + rng.normal(0.0, 3.0, 2)
                if board._is_free(point, 0.65):
                    person.position = point
                    break
    board.measurements[-1] = board.sensor.measure(board)
    board.magnetic_scan = board.sensor.scan(board)
    board.perception.update(board.robot.position, board.magnetic_scan, board.range_scan)
