from collections import deque

import numpy as np

from config import PERCEPTION_CELL_SIZE


class FrontierExpert:
    def decide(self, environment) -> int:
        if environment.last_collision:
            return self._clearance_action(environment)
        coverage_map = environment.coverage_map
        robot = environment.board.robot
        rows = coverage_map.rows
        columns = coverage_map.columns
        start_column = min(columns - 1, max(0, int(robot.position[0] / PERCEPTION_CELL_SIZE)))
        start_row = min(rows - 1, max(0, int(robot.position[1] / PERCEPTION_CELL_SIZE)))
        traversable = ~coverage_map.obstacles
        targets = ~coverage_map.observed
        targets[start_row, start_column] = False
        queue = deque([(start_row, start_column)])
        parent = {(start_row, start_column): None}
        target = None
        while queue:
            cell = queue.popleft()
            if targets[cell]:
                target = cell
                break
            row, column = cell
            for neighbor in ((row - 1, column), (row + 1, column), (row, column - 1), (row, column + 1)):
                next_row, next_column = neighbor
                if 0 <= next_row < rows and 0 <= next_column < columns and traversable[next_row, next_column] and neighbor not in parent:
                    parent[neighbor] = cell
                    queue.append(neighbor)
        if target is None:
            target_rows, target_columns = np.where(targets)
            if len(target_rows) == 0:
                return self._clearance_action(environment)
            positions = np.column_stack(
                [
                    (target_columns + 0.5) * PERCEPTION_CELL_SIZE,
                    (target_rows + 0.5) * PERCEPTION_CELL_SIZE,
                ]
            )
            nearest = int(np.argmin(np.linalg.norm(positions - robot.position, axis=1)))
            next_cell = int(target_rows[nearest]), int(target_columns[nearest])
        else:
            next_cell = target
            while parent[next_cell] is not None and parent[next_cell] != (start_row, start_column):
                next_cell = parent[next_cell]
        target_position = np.array([(next_cell[1] + 0.5) * PERCEPTION_CELL_SIZE, (next_cell[0] + 0.5) * PERCEPTION_CELL_SIZE])
        desired = np.arctan2(target_position[1] - robot.position[1], target_position[0] - robot.position[0])
        difference = (desired - robot.heading + np.pi) % (2.0 * np.pi) - np.pi
        if abs(difference) > np.pi / 12.0:
            return 1 if difference > 0.0 else 2
        forward_index = int(np.argmin(np.abs((environment.board.range_scan.angles - robot.heading + np.pi) % (2.0 * np.pi) - np.pi)))
        if environment.board.range_scan.distances[forward_index] < environment.forward_distance + robot.radius + 0.5:
            return self._clearance_action(environment)
        return 0

    def _clearance_action(self, environment) -> int:
        angles = environment.board.range_scan.angles
        distances = environment.board.range_scan.distances
        heading = environment.board.robot.heading
        relative = (angles - heading + np.pi) % (2.0 * np.pi) - np.pi
        left = distances[(relative > 0.15) & (relative < 1.5)].mean()
        right = distances[(relative < -0.15) & (relative > -1.5)].mean()
        return 1 if left >= right else 2


class OracleCoverageExpert:
    def decide(self, environment) -> int:
        robot = environment.board.robot
        rows, columns = environment.accessible.shape
        start_column = min(columns - 1, max(0, int(robot.position[0] / PERCEPTION_CELL_SIZE)))
        start_row = min(rows - 1, max(0, int(robot.position[1] / PERCEPTION_CELL_SIZE)))
        targets = environment.accessible & ~environment.covered
        queue = deque([(start_row, start_column)])
        parent = {(start_row, start_column): None}
        target = None
        while queue:
            cell = queue.popleft()
            if targets[cell]:
                target = cell
                break
            row, column = cell
            for neighbor in ((row - 1, column), (row + 1, column), (row, column - 1), (row, column + 1)):
                next_row, next_column = neighbor
                if 0 <= next_row < rows and 0 <= next_column < columns and environment.accessible[next_row, next_column] and neighbor not in parent:
                    parent[neighbor] = cell
                    queue.append(neighbor)
        if target is None:
            return 3
        next_cell = target
        while parent[next_cell] is not None and parent[next_cell] != (start_row, start_column):
            next_cell = parent[next_cell]
        target_position = np.array([(next_cell[1] + 0.5) * PERCEPTION_CELL_SIZE, (next_cell[0] + 0.5) * PERCEPTION_CELL_SIZE])
        desired = np.arctan2(target_position[1] - robot.position[1], target_position[0] - robot.position[0])
        difference = (desired - robot.heading + np.pi) % (2.0 * np.pi) - np.pi
        if abs(difference) > np.pi / 12.0:
            return 1 if difference > 0.0 else 2
        direction = np.array([np.cos(robot.heading), np.sin(robot.heading)])
        candidate = robot.position + environment.forward_distance * direction
        if not environment._path_free(robot.position, candidate):
            return 1 if difference >= 0.0 else 2
        return 0
