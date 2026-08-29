from collections import deque

import numpy as np

from config import BOARD_HEIGHT, BOARD_WIDTH, ENVIRONMENT_PRESETS, HISTORY_LENGTH, PERCEPTION_CELL_SIZE, SENSOR_RANGE
from navigation.coverage_map import CoverageMap
from simulation.board import Board
from simulation.entities import RangeScan


ACTION_NAMES = ("forward", "turn_left", "turn_right", "stop")


class CoverageEnvironment:
    def __init__(self, seed: int = 0, environment: str | None = None, max_steps: int = 1000) -> None:
        self.seed = seed
        self.environment = environment or next(iter(ENVIRONMENT_PRESETS))
        self.max_steps = max_steps
        self.forward_distance = 2.0
        self.turn_angle = np.pi / 6.0
        self.step_count = 0
        self.collision_count = 0
        self.previous_action = 3
        self.consecutive_turns = 0
        self.last_collision = False
        self.board: Board
        self.coverage_map: CoverageMap
        self.accessible: np.ndarray
        self.covered: np.ndarray
        self.reset(seed, self.environment)

    @property
    def map_shape(self) -> tuple[int, int, int]:
        return 4, 20, 30

    @property
    def vector_size(self) -> int:
        return 57

    @property
    def coverage_ratio(self) -> float:
        accessible_regions, covered_regions = self._region_coverage()
        return float(covered_regions.sum() / max(1, accessible_regions.sum()))

    def _region_coverage(self) -> tuple[np.ndarray, np.ndarray]:
        accessible_regions = self.accessible.reshape(20, 2, 30, 2).any(axis=(1, 3))
        covered_regions = self.covered.reshape(20, 2, 30, 2).any(axis=(1, 3)) & accessible_regions
        return accessible_regions, covered_regions

    def reset(self, seed: int | None = None, environment: str | None = None) -> tuple[np.ndarray, np.ndarray]:
        if seed is not None:
            self.seed = seed
        if environment is not None:
            self.environment = environment
        self.board = Board(self.seed, self.environment)
        start_column = min(int(BOARD_WIDTH / PERCEPTION_CELL_SIZE) - 1, max(0, int(self.board.robot.position[0] / PERCEPTION_CELL_SIZE)))
        start_row = min(int(BOARD_HEIGHT / PERCEPTION_CELL_SIZE) - 1, max(0, int(self.board.robot.position[1] / PERCEPTION_CELL_SIZE)))
        center = np.array([(start_column + 0.5) * PERCEPTION_CELL_SIZE, (start_row + 0.5) * PERCEPTION_CELL_SIZE])
        if self.board._is_free(center, self.board.robot.radius):
            self.board.robot.position = center
        self.board.robot.heading = round(self.board.robot.heading / self.turn_angle) * self.turn_angle
        self.board.robot.moving = False
        self.board.robot.path = [self.board.robot.position.copy()]
        self.coverage_map = CoverageMap()
        self.step_count = 0
        self.collision_count = 0
        self.previous_action = 3
        self.consecutive_turns = 0
        self.last_collision = False
        self.board.range_scan = self._range_scan()
        self.coverage_map.update(self.board.robot.position, self.board.range_scan)
        self.accessible = self._reachable_mask()
        self.covered = self.coverage_map.observed & self.accessible
        return self.observation()

    def _range_scan(self) -> RangeScan:
        position = self.board.robot.position
        angles = np.linspace(-np.pi, np.pi, 48, endpoint=False)
        directions = np.column_stack([np.cos(angles), np.sin(angles)])
        distances = np.full(len(angles), SENSOR_RANGE, dtype=float)
        for index, direction in enumerate(directions):
            dx, dy = direction
            boundary_x = (BOARD_WIDTH - position[0]) / dx if dx > 1e-9 else (-position[0] / dx if dx < -1e-9 else np.inf)
            boundary_y = (BOARD_HEIGHT - position[1]) / dy if dy > 1e-9 else (-position[1] / dy if dy < -1e-9 else np.inf)
            distance = min(SENSOR_RANGE, boundary_x, boundary_y)
            for obstacle in self.board.obstacles:
                if abs(dx) <= 1e-9 and not (obstacle.x <= position[0] <= obstacle.x + obstacle.width):
                    continue
                if abs(dy) <= 1e-9 and not (obstacle.y <= position[1] <= obstacle.y + obstacle.height):
                    continue
                tx1 = (obstacle.x - position[0]) / dx if abs(dx) > 1e-9 else -np.inf
                tx2 = (obstacle.x + obstacle.width - position[0]) / dx if abs(dx) > 1e-9 else np.inf
                ty1 = (obstacle.y - position[1]) / dy if abs(dy) > 1e-9 else -np.inf
                ty2 = (obstacle.y + obstacle.height - position[1]) / dy if abs(dy) > 1e-9 else np.inf
                near = max(min(tx1, tx2), min(ty1, ty2), 0.0)
                far = min(max(tx1, tx2), max(ty1, ty2))
                if far >= near and near < distance:
                    distance = near
            distances[index] = max(0.0, distance)
        endpoints = position + directions * distances[:, None]
        hits = distances < SENSOR_RANGE - 1e-6
        return RangeScan(angles, distances, endpoints, hits)

    def _reachable_mask(self) -> np.ndarray:
        rows = self.coverage_map.rows
        columns = self.coverage_map.columns
        free = np.zeros((rows, columns), dtype=bool)
        for row in range(rows):
            for column in range(columns):
                point = np.array([(column + 0.5) * PERCEPTION_CELL_SIZE, (row + 0.5) * PERCEPTION_CELL_SIZE])
                free[row, column] = self.board._is_free(point, self.board.robot.radius)
        start_column = min(columns - 1, max(0, int(self.board.robot.position[0] / PERCEPTION_CELL_SIZE)))
        start_row = min(rows - 1, max(0, int(self.board.robot.position[1] / PERCEPTION_CELL_SIZE)))
        reachable = np.zeros_like(free)
        queue = deque([(start_row, start_column)])
        while queue:
            row, column = queue.popleft()
            if row < 0 or row >= rows or column < 0 or column >= columns or reachable[row, column] or not free[row, column]:
                continue
            reachable[row, column] = True
            queue.extend([(row - 1, column), (row + 1, column), (row, column - 1), (row, column + 1)])
        return reachable

    def _path_free(self, start: np.ndarray, end: np.ndarray) -> bool:
        distance = float(np.linalg.norm(end - start))
        for fraction in np.linspace(0.0, 1.0, max(2, int(np.ceil(distance / 0.25)) + 1))[1:]:
            if not self.board._is_free(start + fraction * (end - start), self.board.robot.radius):
                return False
        return True

    def _apply_action(self, action: int) -> bool:
        robot = self.board.robot
        collision = False
        robot.moving = False
        if action == 0:
            self.consecutive_turns = 0
        elif action == 1:
            robot.heading += self.turn_angle
            self.consecutive_turns += 1
        elif action == 2:
            robot.heading -= self.turn_angle
            self.consecutive_turns += 1
        else:
            self.consecutive_turns = 0
        move_forward = action == 0 or (action in (1, 2) and self.consecutive_turns >= 3)
        if move_forward:
            direction = np.array([np.cos(robot.heading), np.sin(robot.heading)])
            candidate = robot.position + self.forward_distance * direction
            if self._path_free(robot.position, candidate):
                robot.position = candidate
                robot.moving = True
                self.consecutive_turns = 0
            elif action == 0:
                collision = True
        robot.heading = (robot.heading + np.pi) % (2.0 * np.pi) - np.pi
        robot.path.append(robot.position.copy())
        if len(robot.path) > HISTORY_LENGTH:
            robot.path.pop(0)
        return collision

    def observation(self) -> tuple[np.ndarray, np.ndarray]:
        map_observation = self.coverage_map.observation(self.board.robot.position)
        ranges = self.board.range_scan.distances.astype(np.float32) / SENSOR_RANGE
        robot = self.board.robot
        start_column = min(self.coverage_map.columns - 1, max(0, int(robot.position[0] / PERCEPTION_CELL_SIZE)))
        start_row = min(self.coverage_map.rows - 1, max(0, int(robot.position[1] / PERCEPTION_CELL_SIZE)))
        targets = ~self.coverage_map.observed
        traversable = ~self.coverage_map.obstacles
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
                if 0 <= next_row < self.coverage_map.rows and 0 <= next_column < self.coverage_map.columns and traversable[next_row, next_column] and neighbor not in parent:
                    parent[neighbor] = cell
                    queue.append(neighbor)
        if target is not None:
            next_cell = target
            while parent[next_cell] is not None and parent[next_cell] != (start_row, start_column):
                next_cell = parent[next_cell]
            target_position = np.array([(next_cell[1] + 0.5) * PERCEPTION_CELL_SIZE, (next_cell[0] + 0.5) * PERCEPTION_CELL_SIZE])
            target_offset = target_position - robot.position
            target_angle = np.arctan2(target_offset[1], target_offset[0])
            relative_target = (target_angle - robot.heading + np.pi) % (2.0 * np.pi) - np.pi
            target_features = np.array(
                [np.sin(relative_target), np.cos(relative_target), np.linalg.norm(target_offset) / np.hypot(BOARD_WIDTH, BOARD_HEIGHT)],
                dtype=np.float32,
            )
        else:
            target_features = np.zeros(3, dtype=np.float32)
        vector = np.concatenate(
            [
                ranges,
                np.array(
                    [
                        robot.position[0] / BOARD_WIDTH,
                        robot.position[1] / BOARD_HEIGHT,
                        np.sin(robot.heading),
                        np.cos(robot.heading),
                        self.coverage_map.observed.mean(),
                        float(robot.moving),
                    ],
                    dtype=np.float32,
                ),
                target_features,
            ]
        )
        return map_observation, vector.astype(np.float32)

    def step(self, action: int) -> tuple[tuple[np.ndarray, np.ndarray], float, bool, bool, dict]:
        action = int(action)
        self.step_count += 1
        self.board.time += 0.4
        collision = self._apply_action(action)
        self.last_collision = collision
        if collision:
            self.collision_count += 1
        self.board.range_scan = self._range_scan()
        self.coverage_map.update(self.board.robot.position, self.board.range_scan)
        newly_covered = self.coverage_map.observed & self.accessible & ~self.covered
        new_count = int(newly_covered.sum())
        self.covered |= newly_covered
        reward = 0.025 * new_count - 0.015
        if new_count == 0:
            reward -= 0.05
        if collision:
            reward -= 1.0
        if action == 3:
            reward -= 0.1
        accessible_regions, covered_regions = self._region_coverage()
        completed = bool(np.all(covered_regions[accessible_regions]))
        if completed:
            reward += 25.0
        truncated = self.step_count >= self.max_steps
        self.previous_action = action
        info = {
            "coverage": self.coverage_ratio,
            "new_cells": new_count,
            "collisions": self.collision_count,
            "steps": self.step_count,
            "completed": completed,
        }
        return self.observation(), float(reward), completed, truncated, info
