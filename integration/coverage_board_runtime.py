from collections import deque
from pathlib import Path

import numpy as np
import torch

from config import BOARD_HEIGHT, BOARD_WIDTH, PERCEPTION_CELL_SIZE, SENSOR_RANGE
from navigation.coverage_policy import CoverageActorCritic


class CoverageBoardRuntime:
    def __init__(self, model_path: str | Path | None = None, device: str = "cpu") -> None:
        root = Path(__file__).resolve().parents[1]
        self.model_path = Path(model_path) if model_path else root / "models" / "coverage_agent" / "best_coverage_agent.pth"
        if not self.model_path.is_absolute():
            self.model_path = root / self.model_path
        if not self.model_path.exists():
            raise FileNotFoundError(f"Coverage model not found: {self.model_path}")
        checkpoint = torch.load(self.model_path, map_location=device, weights_only=False)
        self.device = torch.device(device)
        if self.device.type == "cpu":
            torch.set_num_threads(1)
        self.actions = tuple(checkpoint.get("actions", ("forward", "turn_left", "turn_right", "stop")))
        self.model = CoverageActorCritic(map_channels=4, vector_size=int(checkpoint.get("vector_size", 57)), actions=len(self.actions)).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.map_shape = tuple(checkpoint.get("map_shape", (4, 20, 30)))
        self.last_action = "stop"
        self.last_coverage = 0.0
        self.stalled_steps = 0
        self.last_target_features = np.zeros(3, dtype=np.float32)
        self.patrol_mode = False
        self.patrol_target: tuple[int, int] | None = None
        self.recent_cells: deque[tuple[int, int]] = deque(maxlen=48)
        self.cell_visits: dict[tuple[int, int], int] = {}
        self.turn_direction = 0
        self.turn_streak = 0

    def reset(self) -> None:
        self.last_action = "stop"
        self.last_coverage = 0.0
        self.stalled_steps = 0
        self.last_target_features = np.zeros(3, dtype=np.float32)
        self.patrol_mode = False
        self.patrol_target = None
        self.recent_cells.clear()
        self.cell_visits.clear()
        self.turn_direction = 0
        self.turn_streak = 0

    def _observation(self, board) -> tuple[np.ndarray, np.ndarray, float]:
        rows = int(BOARD_HEIGHT / PERCEPTION_CELL_SIZE)
        columns = int(BOARD_WIDTH / PERCEPTION_CELL_SIZE)
        observed = board.perception.observed.copy()
        obstacle = np.zeros_like(observed, dtype=bool)
        for point in board.perception.obstacle_points:
            col = int(np.clip(point[0] / PERCEPTION_CELL_SIZE, 0, columns - 1))
            row = int(np.clip(point[1] / PERCEPTION_CELL_SIZE, 0, rows - 1))
            obstacle[max(0, row - 1):min(rows, row + 2), max(0, col - 1):min(columns, col + 2)] = True
        known_free = observed & (~obstacle)
        adjacent_unknown = np.zeros_like(observed)
        adjacent_unknown[1:] |= ~observed[:-1]
        adjacent_unknown[:-1] |= ~observed[1:]
        adjacent_unknown[:, 1:] |= ~observed[:, :-1]
        adjacent_unknown[:, :-1] |= ~observed[:, 1:]
        frontier = known_free & adjacent_unknown
        unknown = ~observed & (~obstacle)
        robot = np.zeros_like(observed, dtype=bool)
        robot[int(np.clip(board.robot.position[1] / PERCEPTION_CELL_SIZE, 0, rows - 1)), int(np.clip(board.robot.position[0] / PERCEPTION_CELL_SIZE, 0, columns - 1))] = True
        fine = np.stack([observed, obstacle, frontier, robot]).astype(np.float32)
        map_obs = fine.reshape(4, 20, 2, 30, 2).max(axis=(2, 4))
        distances = np.asarray(board.range_scan.distances, dtype=np.float32) / SENSOR_RANGE
        start_column = min(columns - 1, max(0, int(board.robot.position[0] / PERCEPTION_CELL_SIZE)))
        start_row = min(rows - 1, max(0, int(board.robot.position[1] / PERCEPTION_CELL_SIZE)))
        current_cell = (start_row, start_column)
        if not self.recent_cells or self.recent_cells[-1] != current_cell:
            self.recent_cells.append(current_cell)
            self.cell_visits[current_cell] = self.cell_visits.get(current_cell, 0) + 1
        traversable = ~obstacle
        queue = deque([(start_row, start_column)])
        parent = {(start_row, start_column): None}
        target = None
        while queue:
            row, column = queue.popleft()
            if unknown[row, column]:
                target = (row, column)
                break
            for next_row, next_column in ((row - 1, column), (row + 1, column), (row, column - 1), (row, column + 1)):
                if 0 <= next_row < rows and 0 <= next_column < columns and traversable[next_row, next_column] and (next_row, next_column) not in parent:
                    parent[(next_row, next_column)] = (row, column)
                    queue.append((next_row, next_column))
        self.patrol_mode = target is None
        if target is None:
            reachable = [cell for cell in parent if known_free[cell]]
            if reachable:
                if self.patrol_target not in reachable or current_cell == self.patrol_target:
                    def patrol_score(cell: tuple[int, int]) -> float:
                        distance = float(np.hypot(cell[0] - start_row, cell[1] - start_column))
                        visits = self.cell_visits.get(cell, 0)
                        recent = 1.0 if cell in self.recent_cells else 0.0
                        return distance + 5.0 / (1.0 + visits) - 4.0 * recent
                    self.patrol_target = max(reachable, key=patrol_score)
                target = self.patrol_target
        else:
            self.patrol_target = None
        if target is not None:
            next_cell = target
            while parent[next_cell] is not None and parent[next_cell] != (start_row, start_column):
                next_cell = parent[next_cell]
            target_position = np.array([(next_cell[1] + 0.5) * PERCEPTION_CELL_SIZE, (next_cell[0] + 0.5) * PERCEPTION_CELL_SIZE])
            target_offset = target_position - board.robot.position
            target_angle = np.arctan2(target_offset[1], target_offset[0])
            relative_target = (target_angle - board.robot.heading + np.pi) % (2.0 * np.pi) - np.pi
            target_features = np.array([np.sin(relative_target), np.cos(relative_target), np.linalg.norm(target_offset) / np.hypot(BOARD_WIDTH, BOARD_HEIGHT)], dtype=np.float32)
        else:
            target_features = np.zeros(3, dtype=np.float32)
        vector = np.concatenate([distances, np.array([board.robot.position[0] / BOARD_WIDTH, board.robot.position[1] / BOARD_HEIGHT, np.sin(board.robot.heading), np.cos(board.robot.heading), observed.mean(), float(board.robot.moving)], dtype=np.float32), target_features])
        self.last_target_features = target_features
        return map_obs, vector.astype(np.float32), float(observed.mean())

    def decide(self, board) -> tuple[str, float]:
        map_obs, vector, coverage = self._observation(board)
        failed_forward = self.last_action == "forward" and not board.robot.moving
        if coverage > self.last_coverage + 1e-5:
            self.stalled_steps = 0
        else:
            self.stalled_steps += 1
        self.last_coverage = coverage
        with torch.inference_mode():
            logits, _ = self.model(torch.from_numpy(map_obs[None]).to(self.device), torch.from_numpy(vector[None]).to(self.device))
            action = self.actions[int(torch.argmax(logits, dim=1)[0].cpu())]
        if self.patrol_mode or self.stalled_steps >= 12 or action == "stop" or failed_forward:
            target_sin, target_cos, _ = self.last_target_features
            relative_angles = (np.asarray(board.range_scan.angles) - board.robot.heading + np.pi) % (2.0 * np.pi) - np.pi
            forward_rays = np.abs(relative_angles) <= np.pi / 12.0
            forward_clearance = float(np.min(board.range_scan.distances[forward_rays])) if np.any(forward_rays) else 0.0
            target_aligned = target_cos >= 0.90 and abs(target_sin) <= 0.35
            if failed_forward:
                left_rays = (relative_angles >= np.pi / 12.0) & (relative_angles <= 5.0 * np.pi / 6.0)
                right_rays = (relative_angles <= -np.pi / 12.0) & (relative_angles >= -5.0 * np.pi / 6.0)
                left_clearance = float(np.mean(board.range_scan.distances[left_rays])) if np.any(left_rays) else 0.0
                right_clearance = float(np.mean(board.range_scan.distances[right_rays])) if np.any(right_rays) else 0.0
                self.turn_direction = 1 if left_clearance >= right_clearance else -1
                self.turn_streak = 1
                action = "turn_left" if self.turn_direction > 0 else "turn_right"
            elif target_aligned and forward_clearance >= 3.5:
                self.turn_direction = 0
                self.turn_streak = 0
                action = "forward"
            else:
                desired_direction = 1 if target_sin >= 0.0 else -1
                if desired_direction != self.turn_direction:
                    self.turn_direction = desired_direction
                    self.turn_streak = 0
                self.turn_streak += 1
                if self.turn_streak >= 3 and forward_clearance >= 2.5:
                    self.turn_streak = 0
                    action = "forward"
                else:
                    action = "turn_left" if self.turn_direction > 0 else "turn_right"
        self.last_action = action
        return action, coverage

