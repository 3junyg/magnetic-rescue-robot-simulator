from collections import deque
from pathlib import Path

import numpy as np
import torch

from config_v2 import BOARD_HEIGHT, BOARD_WIDTH, PERCEPTION_CELL_SIZE, SENSOR_RANGE, NAV_PRECISE_RESOLUTION, NAV_HEADING_TOLERANCE_DEGREES, NAV_REFINEMENT_INTERVAL, NAV_PRECISION_EXIT_STEPS, NAV_CORRIDOR_SIDE_CLEARANCE, PATROL_START_STABLE_STEPS, PATROL_MIN_OBSERVED_FRACTION
from navigation.coverage_policy_v2 import CoverageActorCritic
from navigation.clearance_planner_v2 import ClearancePlanner, NavigationAction
from navigation.patrol_scheduler_v2 import PatrolScheduler
from navigation.progress_guard_v2 import NavigationProgressGuard
from config_v2 import NAV_PROGRESS_WINDOW, NAV_MIN_LOCAL_PROGRESS, NAV_ESCAPE_DISTANCE, NAV_ESCAPE_COOLDOWN, NAV_FRONTIER_RETRY_STEPS, NAV_RELOCATION_MAX_STEPS, NAV_RELOCATION_COVERAGE_GAIN


class CoverageBoardRuntime:
    def __init__(self, model_path: str | Path | None = None, device: str = "cpu") -> None:
        root = Path(__file__).resolve().parents[1]
        self.model_path = Path(model_path) if model_path else root / "models" / "coverage_agent" / "best_coverage_agent_v2.pth"
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
        self.clearance_planner = ClearancePlanner()
        self.precise_planner = ClearancePlanner(resolution=NAV_PRECISE_RESOLUTION)
        self.active_planner = self.clearance_planner
        self.refinement_cooldown = 0
        self.position_history = deque(maxlen=40)
        self.no_translation = 0
        self.recovery_target = None
        self.recovery_steps = 0
        self.precision_active = False
        self.precision_clear_steps = 0
        self.waypoint_cache = None
        self.control_mode = 'NORMAL'
        self.normal_waypoint = None
        self.coarse_waypoint = None
        self.precision_entry_coverage = 0.0
        self.patrol_scheduler = PatrolScheduler()
        self.patrol_active = False
        self.progress_history = deque(maxlen=NAV_PROGRESS_WINDOW)
        self.escape_cooldown = 0
        self.local_escape_count = 0
        self.patrol_progress_target = None
        self.patrol_best_distance = float('inf')
        self.patrol_no_progress = 0
        self.cycle_guard = NavigationProgressGuard()
        self.relocating = False
        self.relocation_steps = 0
        self.relocation_coverage = 0.0

    def reset(self) -> None:
        self.clearance_planner.reset()
        self.precise_planner.reset()
        self.active_planner = self.clearance_planner
        self.refinement_cooldown = 0
        self.position_history.clear()
        self.no_translation = 0
        self.recovery_target = None
        self.recovery_steps = 0
        self.precision_active = False
        self.precision_clear_steps = 0
        self.waypoint_cache = None
        self.control_mode = 'NORMAL'
        self.normal_waypoint = None
        self.coarse_waypoint = None
        self.precision_entry_coverage = 0.0
        self.patrol_scheduler.reset()
        self.patrol_active = False
        self.progress_history.clear()
        self.escape_cooldown = 0
        self.local_escape_count = 0
        self.patrol_progress_target = None
        self.patrol_best_distance = float('inf')
        self.patrol_no_progress = 0
        self.cycle_guard.reset()
        self.relocating = False
        self.relocation_steps = 0
        self.relocation_coverage = 0.0
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
        if board.perception.obstacle_points:
            points = np.asarray(board.perception.obstacle_points)
            cols = np.clip(points[:, 0] / PERCEPTION_CELL_SIZE, 0, columns - 1).astype(int)
            rs = np.clip(points[:, 1] / PERCEPTION_CELL_SIZE, 0, rows - 1).astype(int)
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    valid = (rs + dr >= 0) & (rs + dr < rows) & (cols + dc >= 0) & (cols + dc < columns)
                    obstacle[rs[valid] + dr, cols[valid] + dc] = True
        known_free = observed & (~obstacle)
        adjacent_unknown = np.zeros_like(observed)
        adjacent_unknown[1:] |= ~observed[:-1]
        adjacent_unknown[:-1] |= ~observed[1:]
        adjacent_unknown[:, 1:] |= ~observed[:, :-1]
        adjacent_unknown[:, :-1] |= ~observed[:, 1:]
        frontier = known_free & adjacent_unknown
        unknown = ~observed & (~obstacle)
        for row, column in self.clearance_planner.suppressed_frontiers() | self.precise_planner.suppressed_frontiers():
            unknown[row, column] = False
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
        self.patrol_mode = self.patrol_active or target is None
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
            route = []
            next_cell = target
            while parent[next_cell] is not None:
                route.append(next_cell)
                next_cell = parent[next_cell]
            route.reverse()
            next_cell = route[0] if route else current_cell
            direction = np.asarray(next_cell) - np.asarray(current_cell)
            for cell in route[1:3]:
                offset = np.asarray(cell) - np.asarray(current_cell)
                if direction[0] * offset[1] != direction[1] * offset[0]:
                    break
                next_cell = cell
            target_position = np.array([(next_cell[1] + 0.5) * PERCEPTION_CELL_SIZE, (next_cell[0] + 0.5) * PERCEPTION_CELL_SIZE])
            self.coarse_waypoint = target_position
            target_offset = target_position - board.robot.position
            target_angle = np.arctan2(target_offset[1], target_offset[0])
            relative_target = (target_angle - board.robot.heading + np.pi) % (2.0 * np.pi) - np.pi
            target_features = np.array([np.sin(relative_target), np.cos(relative_target), np.linalg.norm(target_offset) / np.hypot(BOARD_WIDTH, BOARD_HEIGHT)], dtype=np.float32)
        else:
            self.coarse_waypoint = None
            target_features = np.zeros(3, dtype=np.float32)
        vector = np.concatenate([distances, np.array([board.robot.position[0] / BOARD_WIDTH, board.robot.position[1] / BOARD_HEIGHT, np.sin(board.robot.heading), np.cos(board.robot.heading), observed.mean(), float(board.robot.moving)], dtype=np.float32), target_features])
        self.last_target_features = target_features
        return map_obs, vector.astype(np.float32), float(observed.mean())

    def decide(self, board) -> tuple[str, float]:
        action, coverage = self._decide(board)
        heading = getattr(action, 'heading', None)
        if heading is not None and self.cycle_guard.reject_backtrack(board.robot.position, heading):
            self.last_action = 'stop'
            return 'stop', coverage
        return action, coverage

    def _decide(self, board) -> tuple[str, float]:
        self.clearance_planner.advance_clock()
        self.precise_planner.advance_clock()
        map_obs, vector, coverage = self._observation(board)
        position = board.robot.position
        self.progress_history.append((position.copy(), coverage))
        self.escape_cooldown = max(0, self.escape_cooldown - 1)
        self.patrol_scheduler.record(position)
        patrol_progress = (self.patrol_scheduler.round, int(np.sum(self.patrol_scheduler.visited_round == self.patrol_scheduler.round))) if self.patrol_active else None
        if self.cycle_guard.observe(position, coverage, patrol_progress):
            cells = self.clearance_planner.defer_frontier() | self.precise_planner.defer_frontier()
            for planner in (self.clearance_planner, self.precise_planner):
                for cell in cells:
                    planner.deferred_frontiers[cell] = planner.plan_step + NAV_FRONTIER_RETRY_STEPS
                planner.target = None
            self.patrol_scheduler.defer()
            self.patrol_target = None
            self.normal_waypoint = self.waypoint_cache = None
            self.recovery_target = None
            self.recovery_steps = self.no_translation = 0
            self.escape_cooldown = NAV_ESCAPE_COOLDOWN
            self.active_planner = self.precise_planner
            self.precision_active = True
            self.progress_history.clear()
            self.patrol_progress_target = None
            if not self.patrol_active:
                self.relocating = True
                self.relocation_steps = 0
                self.relocation_coverage = coverage
            map_obs, vector, coverage = self._observation(board)
        if not self.position_history or np.linalg.norm(position - self.position_history[-1]) > 0.01:
            self.position_history.append(position.copy())
            self.no_translation = 0
        else:
            self.no_translation += 1
        failed_forward = self.last_action == "forward" and not board.robot.moving
        if failed_forward and isinstance(self.last_action, NavigationAction) and not self.last_action.recovery:
            self.active_planner.report_failure(board.robot.position, self.last_action.heading)
            if self.active_planner is self.clearance_planner:
                self.active_planner = self.precise_planner
                self.precise_planner.report_failure(board.robot.position, self.last_action.heading)
        if self.recovery_target is not None and np.linalg.norm(position - self.recovery_target) < 0.1:
            self.recovery_target = None
            self.no_translation = 0
            self.recovery_steps = 0
        if self.recovery_target is not None and self.recovery_steps >= NAV_PROGRESS_WINDOW:
            self.recovery_target = None
            self.recovery_steps = 0
            self.no_translation = 0
            self.escape_cooldown = NAV_ESCAPE_COOLDOWN
            self.normal_waypoint = self.waypoint_cache = None
            self.progress_history.clear()
            self.patrol_scheduler.defer()
        if len(self.progress_history) == NAV_PROGRESS_WINDOW and self.escape_cooldown == 0 and self.recovery_target is None:
            positions = np.asarray([item[0] for item in self.progress_history])
            local_span = float(np.linalg.norm(np.ptp(positions, axis=0)))
            new_coverage = coverage - self.progress_history[0][1]
            if local_span < NAV_MIN_LOCAL_PROGRESS and new_coverage < 1e-5:
                self.normal_waypoint = self.waypoint_cache = None
                self.patrol_scheduler.defer()
                self.clearance_planner.target = self.precise_planner.target = None
                self.recovery_target = next((point.copy() for point in reversed(self.position_history) if np.linalg.norm(position - point) >= NAV_ESCAPE_DISTANCE and self._sensor_clear(board, float(np.arctan2((point - position)[1], (point - position)[0])), float(np.linalg.norm(point - position)))), None)
                self.escape_cooldown = NAV_ESCAPE_COOLDOWN
                self.local_escape_count += 1
                self.progress_history.clear()
        if self.no_translation >= 12 and self.recovery_target is None and self.escape_cooldown == 0:
            self.recovery_target = next((point.copy() for point in reversed(self.position_history) if np.linalg.norm(position - point) >= 0.25), None)
        if self.recovery_target is not None:
            self.recovery_steps += 1
            self.control_mode = 'RECOVERY'
            self.waypoint_cache = None
            action = self._steer(position, board.robot.heading, self.recovery_target, recovery=True)
            self.last_action = action
            return action, coverage
        if coverage > self.last_coverage + 1e-5:
            self.stalled_steps = 0
        else:
            self.stalled_steps += 1
        self.last_coverage = coverage
        if self.relocating:
            self.relocation_steps += 1
            arrived = self.relocation_steps > 1 and self.patrol_scheduler.target is None
            if arrived or self.relocation_steps >= NAV_RELOCATION_MAX_STEPS or coverage >= self.relocation_coverage + NAV_RELOCATION_COVERAGE_GAIN:
                self.relocating = False
                self.patrol_scheduler.defer()
                self.precision_active = False
                self.normal_waypoint = self.waypoint_cache = None
                self.cycle_guard.clear_history()
            else:
                action = self._patrol_action(board)
                if action is not None:
                    self.control_mode = 'RELOCATING'
                    self.last_action = action
                    return action, coverage
                self.relocating = False
        if not self.patrol_active and self.stalled_steps % 20 == 12 and (self.stalled_steps >= PATROL_START_STABLE_STEPS or self.patrol_mode):
            points = np.asarray(board.perception.obstacle_points, dtype=float).reshape(-1, 2)
            self.clearance_planner.waypoint(position, board.perception.observed, points, board.robot.radius)
            if self.clearance_planner.seeking_unknown:
                unseen = True
            else:
                self.precise_planner.waypoint(position, board.perception.observed, points, board.robot.radius)
                unseen = self.precise_planner.seeking_unknown
            self.patrol_active = not unseen or (self.stalled_steps >= PATROL_START_STABLE_STEPS and coverage >= PATROL_MIN_OBSERVED_FRACTION)
            if self.patrol_active:
                self.normal_waypoint = self.waypoint_cache = None
        if self.patrol_active:
            action = self._patrol_action(board)
            if action is not None:
                self.patrol_mode = True
                self.control_mode = 'PATROL'
                self.last_action = action
                return action, coverage
        with torch.inference_mode():
            logits, _ = self.model(torch.from_numpy(map_obs[None]).to(self.device), torch.from_numpy(vector[None]).to(self.device))
            action = self.actions[int(torch.argmax(logits, dim=1)[0].cpu())]
        relative = (np.asarray(board.range_scan.angles) - board.robot.heading + np.pi) % (2 * np.pi) - np.pi
        left = board.range_scan.distances[(relative > np.pi / 3) & (relative < 2 * np.pi / 3)]
        right = board.range_scan.distances[(relative < -np.pi / 3) & (relative > -2 * np.pi / 3)]
        corridor = bool(len(left) and len(right) and np.median(left) < NAV_CORRIDOR_SIDE_CLEARANCE and np.median(right) < NAV_CORRIDOR_SIDE_CLEARANCE)
        forward_clear = self._sensor_clear(board, board.robot.heading, 2.0)
        needs_precision = failed_forward or (action == 'forward' and not forward_clear) or (self.stalled_steps >= 12 and coverage < 0.95 and self.refinement_cooldown == 0) or (corridor and self.patrol_mode)
        if needs_precision:
            if not self.precision_active:
                self.precision_entry_coverage = coverage
            self.precision_active = True
            self.precision_clear_steps = 0
        elif self.precision_active and forward_clear and not corridor and not self.patrol_mode and coverage > self.precision_entry_coverage + 0.002:
            self.precision_clear_steps += 1
            if self.precision_clear_steps >= NAV_PRECISION_EXIT_STEPS:
                self.precision_active = False
                self.waypoint_cache = None
                self.active_planner = self.clearance_planner
        else:
            self.precision_clear_steps = 0
        self.refinement_cooldown = max(0, self.refinement_cooldown - 1)
        if not self.precision_active:
            self.active_planner = self.clearance_planner
        if self.precision_active:
            self.control_mode = 'PRECISE'
            self.normal_waypoint = None
            points = np.asarray(board.perception.obstacle_points, dtype=float).reshape(-1, 2)
            waypoint = self.waypoint_cache
            if waypoint is not None and not self.active_planner.seeking_unknown and forward_clear and not corridor and not failed_forward:
                self.precision_active = False
                self.refinement_cooldown = NAV_REFINEMENT_INTERVAL
                waypoint = None
            if waypoint is not None:
                offset = waypoint - position
                direction = float(np.arctan2(offset[1], offset[0]))
                if failed_forward or np.linalg.norm(offset) < 0.2 or not self._sensor_clear(board, direction, float(np.linalg.norm(offset))):
                    waypoint = None
            if waypoint is None:
                waypoint = self.active_planner.waypoint(board.robot.position, board.perception.observed, points, board.robot.radius)
            if self.active_planner is self.clearance_planner and (waypoint is None or not self.clearance_planner.seeking_unknown) and not board.perception.observed.all() and self.refinement_cooldown == 0:
                precise = self.precise_planner.waypoint(board.robot.position, board.perception.observed, points, board.robot.radius)
                self.refinement_cooldown = NAV_REFINEMENT_INTERVAL
                if precise is not None and self.precise_planner.seeking_unknown:
                    self.active_planner = self.precise_planner
                    waypoint = precise
            if waypoint is not None:
                self.waypoint_cache = waypoint.copy()
                action = self._steer(position, board.robot.heading, waypoint)
                self.last_action = action
                return action, coverage
        self.control_mode = 'NORMAL'
        self.waypoint_cache = None
        target = self.normal_waypoint
        if target is not None:
            offset = target - position
            direction = float(np.arctan2(offset[1], offset[0]))
            if np.linalg.norm(offset) < 0.3 or failed_forward or not self._sensor_clear(board, direction, min(2.0, float(np.linalg.norm(offset)))):
                target = None
        if target is None:
            target = self.coarse_waypoint
        if target is not None and np.linalg.norm(target - position) >= 0.3:
            self.normal_waypoint = target.copy()
            action = self._steer(position, board.robot.heading, target)
            self.last_action = action
            return action, coverage
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

    def _patrol_action(self, board) -> NavigationAction | None:
        points = np.asarray(board.perception.obstacle_points, dtype=float).reshape(-1, 2)
        self.patrol_scheduler.refresh(board.perception.observed, self.precise_planner, points, board.robot.radius)
        for _ in range(2):
            target = self.patrol_scheduler.choose(board.robot.position)
            if target is None:
                return None
            planners = (self.active_planner, self.precise_planner) if self.active_planner is self.clearance_planner else (self.precise_planner, self.clearance_planner)
            waypoint = None
            for planner in planners:
                waypoint = planner.waypoint_to(board.robot.position, target, points, board.perception.observed, board.robot.radius)
                if waypoint is not None:
                    self.active_planner = planner
                    break
            if waypoint is not None:
                route = np.asarray(self.active_planner._goal_route)
                distance = float(np.linalg.norm(target - board.robot.position))
                if route.size:
                    centers = (route[:, ::-1] + 0.5) * self.active_planner.resolution
                    offsets = np.linalg.norm(centers - board.robot.position, axis=1)
                    index = int(np.argmin(offsets))
                    distance = float(offsets[index] + (len(route) - index - 1) * self.active_planner.resolution)
                if self.patrol_progress_target is None or np.linalg.norm(target - self.patrol_progress_target) > 0.5:
                    self.patrol_progress_target = target.copy()
                    self.patrol_best_distance = distance
                    self.patrol_no_progress = 0
                elif distance < self.patrol_best_distance - 0.15:
                    self.patrol_best_distance = distance
                    self.patrol_no_progress = 0
                else:
                    self.patrol_no_progress += 1
                if self.patrol_no_progress >= NAV_PROGRESS_WINDOW:
                    self.patrol_scheduler.defer()
                    self.patrol_progress_target = None
                    continue
                return self._steer(board.robot.position, board.robot.heading, waypoint)
            alternative = self.precise_planner.alternative_goal(board.robot.position, target, points, board.robot.radius)
            if alternative is not None:
                waypoint = self.precise_planner.waypoint_to(board.robot.position, alternative, points, board.perception.observed, board.robot.radius)
                if waypoint is not None:
                    self.patrol_scheduler.redirect(alternative)
                    self.active_planner = self.precise_planner
                    return self._steer(board.robot.position, board.robot.heading, waypoint)
            self.patrol_scheduler.exclude()
        return None

    @staticmethod
    def _sensor_clear(board, heading: float, distance: float) -> bool:
        hits = board.range_scan.endpoints[board.range_scan.hits]
        if not len(hits):
            return True
        direction = np.array([np.cos(heading), np.sin(heading)])
        samples = board.robot.position + np.linspace(0.1, distance, max(2, int(np.ceil(distance / 0.25))))[:, None] * direction
        distances = np.max(np.abs(samples[:, None, :] - hits[None, :, :]), axis=2)
        return bool(np.all(distances > board.robot.radius + 0.025))

    @staticmethod
    def _steer(position: np.ndarray, heading: float, target: np.ndarray, recovery: bool = False) -> NavigationAction:
        offset = target - position
        desired = float(np.arctan2(offset[1], offset[0]))
        error = (desired - heading + np.pi) % (2 * np.pi) - np.pi
        action = 'forward' if abs(error) <= np.deg2rad(NAV_HEADING_TOLERANCE_DEGREES) else ('turn_left' if error > 0 else 'turn_right')
        return NavigationAction(action, desired, min(2.0, float(np.linalg.norm(offset))), recovery)
