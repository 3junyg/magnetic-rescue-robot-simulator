import numpy as np

from config_v2 import BOARD_WIDTH, BOARD_HEIGHT, PERCEPTION_CELL_SIZE, PATROL_REGION_SIZE, PATROL_GOAL_TOLERANCE, PATROL_TARGET_TIMEOUT, PATROL_REFRESH_STEPS


class PatrolScheduler:
    def __init__(self) -> None:
        self.rows = int(np.ceil(BOARD_HEIGHT / PATROL_REGION_SIZE))
        self.columns = int(np.ceil(BOARD_WIDTH / PATROL_REGION_SIZE))
        self.reset()

    def reset(self) -> None:
        self.round = 1
        self.step = 0
        self.visited_round = np.zeros((self.rows, self.columns), dtype=np.int32)
        self.last_visit = np.full((self.rows, self.columns), -1, dtype=np.int64)
        self.deferred_until = {}
        self.goals = {}
        self.alternative_goals = {}
        self.target_region = None
        self.target = None
        self.target_started = 0
        self.last_refresh = -PATROL_REFRESH_STEPS
        self.unreachable = set()
        self.observed_count = -1

    def record(self, position: np.ndarray) -> None:
        self.step += 1
        col = min(self.columns - 1, max(0, int(position[0] / PATROL_REGION_SIZE)))
        row = min(self.rows - 1, max(0, int(position[1] / PATROL_REGION_SIZE)))
        self.visited_round[row, col] = self.round
        self.last_visit[row, col] = self.step
        if self.target is not None and (row, col) == self.target_region and np.linalg.norm(position - self.target) <= PATROL_GOAL_TOLERANCE:
            self.visited_round[self.target_region] = self.round
            self.last_visit[self.target_region] = self.step
            self.target = self.target_region = None

    def refresh(self, observed: np.ndarray, planner, points: np.ndarray, radius: float) -> None:
        if self.step - self.last_refresh < PATROL_REFRESH_STEPS:
            return
        self.last_refresh = self.step
        count = int(observed.sum())
        if count != self.observed_count:
            self.unreachable.clear()
            self.observed_count = count
        blocked = planner.blocked_map(points, radius)
        rows, cols = np.where(~blocked)
        xs, ys = (cols + 0.5) * planner.resolution, (rows + 0.5) * planner.resolution
        coarse_cols = np.minimum(observed.shape[1] - 1, (xs / PERCEPTION_CELL_SIZE).astype(int))
        coarse_rows = np.minimum(observed.shape[0] - 1, (ys / PERCEPTION_CELL_SIZE).astype(int))
        valid = observed[coarse_rows, coarse_cols]
        xs, ys = xs[valid], ys[valid]
        tile_cols, tile_rows = (xs / PATROL_REGION_SIZE).astype(int), (ys / PATROL_REGION_SIZE).astype(int)
        keys = tile_rows * self.columns + tile_cols
        score = (xs - (tile_cols + 0.5) * PATROL_REGION_SIZE) ** 2 + (ys - (tile_rows + 0.5) * PATROL_REGION_SIZE) ** 2
        order = np.argsort(score, kind='stable')
        _, first = np.unique(keys[order], return_index=True)
        selected = order[first]
        self.goals = {(int(tile_rows[index]), int(tile_cols[index])): np.array([xs[index], ys[index]]) for index in selected}
        self.goals.update({region: goal for region, goal in self.alternative_goals.items() if region in self.goals})
        if self.target_region not in self.goals:
            self.target = self.target_region = None
        elif self.target is not None:
            self.target = self.goals[self.target_region].copy()

    def choose(self, position: np.ndarray) -> np.ndarray | None:
        if self.target is not None:
            if self.step - self.target_started <= PATROL_TARGET_TIMEOUT:
                return self.target
            self.defer()
        eligible = [region for region in self.goals if region not in self.unreachable]
        available = [region for region in eligible if self.deferred_until.get(region, 0) <= self.step]
        pending = [region for region in available if self.visited_round[region] < self.round]
        if not pending:
            if any(self.visited_round[region] < self.round for region in eligible):
                return None
            if not available:
                return None
            self.round += 1
            pending = available
        region = min(pending, key=lambda key: (np.linalg.norm(self.goals[key] - position), self.last_visit[key]))
        self.target_region = region
        self.target = self.goals[region].copy()
        self.target_started = self.step
        return self.target

    def defer(self) -> None:
        if self.target_region is not None:
            self.deferred_until[self.target_region] = self.step + PATROL_REFRESH_STEPS * 3
        self.target = self.target_region = None

    def exclude(self) -> None:
        if self.target_region is not None:
            self.unreachable.add(self.target_region)
        self.target = self.target_region = None

    def redirect(self, goal: np.ndarray) -> None:
        self.target = goal.copy()
        self.goals[self.target_region] = goal.copy()
        self.alternative_goals[self.target_region] = goal.copy()

    def status(self) -> dict:
        eligible = [region for region in self.goals if region not in self.unreachable]
        return {'round': self.round, 'regions': len(eligible), 'visited': int(sum(self.visited_round[region] >= self.round for region in eligible)), 'unreachable': len(self.unreachable), 'target': None if self.target is None else self.target.tolist()}
