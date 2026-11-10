from collections import deque
import heapq

import numpy as np

from config_v2 import BOARD_HEIGHT, BOARD_WIDTH, PERCEPTION_CELL_SIZE, NAV_GRID_RESOLUTION, NAV_CLEARANCE_MARGIN, PATROL_REGION_SIZE
from config_v2 import NAV_FRONTIER_PROBE_DISTANCE, NAV_FRONTIER_PROBE_LIMIT, NAV_FRONTIER_RETRY_STEPS


class NavigationAction(str):
    def __new__(cls, value: str, heading: float, distance: float = 2.0, recovery: bool = False):
        result = super().__new__(cls, value)
        result.heading = heading
        result.distance = distance
        result.recovery = recovery
        return result


class ClearancePlanner:
    def __init__(self, resolution: float = NAV_GRID_RESOLUTION, safety_margin: float = NAV_CLEARANCE_MARGIN) -> None:
        self.resolution = resolution
        self.safety_margin = safety_margin
        self.rows = int(BOARD_HEIGHT / resolution)
        self.columns = int(BOARD_WIDTH / resolution)
        self.visits = np.zeros((self.rows, self.columns), dtype=np.int32)
        self.target = None
        self.last_cell = None
        self.rejected_cells = set()
        self.last_waypoint = None
        self.seeking_unknown = False
        self._blocked_cache = None
        self._blocked_key = None
        self.rejected_edges = set()
        self.last_first_cell = None
        self._goal_route = []
        self._route_goal = None
        self._route_version = None
        self.frontier_cell = None
        self.frontier_probes = {}
        self.deferred_frontiers = {}
        self.plan_step = 0

    def reset(self) -> None:
        self.visits.fill(0)
        self.target = None
        self.last_cell = None
        self.rejected_cells.clear()
        self.last_waypoint = None
        self.seeking_unknown = False
        self._blocked_cache = None
        self._blocked_key = None
        self.rejected_edges.clear()
        self.last_first_cell = None
        self._goal_route = []
        self._route_goal = None
        self._route_version = None
        self.frontier_cell = None
        self.frontier_probes.clear()
        self.deferred_frontiers.clear()
        self.plan_step = 0

    def suppressed_frontiers(self) -> set:
        return {cell for cell, until in self.deferred_frontiers.items() if until > self.plan_step}

    def advance_clock(self) -> None:
        self.plan_step += 1

    def defer_frontier(self) -> set:
        if self.frontier_cell is None:
            return set()
        point = self.center(self.frontier_cell)
        column, row = (point / PERCEPTION_CELL_SIZE).astype(int)
        cells = {(r, c) for r in range(max(0, row - 1), min(int(BOARD_HEIGHT / PERCEPTION_CELL_SIZE), row + 2)) for c in range(max(0, column - 1), min(int(BOARD_WIDTH / PERCEPTION_CELL_SIZE), column + 2))}
        for cell in cells:
            self.deferred_frontiers[cell] = self.plan_step + NAV_FRONTIER_RETRY_STEPS
        self.frontier_cell = None
        return cells

    def report_failure(self, position: np.ndarray, heading: float) -> None:
        start = tuple(np.clip((position[::-1] / self.resolution).astype(int), [0, 0], [self.rows - 1, self.columns - 1]))
        if self.last_first_cell is not None:
            self.rejected_edges.add((start, self.last_first_cell))
        point = position + self.resolution * np.array([np.cos(heading), np.sin(heading)])
        row, column = np.clip((point[::-1] / self.resolution).astype(int), [0, 0], [self.rows - 1, self.columns - 1])
        self.rejected_cells.add((int(row), int(column)))

    def blocked_map(self, points: np.ndarray, radius: float) -> np.ndarray:
        key = (points.shape, radius, hash(points.tobytes()))
        if key == self._blocked_key:
            return self._blocked_cache.copy()
        blocked = np.zeros((self.rows, self.columns), dtype=bool)
        margin = radius + self.safety_margin
        xs = (np.arange(self.columns) + 0.5) * self.resolution
        ys = (np.arange(self.rows) + 0.5) * self.resolution
        blocked |= (ys[:, None] < margin) | (ys[:, None] > BOARD_HEIGHT - margin)
        blocked |= (xs[None, :] < margin) | (xs[None, :] > BOARD_WIDTH - margin)
        if len(points):
            columns = np.floor(points[:, 0] / self.resolution).astype(int)
            rows = np.floor(points[:, 1] / self.resolution).astype(int)
            extent = int(np.ceil(margin / self.resolution)) + 1
            for dr in range(-extent, extent + 1):
                for dc in range(-extent, extent + 1):
                    rs, cs = rows + dr, columns + dc
                    valid = (rs >= 0) & (rs < self.rows) & (cs >= 0) & (cs < self.columns)
                    valid &= np.abs((rs + 0.5) * self.resolution - points[:, 1]) <= margin
                    valid &= np.abs((cs + 0.5) * self.resolution - points[:, 0]) <= margin
                    blocked[rs[valid], cs[valid]] = True
        self._blocked_key = key
        self._blocked_cache = blocked.copy()
        return blocked

    def waypoint(self, position: np.ndarray, observed: np.ndarray, points: np.ndarray, radius: float) -> np.ndarray | None:
        start = tuple(np.clip((position[::-1] / self.resolution).astype(int), [0, 0], [self.rows - 1, self.columns - 1]))
        if start != self.last_cell:
            self.visits[start] += 1
            self.last_cell = start
        blocked = self.blocked_map(points, radius)
        for cell in self.rejected_cells:
            blocked[cell] = True
        blocked[start] = False
        coarse_rows = np.minimum(observed.shape[0] - 1, ((np.arange(self.rows) + 0.5) * self.resolution / PERCEPTION_CELL_SIZE).astype(int))
        coarse_columns = np.minimum(observed.shape[1] - 1, ((np.arange(self.columns) + 0.5) * self.resolution / PERCEPTION_CELL_SIZE).astype(int))
        known = observed[coarse_rows[:, None], coarse_columns[None, :]]
        suppressed = self.suppressed_frontiers()
        ignored = np.zeros_like(observed, dtype=bool)
        for row, column in suppressed:
            ignored[row, column] = True
        unavailable = ignored[coarse_rows[:, None], coarse_columns[None, :]]
        held = self.frontier_cell
        if held is not None and (blocked[held] or known[held] or unavailable[held]):
            held = None
        if held is not None:
            key = (int(coarse_rows[held[0]]), int(coarse_columns[held[1]]))
            if np.linalg.norm(self.center(held) - position) <= NAV_FRONTIER_PROBE_DISTANCE:
                self.frontier_probes[key] = self.frontier_probes.get(key, 0) + 1
                if self.frontier_probes[key] >= NAV_FRONTIER_PROBE_LIMIT:
                    self.deferred_frontiers[key] = self.plan_step + NAV_FRONTIER_RETRY_STEPS
                    self.frontier_probes.pop(key, None)
                    unavailable |= (coarse_rows[:, None] == key[0]) & (coarse_columns[None, :] == key[1])
                    held = None
        queue = deque([start])
        parent = {start: None}
        target = None
        reachable = []
        while queue:
            cell = queue.popleft()
            row, column = cell
            if cell != start and not known[cell] and not unavailable[cell] and (held is None or cell == held):
                target = cell
                break
            reachable.append(cell)
            for neighbor in ((row - 1, column), (row + 1, column), (row, column - 1), (row, column + 1)):
                r, c = neighbor
                if 0 <= r < self.rows and 0 <= c < self.columns and not blocked[r, c] and neighbor not in parent and (cell, neighbor) not in self.rejected_edges:
                    parent[neighbor] = cell
                    queue.append(neighbor)
        self.seeking_unknown = target is not None
        self.frontier_cell = target
        if target is None and reachable:
            if self.target not in parent or self.target == start:
                cells = np.asarray(reachable)
                scores = np.linalg.norm(cells - np.asarray(start), axis=1) / (1 + self.visits[cells[:, 0], cells[:, 1]])
                target = reachable[int(np.argmax(scores))]
                self.target = target
            else:
                target = self.target
        if target is None or target == start:
            return None
        path = []
        cell = target
        while cell != start:
            path.append(cell)
            cell = parent[cell]
        path.reverse()
        return self._follow_path(position, start, path, blocked)

    def _follow_path(self, position, start, path, blocked, align: bool = False):
        self.last_first_cell = path[0]
        waypoint = self.center(path[0])
        direction = np.asarray(path[0]) - np.asarray(start)
        lateral_axis = 0 if direction[0] else 1
        samples = position + np.linspace(0.0, 1.0, 12)[:, None] * (waypoint - position)
        indices = np.clip((samples[:, ::-1] / self.resolution).astype(int), [0, 0], [self.rows - 1, self.columns - 1])
        needs_alignment = bool(blocked[indices[:, 0], indices[:, 1]].any())
        if align and needs_alignment and abs(position[lateral_axis] - waypoint[lateral_axis]) > 0.06:
            aligned = position.copy()
            aligned[lateral_axis] = waypoint[lateral_axis]
            self.last_waypoint = aligned.copy()
            return aligned
        for cell in path[:12]:
            offset = np.asarray(cell) - np.asarray(start)
            if direction[0] * offset[1] != direction[1] * offset[0]:
                break
            candidate = self.center(cell)
            if np.linalg.norm(candidate - position) > 2.0:
                break
            samples = position + np.linspace(0.0, 1.0, max(12, int(np.ceil(np.linalg.norm(candidate - position) / 0.05)) + 1))[:, None] * (candidate - position)
            indices = np.clip((samples[:, ::-1] / self.resolution).astype(int), [0, 0], [self.rows - 1, self.columns - 1])
            if blocked[indices[:, 0], indices[:, 1]].any():
                break
            waypoint = candidate
        self.last_waypoint = waypoint.copy()
        return waypoint

    def waypoint_to(self, position: np.ndarray, goal: np.ndarray, points: np.ndarray, observed: np.ndarray, radius: float) -> np.ndarray | None:
        start = tuple(np.clip((position[::-1] / self.resolution).astype(int), [0, 0], [self.rows - 1, self.columns - 1]))
        target = tuple(np.clip((goal[::-1] / self.resolution).astype(int), [0, 0], [self.rows - 1, self.columns - 1]))
        if start == target:
            return np.asarray(goal).copy()
        blocked = self.blocked_map(points, radius)
        for cell in self.rejected_cells:
            blocked[cell] = True
        coarse_rows = np.minimum(observed.shape[0] - 1, ((np.arange(self.rows) + 0.5) * self.resolution / PERCEPTION_CELL_SIZE).astype(int))
        coarse_cols = np.minimum(observed.shape[1] - 1, ((np.arange(self.columns) + 0.5) * self.resolution / PERCEPTION_CELL_SIZE).astype(int))
        known = observed[coarse_rows[:, None], coarse_cols[None, :]]
        blocked[start] = False
        if blocked[target]:
            return None
        version = (len(self.rejected_cells), len(self.rejected_edges))
        if target == self._route_goal and version == self._route_version and start in self._goal_route:
            index = self._goal_route.index(start)
            path = self._goal_route[index + 1:]
            if path and not any(blocked[cell] for cell in path):
                return self._follow_path(position, start, path, blocked, align=True)
        queue = [(abs(start[0] - target[0]) + abs(start[1] - target[1]), 0, start)]
        cost = {start: 0}
        parent = {start: None}
        while queue:
            _, distance, cell = heapq.heappop(queue)
            if distance != cost[cell]:
                continue
            if cell == target:
                path = []
                while cell != start:
                    path.append(cell)
                    cell = parent[cell]
                path.reverse()
                self._goal_route = [start, *path]
                self._route_goal, self._route_version = target, version
                return self._follow_path(position, start, path, blocked, align=True)
            row, column = cell
            for neighbor in ((row - 1, column), (row + 1, column), (row, column - 1), (row, column + 1)):
                r, c = neighbor
                if not (0 <= r < self.rows and 0 <= c < self.columns) or blocked[r, c] or (cell, neighbor) in self.rejected_edges:
                    continue
                candidate = distance + (1 if known[neighbor] else 3)
                if candidate < cost.get(neighbor, float('inf')):
                    cost[neighbor] = candidate
                    parent[neighbor] = cell
                    heuristic = abs(r - target[0]) + abs(c - target[1])
                    heapq.heappush(queue, (candidate + heuristic, candidate, neighbor))
        return None

    def center(self, cell: tuple[int, int]) -> np.ndarray:
        return (np.array([cell[1], cell[0]], dtype=float) + 0.5) * self.resolution

    def alternative_goal(self, position: np.ndarray, goal: np.ndarray, points: np.ndarray, radius: float) -> np.ndarray | None:
        blocked = self.blocked_map(points, radius)
        for cell in self.rejected_cells:
            blocked[cell] = True
        start = tuple(np.clip((position[::-1] / self.resolution).astype(int), [0, 0], [self.rows - 1, self.columns - 1]))
        region = (np.asarray(goal) / PATROL_REGION_SIZE).astype(int)
        queue = deque([start])
        seen = {start}
        while queue:
            cell = queue.popleft()
            row, col = cell
            if int((row + 0.5) * self.resolution / PATROL_REGION_SIZE) == region[1] and int((col + 0.5) * self.resolution / PATROL_REGION_SIZE) == region[0]:
                return self.center(cell)
            for neighbor in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                r, c = neighbor
                if 0 <= r < self.rows and 0 <= c < self.columns and not blocked[r, c] and neighbor not in seen and (cell, neighbor) not in self.rejected_edges:
                    seen.add(neighbor)
                    queue.append(neighbor)
        return None
