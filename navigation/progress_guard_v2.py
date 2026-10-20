from collections import deque

import numpy as np

from config_v2 import NAV_CYCLE_WINDOW, NAV_CYCLE_MAX_SPAN
from config_v2 import NAV_BACKTRACK_MAX_STEPS, NAV_BACKTRACK_MAX_DISTANCE, NAV_BACKTRACK_MIN_MOVE, NAV_BACKTRACK_OPPOSITE_COS, NAV_BACKTRACK_TURN_COS


class NavigationProgressGuard:
    def __init__(self) -> None:
        self.history = deque(maxlen=NAV_CYCLE_WINDOW)
        self.interventions = 0
        self.last_reason = None
        self.clear_history()

    def clear_history(self) -> None:
        self.history.clear()
        self.motion_position = None
        self.motion_direction = None
        self.reversals = deque(maxlen=2)
        self.motion_step = 0
        self.previous_coverage = None
        self.previous_patrol_progress = None
        self.pending_replan = False

    def reset(self) -> None:
        self.clear_history()
        self.interventions = 0
        self.last_reason = None

    def _short_backtrack(self, position: np.ndarray, coverage: float, patrol_progress: tuple | None) -> bool:
        self.motion_step += 1
        if self.previous_coverage is not None and (coverage > self.previous_coverage + 1e-5 or patrol_progress != self.previous_patrol_progress):
            self.reversals.clear()
        self.previous_coverage = coverage
        self.previous_patrol_progress = patrol_progress
        if self.motion_position is None:
            self.motion_position = position.copy()
            return False
        offset = position - self.motion_position
        distance = float(np.linalg.norm(offset))
        if distance < NAV_BACKTRACK_MIN_MOVE:
            return False
        direction = offset / distance
        if self.motion_direction is not None:
            alignment = float(np.dot(direction, self.motion_direction))
            if alignment <= NAV_BACKTRACK_OPPOSITE_COS:
                self.reversals.append((self.motion_step, self.motion_position.copy()))
            elif alignment < NAV_BACKTRACK_TURN_COS:
                self.reversals.clear()
        self.motion_position = position.copy()
        self.motion_direction = direction
        if len(self.reversals) < 2:
            return False
        first, second = self.reversals
        return second[0] - first[0] <= NAV_BACKTRACK_MAX_STEPS and np.linalg.norm(second[1] - first[1]) <= NAV_BACKTRACK_MAX_DISTANCE

    def reject_backtrack(self, position: np.ndarray, heading: float) -> bool:
        if not self.reversals or self.motion_direction is None:
            return False
        step, point = self.reversals[-1]
        direction = np.array([np.cos(heading), np.sin(heading)])
        repeat = self.motion_step - step <= NAV_BACKTRACK_MAX_STEPS and np.linalg.norm(position - point) <= NAV_BACKTRACK_MAX_DISTANCE and np.dot(direction, self.motion_direction) <= NAV_BACKTRACK_OPPOSITE_COS
        if repeat:
            self.pending_replan = True
        return bool(repeat)

    def observe(self, position: np.ndarray, coverage: float, patrol_progress: tuple | None) -> bool:
        if self.pending_replan or self._short_backtrack(position, coverage, patrol_progress):
            self.clear_history()
            self.interventions += 1
            self.last_reason = 'SHORT_BACKTRACK'
            return True
        self.history.append((position.copy(), coverage, patrol_progress))
        if len(self.history) < NAV_CYCLE_WINDOW:
            return False
        positions = np.asarray([entry[0] for entry in self.history])
        span = float(np.linalg.norm(np.ptp(positions, axis=0)))
        changed = coverage > self.history[0][1] + 1e-5 or patrol_progress != self.history[0][2]
        if changed or span >= NAV_CYCLE_MAX_SPAN:
            return False
        self.clear_history()
        self.interventions += 1
        self.last_reason = 'NO_PROGRESS'
        return True
