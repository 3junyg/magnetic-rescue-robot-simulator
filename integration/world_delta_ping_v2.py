import numpy as np

from integration.ping_decision_agent_v2 import PingDecision


class WorldDeltaPingAgent:
    def __init__(self, threshold: float = 6.0) -> None:
        self.threshold = threshold
        self.previous_scan = None
        self.pending_position = None
        self.pending_hits = 0
        self.last_ping_time = -float('inf')
        self.pins: list[np.ndarray] = []

    def reset(self) -> None:
        self.previous_scan = None
        self.pending_position = None
        self.pending_hits = 0
        self.last_ping_time = -float('inf')
        self.pins.clear()

    def process(self, scan, robot_position, timestamp: float, existing_pins, verify=None) -> PingDecision | None:
        previous = self.previous_scan
        self.previous_scan = scan
        if previous is None:
            return None
        old = {tuple(point): vector for point, vector in zip(previous.positions, previous.vectors)}
        points = []
        differences = []
        for point, vector in zip(scan.positions, scan.vectors):
            prior = old.get(tuple(point))
            if prior is not None:
                points.append(point)
                differences.append(vector - prior)
        if not points:
            self.pending_hits = 0
            return None
        differences = np.asarray(differences)
        differences -= np.median(differences, axis=0)
        points = np.asarray(points)
        active = np.linalg.norm(points - robot_position, axis=1) <= 8.0
        if not active.any():
            self.pending_hits = 0
            return None
        strengths = np.linalg.norm(differences[active], axis=1)
        index = int(np.argmax(strengths))
        point = np.asarray(points[active][index], dtype=float)
        strength = float(strengths[index])
        if strength < self.threshold:
            self.pending_hits = 0
            return None
        if self.pending_position is not None and np.linalg.norm(point - self.pending_position) <= 4.0:
            self.pending_hits += 1
            self.pending_position = 0.5 * (self.pending_position + point)
        else:
            self.pending_position = point
            self.pending_hits = 1
        if self.pending_hits < 2 or timestamp - self.last_ping_time < 8.0:
            return None
        if any(np.linalg.norm(self.pending_position - pin) <= 10.0 for pin in (*self.pins, *existing_pins)):
            return None
        if verify is not None and not verify(self.pending_position):
            return None
        result = PingDecision(self.pending_position.copy(), min(1.0, strength / (self.threshold * 2.0)), 'WORLD_DELTA')
        self.pins.append(result.position.copy())
        self.last_ping_time = timestamp
        self.pending_hits = 0
        return result
