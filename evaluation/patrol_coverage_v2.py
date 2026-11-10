import argparse
from collections import deque
import json
from pathlib import Path

import numpy as np

from config_v2 import PATROL_REGION_SIZE
from integration.controlled_stepper_v2 import ControlledBoardStepper
from integration.coverage_board_runtime_v2 import CoverageBoardRuntime
from simulation.board_v2 import Board
from simulation.entities_v2 import MagneticScan, Obstacle
from simulation.perception_v2 import RobotPerception


class OldPatrolRuntime(CoverageBoardRuntime):
    def _patrol_action(self, board):
        return None


def known_scene(seed: int, kind: str) -> Board:
    board = Board(seed)
    board.people = []
    if kind == 'open':
        board.obstacles = []
    elif kind == 'split':
        board.obstacles = [Obstacle(58, 0, 4, 39.07), Obstacle(58, 41.67, 4, 38.33)]
    board.robot.position = np.array([20.0, 40.375]) if kind in ('open', 'split') else board.robot.position
    board.robot.path = [board.robot.position.copy()]
    board.perception = RobotPerception()
    empty = MagneticScan(np.empty((0, 2)), np.empty((0, 3)), np.empty(0))
    for y in np.arange(3.0, 80.0, 6.0):
        for x in np.arange(3.0, 120.0, 6.0):
            point = np.array([x, y])
            if board._is_free(point, board.robot.radius):
                scan = board.range_sensor.scan(point, board.obstacles)
                board.perception.update(point, empty, scan)
    board.range_scan = board.range_sensor.scan(board.robot.position, board.obstacles)
    board.magnetic_scan = board.sensor.scan(board)
    board.perception.update(board.robot.position, board.magnetic_scan, board.range_scan)
    return board


def actual_reachable_regions(board) -> set:
    resolution = 0.25
    xs = (np.arange(480) + 0.5) * resolution
    ys = (np.arange(320) + 0.5) * resolution
    radius = board.robot.radius
    blocked = (xs[None, :] <= radius) | (xs[None, :] >= 120 - radius) | (ys[:, None] <= radius) | (ys[:, None] >= 80 - radius)
    for wall in board.obstacles:
        blocked |= (xs[None, :] >= wall.x - radius) & (xs[None, :] <= wall.x + wall.width + radius) & (ys[:, None] >= wall.y - radius) & (ys[:, None] <= wall.y + wall.height + radius)
    start = tuple((board.robot.position[::-1] / resolution).astype(int))
    queue = deque([start])
    seen = {start}
    regions = set()
    while queue:
        row, col = queue.popleft()
        regions.add((int((row + 0.5) * resolution / PATROL_REGION_SIZE), int((col + 0.5) * resolution / PATROL_REGION_SIZE)))
        for r, c in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            if 0 <= r < 320 and 0 <= c < 480 and not blocked[r, c] and (r, c) not in seen:
                seen.add((r, c))
                queue.append((r, c))
    return regions


def run(seed: int, kind: str, steps: int, baseline: bool) -> dict:
    board = known_scene(seed, kind)
    expected = actual_reachable_regions(board)
    agent = OldPatrolRuntime() if baseline else CoverageBoardRuntime()
    agent.patrol_active = True
    stepper = ControlledBoardStepper()
    visited = {tuple((board.robot.position[::-1] / PATROL_REGION_SIZE).astype(int))}
    collisions = longest_stall = stall = 0
    all_visited_at = None
    right_room = False
    for step in range(steps):
        before = board.robot.position.copy()
        action, _ = agent.decide(board)
        stepper.step(board, action, 0.4)
        position = board.robot.position
        visited.add(tuple((position[::-1] / PATROL_REGION_SIZE).astype(int)))
        right_room |= bool(position[0] > 63.2)
        collisions += int(not board._is_free(position, board.robot.radius))
        stall = stall + 1 if np.linalg.norm(position - before) < 0.01 else 0
        longest_stall = max(longest_stall, stall)
        if all_visited_at is None and expected <= visited:
            all_visited_at = step + 1
    return {'seed': seed, 'kind': kind, 'baseline': baseline, 'steps': steps, 'reachable_regions': len(expected), 'visited_regions': len(expected & visited), 'missing_regions': [list(region) for region in sorted(expected - visited)], 'region_coverage': len(expected & visited) / max(1, len(expected)), 'all_visited_step': all_visited_at, 'right_room_visited': right_room, 'collisions': collisions, 'longest_stall': longest_stall, 'scheduler': agent.patrol_scheduler.status()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=2200)
    parser.add_argument('--seeds', nargs='+', type=int, default=[11, 73, 99])
    parser.add_argument('--kinds', nargs='+', choices=('open', 'split', 'random'), default=['open', 'split', 'random'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--only-new', action='store_true')
    args = parser.parse_args()
    records = []
    for kind in args.kinds:
        for seed in args.seeds:
            for baseline in ((False,) if args.only_new else (True, False)):
                result = run(seed, kind, args.steps, baseline)
                records.append(result)
                print(result, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
