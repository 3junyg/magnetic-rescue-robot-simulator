import argparse
from collections import deque
import json
from pathlib import Path

import numpy as np

from evaluation.patrol_coverage_v2 import known_scene
from integration.controlled_stepper_v2 import ControlledBoardStepper
from integration.coverage_board_runtime_v2 import CoverageBoardRuntime
from simulation.board_v2 import Board


def run(seed: int, patrol: bool, steps: int) -> dict:
    board = known_scene(seed, 'random') if patrol else Board(seed)
    agent = CoverageBoardRuntime()
    agent.patrol_active = patrol
    stepper = ControlledBoardStepper()
    turns = short_moves = blocked = wall_intrusions = loops = 0
    traveled = 0.0
    visited = set()
    recent = deque(maxlen=24)
    for _ in range(steps):
        before = board.robot.position.copy()
        action, coverage = agent.decide(board)
        stepper.step(board, action, 0.4)
        distance = float(np.linalg.norm(board.robot.position - before))
        traveled += distance
        turns += int(action in ('turn_left', 'turn_right'))
        short_moves += int(0.01 < distance < 0.5)
        blocked += int(action == 'forward' and distance < 0.01)
        wall_intrusions += int(not board._is_free(board.robot.position, board.robot.radius))
        visited.add(tuple((board.robot.position / 8).astype(int)))
        recent.append(board.robot.position.copy())
        loops += int(len(recent) == 24 and np.linalg.norm(np.ptp(np.asarray(recent), axis=0)) < 1.2)
    return dict(seed=seed, patrol=patrol, steps=steps, turns=turns, short_moves=short_moves, blocked=blocked, local_loop_steps=loops, wall_intrusions=wall_intrusions, distance=traveled, regions=len(visited), coverage=coverage)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=1200)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    results = []
    for patrol in (False, True):
        for seed in (11, 73, 99):
            record = run(seed, patrol, args.steps)
            results.append(record)
            print(record, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
