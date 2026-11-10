import argparse
import json
from pathlib import Path
import time

import numpy as np

from config_v2 import ENVIRONMENT_PRESETS
from integration.controlled_stepper_v2 import ControlledBoardStepper
from integration.coverage_board_runtime_v2 import CoverageBoardRuntime
from simulation.board_v2 import Board


def run(seed: int, environment: str, steps: int) -> dict:
    board = Board(seed, environment)
    agent, stepper = CoverageBoardRuntime(), ControlledBoardStepper()
    turns = blocked = short_moves = collisions = reversals = 0
    stall = max_stall = 0
    traveled = 0.0
    previous_turn = None
    controls = {}
    started = time.perf_counter()
    for _ in range(steps):
        before = board.robot.position.copy()
        action, coverage = agent.decide(board)
        controls[getattr(agent, 'control_mode', 'OLD')] = controls.get(getattr(agent, 'control_mode', 'OLD'), 0) + 1
        stepper.step(board, action, 0.4)
        distance = float(np.linalg.norm(board.robot.position - before))
        traveled += distance
        turns += int(action in ('turn_left', 'turn_right'))
        blocked += int(action == 'forward' and distance < 0.01)
        short_moves += int(0.01 <= distance < 0.5)
        collisions += int(not board._is_free(board.robot.position, board.robot.radius))
        stall = stall + 1 if distance < 0.01 else 0
        max_stall = max(max_stall, stall)
        if action in ('turn_left', 'turn_right'):
            reversals += int(previous_turn is not None and action != previous_turn)
            previous_turn = action
        else:
            previous_turn = None
    return {'seed': seed, 'environment': environment, 'steps': steps, 'turns': turns, 'blocked': blocked, 'short_moves': short_moves, 'turn_reversals': reversals, 'max_stall': max_stall, 'distance': traveled, 'coverage': coverage, 'collisions': collisions, 'controls': controls, 'seconds': time.perf_counter() - started}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=600)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    records = []
    for environment in ENVIRONMENT_PRESETS:
        for seed in (11, 73, 99):
            result = run(seed, environment, args.steps)
            records.append(result)
            print(result, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    main()
