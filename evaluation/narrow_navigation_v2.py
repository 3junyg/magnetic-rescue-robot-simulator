import argparse
import json
from pathlib import Path

import numpy as np

from config_v2 import ENVIRONMENT_PRESETS
from integration.controlled_stepper_v2 import ControlledBoardStepper
from integration.coverage_board_runtime_v2 import CoverageBoardRuntime
from simulation.board_v2 import Board
from evaluation.narrow_scenes_v2 import create_narrow_scene


def run(seed: int, width: float, steps: int, corner: bool = False) -> dict:
    board = create_narrow_scene(seed, width, 'corner' if corner else 'straight')
    agent, stepper = CoverageBoardRuntime(), ControlledBoardStepper()
    failures = collisions = 0
    crossed = None
    positions = []
    for step in range(steps):
        action, _ = agent.decide(board)
        stepper.step(board, action, 0.4)
        failures += int(action == 'forward' and not board.robot.moving)
        collisions += int(not board._is_free(board.robot.position, board.robot.radius))
        positions.append(board.robot.position.copy())
        if board.robot.position[0] > 64:
            crossed = step + 1
            break
    recent = np.asarray(positions[-60:])
    spread = float(np.linalg.norm(np.ptp(recent, axis=0)))
    return {'seed': seed, 'width': width, 'corner': corner, 'crossed_step': crossed, 'collisions': collisions, 'blocked_forward': failures, 'recent_spread': spread, 'position': board.robot.position.tolist()}


def random_run(seed: int, environment: str, steps: int) -> dict:
    board = Board(seed, environment)
    agent, stepper = CoverageBoardRuntime(), ControlledBoardStepper()
    collisions = 0
    longest_stall = stall = 0
    positions = []
    for _ in range(steps):
        before = board.robot.position.copy()
        action, coverage = agent.decide(board)
        stepper.step(board, action, 0.4)
        stall = stall + 1 if np.linalg.norm(before - board.robot.position) < 0.05 else 0
        longest_stall = max(longest_stall, stall)
        collisions += int(not board._is_free(board.robot.position, board.robot.radius))
        positions.append(board.robot.position.copy())
    return {'seed': seed, 'environment': environment, 'steps': steps, 'collisions': collisions, 'longest_no_translation': longest_stall, 'coverage': coverage}


def bend_run(seed: int, width: float, steps: int) -> dict:
    board = create_narrow_scene(seed, width, 'bend')
    agent, stepper = CoverageBoardRuntime(), ControlledBoardStepper()
    collisions = failures = 0
    reached = None
    for step in range(steps):
        action, _ = agent.decide(board)
        stepper.step(board, action, 0.4)
        collisions += int(not board._is_free(board.robot.position, board.robot.radius))
        failures += int(action == 'forward' and not board.robot.moving)
        if board.robot.position[1] > 62:
            reached = step + 1
            break
    return {'seed': seed, 'width': width, 'bend': True, 'reached_step': reached, 'collisions': collisions, 'blocked_forward': failures, 'position': board.robot.position.tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=250)
    parser.add_argument('--random', action='store_true')
    parser.add_argument('--bend', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    records = []
    if args.bend:
        for width in (2.6, 2.8, 3.2, 4.0):
            for seed in (11, 73, 99):
                record = bend_run(seed, width, args.steps)
                records.append(record)
                print(record, flush=True)
    elif args.random:
        for environment in ENVIRONMENT_PRESETS:
            for seed in (11, 73, 99):
                record = random_run(seed, environment, args.steps)
                records.append(record)
                print(record, flush=True)
    else:
        for width in (2.0, 2.6, 2.8, 3.2, 4.0):
            for corner in (False, True):
                for seed in (11, 73, 99):
                    record = run(seed, width, args.steps, corner)
                    records.append(record)
                    print(record, flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(records, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
