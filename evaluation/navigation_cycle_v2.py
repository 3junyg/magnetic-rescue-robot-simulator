import argparse
from collections import deque
import json
from pathlib import Path

import numpy as np

from config_v2 import ENVIRONMENT_PRESETS
from integration.controlled_stepper_v2 import ControlledBoardStepper
from integration.coverage_board_runtime_v2 import CoverageBoardRuntime
from simulation.board_v2 import Board


def run(seed: int, environment: str, steps: int) -> dict:
    board = Board(seed, environment)
    agent, stepper = CoverageBoardRuntime(), ControlledBoardStepper()
    trace = []
    history = deque(maxlen=120)
    longest_stagnation = stagnation = repeated_frames = collisions = 0
    best = float(board.perception.observed.mean())
    cancelled_backtracks = 0
    for index in range(steps):
        action, coverage = agent.decide(board)
        cancelled = bool(action == 'stop' and agent.cycle_guard.pending_replan)
        cancelled_backtracks += int(cancelled)
        planner = agent.active_planner
        stepper.step(board, action, 0.4)
        point = board.robot.position.copy()
        history.append((point, coverage))
        if coverage > best + 1e-5:
            best, stagnation = coverage, 0
        else:
            stagnation += 1
        longest_stagnation = max(longest_stagnation, stagnation)
        if len(history) == 120:
            span = float(np.linalg.norm(np.ptp(np.asarray([entry[0] for entry in history]), axis=0)))
            repeated_frames += int(span < 15 and coverage - history[0][1] < 1e-5)
        collisions += int(not board._is_free(point, board.robot.radius))
        trace.append(dict(step=index, position=point.tolist(), heading=board.robot.heading, action=str(action), mode=agent.control_mode, coverage=coverage, waypoint=None if planner.last_waypoint is None else planner.last_waypoint.tolist(), seeking_unknown=planner.seeking_unknown, patrol=agent.patrol_active, cancelled_backtrack=cancelled))
        if index % 200 == 199:
            print(seed, environment, index + 1, 'coverage', round(coverage, 4), 'stagnation', stagnation, 'position', point.round(2), 'mode', agent.control_mode, flush=True)
    return dict(seed=seed, environment=environment, steps=steps, coverage=best, longest_stagnation=longest_stagnation, repeated_frames=repeated_frames, wall_intrusions=collisions, cancelled_backtracks=cancelled_backtracks, guard_interventions=agent.cycle_guard.interventions, trace=trace)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 73, 99])
    parser.add_argument('--environments', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    results = [run(seed, list(ENVIRONMENT_PRESETS)[env], args.steps) for env in args.environments for seed in args.seeds]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
