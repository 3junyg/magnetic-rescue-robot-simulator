import argparse

import numpy as np

from config_v2 import ENVIRONMENT_PRESETS
from web_runtime_v2 import WebSimulationSession


def run(seed: int, environment: str, enabled: bool, steps: int):
    session = WebSimulationSession()
    session.detector_runtime.world_delta_enabled = enabled
    session.reset(seed, environment)
    active = {}
    completed = set()
    encountered = 0
    rescued_on_first_pass = 0
    for _ in range(steps):
        if session.rescue_manager.all_rescued(session.board):
            break
        session.step()
        if session.detector_result.status != 'READY':
            continue
        for index, person in enumerate(session.board.people):
            if index in completed:
                continue
            distance = float(np.linalg.norm(person.position - session.board.robot.position))
            if distance <= 8.0 and index not in active:
                active[index] = [0, False]
            if index in active:
                active[index][0] += 1
                active[index][1] |= index in session.rescue_manager.rescued_indices
                if distance > 8.0 or active[index][1]:
                    length, successful = active.pop(index)
                    if length >= 2:
                        encountered += 1
                        rescued_on_first_pass += int(successful)
                        completed.add(index)
    for index, (length, successful) in active.items():
        if length >= 2 and index not in completed:
            encountered += 1
            rescued_on_first_pass += int(successful)
    return encountered, rescued_on_first_pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 3, 42, 4, 8, 11])
    parser.add_argument('--steps', type=int, default=400)
    args = parser.parse_args()
    totals = {False: np.zeros(2, dtype=int), True: np.zeros(2, dtype=int)}
    for environment in ENVIRONMENT_PRESETS:
        for seed in args.seeds:
            for enabled in (False, True):
                result = run(seed, environment, enabled, args.steps)
                totals[enabled] += result
                print(f'environment={environment} seed={seed} delta={enabled} encounters,first_pass_rescued={result}', flush=True)
    for enabled, values in totals.items():
        print(f'total delta={enabled} encounters,first_pass_rescued={values.tolist()}')


if __name__ == '__main__':
    main()
