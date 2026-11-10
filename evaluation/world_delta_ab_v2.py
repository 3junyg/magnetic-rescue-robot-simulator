import argparse

import numpy as np

from config_v2 import ENVIRONMENT_PRESETS
from web_runtime_v2 import WebSimulationSession


def run(seed: int, environment: str, enabled: bool, steps: int, threshold: float):
    session = WebSimulationSession()
    session.detector_runtime.world_delta_enabled = enabled
    session.detector_runtime.world_delta_agent.threshold = threshold
    session.reset(seed, environment)
    pings = 0
    near_pings = 0
    delta_pings = 0
    for _ in range(steps):
        if session.rescue_manager.all_rescued(session.board):
            break
        session.step()
        position = session.detector_result.estimated_position
        if position is None:
            continue
        pings += 1
        if min((np.linalg.norm(person.position - position) for person in session.board.people), default=float('inf')) <= 6.5:
            near_pings += 1
        delta_pings += int(session.detector_result.decision_mode == 'WORLD_DELTA')
    return pings, near_pings, len(session.rescue_manager.rescued_indices), delta_pings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 3, 42, 4, 8, 11])
    parser.add_argument('--steps', type=int, default=400)
    parser.add_argument('--threshold', type=float, default=6.0)
    args = parser.parse_args()
    totals = {False: np.zeros(4, dtype=int), True: np.zeros(4, dtype=int)}
    for environment in ENVIRONMENT_PRESETS:
        for seed in args.seeds:
            for enabled in (False, True):
                result = run(seed, environment, enabled, args.steps, args.threshold)
                totals[enabled] += result
                print(f'environment={environment} seed={seed} delta={enabled} pins,near,rescued,delta_pins={result}', flush=True)
    for enabled, result in totals.items():
        print(f'total delta={enabled} pins,near,rescued,delta_pins={result.tolist()}')


if __name__ == '__main__':
    main()
