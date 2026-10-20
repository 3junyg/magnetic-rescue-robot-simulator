import argparse

import numpy as np

from config_v2 import ENVIRONMENT_PRESETS
from evaluation.cluster_scenarios_v2 import configure_cluster_scene
from integration.candidate_verifier_runtime_v2 import CandidateVerifierRuntime
from web_runtime_v2 import WebSimulationSession


def run(seed, environment, people_mode, steps, verifier=True, model=None):
    session = WebSimulationSession()
    session.reset(seed, environment)
    configure_cluster_scene(session.board, people_mode)
    if model:
        session.detector_runtime.verifier = CandidateVerifierRuntime(model)
    session.detector_runtime.reset()
    if hasattr(session.detector_runtime, 'verifier_enabled'):
        session.detector_runtime.verifier_enabled = verifier
    session.rescue_manager.reset()
    session._update_detector()
    pins = near = delta = 0
    for _ in range(steps):
        if session.rescue_manager.all_rescued(session.board):
            break
        session.step()
        result = session.detector_result
        if result.estimated_position is None:
            continue
        pins += 1
        delta += int(result.decision_mode == 'WORLD_DELTA')
        near += int(min((np.linalg.norm(person.position - result.estimated_position) for person in session.board.people), default=float('inf')) <= 6.5)
    return pins, near, delta, len(session.rescue_manager.rescued_indices)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 3, 42])
    parser.add_argument('--steps', type=int, default=400)
    parser.add_argument('--people', choices=('none', 'mixed'), default='none')
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--model')
    args = parser.parse_args()
    totals = np.zeros(4, dtype=int)
    for environment in ENVIRONMENT_PRESETS:
        for seed in args.seeds:
            result = run(seed, environment, args.people, args.steps, not args.baseline, args.model)
            totals += result
            print(f'env={environment} seed={seed} pins,near,delta,rescued={result}', flush=True)
    print(f'total={totals.tolist()}', flush=True)


if __name__ == '__main__':
    main()
