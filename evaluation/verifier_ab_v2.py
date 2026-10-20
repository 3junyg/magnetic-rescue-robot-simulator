import argparse
import json
from pathlib import Path

import numpy as np

from config_v2 import ENVIRONMENT_PRESETS
from evaluation.cluster_scenarios_v2 import configure_cluster_scene
from integration.candidate_verifier_runtime_v2 import CandidateVerifierRuntime
from web_runtime_v2 import WebSimulationSession


def run(seed, environment, scenario, steps, model_path=None):
    session = WebSimulationSession()
    session.reset(seed, environment)
    if scenario != 'regular':
        configure_cluster_scene(session.board, 'none' if scenario == 'cluster_none' else 'mixed')
    session.detector_runtime.verifier_enabled = model_path is not None
    if model_path:
        session.detector_runtime.verifier = CandidateVerifierRuntime(model_path)
    session.detector_runtime.reset()
    session.rescue_manager.reset()
    session._update_detector()
    pins = correct = delta = 0
    events = []
    for step in range(steps):
        if session.board.people and session.rescue_manager.all_rescued(session.board):
            break
        rescued_before = session.rescue_manager.rescued_indices.copy()
        if session.board.people:
            session.step()
        else:
            action, coverage = session.coverage_runtime.decide(session.board)
            session.coverage_ratio = coverage
            session.last_action = action
            session.stepper.step(session.board, action, 0.4)
            session._update_detector()
            session.frame += 1
        result = session.detector_result
        if result.estimated_position is None:
            continue
        position = result.estimated_position
        distance = min((float(np.linalg.norm(person.position - position)) for index, person in enumerate(session.board.people) if index not in rescued_before), default=float('inf'))
        pins += 1
        correct += int(distance <= 6.5)
        delta += int(result.decision_mode == 'WORLD_DELTA')
        events.append({'step': step, 'position': position.tolist(), 'distance': distance if np.isfinite(distance) else None, 'mode': result.decision_mode})
    return {'pins': pins, 'correct': correct, 'false': pins - correct, 'delta': delta, 'rescued': len(session.rescue_manager.rescued_indices), 'events': events}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 3, 42])
    parser.add_argument('--steps', type=int, default=400)
    parser.add_argument('--scenarios', nargs='+', choices=('regular', 'cluster_none', 'cluster_mixed'), default=['regular', 'cluster_none', 'cluster_mixed'])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    records = []
    for scenario in args.scenarios:
        totals = {False: np.zeros(4, dtype=int), True: np.zeros(4, dtype=int)}
        for env_index, environment in enumerate(ENVIRONMENT_PRESETS):
            for seed in args.seeds:
                for enabled in (False, True):
                    result = run(seed, environment, scenario, args.steps, args.model if enabled else None)
                    values = [result['pins'], result['correct'], result['false'], result['rescued']]
                    totals[enabled] += values
                    records.append({'scenario': scenario, 'environment': environment, 'seed': seed, 'verifier': enabled, **result})
                    print(f'scenario={scenario} env={env_index} seed={seed} verifier={enabled} pins,correct,false,rescued={values}', flush=True)
        for enabled, result in totals.items():
            print(f'TOTAL scenario={scenario} verifier={enabled} pins,correct,false,rescued={result.tolist()}', flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({'model': str(args.model), 'steps': args.steps, 'records': records}, indent=2, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    main()
