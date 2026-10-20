import argparse
from pathlib import Path

import numpy as np

from config_v2 import ENVIRONMENT_PRESETS
from evaluation.cluster_scenarios_v2 import configure_cluster_scene
from integration.controlled_stepper_v2 import ControlledBoardStepper
from integration.coverage_board_runtime_v2 import CoverageBoardRuntime
from integration.human_detector_runtime_v2 import HumanDetectorRuntime
from integration.world_delta_ping_v2 import WorldDeltaPingAgent
from integration.candidate_verifier_runtime_v2 import CandidateVerifierRuntime
from ml.candidate_patch_v2 import MagneticPatchHistory
from simulation.board_v2 import Board
from simulation.observations_v2 import observation_from_board


def candidate_points(previous, scan, robot_position, detector, result):
    previous_values = {tuple(point): value for point, value in zip(previous.positions, previous.vectors)}
    overlap_indices = []
    differences = []
    for index, (point, vector) in enumerate(zip(scan.positions, scan.vectors)):
        prior = previous_values.get(tuple(point))
        if prior is not None:
            overlap_indices.append(index)
            differences.append(vector - prior)
    ranked = []
    if differences:
        differences = np.asarray(differences)
        differences -= np.median(differences, axis=0)
        indices = np.asarray(overlap_indices)
        scores = np.linalg.norm(differences, axis=1)
        near = np.linalg.norm(scan.positions[indices] - robot_position, axis=1) <= 10.0
        for index in np.argsort(np.where(near, scores, -1.0))[::-1]:
            if scores[index] < 0.8 or not near[index]:
                continue
            point = scan.positions[indices[index]]
            if all(np.linalg.norm(point - prior) >= 5.0 for prior in ranked):
                ranked.append(point.copy())
            if len(ranked) >= 2:
                break
    if result.location_probabilities is not None:
        cell = int(np.argmax(result.location_probabilities))
        expected = detector.estimator.predict_at(scan.positions)
        ranked.append(detector._location_from_scan(cell, scan.positions, scan.vectors - expected, robot_position, detector.previous_heading))
    anomaly = np.linalg.norm(scan.vectors - np.median(scan.vectors, axis=0), axis=1)
    if len(anomaly):
        ranked.append(scan.positions[int(np.argmax(anomaly))].copy())
    if result.estimated_position is not None:
        ranked.insert(0, result.estimated_position.copy())
    unique = []
    for point in ranked:
        if all(np.linalg.norm(point - saved) >= 3.0 for saved in unique):
            unique.append(point)
    return unique[:4]


def generate(args):
    environments = list(ENVIRONMENT_PRESETS)
    patches, motions, labels, groups, scenarios = [], [], [], [], []
    mining_model = CandidateVerifierRuntime(args.mine_model) if args.mine_model else None
    for episode in range(args.episodes):
        seed = args.seed + episode
        environment = environments[episode % len(environments)]
        scenario = (episode // len(environments)) % 6
        board = Board(seed, environment)
        if scenario in (0, 2, 3, 5):
            if scenario == 2:
                board.people = board.people[:1]
            elif scenario == 3:
                board.people = board.people[:3]
            configure_cluster_scene(board, 'none' if scenario == 0 else 'mixed')
        elif scenario == 1:
            board.people = []
            board.measurements[-1] = board.sensor.measure(board)
            board.magnetic_scan = board.sensor.scan(board)
        detector = HumanDetectorRuntime(add_markers=False, verifier_enabled=False)
        detector.world_delta_enabled = False
        if hasattr(detector, 'verifier_enabled'):
            detector.verifier_enabled = False
        coverage = CoverageBoardRuntime()
        stepper = ControlledBoardStepper()
        history = MagneticPatchHistory()
        delta_agent = WorldDeltaPingAgent()
        sample_rng = np.random.default_rng(seed + 7900)
        if mining_model is not None:
            mining_model.history = history
        observation = observation_from_board(board)
        history.observe(observation)
        detector.process_observation(observation)
        delta_agent.process(board.magnetic_scan, board.robot.position, board.time, [])
        mode = int(np.random.default_rng(seed + 13).integers(0, 3))
        if args.coverage_only:
            mode = 0
        before = len(labels)
        for step in range(args.steps):
            previous = board.magnetic_scan
            if mode == 0:
                action, _ = coverage.decide(board)
            elif mode == 1:
                action = ('forward', 'forward', 'turn_left', 'forward', 'turn_right', 'stop')[step % 6]
            else:
                action, _ = coverage.decide(board)
                if step % 14 in (0, 1, 2):
                    action = 'stop'
            stepper.step(board, action, 0.4)
            observation = observation_from_board(board)
            history.observe(observation)
            result = detector.process_observation(observation)
            delta_decision = delta_agent.process(board.magnetic_scan, board.robot.position, board.time, [])
            if step < 2 or (step % 3 != 0 and result.estimated_position is None and delta_decision is None):
                continue
            points = candidate_points(previous, board.magnetic_scan, board.robot.position, detector, result)
            if delta_decision is not None:
                points.insert(0, delta_decision.position)
            for point in points:
                distance = min((float(np.linalg.norm(person.position - point)) for person in board.people), default=float('inf'))
                if 6.5 < distance < 9.0:
                    continue
                if mining_model is not None and distance > 9.0:
                    score = float(mining_model.scores([point])[0])
                    if score < 0.15 and sample_rng.random() > 0.25:
                        continue
                patch, motion = history.at(point)
                patches.append(patch.astype(np.float16))
                motions.append(motion.astype(np.float16))
                labels.append(int(distance <= 6.5))
                groups.append(seed)
                scenarios.append(scenario)
        print(f'episode={episode + 1}/{args.episodes} seed={seed} scenario={scenario} samples={len(labels) - before}', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, patches=np.asarray(patches), motion=np.asarray(motions), labels=np.asarray(labels, dtype=np.uint8), episode_ids=np.asarray(groups), scenario_ids=np.asarray(scenarios, dtype=np.uint8), seed=args.seed)
    print(f'output={args.output} samples={len(labels)} positive={sum(labels)} negative={len(labels) - sum(labels)}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=108)
    parser.add_argument('--steps', type=int, default=150)
    parser.add_argument('--seed', type=int, default=20000)
    parser.add_argument('--output', type=Path, default=Path('data/candidate_clusters_v2.npz'))
    parser.add_argument('--mine-model', type=Path)
    parser.add_argument('--coverage-only', action='store_true')
    generate(parser.parse_args())


if __name__ == '__main__':
    main()
