import argparse
from pathlib import Path

import numpy as np

from config import SENSOR_RANGE, TIME_STEP
from integration.controlled_stepper import ControlledBoardStepper
from integration.human_detector_runtime import HumanDetectorRuntime
from simulation.board import Board
from training.generate_hard_human_dataset import controlled_action, place_hard_metals


SCENARIOS = ("none", "hard_metal_none", "weak_single", "multi", "random")


def configure(board: Board, scenario: str) -> None:
    if scenario in ("none", "hard_metal_none"):
        board.people = []
    elif scenario == "weak_single":
        board.people = board.people[:1]
        person = board.people[0]
        person.position = board.robot.position + np.array([6.0, 2.0])
        person.signal_strength *= 0.68
    elif scenario == "multi":
        board.people = board.people[:3]
        offsets = (np.array([4.0, 2.0]), np.array([-5.0, 3.0]), np.array([2.0, -7.0]))
        for person, offset in zip(board.people, offsets):
            candidate = board.robot.position + offset
            if board._is_free(candidate, 0.65):
                person.position = candidate
    if scenario == "hard_metal_none":
        place_hard_metals(board, min(10, len(board.metals)))
    board.measurements[-1] = board.sensor.measure(board)
    board.magnetic_scan = board.sensor.scan(board)


def collect(output: Path, seeds: int, steps: int, base_seed: int) -> None:
    rows = []
    for seed_offset in range(seeds):
        for scenario_index, scenario in enumerate(SCENARIOS):
            seed = base_seed + seed_offset * len(SCENARIOS) + scenario_index
            board = Board(seed)
            configure(board, scenario)
            detector = HumanDetectorRuntime()
            stepper = ControlledBoardStepper()
            confusion = np.zeros((2, 2), dtype=np.int64)
            probabilities = []
            for step in range(steps):
                result = detector.process_board(board)
                actual = int(any(np.linalg.norm(person.position - board.robot.position) <= SENSOR_RANGE for person in board.people))
                if result.status == "READY":
                    confusion[actual, result.predicted_label] += 1
                    probabilities.append(result.person_probability)
                if (seed_offset + scenario_index) % 2 == 0:
                    board.step(TIME_STEP, update_perception=False)
                else:
                    stepper.step(board, controlled_action(board, step + 1), 0.4, update_perception=False)
            tn, fp, fn, tp = confusion.ravel()
            total = int(confusion.sum())
            rows.append((seed, scenario_index, tn, fp, fn, tp, (fp + fn) / max(1, total), float(np.mean(probabilities)) if probabilities else 0.0))
        print(f"seed_groups={seed_offset + 1}/{seeds}", flush=True)
    values = np.asarray(rows, dtype=object)
    ranking = np.argsort(values[:, 6].astype(float))[::-1]
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        seeds=values[:, 0].astype(np.int64),
        scenario_indices=values[:, 1].astype(np.int64),
        scenarios=np.asarray(SCENARIOS),
        tn=values[:, 2].astype(np.int64),
        fp=values[:, 3].astype(np.int64),
        fn=values[:, 4].astype(np.int64),
        tp=values[:, 5].astype(np.int64),
        error_rates=values[:, 6].astype(np.float32),
        mean_probabilities=values[:, 7].astype(np.float32),
        hard_seeds=values[ranking[: min(30, len(ranking))], 0].astype(np.int64),
        hard_scenario_indices=values[ranking[: min(30, len(ranking))], 1].astype(np.int64),
    )
    print(f"saved={output}")
    for index in ranking[:10]:
        seed, scenario_index, tn, fp, fn, tp, error_rate, mean_probability = rows[index]
        print(f"seed={seed} scenario={SCENARIOS[scenario_index]} error={error_rate:.3f} fp={fp} fn={fn} tp={tp} tn={tn} mean_p={mean_probability:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/detection_failures_v1.npz"))
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--base-seed", type=int, default=92000)
    args = parser.parse_args()
    collect(args.output, args.seeds, args.steps, args.base_seed)


if __name__ == "__main__":
    main()
