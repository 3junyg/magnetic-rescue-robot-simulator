import argparse
from pathlib import Path

import numpy as np

from config import ENVIRONMENT_PRESETS, HISTORY_LENGTH, SENSOR_RANGE, TIME_STEP
from integration.controlled_stepper import ControlledBoardStepper
from ml.features import extract_scan_features, local_motion
from simulation.board import Board
from simulation.metal_field_estimator import MetalFieldEstimator
from training.generate_human_dataset import people_in_range, people_location_grid, residual_features


def place_people(board: Board, count: int) -> None:
    board.people = board.people[:count]
    placed = []
    for person in board.people:
        person.signal_strength *= board.rng.uniform(0.68, 1.05)
        for _ in range(500):
            angle = board.rng.uniform(-np.pi, np.pi)
            distance = board.rng.uniform(SENSOR_RANGE * 0.18, SENSOR_RANGE * 0.94)
            position = board.robot.position + distance * np.array([np.cos(angle), np.sin(angle)])
            if board._is_free(position, 0.65) and all(np.linalg.norm(position - other) >= 2.0 for other in placed):
                person.position = position
                person.heading = board.rng.uniform(-np.pi, np.pi)
                speed = np.linalg.norm(person.velocity)
                person.velocity = speed * np.array([np.cos(person.heading), np.sin(person.heading)])
                placed.append(position.copy())
                break


def place_hard_metals(board: Board, count: int) -> None:
    for metal in board.metals[:count]:
        for _ in range(300):
            angle = board.rng.uniform(-np.pi, np.pi)
            distance = board.rng.uniform(3.0, SENSOR_RANGE * 0.95)
            position = board.robot.position + distance * np.array([np.cos(angle), np.sin(angle)])
            if board._is_free(position, 0.5):
                metal.position = position
                metal.strength = board.rng.uniform(30.0, 52.0)
                metal.size = board.rng.uniform(1.5, 4.5)
                break


def stationary_step(board: Board, dt: float) -> None:
    board.time += dt
    board._update_people(dt)
    board.robot.moving = False
    board.measurements.append(board.sensor.measure(board))
    board.magnetic_scan = board.sensor.scan(board)
    if len(board.measurements) > HISTORY_LENGTH:
        board.measurements.pop(0)


def controlled_action(board: Board, step: int) -> str:
    if step % 17 == 0:
        return "stop"
    if step % 11 in (0, 1):
        return "turn_left"
    if step % 13 in (0, 1):
        return "turn_right"
    value = board.rng.random()
    if value < 0.72:
        return "forward"
    return "turn_left" if value < 0.86 else "turn_right"


def people_location_grid_highres(board: Board) -> np.ndarray:
    grid = np.zeros((8, 16), dtype=np.float32)
    cosine = np.cos(board.robot.heading)
    sine = np.sin(board.robot.heading)
    for person in board.people:
        relative = person.position - board.robot.position
        distance = float(np.linalg.norm(relative))
        if distance > SENSOR_RANGE:
            continue
        local_x = cosine * relative[0] + sine * relative[1]
        local_y = -sine * relative[0] + cosine * relative[1]
        angle = (np.arctan2(local_y, local_x) + 2.0 * np.pi) % (2.0 * np.pi)
        radial = min(7, int(distance / SENSOR_RANGE * 8))
        angular = min(15, int(angle / (2.0 * np.pi) * 16))
        grid[radial, angular] = 1.0
    return grid


def generate_episode(seed: int, environment: str, scenario_count: int, steps: int, motion_mode: int, hard_negative: bool):
    board = Board(seed, environment)
    place_people(board, scenario_count)
    if hard_negative:
        place_hard_metals(board, min(8, len(board.metals)))
    board.measurements[-1] = board.sensor.measure(board)
    board.magnetic_scan = board.sensor.scan(board)
    estimator = MetalFieldEstimator()
    stepper = ControlledBoardStepper()
    scans = []
    physics = []
    centers = []
    motions = [np.zeros(3, dtype=np.float32)]
    labels = []
    locations = []
    highres_locations = []

    def save_state() -> None:
        scans.append(extract_scan_features(board.magnetic_scan, board.robot.position, board.robot.heading))
        field_features, center = residual_features(board, estimator)
        physics.append(field_features)
        centers.append(center)
        labels.append(people_in_range(board))
        locations.append(people_location_grid(board))
        highres_locations.append(people_location_grid_highres(board))

    save_state()
    for step in range(1, steps):
        previous_position = board.robot.position.copy()
        previous_heading = board.robot.heading
        if motion_mode == 0:
            board.step(TIME_STEP, update_perception=False)
        elif motion_mode == 1:
            stepper.step(board, controlled_action(board, step), 0.4, update_perception=False)
        elif motion_mode == 2:
            stationary_step(board, TIME_STEP)
        elif step % 4 == 0:
            stationary_step(board, TIME_STEP)
        else:
            stepper.step(board, controlled_action(board, step), 0.4, update_perception=False)
        motions.append(local_motion(previous_position, board.robot.position, previous_heading, board.robot.heading))
        save_state()
    return (
        np.asarray(scans, dtype=np.float32),
        np.asarray(motions, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(locations, dtype=np.float32),
        np.asarray(highres_locations, dtype=np.float32),
        np.asarray(physics, dtype=np.float32),
        np.asarray(centers, dtype=np.float32),
    )


def generate_dataset(output: Path, episodes: int, steps: int, seed: int, failure_file: Path | None = None) -> None:
    environments = list(ENVIRONMENT_PRESETS)
    scans = []
    motions = []
    labels = []
    locations = []
    highres_locations = []
    physics = []
    centers = []
    scenario_counts = []
    episode_ids = []
    motion_modes = []
    hard_negatives = []
    failure_seeds = np.empty(0, dtype=np.int64)
    failure_scenarios = np.empty(0, dtype=np.int64)
    if failure_file is not None and failure_file.exists():
        failures = np.load(failure_file)
        failure_seeds = failures["hard_seeds"]
        failure_scenarios = failures["hard_scenario_indices"]
    for episode in range(episodes):
        if episode < len(failure_seeds):
            episode_seed = int(failure_seeds[episode])
            failure_scenario = int(failure_scenarios[episode])
            scenario_count = (0, 0, 1, 3, 2)[failure_scenario]
            hard_negative = failure_scenario == 1
        else:
            episode_seed = seed + episode
            scenario_class = episode % 4
            scenario_count = scenario_class if scenario_class < 3 else 3 + episode % 2
            hard_negative = scenario_count == 0 or episode % 5 == 0
        motion_mode = (episode // 4) % 4
        result = generate_episode(episode_seed, environments[episode % len(environments)], scenario_count, steps, motion_mode, hard_negative)
        episode_scans, episode_motions, episode_labels, episode_locations, episode_highres, episode_physics, episode_centers = result
        scans.append(episode_scans)
        motions.append(episode_motions)
        labels.append(episode_labels)
        locations.append(episode_locations)
        highres_locations.append(episode_highres)
        physics.append(episode_physics)
        centers.append(episode_centers)
        scenario_counts.append(scenario_count)
        episode_ids.append(episode)
        motion_modes.append(motion_mode)
        hard_negatives.append(hard_negative)
        if (episode + 1) % 20 == 0:
            print(f"episodes={episode + 1}/{episodes}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        scans=np.asarray(scans, dtype=np.float32),
        motions=np.asarray(motions, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        locations=np.asarray(locations, dtype=np.float32),
        locations_highres=np.asarray(highres_locations, dtype=np.float32),
        physics_features=np.asarray(physics, dtype=np.float32),
        center_residuals=np.asarray(centers, dtype=np.float32),
        episode_ids=np.asarray(episode_ids, dtype=np.int64),
        scenario_counts=np.asarray(scenario_counts, dtype=np.int64),
        motion_modes=np.asarray(motion_modes, dtype=np.int64),
        hard_negatives=np.asarray(hard_negatives, dtype=bool),
        seed=np.asarray(seed, dtype=np.int64),
    )
    values, counts = np.unique(np.clip(np.asarray(labels), 0, 3), return_counts=True)
    print(f"saved={output}")
    print(f"scans={np.asarray(scans).shape}")
    print("class_counts=" + ", ".join(f"{value}:{count}" for value, count in zip(values, counts)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/human_transition_dataset_v2.npz"))
    parser.add_argument("--episodes", type=int, default=480)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=32026)
    parser.add_argument("--failure-file", type=Path)
    args = parser.parse_args()
    generate_dataset(args.output, args.episodes, args.steps, args.seed, args.failure_file)


if __name__ == "__main__":
    main()
