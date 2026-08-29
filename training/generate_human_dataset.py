import argparse
from pathlib import Path

import numpy as np

from config import ENVIRONMENT_PRESETS, SENSOR_RANGE, TIME_STEP
from ml.features import ANGULAR_BINS, RADIAL_BINS, extract_scan_features, extract_vector_features, local_motion
from simulation.board import Board
from simulation.metal_field_estimator import MetalFieldEstimator


def place_people(board: Board, count: int) -> None:
    board.people = board.people[:count]
    placed = []
    for person in board.people:
        for _ in range(500):
            angle = board.rng.uniform(-np.pi, np.pi)
            distance = board.rng.uniform(SENSOR_RANGE * 0.25, SENSOR_RANGE * 0.78)
            position = board.robot.position + distance * np.array([np.cos(angle), np.sin(angle)])
            if board._is_free(position, 0.65) and all(np.linalg.norm(position - other) >= 2.0 for other in placed):
                person.position = position
                person.heading = board.rng.uniform(-np.pi, np.pi)
                speed = np.linalg.norm(person.velocity)
                person.velocity = speed * np.array([np.cos(person.heading), np.sin(person.heading)])
                placed.append(position.copy())
                break


def people_in_range(board: Board) -> int:
    return sum(np.linalg.norm(person.position - board.robot.position) <= SENSOR_RANGE for person in board.people)


def people_location_grid(board: Board) -> np.ndarray:
    grid = np.zeros((RADIAL_BINS, ANGULAR_BINS), dtype=np.float32)
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
        radial = min(RADIAL_BINS - 1, int(distance / SENSOR_RANGE * RADIAL_BINS))
        angular = min(ANGULAR_BINS - 1, int(angle / (2.0 * np.pi) * ANGULAR_BINS))
        grid[radial, angular] = 1.0
    return grid


def residual_features(board: Board, estimator: MetalFieldEstimator) -> tuple[np.ndarray, np.ndarray]:
    scan = board.magnetic_scan
    expected = estimator.predict_at(scan.positions)
    residual_vectors = scan.vectors - expected
    features = extract_vector_features(
        scan.positions,
        residual_vectors,
        board.robot.position,
        board.robot.heading,
        reference=np.zeros(3, dtype=np.float64),
    )
    expected_center = estimator.predict_at(board.robot.position.reshape(1, 2))[0]
    measurement = board.measurements[-1]
    center_residual = np.asarray(
        [measurement.bx, measurement.by, measurement.bz],
        dtype=np.float32,
    ) - expected_center.astype(np.float32)
    estimator.update(scan)
    return features, center_residual


def generate_episode(seed: int, environment: str, scenario_count: int, steps: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    board = Board(seed, environment)
    place_people(board, scenario_count)
    board.measurements[-1] = board.sensor.measure(board)
    board.magnetic_scan = board.sensor.scan(board)
    estimator = MetalFieldEstimator()
    scans = []
    physics = []
    center_residuals = []
    motions = [np.zeros(3, dtype=np.float32)]
    counts = []
    locations = []
    scans.append(extract_scan_features(board.magnetic_scan, board.robot.position, board.robot.heading))
    episode_physics, episode_center_residual = residual_features(board, estimator)
    physics.append(episode_physics)
    center_residuals.append(episode_center_residual)
    counts.append(people_in_range(board))
    locations.append(people_location_grid(board))
    for _ in range(steps - 1):
        previous_position = board.robot.position.copy()
        previous_heading = board.robot.heading
        board.step(TIME_STEP, update_perception=False)
        motions.append(local_motion(previous_position, board.robot.position, previous_heading, board.robot.heading))
        scans.append(extract_scan_features(board.magnetic_scan, board.robot.position, board.robot.heading))
        episode_physics, episode_center_residual = residual_features(board, estimator)
        physics.append(episode_physics)
        center_residuals.append(episode_center_residual)
        counts.append(people_in_range(board))
        locations.append(people_location_grid(board))
    return (
        np.asarray(scans, dtype=np.float32),
        np.asarray(motions, dtype=np.float32),
        np.asarray(counts, dtype=np.int64),
        np.asarray(locations, dtype=np.float32),
        np.asarray(physics, dtype=np.float32),
        np.asarray(center_residuals, dtype=np.float32),
    )


def generate_dataset(output: Path, episodes: int, steps: int, seed: int) -> None:
    environments = list(ENVIRONMENT_PRESETS)
    scans = []
    motions = []
    labels = []
    scenario_counts = []
    episode_ids = []
    location_labels = []
    physics_features = []
    center_residuals = []
    for episode in range(episodes):
        scenario_class = episode % 4
        scenario_count = scenario_class if scenario_class < 3 else 3 + (episode % 2)
        environment = environments[episode % len(environments)]
        episode_scans, episode_motions, episode_labels, episode_locations, episode_physics, episode_center_residuals = generate_episode(seed + episode, environment, scenario_count, steps)
        scans.append(episode_scans)
        motions.append(episode_motions)
        labels.append(episode_labels)
        location_labels.append(episode_locations)
        physics_features.append(episode_physics)
        center_residuals.append(episode_center_residuals)
        scenario_counts.append(scenario_count)
        episode_ids.append(episode)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        scans=np.asarray(scans, dtype=np.float32),
        motions=np.asarray(motions, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        locations=np.asarray(location_labels, dtype=np.float32),
        physics_features=np.asarray(physics_features, dtype=np.float32),
        center_residuals=np.asarray(center_residuals, dtype=np.float32),
        episode_ids=np.asarray(episode_ids, dtype=np.int64),
        scenario_counts=np.asarray(scenario_counts, dtype=np.int64),
        seed=np.asarray(seed, dtype=np.int64),
        time_step=np.asarray(TIME_STEP, dtype=np.float32),
    )
    values, counts_per_class = np.unique(np.clip(np.asarray(labels), 0, 3), return_counts=True)
    print(f"saved={output}")
    print(f"scans={np.asarray(scans).shape}")
    print(f"motions={np.asarray(motions).shape}")
    print(f"labels={np.asarray(labels).shape}")
    print(f"locations={np.asarray(location_labels).shape}")
    print(f"physics_features={np.asarray(physics_features).shape}")
    print(f"center_residuals={np.asarray(center_residuals).shape}")
    print("class_counts=" + ", ".join(f"{value}:{count}" for value, count in zip(values, counts_per_class)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/human_transition_dataset.npz"))
    parser.add_argument("--episodes", type=int, default=160)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    generate_dataset(args.output, args.episodes, args.steps, args.seed)


if __name__ == "__main__":
    main()
