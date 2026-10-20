import argparse
from pathlib import Path

import numpy as np

from config_v2 import ENVIRONMENT_PRESETS
from train_enhanced_noise_detector_v2 import EnhancedNoiseBoard
import training.generate_hard_human_dataset_v2 as hard_generator


SCENARIO_PATTERN = (0, 0, 0, 1, 1, 2, 2, 3, 4)


def generate_dataset(output: Path, episodes: int, steps: int, seed: int, noise_min: float, noise_max: float) -> None:
    EnhancedNoiseBoard.noise_level_min = max(0.0, min(noise_min, noise_max))
    EnhancedNoiseBoard.noise_level_max = max(EnhancedNoiseBoard.noise_level_min, max(noise_min, noise_max))
    hard_generator.Board = EnhancedNoiseBoard
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
    noise_levels = []
    for episode in range(episodes):
        scenario_count = SCENARIO_PATTERN[episode % len(SCENARIO_PATTERN)]
        motion_mode = (episode // len(SCENARIO_PATTERN)) % 4
        hard_negative = scenario_count == 0 or episode % 3 == 0
        episode_seed = seed + episode
        result = hard_generator.generate_episode(
            episode_seed,
            environments[episode % len(environments)],
            scenario_count,
            steps,
            motion_mode,
            hard_negative,
        )
        episode_scans, episode_motions, episode_labels, episode_locations, episode_highres, episode_physics, episode_centers = result
        level_rng = np.random.default_rng(episode_seed + 417731)
        noise_level = float(level_rng.uniform(EnhancedNoiseBoard.noise_level_min, EnhancedNoiseBoard.noise_level_max))
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
        noise_levels.append(noise_level)
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
        noise_levels=np.asarray(noise_levels, dtype=np.float32),
        seed=np.asarray(seed, dtype=np.int64),
    )
    label_array = np.asarray(labels)
    presence = label_array > 0
    print(f"saved={output}")
    print(f"scans={np.asarray(scans).shape}")
    print(f"presence_absent={int((~presence).sum())} presence_present={int(presence.sum())}")
    print(f"hard_negative_episodes={int(np.sum(hard_negatives))}")
    print(f"noise_range={min(noise_levels):.3f}..{max(noise_levels):.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/human_noise_robust_v1_v2.npz"))
    parser.add_argument("--episodes", type=int, default=540)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=72026)
    parser.add_argument("--noise-min", type=float, default=0.7)
    parser.add_argument("--noise-max", type=float, default=2.0)
    args = parser.parse_args()
    generate_dataset(args.output, args.episodes, args.steps, args.seed, args.noise_min, args.noise_max)


if __name__ == "__main__":
    main()
