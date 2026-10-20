import argparse
from pathlib import Path

import numpy as np


ARRAY_KEYS = ("scans", "motions", "labels", "locations", "physics_features", "center_residuals", "scenario_counts")


def combine(inputs: list[Path], output: Path) -> None:
    datasets = [np.load(path) for path in inputs]
    steps = {data["scans"].shape[1] for data in datasets}
    if len(steps) != 1:
        raise ValueError("모든 데이터셋의 timestep 수가 같아야 합니다.")
    combined = {key: np.concatenate([data[key] for data in datasets], axis=0) for key in ARRAY_KEYS}
    episode_count = combined["scans"].shape[0]
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        **combined,
        episode_ids=np.arange(episode_count, dtype=np.int64),
    )
    values, counts = np.unique(np.clip(combined["labels"], 0, 3), return_counts=True)
    print(f"saved={output}")
    print(f"episodes={episode_count}")
    print(f"scans={combined['scans'].shape}")
    print("class_counts=" + ", ".join(f"{value}:{count}" for value, count in zip(values, counts)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    combine(args.inputs, args.output)


if __name__ == "__main__":
    main()
