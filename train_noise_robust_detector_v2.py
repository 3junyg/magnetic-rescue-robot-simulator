import argparse
from pathlib import Path
import subprocess
import sys

from training.generate_noise_robust_dataset_v2 import generate_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/human_noise_robust_v1_v2.npz"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/human_detector_noise_robust"))
    parser.add_argument("--episodes", type=int, default=540)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=72026)
    parser.add_argument("--noise-min", type=float, default=0.7)
    parser.add_argument("--noise-max", type=float, default=2.0)
    parser.add_argument("--predictor-epochs", type=int, default=25)
    parser.add_argument("--classifier-epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()
    if not args.skip_generation:
        generate_dataset(args.data, args.episodes, args.steps, args.seed, args.noise_min, args.noise_max)
    if args.skip_training:
        return
    command = [
        sys.executable,
        "-m",
        "training.train_human_detector_v2",
        "--data",
        str(args.data),
        "--output-dir",
        str(args.model_dir),
        "--predictor-epochs",
        str(args.predictor_epochs),
        "--classifier-epochs",
        str(args.classifier_epochs),
        "--batch-size",
        str(args.batch_size),
        "--window",
        str(args.window),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
