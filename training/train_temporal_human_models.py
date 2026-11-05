import argparse
import copy
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ml.features import ANGULAR_BINS, CHANNELS, RADIAL_BINS
from ml.models import MagneticTransitionPredictor, TemporalLocationModel, TemporalPresenceDetector
from training.train_human_detector import prediction_residuals, split_episodes, train_predictor, transition_dataset


class TemporalDataset(Dataset):
    def __init__(self, features, labels, locations, episodes, window: int) -> None:
        self.features = features
        self.labels = labels
        self.locations = locations
        self.window = window
        self.indices = [(int(episode), start) for episode in episodes for start in range(features.shape[1] - window + 1)]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        episode, start = self.indices[index]
        end = start + self.window
        sequence = self.features[episode, start:end].transpose(1, 2, 0)
        presence = int(self.labels[episode, end] > 0)
        return torch.from_numpy(sequence), torch.tensor(presence), torch.from_numpy(self.locations[episode, end])


def metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    tp = int(np.sum((targets == 1) & (predictions == 1)))
    fp = int(np.sum((targets == 0) & (predictions == 1)))
    fn = int(np.sum((targets == 1) & (predictions == 0)))
    tn = int(np.sum((targets == 0) & (predictions == 0)))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {
        "accuracy": (tp + tn) / max(1, tp + tn + fp + fn),
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / max(1e-9, precision + recall),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def evaluate_detector(model, loader, device, threshold: float = 0.5):
    probabilities = []
    targets = []
    model.eval()
    with torch.inference_mode():
        for sequence, presence, location in loader:
            logits = model(sequence.to(device))
            probabilities.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            targets.append(presence.numpy())
    probabilities = np.concatenate(probabilities)
    targets = np.concatenate(targets)
    return metrics(targets, (probabilities >= threshold).astype(np.int64)), probabilities, targets


def evaluate_localizer(model, loader, device, threshold: float = 0.5):
    probabilities = []
    targets = []
    model.eval()
    with torch.inference_mode():
        for sequence, presence, location in loader:
            active = presence > 0
            if not torch.any(active):
                continue
            logits = model(sequence[active].to(device))
            probabilities.append(torch.sigmoid(logits).cpu().numpy().reshape(-1))
            targets.append(location[active].numpy().reshape(-1))
    probabilities = np.concatenate(probabilities)
    targets = np.concatenate(targets) >= 0.5
    predictions = probabilities >= threshold
    tp = np.sum(predictions & targets)
    fp = np.sum(predictions & ~targets)
    fn = np.sum(~predictions & targets)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return 2.0 * precision * recall / max(1e-9, precision + recall), probabilities, targets


def train_detector(model, train_loader, validation_loader, device, epochs: int):
    counts = np.zeros(2, dtype=float)
    for _, presence, _ in train_loader:
        counts += np.bincount(presence.numpy(), minlength=2)
    weights = counts.sum() / (2.0 * np.maximum(counts, 1.0))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=8e-4)
    history = []
    best_state = None
    best_f1 = -1.0
    for epoch in range(epochs):
        model.train()
        total = 0.0
        samples = 0
        for sequence, presence, location in train_loader:
            sequence, presence = sequence.to(device), presence.to(device)
            optimizer.zero_grad()
            loss = criterion(model(sequence), presence)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += loss.item() * len(sequence)
            samples += len(sequence)
        validation, _, _ = evaluate_detector(model, validation_loader, device)
        history.append((total / samples, validation["f1"]))
        if validation["f1"] > best_f1:
            best_f1 = validation["f1"]
            best_state = copy.deepcopy(model.state_dict())
        print(f"detector epoch={epoch + 1} loss={total / samples:.6f} accuracy={validation['accuracy']:.4f} f1={validation['f1']:.4f}", flush=True)
    model.load_state_dict(best_state)
    return history


def train_localizer(model, train_loader, validation_loader, device, epochs: int):
    positives = 0.0
    total_values = 0.0
    for _, presence, location in train_loader:
        active = presence > 0
        positives += float(location[active].sum())
        total_values += float(location[active].numel())
    weight = np.sqrt(np.clip((total_values - positives) / max(1.0, positives), 1.0, 40.0))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=8e-4)
    history = []
    best_state = None
    best_f1 = -1.0
    for epoch in range(epochs):
        model.train()
        total = 0.0
        samples = 0
        for sequence, presence, location in train_loader:
            active = presence > 0
            if not torch.any(active):
                continue
            sequence, location = sequence[active].to(device), location[active].to(device)
            optimizer.zero_grad()
            loss = criterion(model(sequence), location)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += loss.item() * len(sequence)
            samples += len(sequence)
        location_f1, _, _ = evaluate_localizer(model, validation_loader, device)
        history.append((total / max(1, samples), location_f1))
        if location_f1 > best_f1:
            best_f1 = location_f1
            best_state = copy.deepcopy(model.state_dict())
        print(f"localizer epoch={epoch + 1} loss={total / max(1, samples):.6f} f1={location_f1:.4f}", flush=True)
    model.load_state_dict(best_state)
    return history


def tune(probabilities, targets) -> float:
    best = (-1.0, 0.5)
    for threshold in np.linspace(0.2, 0.8, 25):
        value = metrics(targets.astype(np.int64), (probabilities >= threshold).astype(np.int64))["f1"]
        if value > best[0]:
            best = (value, float(threshold))
    return best[1]


def tune_location(probabilities, targets) -> float:
    best = (-1.0, 0.5)
    for threshold in np.linspace(0.1, 0.8, 29):
        predictions = probabilities >= threshold
        tp = np.sum(predictions & targets)
        fp = np.sum(predictions & ~targets)
        fn = np.sum(~predictions & targets)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2.0 * precision * recall / max(1e-9, precision + recall)
        if f1 > best[0]:
            best = (f1, float(threshold))
    return best[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/human_temporal_v4.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/human_detector_v4"))
    parser.add_argument("--predictor-epochs", type=int, default=6)
    parser.add_argument("--detector-epochs", type=int, default=10)
    parser.add_argument("--localizer-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    data = np.load(args.data)
    raw_scans = data["scans"].reshape(data["scans"].shape[0], data["scans"].shape[1], -1)
    raw_motions = data["motions"].astype(np.float32)
    raw_physics = data["physics_features"].astype(np.float32)
    raw_centers = data["center_residuals"].astype(np.float32)
    labels = data["labels"]
    locations = data["locations_highres"].astype(np.float32)
    train_episodes, validation_episodes, test_episodes = split_episodes(data["scenario_counts"], args.seed)
    scan_mean = raw_scans[train_episodes].mean(axis=(0, 1))
    scan_std = np.maximum(raw_scans[train_episodes].std(axis=(0, 1)), 1e-5)
    motion_mean = raw_motions[train_episodes].mean(axis=(0, 1))
    motion_std = np.maximum(raw_motions[train_episodes].std(axis=(0, 1)), 1e-5)
    physics_mean = raw_physics[train_episodes].mean(axis=(0, 1, 2, 3))
    physics_std = np.maximum(raw_physics[train_episodes].std(axis=(0, 1, 2, 3)), 1e-5)
    center_mean = raw_centers[train_episodes].mean(axis=(0, 1))
    center_std = np.maximum(raw_centers[train_episodes].std(axis=(0, 1)), 1e-5)
    scans = ((raw_scans - scan_mean) / scan_std).astype(np.float32)
    motions = ((raw_motions - motion_mean) / motion_std).astype(np.float32)
    physics = ((raw_physics - physics_mean) / physics_std).astype(np.float32)
    centers = ((raw_centers - center_mean) / center_std).astype(np.float32)
    predictor = MagneticTransitionPredictor(scans.shape[2]).to(device)
    predictor_history = train_predictor(
        predictor,
        DataLoader(transition_dataset(scans, motions, labels, train_episodes), batch_size=args.batch_size, shuffle=True),
        DataLoader(transition_dataset(scans, motions, labels, validation_episodes), batch_size=args.batch_size),
        args.predictor_epochs,
        device,
    )
    features = prediction_residuals(predictor, scans, motions, physics, centers, device)
    train_set = TemporalDataset(features, labels, locations, train_episodes, args.window)
    validation_set = TemporalDataset(features, labels, locations, validation_episodes, args.window)
    test_set = TemporalDataset(features, labels, locations, test_episodes, args.window)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=args.batch_size)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)
    channels = features.shape[3]
    detector = TemporalPresenceDetector(channels).to(device)
    localizer = TemporalLocationModel(channels).to(device)
    detector_history = train_detector(detector, train_loader, validation_loader, device, args.detector_epochs)
    localizer_history = train_localizer(localizer, train_loader, validation_loader, device, args.localizer_epochs)
    _, detector_probabilities, detector_targets = evaluate_detector(detector, validation_loader, device)
    presence_threshold = tune(detector_probabilities, detector_targets)
    _, location_probabilities, location_targets = evaluate_localizer(localizer, validation_loader, device)
    location_threshold = tune_location(location_probabilities, location_targets)
    test_metrics, _, _ = evaluate_detector(detector, test_loader, device, presence_threshold)
    location_f1, _, _ = evaluate_localizer(localizer, test_loader, device, location_threshold)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "best_temporal_human_models.pth"
    torch.save(
        {
            "model_type": "separate_cnn_gru_v1",
            "predictor_state_dict": predictor.cpu().state_dict(),
            "detector_state_dict": detector.cpu().state_dict(),
            "localizer_state_dict": localizer.cpu().state_dict(),
            "scan_mean": torch.from_numpy(scan_mean.astype(np.float32)),
            "scan_std": torch.from_numpy(scan_std.astype(np.float32)),
            "motion_mean": torch.from_numpy(motion_mean.astype(np.float32)),
            "motion_std": torch.from_numpy(motion_std.astype(np.float32)),
            "physics_mean": torch.from_numpy(physics_mean.astype(np.float32)),
            "physics_std": torch.from_numpy(physics_std.astype(np.float32)),
            "center_mean": torch.from_numpy(center_mean.astype(np.float32)),
            "center_std": torch.from_numpy(center_std.astype(np.float32)),
            "window": args.window,
            "scan_size": scans.shape[2],
            "classifier_channels": channels,
            "presence_threshold": presence_threshold,
            "location_threshold": location_threshold,
            "radial_bins": 8,
            "angular_bins": 16,
        },
        model_path,
    )
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(predictor_history["train"], label="train")
    axes[0].plot(predictor_history["validation"], label="validation")
    axes[1].plot([item[0] for item in detector_history], label="loss")
    axes[1].plot([item[1] for item in detector_history], label="F1")
    axes[2].plot([item[0] for item in localizer_history], label="loss")
    axes[2].plot([item[1] for item in localizer_history], label="F1")
    for axis in axes:
        axis.legend()
    figure.tight_layout()
    figure.savefig(args.output_dir / "training_curves.png", dpi=160)
    plt.close(figure)
    print(f"accuracy={test_metrics['accuracy']:.6f}")
    print(f"precision={test_metrics['precision']:.6f}")
    print(f"recall={test_metrics['recall']:.6f}")
    print(f"f1={test_metrics['f1']:.6f}")
    print(f"location_f1={location_f1:.6f}")
    print(f"presence_threshold={presence_threshold:.3f}")
    print(f"location_threshold={location_threshold:.3f}")
    print(f"confusion_matrix=({test_metrics['tn']}, {test_metrics['fp']}, {test_metrics['fn']}, {test_metrics['tp']})")
    print(f"model={model_path}")


if __name__ == "__main__":
    main()
