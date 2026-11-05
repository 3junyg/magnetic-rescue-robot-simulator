import argparse
import copy
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from ml.features import ANGULAR_BINS, CHANNELS, RADIAL_BINS
from ml.models import MagneticTransitionPredictor, ResidualHumanCNN


COUNT_LOSS_WEIGHT = 0.2
LOCATION_LOSS_WEIGHT = 0.2


class ResidualWindowDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray, locations: np.ndarray, episode_indices: np.ndarray, window: int) -> None:
        self.features = features
        self.labels = labels
        self.locations = locations
        self.window = window
        self.indices = [(int(episode), start) for episode in episode_indices for start in range(features.shape[1] - window + 1)]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        episode, start = self.indices[index]
        end = start + self.window
        sequence = self.features[episode, start:end].transpose(1, 2, 0)
        count = min(3, int(self.labels[episode, end]))
        presence = int(count > 0)
        location = self.locations[episode, end].reshape(-1)
        return torch.from_numpy(sequence), torch.tensor(presence), torch.tensor(count), torch.from_numpy(location)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def split_episodes(scenario_counts: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train = []
    validation = []
    test = []
    for value in np.unique(np.clip(scenario_counts, 0, 3)):
        indices = np.where(np.clip(scenario_counts, 0, 3) == value)[0]
        rng.shuffle(indices)
        if len(indices) < 3:
            raise ValueError("각 scenario class에 최소 3개 episode가 필요합니다.")
        validation_size = max(1, int(round(len(indices) * 0.15)))
        test_size = max(1, int(round(len(indices) * 0.15)))
        validation.append(indices[:validation_size])
        test.append(indices[validation_size:validation_size + test_size])
        train.append(indices[validation_size + test_size:])
    return np.concatenate(train), np.concatenate(validation), np.concatenate(test)


def transition_dataset(scans: np.ndarray, motions: np.ndarray, labels: np.ndarray, episodes: np.ndarray) -> TensorDataset:
    scan_inputs = []
    motion_inputs = []
    targets = []
    for episode in episodes:
        valid = (labels[episode, :-1] == 0) & (labels[episode, 1:] == 0)
        scan_inputs.append(scans[episode, :-1][valid])
        motion_inputs.append(motions[episode, 1:][valid])
        targets.append((scans[episode, 1:] - scans[episode, :-1])[valid])
    scan_array = np.concatenate(scan_inputs)
    motion_array = np.concatenate(motion_inputs)
    target_array = np.concatenate(targets)
    if len(scan_array) == 0:
        raise ValueError("사람이 없는 predictor 학습 transition이 없습니다.")
    return TensorDataset(torch.from_numpy(scan_array), torch.from_numpy(motion_array), torch.from_numpy(target_array))


def train_predictor(model, train_loader, validation_loader, epochs, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.SmoothL1Loss()
    history = {"train": [], "validation": []}
    best_loss = float("inf")
    best_state = None
    for epoch in range(epochs):
        model.train()
        total = 0.0
        samples = 0
        for scan, motion, target in train_loader:
            scan, motion, target = scan.to(device), motion.to(device), target.to(device)
            optimizer.zero_grad()
            loss = criterion(model(scan, motion), target)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(scan)
            samples += len(scan)
        train_loss = total / samples
        model.eval()
        total = 0.0
        samples = 0
        with torch.no_grad():
            for scan, motion, target in validation_loader:
                scan, motion, target = scan.to(device), motion.to(device), target.to(device)
                loss = criterion(model(scan, motion), target)
                total += loss.item() * len(scan)
                samples += len(scan)
        validation_loss = total / samples
        history["train"].append(train_loss)
        history["validation"].append(validation_loss)
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
        print(f"predictor epoch={epoch + 1} train_loss={train_loss:.6f} validation_loss={validation_loss:.6f}")
    model.load_state_dict(best_state)
    return history


def prediction_residuals(model, scans: np.ndarray, motions: np.ndarray, physics_features: np.ndarray, center_residuals: np.ndarray, device) -> np.ndarray:
    residuals = []
    model.eval()
    with torch.no_grad():
        for episode in range(scans.shape[0]):
            current = torch.from_numpy(scans[episode, :-1]).to(device)
            movement = torch.from_numpy(motions[episode, 1:]).to(device)
            temporal_change = scans[episode, 1:] - scans[episode, :-1]
            predicted = model(current, movement).cpu().numpy()
            residual = temporal_change - predicted
            cells = RADIAL_BINS * ANGULAR_BINS
            residual = residual.reshape(residual.shape[0], cells, CHANNELS)
            current_scan = scans[episode, 1:].reshape(residual.shape[0], cells, CHANNELS)
            temporal_change = temporal_change.reshape(residual.shape[0], cells, CHANNELS)
            movement_features = np.repeat(motions[episode, 1:, None, :], cells, axis=1)
            physics = physics_features[episode, 1:].reshape(residual.shape[0], cells, CHANNELS)
            center = np.repeat(center_residuals[episode, 1:, None, :], cells, axis=1)
            residuals.append(np.concatenate([residual, current_scan, temporal_change, movement_features, physics, center], axis=2))
    return np.asarray(residuals, dtype=np.float32)


def class_weights(dataset: ResidualWindowDataset, classes: int, position: int) -> torch.Tensor:
    counts = np.zeros(classes, dtype=np.float64)
    for episode, start in dataset.indices:
        count = min(3, int(dataset.labels[episode, start + dataset.window]))
        label = int(count > 0) if position == 0 else count
        counts[label] += 1
    weights = counts.sum() / (classes * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32)


def binary_metrics(confusion: np.ndarray) -> dict[str, float]:
    tn, fp, fn, tp = confusion.ravel()
    accuracy = (tp + tn) / max(1, confusion.sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    return {"accuracy": float(accuracy), "precision": float(precision), "recall": float(recall), "f1": float(f1)}


def evaluate_classifier(model, loader, presence_criterion, count_criterion, location_criterion, device, presence_threshold: float = 0.5, location_threshold: float = 0.5):
    model.eval()
    total_loss = 0.0
    total = 0
    confusion = np.zeros((2, 2), dtype=np.int64)
    count_correct = 0
    location_tp = 0
    location_fp = 0
    location_fn = 0
    with torch.no_grad():
        for sequence, presence, count, location in loader:
            sequence, presence, count, location = sequence.to(device), presence.to(device), count.to(device), location.to(device)
            presence_logits, count_logits, location_logits = model(sequence)
            loss = presence_criterion(presence_logits, presence) + COUNT_LOSS_WEIGHT * count_criterion(count_logits, count) + LOCATION_LOSS_WEIGHT * location_criterion(location_logits, location)
            total_loss += loss.item() * len(sequence)
            total += len(sequence)
            presence_prediction = (torch.softmax(presence_logits, dim=1)[:, 1] >= presence_threshold).long()
            count_prediction = count_logits.argmax(dim=1)
            count_correct += int((count_prediction == count).sum().item())
            location_prediction = torch.sigmoid(location_logits) >= location_threshold
            location_target = location >= 0.5
            location_tp += int((location_prediction & location_target).sum().item())
            location_fp += int((location_prediction & ~location_target).sum().item())
            location_fn += int((~location_prediction & location_target).sum().item())
            for target, prediction in zip(presence.cpu().numpy(), presence_prediction.cpu().numpy()):
                confusion[target, prediction] += 1
    metrics = binary_metrics(confusion)
    metrics["count_accuracy"] = count_correct / max(1, total)
    location_precision = location_tp / max(1, location_tp + location_fp)
    location_recall = location_tp / max(1, location_tp + location_fn)
    metrics["location_f1"] = 2.0 * location_precision * location_recall / max(1e-12, location_precision + location_recall)
    return total_loss / max(1, total), metrics, confusion


def tune_thresholds(model, loader, device) -> tuple[float, float]:
    presence_probabilities = []
    presence_targets = []
    location_probabilities = []
    location_targets = []
    model.eval()
    with torch.no_grad():
        for sequence, presence, count, location in loader:
            presence_logits, count_logits, location_logits = model(sequence.to(device))
            presence_probabilities.append(torch.softmax(presence_logits, dim=1)[:, 1].cpu().numpy())
            presence_targets.append(presence.numpy())
            location_probabilities.append(torch.sigmoid(location_logits).cpu().numpy().reshape(-1))
            location_targets.append(location.numpy().reshape(-1))
    presence_probabilities = np.concatenate(presence_probabilities)
    presence_targets = np.concatenate(presence_targets)
    location_probabilities = np.concatenate(location_probabilities)
    location_targets = np.concatenate(location_targets)
    best_presence_threshold = 0.5
    best_presence_f1 = -1.0
    best_location_threshold = 0.5
    best_location_f1 = -1.0
    for threshold in np.linspace(0.1, 0.9, 33):
        predictions = presence_probabilities >= threshold
        tp = np.sum(predictions & (presence_targets == 1))
        fp = np.sum(predictions & (presence_targets == 0))
        fn = np.sum(~predictions & (presence_targets == 1))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
        if f1 > best_presence_f1:
            best_presence_f1 = f1
            best_presence_threshold = float(threshold)
        location_predictions = location_probabilities >= threshold
        location_target = location_targets >= 0.5
        tp = np.sum(location_predictions & location_target)
        fp = np.sum(location_predictions & ~location_target)
        fn = np.sum(~location_predictions & location_target)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
        if f1 > best_location_f1:
            best_location_f1 = f1
            best_location_threshold = float(threshold)
    return best_presence_threshold, best_location_threshold


def train_classifier(model, train_loader, validation_loader, presence_criterion, count_criterion, location_criterion, epochs, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    history = {"train_loss": [], "validation_loss": [], "validation_accuracy": [], "validation_f1": [], "validation_location_f1": []}
    best_f1 = -1.0
    best_loss = float("inf")
    best_state = None
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total = 0
        for sequence, presence, count, location in train_loader:
            sequence, presence, count, location = sequence.to(device), presence.to(device), count.to(device), location.to(device)
            optimizer.zero_grad()
            presence_logits, count_logits, location_logits = model(sequence)
            loss = presence_criterion(presence_logits, presence) + COUNT_LOSS_WEIGHT * count_criterion(count_logits, count) + LOCATION_LOSS_WEIGHT * location_criterion(location_logits, location)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item() * len(sequence)
            total += len(sequence)
        train_loss = total_loss / total
        validation_loss, metrics, _ = evaluate_classifier(model, validation_loader, presence_criterion, count_criterion, location_criterion, device)
        history["train_loss"].append(train_loss)
        history["validation_loss"].append(validation_loss)
        history["validation_accuracy"].append(metrics["accuracy"])
        history["validation_f1"].append(metrics["f1"])
        history["validation_location_f1"].append(metrics["location_f1"])
        if metrics["f1"] > best_f1 or (metrics["f1"] == best_f1 and validation_loss < best_loss):
            best_f1 = metrics["f1"]
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
        print(f"classifier epoch={epoch + 1} train_loss={train_loss:.6f} validation_loss={validation_loss:.6f} accuracy={metrics['accuracy']:.4f} f1={metrics['f1']:.4f} location_f1={metrics['location_f1']:.4f}")
    model.load_state_dict(best_state)
    return history


def save_curves(output: Path, predictor_history: dict, classifier_history: dict) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(predictor_history["train"], label="train")
    axes[0].plot(predictor_history["validation"], label="validation")
    axes[0].set_title("Transition predictor loss")
    axes[0].legend()
    axes[1].plot(classifier_history["train_loss"], label="train")
    axes[1].plot(classifier_history["validation_loss"], label="validation")
    axes[1].set_title("Human classifier loss")
    axes[1].legend()
    axes[2].plot(classifier_history["validation_accuracy"], label="accuracy")
    axes[2].plot(classifier_history["validation_f1"], label="F1")
    axes[2].plot(classifier_history["validation_location_f1"], label="location F1")
    axes[2].set_title("Validation metrics")
    axes[2].legend()
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/human_transition_dataset.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/human_detector"))
    parser.add_argument("--predictor-epochs", type=int, default=25)
    parser.add_argument("--classifier-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    set_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    data = np.load(args.data)
    raw_scans = data["scans"].reshape(data["scans"].shape[0], data["scans"].shape[1], -1)
    raw_motions = data["motions"]
    if "physics_features" not in data.files:
        raise ValueError("데이터셋에 physics_features가 없습니다. generate_human_dataset.py로 데이터를 다시 생성하세요.")
    if "center_residuals" not in data.files:
        raise ValueError("데이터셋에 center_residuals가 없습니다. generate_human_dataset.py로 데이터를 다시 생성하세요.")
    raw_physics = data["physics_features"].astype(np.float32)
    raw_center_residuals = data["center_residuals"].astype(np.float32)
    labels = data["labels"]
    locations = data["locations"]
    train_episodes, validation_episodes, test_episodes = split_episodes(data["scenario_counts"], args.seed)
    scan_mean = raw_scans[train_episodes].mean(axis=(0, 1))
    scan_std = np.maximum(raw_scans[train_episodes].std(axis=(0, 1)), 1e-5)
    motion_mean = raw_motions[train_episodes].mean(axis=(0, 1))
    motion_std = np.maximum(raw_motions[train_episodes].std(axis=(0, 1)), 1e-5)
    physics_mean = raw_physics[train_episodes].mean(axis=(0, 1, 2, 3))
    physics_std = np.maximum(raw_physics[train_episodes].std(axis=(0, 1, 2, 3)), 1e-5)
    center_mean = raw_center_residuals[train_episodes].mean(axis=(0, 1))
    center_std = np.maximum(raw_center_residuals[train_episodes].std(axis=(0, 1)), 1e-5)
    scans = ((raw_scans - scan_mean) / scan_std).astype(np.float32)
    motions = ((raw_motions - motion_mean) / motion_std).astype(np.float32)
    physics_features = ((raw_physics - physics_mean) / physics_std).astype(np.float32)
    center_residuals = ((raw_center_residuals - center_mean) / center_std).astype(np.float32)
    train_transitions = transition_dataset(scans, motions, labels, train_episodes)
    validation_transitions = transition_dataset(scans, motions, labels, validation_episodes)
    predictor = MagneticTransitionPredictor(scans.shape[2]).to(device)
    predictor_history = train_predictor(
        predictor,
        DataLoader(train_transitions, batch_size=args.batch_size, shuffle=True),
        DataLoader(validation_transitions, batch_size=args.batch_size),
        args.predictor_epochs,
        device,
    )
    residual_features = prediction_residuals(predictor, scans, motions, physics_features, center_residuals, device)
    train_dataset = ResidualWindowDataset(residual_features, labels, locations, train_episodes, args.window)
    validation_dataset = ResidualWindowDataset(residual_features, labels, locations, validation_episodes, args.window)
    test_dataset = ResidualWindowDataset(residual_features, labels, locations, test_episodes, args.window)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    classifier = ResidualHumanCNN(residual_features.shape[3], location_cells=residual_features.shape[2]).to(device)
    presence_weights = class_weights(train_dataset, 2, 0)
    presence_criterion = nn.CrossEntropyLoss(weight=presence_weights.to(device))
    count_criterion = nn.CrossEntropyLoss(weight=torch.sqrt(class_weights(train_dataset, 4, 1)).to(device), label_smoothing=0.03)
    location_targets = np.asarray([locations[episode, start + args.window].reshape(-1) for episode, start in train_dataset.indices])
    positives = location_targets.sum(axis=0)
    negatives = len(location_targets) - positives
    location_weight = torch.from_numpy(np.sqrt(np.clip(negatives / np.maximum(positives, 1.0), 1.0, 20.0)).astype(np.float32)).to(device)
    location_criterion = nn.BCEWithLogitsLoss(pos_weight=location_weight)
    classifier_history = train_classifier(
        classifier,
        train_loader,
        validation_loader,
        presence_criterion,
        count_criterion,
        location_criterion,
        args.classifier_epochs,
        device,
    )
    presence_threshold, location_threshold = tune_thresholds(classifier, validation_loader, device)
    test_loss, test_metrics, confusion = evaluate_classifier(classifier, test_loader, presence_criterion, count_criterion, location_criterion, device, presence_threshold, location_threshold)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "best_human_transition_detector.pth"
    torch.save(
        {
            "predictor_state_dict": predictor.cpu().state_dict(),
            "classifier_state_dict": classifier.cpu().state_dict(),
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
            "classifier_channels": residual_features.shape[3],
            "radial_bins": RADIAL_BINS,
            "angular_bins": ANGULAR_BINS,
            "feature_channels": CHANNELS,
            "count_classes": ["0", "1", "2", "3+"],
            "location_cells": RADIAL_BINS * ANGULAR_BINS,
            "presence_threshold": presence_threshold,
            "location_threshold": location_threshold,
            "seed": args.seed,
        },
        model_path,
    )
    curves_path = args.output_dir / "training_curves.png"
    save_curves(curves_path, predictor_history, classifier_history)
    print(f"device={device}")
    print(f"episodes train={len(train_episodes)} validation={len(validation_episodes)} test={len(test_episodes)}")
    print(f"windows train={len(train_dataset)} validation={len(validation_dataset)} test={len(test_dataset)}")
    print(f"test_loss={test_loss:.6f}")
    print(f"accuracy={test_metrics['accuracy']:.6f}")
    print(f"precision={test_metrics['precision']:.6f}")
    print(f"recall={test_metrics['recall']:.6f}")
    print(f"f1={test_metrics['f1']:.6f}")
    print(f"count_accuracy={test_metrics['count_accuracy']:.6f}")
    print(f"location_f1={test_metrics['location_f1']:.6f}")
    print(f"presence_threshold={presence_threshold:.3f}")
    print(f"location_threshold={location_threshold:.3f}")
    print("confusion_matrix=")
    print(confusion)
    print(f"model={model_path}")
    print(f"curves={curves_path}")


if __name__ == "__main__":
    main()
