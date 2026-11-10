import argparse
import copy
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from ml.candidate_patch_v2 import CandidatePatchCNN


class PatchDataset(Dataset):
    def __init__(self, patches, motion, labels, indices, normalization, augment=False):
        self.patches, self.motion, self.labels = patches, motion, labels
        self.indices = indices
        self.normalization = normalization
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        source = self.indices[index]
        patch = self.patches[source].astype(np.float32)
        motion = self.motion[source].astype(np.float32)
        if self.augment:
            patch[:, :6] *= np.random.uniform(0.8, 1.2)
        patch = (patch - self.normalization['patch_mean'][None, :, None, None]) / self.normalization['patch_std'][None, :, None, None]
        motion = (motion - self.normalization['motion_mean']) / self.normalization['motion_std']
        return torch.from_numpy(patch), torch.from_numpy(motion), torch.tensor(int(self.labels[source]), dtype=torch.long)


def probabilities(model, loader):
    model.eval()
    values, labels = [], []
    with torch.inference_mode():
        for patch, motion, target in loader:
            values.append(torch.softmax(model(patch, motion), dim=1)[:, 1].numpy())
            labels.append(target.numpy())
    return np.concatenate(values), np.concatenate(labels)


def metrics(scores, target, threshold):
    prediction = scores >= threshold
    tp = int(np.sum(prediction & (target == 1)))
    fp = int(np.sum(prediction & (target == 0)))
    fn = int(np.sum(~prediction & (target == 1)))
    tn = int(np.sum(~prediction & (target == 0)))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {'accuracy': (tp + tn) / max(1, len(target)), 'precision': precision, 'recall': recall, 'f1': f1, 'confusion': [[tn, fp], [fn, tp]]}


def choose_threshold(scores, labels, minimum_precision=0.95):
    candidates = []
    for threshold in np.arange(0.05, 0.981, 0.025):
        result = metrics(scores, labels, threshold)
        candidates.append((threshold, result))
    precise = [(threshold, result) for threshold, result in candidates if result['precision'] >= minimum_precision]
    if precise:
        return max(precise, key=lambda item: (item[1]['recall'], item[1]['f1']))
    return max(candidates, key=lambda item: item[1]['f1'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, nargs='+', default=[Path('data/candidate_clusters_v2.npz')])
    parser.add_argument('--output', type=Path, default=Path('models/candidate_verifier/round1_v2.pth'))
    parser.add_argument('--init', type=Path)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=2029)
    parser.add_argument('--minimum-precision', type=float, default=0.95)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    patches, motions, labels, groups, splits = [], [], [], [], []
    for path in args.data:
        data = np.load(path)
        ids = data['episode_ids']
        blocks = ((ids - int(data['seed'])) // 18) % 6
        patches.append(data['patches'])
        motions.append(data['motion'])
        labels.append(data['labels'])
        groups.append(ids)
        splits.append(np.where(blocks < 4, 0, np.where(blocks == 4, 1, 2)))
    patches, motion = np.concatenate(patches), np.concatenate(motions)
    labels, groups, split = np.concatenate(labels), np.concatenate(groups), np.concatenate(splits)
    index_sets = [np.where(split == value)[0] for value in (0, 1, 2)]
    group_sets = [set(groups[index]) for index in index_sets]
    if any(group_sets[a] & group_sets[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise ValueError('Episode overlap detected')
    if any(len(indices) == 0 for indices in index_sets):
        raise ValueError('Train/validation/test episodes are required')
    model = CandidatePatchCNN()
    if args.init:
        checkpoint = torch.load(args.init, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])
        normalization = {key: np.asarray(checkpoint[key], dtype=np.float32) for key in ('patch_mean', 'patch_std', 'motion_mean', 'motion_std')}
    else:
        train_patch = patches[index_sets[0]].astype(np.float32)
        normalization = {
            'patch_mean': train_patch.mean(axis=(0, 1, 3, 4)),
            'patch_std': np.maximum(train_patch.std(axis=(0, 1, 3, 4)), 0.05),
            'motion_mean': motion[index_sets[0]].astype(np.float32).mean(axis=(0, 1)),
            'motion_std': np.maximum(motion[index_sets[0]].astype(np.float32).std(axis=(0, 1)), 0.05),
        }
        normalization['patch_mean'][6:] = 0.0
        normalization['patch_std'][6:] = 1.0
        del train_patch
    datasets = [PatchDataset(patches, motion, labels, indices, normalization, augment=(number == 0)) for number, indices in enumerate(index_sets)]
    weights = np.ones(len(datasets[0]), dtype=np.float64)
    train_labels = labels[index_sets[0]]
    weights[train_labels == 1] = max(1.0, (train_labels == 0).sum() / max(1, (train_labels == 1).sum()))
    if args.init:
        mining = PatchDataset(patches, motion, labels, index_sets[0], normalization)
        scores, _ = probabilities(model, DataLoader(mining, batch_size=256))
        weights[train_labels == 0] *= 1.0 + 6.0 * scores[train_labels == 0] ** 2
        print(f'hard_negatives={int(np.sum((scores >= 0.5) & (train_labels == 0)))}', flush=True)
    sampler = WeightedRandomSampler(torch.from_numpy(weights), num_samples=len(weights), replacement=True)
    train_loader = DataLoader(datasets[0], batch_size=128, sampler=sampler)
    val_loader = DataLoader(datasets[1], batch_size=256)
    test_loader = DataLoader(datasets[2], batch_size=256)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4 if not args.init else 1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    best_score, best_state, best_threshold = -1.0, None, 0.5
    history = []
    for epoch in range(args.epochs):
        model.train()
        loss_total = 0.0
        for patch, movement, target in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(patch, movement), target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            loss_total += loss.item() * len(target)
        scores, targets = probabilities(model, val_loader)
        threshold, result = choose_threshold(scores, targets, args.minimum_precision)
        objective = result['recall'] if result['precision'] >= args.minimum_precision else result['f1'] * 0.5
        if objective > best_score:
            best_score, best_threshold, best_state = objective, float(threshold), copy.deepcopy(model.state_dict())
        history.append({'loss': loss_total / len(datasets[0]), **result, 'threshold': float(threshold)})
        print(f'epoch={epoch + 1} loss={history[-1]["loss"]:.4f} val_precision={result["precision"]:.3f} val_recall={result["recall"]:.3f} threshold={threshold:.3f}', flush=True)
    model.load_state_dict(best_state)
    scores, targets = probabilities(model, test_loader)
    test_metrics = metrics(scores, targets, best_threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {'state_dict': best_state, 'window': 8, 'patch_size': 5, 'threshold': best_threshold, 'seed': args.seed, 'feature_version': 1, 'test_metrics': test_metrics, 'minimum_precision': args.minimum_precision}
    checkpoint.update({key: torch.from_numpy(value.astype(np.float32)) for key, value in normalization.items()})
    torch.save(checkpoint, args.output)
    report = {'data': [str(path) for path in args.data], 'train_episodes': [int(value) for value in sorted(group_sets[0])], 'validation_episodes': [int(value) for value in sorted(group_sets[1])], 'test_episodes': [int(value) for value in sorted(group_sets[2])], 'samples': [len(dataset) for dataset in datasets], 'threshold': best_threshold, 'test_metrics': test_metrics, 'history': history}
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot([item['loss'] for item in history])
    axes[0].set_title('Training loss')
    for metric in ('precision', 'recall', 'f1'):
        axes[1].plot([item[metric] for item in history], label=metric)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(args.output.with_suffix('.png'), dpi=140)
    plt.close(figure)
    print(f'episodes={[len(values) for values in group_sets]} samples={report["samples"]} threshold={best_threshold:.3f} test={test_metrics} output={args.output}', flush=True)


if __name__ == '__main__':
    main()
