from pathlib import Path

import numpy as np
import torch

from ml.candidate_patch_v2 import CandidatePatchCNN, MagneticPatchHistory
from simulation.observations_v2 import RobotObservation


class CandidateVerifierRuntime:
    def __init__(self, model_path: str | Path | None = None) -> None:
        root = Path(__file__).resolve().parents[1]
        path = Path(model_path) if model_path else root / 'models' / 'candidate_verifier' / 'best_candidate_verifier_v2.pth'
        if not path.is_absolute():
            path = root / path
        if not path.exists():
            raise FileNotFoundError(f'Candidate verifier model not found: {path}')
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        if int(checkpoint.get('feature_version', 0)) != 1:
            raise ValueError('Unsupported candidate verifier feature version')
        self.model_path = path
        self.model = CandidatePatchCNN(int(checkpoint['patch_size']))
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        self.threshold = float(checkpoint['threshold'])
        self.history = MagneticPatchHistory(int(checkpoint['window']), int(checkpoint['patch_size']))
        self.patch_mean = np.asarray(checkpoint['patch_mean'], dtype=np.float32)[None, None, :, None, None]
        self.patch_std = np.asarray(checkpoint['patch_std'], dtype=np.float32)[None, None, :, None, None]
        self.motion_mean = np.asarray(checkpoint['motion_mean'], dtype=np.float32)
        self.motion_std = np.asarray(checkpoint['motion_std'], dtype=np.float32)
        self.last_probability = 0.0
        self.rejected_candidates = 0

    def reset(self) -> None:
        self.history.reset()
        self.last_probability = 0.0
        self.rejected_candidates = 0

    def observe(self, observation: RobotObservation) -> None:
        self.history.observe(observation)

    def scores(self, positions) -> np.ndarray:
        positions = list(positions)
        if not positions:
            return np.zeros(0, dtype=np.float32)
        if len(self.history.frames) < 3:
            return np.zeros(len(positions), dtype=np.float32)
        features = [self.history.at(np.asarray(position)) for position in positions]
        patches = (np.stack([feature[0] for feature in features]) - self.patch_mean) / self.patch_std
        motion = (np.stack([feature[1] for feature in features]) - self.motion_mean) / self.motion_std
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(torch.from_numpy(patches), torch.from_numpy(motion)), dim=1)[:, 1].numpy()
        self.last_probability = float(probabilities.max())
        return probabilities

    def accepts(self, position: np.ndarray) -> bool:
        probability = float(self.scores([position])[0])
        accepted = probability >= self.threshold
        self.rejected_candidates += int(not accepted)
        return accepted
