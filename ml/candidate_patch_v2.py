from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from config_v2 import BOARD_HEIGHT, BOARD_WIDTH, PERCEPTION_CELL_SIZE
from simulation.observations_v2 import RobotObservation


@dataclass
class PatchFrame:
    time: float
    grid: np.ndarray
    robot_position: np.ndarray
    heading: float
    moving: bool
    dt: float


class MagneticPatchHistory:
    def __init__(self, window: int = 8, patch_size: int = 5) -> None:
        self.window = window
        self.patch_size = patch_size
        self.frames: deque[PatchFrame] = deque(maxlen=window)
        self.previous_vectors = None
        self.previous_mask = None
        self.last_time = None
        self.rows = int(BOARD_HEIGHT / PERCEPTION_CELL_SIZE)
        self.columns = int(BOARD_WIDTH / PERCEPTION_CELL_SIZE)

    def reset(self) -> None:
        self.frames.clear()
        self.previous_vectors = None
        self.previous_mask = None
        self.last_time = None

    def observe(self, observation: RobotObservation) -> None:
        if self.last_time is not None and observation.time <= self.last_time:
            return
        scan = observation.magnetic_scan
        vectors = np.zeros((self.rows, self.columns, 3), dtype=np.float32)
        mask = np.zeros((self.rows, self.columns), dtype=bool)
        indices = np.floor(scan.positions / PERCEPTION_CELL_SIZE).astype(int)
        valid = (indices[:, 0] >= 0) & (indices[:, 0] < self.columns) & (indices[:, 1] >= 0) & (indices[:, 1] < self.rows)
        columns, rows = indices[valid].T
        vectors[rows, columns] = scan.vectors[valid]
        mask[rows, columns] = True
        background = np.median(scan.vectors, axis=0) if len(scan.vectors) else np.zeros(3)
        relative = np.zeros_like(vectors)
        relative[mask] = np.arcsinh((vectors[mask] - background) / 10.0)
        difference = np.zeros_like(vectors)
        overlap = np.zeros_like(mask)
        if self.previous_vectors is not None:
            overlap = mask & self.previous_mask
            if overlap.any():
                delta = vectors[overlap] - self.previous_vectors[overlap]
                delta -= np.median(delta, axis=0)
                difference[overlap] = np.arcsinh(delta / 2.0)
        grid = np.concatenate((relative, difference, mask[..., None], overlap[..., None]), axis=2).astype(np.float32)
        dt = 0.0 if self.last_time is None else observation.time - self.last_time
        self.frames.append(PatchFrame(float(observation.time), grid, observation.robot_position.copy(), float(observation.robot_heading), bool(observation.robot_moving), dt))
        self.previous_vectors = vectors
        self.previous_mask = mask
        self.last_time = float(observation.time)

    def at(self, position: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.zeros((self.window, 8, self.patch_size, self.patch_size), dtype=np.float32)
        motion = np.zeros((self.window, 6), dtype=np.float32)
        column, row = np.floor(np.asarray(position) / PERCEPTION_CELL_SIZE).astype(int)
        half = self.patch_size // 2
        start_row, start_column = row - half, column - half
        r0, r1 = max(0, start_row), min(self.rows, row + half + 1)
        c0, c1 = max(0, start_column), min(self.columns, column + half + 1)
        for index, frame in enumerate(self.frames, start=self.window - len(self.frames)):
            if r1 > r0 and c1 > c0:
                x[index, :, r0 - start_row:r1 - start_row, c0 - start_column:c1 - start_column] = frame.grid[r0:r1, c0:c1].transpose(2, 0, 1)
            offset = (frame.robot_position - position) / 16.0
            motion[index] = [offset[0], offset[1], np.sin(frame.heading), np.cos(frame.heading), min(4.0, frame.dt / 0.4), float(frame.moving)]
        return x, motion


class CandidatePatchCNN(nn.Module):
    def __init__(self, patch_size: int = 5) -> None:
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1), nn.GroupNorm(4, 16), nn.SiLU(),
            nn.Conv2d(16, 24, 3, padding=1), nn.GroupNorm(4, 24), nn.SiLU(),
            nn.Flatten(), nn.Linear(24 * patch_size * patch_size, 48), nn.SiLU(),
        )
        self.temporal = nn.Sequential(
            nn.Conv1d(54, 48, 3, padding=1), nn.SiLU(), nn.Dropout(0.15),
            nn.Conv1d(48, 32, 3, padding=1), nn.SiLU(),
        )
        self.head = nn.Sequential(nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 2))

    def forward(self, patches: torch.Tensor, motion: torch.Tensor) -> torch.Tensor:
        batch, time, channels, height, width = patches.shape
        spatial = self.spatial(patches.reshape(batch * time, channels, height, width)).reshape(batch, time, 48)
        encoded = self.temporal(torch.cat((spatial, motion), dim=2).transpose(1, 2))
        return self.head(torch.cat((encoded[:, :, -1], encoded.mean(dim=2)), dim=1))
