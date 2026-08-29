import torch
from torch import nn


class MagneticTransitionPredictor(nn.Module):
    def __init__(self, scan_size: int, motion_size: int = 3) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(scan_size + motion_size, 256),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, scan_size),
        )

    def forward(self, scan, motion):
        return self.network(torch.cat([scan, motion], dim=1))


class ResidualHumanCNN(nn.Module):
    def __init__(self, input_channels: int, count_classes: int = 4, location_cells: int = 32) -> None:
        super().__init__()
        self.location_cells = location_cells
        self.cell_encoder = nn.Sequential(
            nn.Conv2d(input_channels, 48, (1, 3), padding=(0, 1)),
            nn.BatchNorm2d(48),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Conv2d(48, 64, (3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Dropout(0.2),
        )
        self.location_head = nn.Conv1d(64, 1, 1)
        self.presence_head = nn.Linear(128, 2)
        self.count_head = nn.Linear(128, count_classes)

    def forward(self, sequence):
        encoded = self.cell_encoder(sequence.permute(0, 2, 1, 3)).mean(dim=3)
        location_logits = self.location_head(encoded).squeeze(1)
        global_features = torch.cat([encoded.mean(dim=2), encoded.amax(dim=2)], dim=1)
        return self.presence_head(global_features), self.count_head(global_features), location_logits


class TemporalPresenceDetector(nn.Module):
    def __init__(self, input_channels: int) -> None:
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(input_channels, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.SiLU(),
            nn.Conv2d(48, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.temporal = nn.GRU(64, 96, batch_first=True)
        self.head = nn.Sequential(nn.Linear(96, 64), nn.SiLU(), nn.Dropout(0.2), nn.Linear(64, 2))

    def forward(self, sequence):
        batch, cells, channels, timesteps = sequence.shape
        frames = sequence.permute(0, 3, 2, 1).reshape(batch * timesteps, channels, 4, 8)
        features = self.spatial(frames).flatten(1).reshape(batch, timesteps, 64)
        output, _ = self.temporal(features)
        return self.head(output[:, -1])


class TemporalLocationModel(nn.Module):
    def __init__(self, input_channels: int) -> None:
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(input_channels, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.SiLU(),
            nn.Conv2d(48, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
        )
        self.temporal = nn.GRU(64, 64, batch_first=True)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.SiLU(),
            nn.Conv2d(32, 1, 3, padding=1),
        )

    def forward(self, sequence):
        batch, cells, channels, timesteps = sequence.shape
        frames = sequence.permute(0, 3, 2, 1).reshape(batch * timesteps, channels, 4, 8)
        encoded = self.spatial(frames).reshape(batch, timesteps, 64, 4, 8)
        cell_sequences = encoded.permute(0, 3, 4, 1, 2).reshape(batch * 32, timesteps, 64)
        output, _ = self.temporal(cell_sequences)
        map_features = output[:, -1].reshape(batch, 4, 8, 64).permute(0, 3, 1, 2)
        return self.decoder(map_features).squeeze(1)
