import torch
from torch import nn


class CoverageActorCritic(nn.Module):
    def __init__(self, map_channels: int = 4, vector_size: int = 57, actions: int = 4) -> None:
        super().__init__()
        self.map_encoder = nn.Sequential(
            nn.Conv2d(map_channels, 24, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(24, 48, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(48, 64, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((3, 4)),
            nn.Flatten(),
        )
        self.vector_encoder = nn.Sequential(nn.Linear(vector_size, 96), nn.SiLU(), nn.Linear(96, 96), nn.SiLU())
        self.shared = nn.Sequential(nn.Linear(64 * 3 * 4 + 96, 256), nn.SiLU(), nn.Linear(256, 128), nn.SiLU())
        self.actor = nn.Linear(128, actions)
        self.critic = nn.Linear(128, 1)

    def forward(self, map_observation, vector_observation):
        features = torch.cat([self.map_encoder(map_observation), self.vector_encoder(vector_observation)], dim=1)
        features = self.shared(features)
        return self.actor(features), self.critic(features).squeeze(1)
