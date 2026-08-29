import numpy as np
import torch

from navigation.frontier_expert import FrontierExpert


class CoverageAgentRuntime:
    def __init__(self, model, device, stall_limit: int = 40, recovery_steps: int = 24) -> None:
        self.model = model
        self.device = device
        self.stall_limit = stall_limit
        self.recovery_steps = recovery_steps
        self.frontier = FrontierExpert()
        self.stalled = 0
        self.recovery_remaining = 0

    def reset(self) -> None:
        self.stalled = 0
        self.recovery_remaining = 0

    def decide(self, observation: tuple[np.ndarray, np.ndarray], environment) -> int:
        if self.recovery_remaining > 0:
            self.recovery_remaining -= 1
            return self.frontier.decide(environment)
        with torch.no_grad():
            map_observation = torch.from_numpy(observation[0][None]).to(self.device)
            vector_observation = torch.from_numpy(observation[1][None]).to(self.device)
            logits, _ = self.model(map_observation, vector_observation)
            return int(logits.argmax(dim=1).item())

    def update(self, info: dict) -> None:
        if info["new_cells"] > 0:
            self.stalled = 0
            return
        self.stalled += 1
        if self.stalled >= self.stall_limit and self.recovery_remaining == 0:
            self.stalled = 0
            self.recovery_remaining = self.recovery_steps
