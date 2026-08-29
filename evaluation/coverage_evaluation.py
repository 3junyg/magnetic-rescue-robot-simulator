from dataclasses import dataclass

import numpy as np

from config import ENVIRONMENT_PRESETS
from navigation.coverage_env import CoverageEnvironment
from navigation.coverage_runtime import CoverageAgentRuntime


@dataclass
class CoverageEvaluation:
    mean_coverage: float
    completion_rate: float
    mean_steps: float
    mean_collisions: float
    results: list[dict]


def evaluate_policy(model, seeds: list[int], device, max_steps: int = 1000, use_recovery: bool = False) -> CoverageEvaluation:
    environments = list(ENVIRONMENT_PRESETS)
    results = []
    model.eval()
    import torch
    with torch.no_grad():
        for index, seed in enumerate(seeds):
            environment = CoverageEnvironment(seed, environments[index % len(environments)], max_steps)
            observation = environment.observation()
            runtime = CoverageAgentRuntime(model, device) if use_recovery else None
            done = False
            info = {"coverage": environment.coverage_ratio, "steps": 0, "collisions": 0, "completed": False}
            while not done:
                if runtime is None:
                    map_observation = torch.from_numpy(observation[0][None]).to(device)
                    vector_observation = torch.from_numpy(observation[1][None]).to(device)
                    logits, _ = model(map_observation, vector_observation)
                    action = int(logits.argmax(dim=1).item())
                else:
                    action = runtime.decide(observation, environment)
                observation, _, terminated, truncated, info = environment.step(action)
                if runtime is not None:
                    runtime.update(info)
                done = terminated or truncated
            results.append(info)
    return CoverageEvaluation(
        mean_coverage=float(np.mean([item["coverage"] for item in results])),
        completion_rate=float(np.mean([item["completed"] for item in results])),
        mean_steps=float(np.mean([item["steps"] for item in results])),
        mean_collisions=float(np.mean([item["collisions"] for item in results])),
        results=results,
    )
