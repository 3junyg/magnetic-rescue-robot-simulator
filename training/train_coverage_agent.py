import argparse
import copy
import random
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from config import ENVIRONMENT_PRESETS
from evaluation.coverage_evaluation import evaluate_policy
from navigation.coverage_env import ACTION_NAMES, CoverageEnvironment
from navigation.coverage_policy import CoverageActorCritic
from navigation.frontier_expert import FrontierExpert


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def collect_expert_data(environments, steps: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    expert = FrontierExpert()
    maps = []
    vectors = []
    actions = []
    observations = [environment.observation() for environment in environments]
    reset_index = 0
    while len(actions) < steps:
        for index, environment in enumerate(environments):
            action = expert.decide(environment)
            maps.append(observations[index][0])
            vectors.append(observations[index][1])
            actions.append(action)
            observation, _, terminated, truncated, _ = environment.step(action)
            if terminated or truncated:
                reset_index += 1
                observation = environment.reset(environment.seed + 1000 + reset_index, list(ENVIRONMENT_PRESETS)[reset_index % len(ENVIRONMENT_PRESETS)])
            observations[index] = observation
            if len(actions) >= steps:
                break
    return np.asarray(maps, dtype=np.float32), np.asarray(vectors, dtype=np.float32), np.asarray(actions, dtype=np.int64)


def collect_dagger_data(model, environments, steps: int, device, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    expert = FrontierExpert()
    maps = []
    vectors = []
    labels = []
    observations = [environment.observation() for environment in environments]
    reset_index = 0
    rng = np.random.default_rng(seed)
    model.eval()
    while len(labels) < steps:
        map_batch = np.asarray([item[0] for item in observations], dtype=np.float32)
        vector_batch = np.asarray([item[1] for item in observations], dtype=np.float32)
        with torch.no_grad():
            logits, _ = model(torch.from_numpy(map_batch).to(device), torch.from_numpy(vector_batch).to(device))
            policy_actions = logits.argmax(dim=1).cpu().numpy()
        for index, environment in enumerate(environments):
            label = expert.decide(environment)
            maps.append(observations[index][0])
            vectors.append(observations[index][1])
            labels.append(label)
            action = label if rng.random() < 0.2 else int(policy_actions[index])
            observation, _, terminated, truncated, _ = environment.step(action)
            if terminated or truncated:
                reset_index += 1
                observation = environment.reset(environment.seed + 3000 + reset_index, list(ENVIRONMENT_PRESETS)[reset_index % len(ENVIRONMENT_PRESETS)])
            observations[index] = observation
            if len(labels) >= steps:
                break
    return np.asarray(maps, dtype=np.float32), np.asarray(vectors, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def behavior_cloning(model, maps, vectors, actions, epochs, batch_size, device) -> list[float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    counts = np.bincount(actions, minlength=len(ACTION_NAMES)).astype(np.float64)
    weights = np.zeros(len(ACTION_NAMES), dtype=np.float32)
    present = counts > 0
    weights[present] = counts.sum() / (present.sum() * counts[present])
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).to(device))
    history = []
    indices = np.arange(len(actions))
    for epoch in range(epochs):
        np.random.shuffle(indices)
        total_loss = 0.0
        for start in range(0, len(indices), batch_size):
            batch = indices[start:start + batch_size]
            map_batch = torch.from_numpy(maps[batch]).to(device)
            vector_batch = torch.from_numpy(vectors[batch]).to(device)
            action_batch = torch.from_numpy(actions[batch]).to(device)
            optimizer.zero_grad()
            logits, _ = model(map_batch, vector_batch)
            loss = criterion(logits, action_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(batch)
        average = total_loss / len(indices)
        history.append(average)
        with torch.no_grad():
            sample = indices[: min(2048, len(indices))]
            logits, _ = model(torch.from_numpy(maps[sample]).to(device), torch.from_numpy(vectors[sample]).to(device))
            accuracy = float((logits.argmax(dim=1).cpu().numpy() == actions[sample]).mean())
        print(f"behavior epoch={epoch + 1} loss={average:.6f} accuracy={accuracy:.4f}")
    return history


def ppo_train(model, environments, updates, rollout_steps, ppo_epochs, batch_size, device, seed):
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-4, weight_decay=1e-5)
    gamma = 0.995
    gae_lambda = 0.95
    clip_ratio = 0.2
    entropy_weight = 0.012
    value_weight = 0.5
    observations = [environment.observation() for environment in environments]
    recent_coverage = deque(maxlen=100)
    recent_completion = deque(maxlen=100)
    recent_collisions = deque(maxlen=100)
    history = {"coverage": [], "completion": [], "collisions": [], "policy_loss": [], "value_loss": []}
    best_score = 0.95
    best_state = copy.deepcopy(model.state_dict())
    reset_counter = 0
    rng = np.random.default_rng(seed)
    for update in range(updates):
        map_buffer = []
        vector_buffer = []
        action_buffer = []
        log_probability_buffer = []
        value_buffer = []
        reward_buffer = []
        done_buffer = []
        for _ in range(rollout_steps):
            maps = np.asarray([item[0] for item in observations], dtype=np.float32)
            vectors = np.asarray([item[1] for item in observations], dtype=np.float32)
            with torch.no_grad():
                logits, values = model(torch.from_numpy(maps).to(device), torch.from_numpy(vectors).to(device))
                distribution = torch.distributions.Categorical(logits=logits)
                actions = distribution.sample()
                log_probabilities = distribution.log_prob(actions)
            rewards = []
            dones = []
            next_observations = []
            for index, environment in enumerate(environments):
                observation, reward, terminated, truncated, info = environment.step(int(actions[index].item()))
                done = terminated or truncated
                if done:
                    recent_coverage.append(info["coverage"])
                    recent_completion.append(float(info["completed"]))
                    recent_collisions.append(info["collisions"])
                    reset_counter += 1
                    environment_name = list(ENVIRONMENT_PRESETS)[reset_counter % len(ENVIRONMENT_PRESETS)]
                    observation = environment.reset(seed + 10000 + reset_counter, environment_name)
                next_observations.append(observation)
                rewards.append(reward)
                dones.append(float(done))
            map_buffer.append(maps)
            vector_buffer.append(vectors)
            action_buffer.append(actions.cpu().numpy())
            log_probability_buffer.append(log_probabilities.cpu().numpy())
            value_buffer.append(values.cpu().numpy())
            reward_buffer.append(np.asarray(rewards, dtype=np.float32))
            done_buffer.append(np.asarray(dones, dtype=np.float32))
            observations = next_observations
        with torch.no_grad():
            next_maps = torch.from_numpy(np.asarray([item[0] for item in observations], dtype=np.float32)).to(device)
            next_vectors = torch.from_numpy(np.asarray([item[1] for item in observations], dtype=np.float32)).to(device)
            _, next_values_tensor = model(next_maps, next_vectors)
            next_values = next_values_tensor.cpu().numpy()
        rewards = np.asarray(reward_buffer)
        dones = np.asarray(done_buffer)
        values = np.asarray(value_buffer)
        advantages = np.zeros_like(rewards)
        gae = np.zeros(len(environments), dtype=np.float32)
        running_next_values = next_values
        for step in range(rollout_steps - 1, -1, -1):
            mask = 1.0 - dones[step]
            delta = rewards[step] + gamma * running_next_values * mask - values[step]
            gae = delta + gamma * gae_lambda * mask * gae
            advantages[step] = gae
            running_next_values = values[step]
        returns = advantages + values
        flat_maps = np.asarray(map_buffer).reshape(-1, *environments[0].map_shape)
        flat_vectors = np.asarray(vector_buffer).reshape(-1, environments[0].vector_size)
        flat_actions = np.asarray(action_buffer).reshape(-1)
        flat_log_probabilities = np.asarray(log_probability_buffer).reshape(-1)
        flat_advantages = advantages.reshape(-1)
        flat_returns = returns.reshape(-1)
        flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)
        indices = np.arange(len(flat_actions))
        policy_losses = []
        value_losses = []
        for _ in range(ppo_epochs):
            rng.shuffle(indices)
            for start in range(0, len(indices), batch_size):
                batch = indices[start:start + batch_size]
                logits, predicted_values = model(
                    torch.from_numpy(flat_maps[batch]).to(device),
                    torch.from_numpy(flat_vectors[batch]).to(device),
                )
                distribution = torch.distributions.Categorical(logits=logits)
                batch_actions = torch.from_numpy(flat_actions[batch]).to(device)
                new_log_probabilities = distribution.log_prob(batch_actions)
                ratio = torch.exp(new_log_probabilities - torch.from_numpy(flat_log_probabilities[batch]).to(device))
                batch_advantages = torch.from_numpy(flat_advantages[batch]).to(device)
                unclipped = ratio * batch_advantages
                clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * batch_advantages
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = nn.functional.smooth_l1_loss(predicted_values, torch.from_numpy(flat_returns[batch]).to(device))
                loss = policy_loss + value_weight * value_loss - entropy_weight * distribution.entropy().mean()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.8)
                optimizer.step()
                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
        mean_coverage = float(np.mean(recent_coverage)) if recent_coverage else float(np.mean([environment.coverage_ratio for environment in environments]))
        completion_rate = float(np.mean(recent_completion)) if recent_completion else 0.0
        mean_collisions = float(np.mean(recent_collisions)) if recent_collisions else float(np.mean([environment.collision_count for environment in environments]))
        history["coverage"].append(mean_coverage)
        history["completion"].append(completion_rate)
        history["collisions"].append(mean_collisions)
        history["policy_loss"].append(float(np.mean(policy_losses)))
        history["value_loss"].append(float(np.mean(value_losses)))
        score = mean_coverage + completion_rate - 0.001 * mean_collisions
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
        print(f"ppo update={update + 1} coverage={mean_coverage:.4f} completion={completion_rate:.3f} collisions={mean_collisions:.2f} policy_loss={history['policy_loss'][-1]:.5f} value_loss={history['value_loss'][-1]:.5f}")
    model.load_state_dict(best_state)
    return history


def save_curves(path: Path, behavior_history: list[float], ppo_history: dict) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(behavior_history)
    axes[0, 0].set_title("Behavior cloning loss")
    axes[0, 1].plot(ppo_history["coverage"], label="coverage")
    axes[0, 1].plot(ppo_history["completion"], label="completion")
    axes[0, 1].legend()
    axes[1, 0].plot(ppo_history["policy_loss"], label="policy")
    axes[1, 0].plot(ppo_history["value_loss"], label="value")
    axes[1, 0].legend()
    axes[1, 1].plot(ppo_history["collisions"])
    axes[1, 1].set_title("Mean collisions")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("models/coverage_agent"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--expert-steps", type=int, default=24000)
    parser.add_argument("--behavior-epochs", type=int, default=8)
    parser.add_argument("--dagger-rounds", type=int, default=3)
    parser.add_argument("--dagger-steps", type=int, default=6000)
    parser.add_argument("--dagger-epochs", type=int, default=3)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    environments = [CoverageEnvironment(args.seed + index, list(ENVIRONMENT_PRESETS)[index % len(ENVIRONMENT_PRESETS)], args.max_steps) for index in range(args.num_envs)]
    model = CoverageActorCritic(vector_size=environments[0].vector_size, actions=len(ACTION_NAMES)).to(device)
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    behavior_history = []
    if args.expert_steps > 0 and args.behavior_epochs > 0:
        expert_maps, expert_vectors, expert_actions = collect_expert_data(environments, args.expert_steps)
        behavior_history = behavior_cloning(model, expert_maps, expert_vectors, expert_actions, args.behavior_epochs, args.batch_size, device)
        for dagger_round in range(args.dagger_rounds):
            dagger_maps, dagger_vectors, dagger_actions = collect_dagger_data(
                model,
                environments,
                args.dagger_steps,
                device,
                args.seed + 20000 + dagger_round,
            )
            expert_maps = np.concatenate([expert_maps, dagger_maps])
            expert_vectors = np.concatenate([expert_vectors, dagger_vectors])
            expert_actions = np.concatenate([expert_actions, dagger_actions])
            round_history = behavior_cloning(
                model,
                expert_maps,
                expert_vectors,
                expert_actions,
                args.dagger_epochs,
                args.batch_size,
                device,
            )
            behavior_history.extend(round_history)
            print(f"dagger round={dagger_round + 1} samples={len(expert_actions)}")
    for index, environment in enumerate(environments):
        environment.reset(args.seed + 5000 + index, list(ENVIRONMENT_PRESETS)[index % len(ENVIRONMENT_PRESETS)])
    ppo_history = ppo_train(model, environments, args.updates, args.rollout_steps, args.ppo_epochs, args.batch_size, device, args.seed)
    evaluation = evaluate_policy(model, list(range(args.seed + 50000, args.seed + 50012)), device, args.max_steps)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "best_coverage_agent.pth"
    torch.save(
        {
            "model_state_dict": model.cpu().state_dict(),
            "map_shape": environments[0].map_shape,
            "vector_size": environments[0].vector_size,
            "actions": ACTION_NAMES,
            "forward_distance": environments[0].forward_distance,
            "turn_angle": environments[0].turn_angle,
            "max_steps": args.max_steps,
            "seed": args.seed,
        },
        model_path,
    )
    curves_path = args.output_dir / "training_curves.png"
    save_curves(curves_path, behavior_history, ppo_history)
    print(f"device={device}")
    print(f"evaluation_seeds={len(evaluation.results)}")
    print(f"mean_coverage={evaluation.mean_coverage:.6f}")
    print(f"completion_rate={evaluation.completion_rate:.6f}")
    print(f"mean_steps={evaluation.mean_steps:.2f}")
    print(f"mean_collisions={evaluation.mean_collisions:.2f}")
    print(f"model={model_path}")
    print(f"curves={curves_path}")


if __name__ == "__main__":
    main()
