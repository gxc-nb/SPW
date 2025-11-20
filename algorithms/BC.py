import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import wandb
from dataclasses import asdict, dataclass
import pyrallis
import os
import rich
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
sys.path.append("./Reward_learning")
import reward_utils
import utils_env
from iql import GaussianPolicy
from iql import wrap_env
from iql import eval_actor

@dataclass
class TrainConfig:
    # Experiment setup
    device: str = "cuda"
    env: str = "metaworld_box-close-v2"
    data_quality: float = 5.0
    seed: int = 0
    checkpoints_path: Optional[str] = None
    human: bool = False

    # BC settings
    epochs: int = 50
    batch_size: int = 256
    lr: float = 1e-3
    hidden_sizes: int = 256
    normalize: bool = True

    # Data
    num_top_episodes: int = 1

    def __post_init__(self):
        checkpoint_name = f"BC/{self.env}/seed_{self.seed}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, checkpoint_name)
            if not os.path.exists(self.checkpoints_path):
                os.makedirs(self.checkpoints_path)

def wandb_init(config: TrainConfig) -> None:
    # wandb.init(
    #     project="",
    #     entity="",
    #     name=f"{config.env}_seed{config.seed}",
    #     config=asdict(config),
    # )
    # wandb.run.save()
    pass

@pyrallis.wrap()
def train(config: TrainConfig):
    rich.print(config)
    reward_utils.set_seed(config.seed)

    # Load dataset
    if "metaworld" in config.env:
        dataset = utils_env.MetaWorld_dataset(config)
        top_n_dataset = utils_env.metaworld_extract_top_episodes(config)
    elif "dmc" in config.env:
        dataset = utils_env.DMC_dataset(config)
        top_n_dataset = utils_env.dmc_extract_top_episodes(config)

    # Normalize dataset
    if config.normalize:
        state_mean, state_std = reward_utils.compute_mean_std(dataset["observations"], eps=1e-3)
    else:
        state_mean, state_std = 0, 1

    top_n_dataset["observations"] = reward_utils.normalize_states(
        top_n_dataset["observations"], state_mean, state_std
    )

    # Init policy network
    state_dim = top_n_dataset["observations"].shape[1]
    action_dim = top_n_dataset["actions"].shape[1]
    max_action = 1.0

    policy = GaussianPolicy(
        state_dim=state_dim,
        act_dim=action_dim,
        max_action=max_action,
        hidden_dim=config.hidden_sizes,
        n_hidden=2
    ).to(config.device)

    optimizer = torch.optim.Adam(policy.parameters(), lr=config.lr)
    scheduler = StepLR(optimizer, step_size=100, gamma=1.0)

    # Convert to tensor
    states = torch.FloatTensor(top_n_dataset["observations"]).to(config.device)
    actions = torch.FloatTensor(top_n_dataset["actions"]).to(config.device)

    n_samples = len(states)

    # Evaluation env
    if "metaworld" in config.env:
        eval_env = utils_env.make_metaworld_env(config.env, config.seed)
    elif "dmc" in config.env:
        eval_env = utils_env.make_dmc_env(config.env, config.seed)
    else:
        import gym
        eval_env = gym.make(config.env)

    if config.normalize:
        eval_env = wrap_env(eval_env, state_mean=state_mean, state_std=state_std)

    # Training loop
    for epoch in range(config.epochs):
        perm = torch.randperm(n_samples)
        total_loss = 0

        for i in range(0, n_samples, config.batch_size):
            batch_indices = perm[i:i + config.batch_size]
            state_batch = states[batch_indices]
            action_batch = actions[batch_indices]

            dist = policy(state_batch)
            log_prob = dist.log_prob(action_batch).sum(-1)
            loss = -log_prob.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / (n_samples / config.batch_size)
        current_lr = scheduler.get_last_lr()[0]

        # ✅ Keep only essential print
        print(f"[Epoch {epoch+1}/{config.epochs}] Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")

    # Final evaluation
    eval_scores, eval_success = eval_actor(
        eval_env,
        config.env,
        policy,
        device=config.device,
        n_episodes=50,
        seed=config.seed,
    )
    eval_score = eval_scores.mean()
    eval_success = eval_success.mean() * 100

    print("\n========== Final Evaluation ==========")
    print(f"Score: {eval_score:.3f} | Success: {eval_success:.1f}%")
    print("=====================================")

    # Save evaluation results
    results_dir = "/aiarena/nas/KDPrior"
    results_file = os.path.join(results_dir, "bc_evaluation_results.txt")
    with open(results_file, "a") as f:
        f.write(f"Env: {config.env}, num_top_episodes: {config.num_top_episodes}, seed: {config.seed}\n")
        f.write(f"Score: {eval_score:.3f}, Success: {eval_success:.1f}%\n")
        f.write("-"*50 + "\n")

    print(f"Results saved to: {results_file}")

    # Save model
    if config.checkpoints_path is not None:
        save_path = os.path.join(config.checkpoints_path, "bc_policy.pt")
        torch.save(policy.state_dict(), save_path)
        print(f"Model saved to: {save_path}")

if __name__ == "__main__":
    train()
