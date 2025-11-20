import numpy as np
import gym
import pyrallis
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import random, os, tqdm, copy, rich
# import wandb  # commented out for now
import uuid
from dataclasses import asdict, dataclass
from scipy.spatial import KDTree
import reward_utils
from reward_utils import collect_feedback, consist_test_dataset, attention, load_human_preference
from reward_model import RewardModel
import sys

sys.path.append("./algorithms")
import utils_env


@dataclass
class TrainConfig:
    device: str = "cuda"
    env: str = "metaworld_box-close-v2"
    seed: int = 0
    checkpoints_path: Optional[str] = None
    load_model: str = ""
    feedback_num: int = 200
    data_quality: float = 1.0
    segment_size: int = 25
    normalize: bool = True
    threshold: float = 0.5
    model_type: str = "BT"
    noise: float = 0.0
    human: bool = False

    # MLP settings
    epochs: int = int(1e3)
    batch_size: int = 512
    activation: str = "tanh"
    lr: float = 1e-3
    hidden_sizes: int = 128
    ensemble_num: int = 3
    ensemble_method: str = "mean"

    num_top_episodes: int = 1
    k: int = 1
    rd_gamma: float = 0
    spw_tau: float = 0.7
    mode: str = "MR"
    attention_type: str = "exp"

    def __post_init__(self):
        checkpoints_name = (
            f"{self.env}/fn_{self.feedback_num}/mode_{self.mode}/at_{self.attention_type}/n_{self.noise}/e_{self.epochs}/s_{self.seed}"
        )
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, checkpoints_name)
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
    """ Main training function """
    rich.print(config)
    reward_utils.set_seed(config.seed)

    # Load dataset
    if "metaworld" in config.env:
        env_name = config.env.replace("metaworld-", "")
        dataset = utils_env.MetaWorld_dataset(config)
        top_n_dataset, expert_episode_lengths = utils_env.metaworld_extract_top_episodes(config)
    elif "dmc" in config.env:
        env_name = config.env.replace("dmc-", "")
        dataset = utils_env.DMC_dataset(config)
        config.threshold *= 0.1
        top_n_dataset = utils_env.dmc_extract_top_episodes(config)
        traj_total_expert = top_n_dataset["observations"].shape[0] // 500

    traj_total = dataset["observations"].shape[0] // 500

    # Normalize dataset
    if config.normalize:
        state_mean, state_std = reward_utils.compute_mean_std(dataset["observations"], eps=1e-3)
    else:
        state_mean, state_std = 0, 1

    dataset["observations"] = reward_utils.normalize_states(dataset["observations"], state_mean, state_std)
    dataset["next_observations"] = reward_utils.normalize_states(dataset["next_observations"], state_mean, state_std)

    top_n_dataset["observations"] = reward_utils.normalize_states(top_n_dataset["observations"], state_mean, state_std)
    top_n_dataset["next_observations"] = reward_utils.normalize_states(top_n_dataset["next_observations"], state_mean, state_std)

    # Build KDTree for expert data
    data = np.hstack([top_n_dataset["observations"], top_n_dataset["actions"]])
    kd_tree = KDTree(data)

    # Compute attention weights for dataset
    key = np.hstack([dataset["observations"], dataset["actions"]])
    action_dim = dataset["actions"].shape[-1]
    beta, alpha = 0.5, 1.0
    dataset["attention"] = attention(
        kd_tree,
        key,
        num_k=config.k,
        action_dim=action_dim,
        beta=beta,
        alpha=alpha,
        no_action_dim=False,
        attention_type=config.attention_type
    )

    # Load human preference if needed
    if config.human:
        loaded_data = load_human_preference(config)
        idx_st_1, idx_st_2, labels, obs_act_1, obs_act_2 = loaded_data
    else:
        idx_st_1, idx_st_2, labels = collect_feedback(dataset, traj_total, config)

    # D-REX data augmentation mode
    if config.mode == 'D-REX':
        aug_feedback_num = 100
        idx_st_1_aug = []
        idx_st_2_aug = []
        labels_aug = []

        collected_pairs = 0
        print(f"Starting data augmentation, target: {aug_feedback_num} samples")

        while collected_pairs < aug_feedback_num:
            if "metaworld" in config.env:
                idx_1 = reward_utils.get_indices_expert(expert_episode_lengths, config)
            elif "dmc" in config.env:
                idx_1 = reward_utils.get_indices(traj_total_expert, config)

            idx_2 = reward_utils.get_indices(traj_total, config)

            idx_st_1_aug.append(idx_1[0][0])
            idx_st_2_aug.append(idx_2[0][0])
            labels_aug.append([1, 0])
            collected_pairs += 1

        print(f"Data augmentation done! Collected: {collected_pairs} samples")

        # Build index ranges
        idx_1 = [[j for j in range(i, i + config.segment_size)] for i in idx_st_1]
        idx_2 = [[j for j in range(i, i + config.segment_size)] for i in idx_st_2]
        idx_1_aug = [[j for j in range(i, i + config.segment_size)] for i in idx_st_1_aug]
        idx_2_aug = [[j for j in range(i, i + config.segment_size)] for i in idx_st_2_aug]

        # Merge original and augmented data
        obs_act_1 = np.concatenate([
            np.concatenate((dataset["observations"][idx_1], dataset["actions"][idx_1]), axis=-1),
            np.concatenate((top_n_dataset["observations"][idx_1_aug], top_n_dataset["actions"][idx_1_aug]), axis=-1)
        ], axis=0)

        obs_act_2 = np.concatenate([
            np.concatenate((dataset["observations"][idx_2], dataset["actions"][idx_2]), axis=-1),
            np.concatenate((dataset["observations"][idx_2_aug], dataset["actions"][idx_2_aug]), axis=-1)
        ], axis=0)

        attention_1 = np.ones((len(idx_1) + len(idx_1_aug), config.segment_size, 1), dtype=np.float32)
        attention_2 = np.ones((len(idx_2) + len(idx_2_aug), config.segment_size, 1), dtype=np.float32)
        labels = np.concatenate([labels, np.array(labels_aug)])

    else:
        # No augmentation
        idx_1 = [[j for j in range(i, i + config.segment_size)] for i in idx_st_1]
        idx_2 = [[j for j in range(i, i + config.segment_size)] for i in idx_st_2]

        obs_act_1 = np.concatenate((dataset["observations"][idx_1], dataset["actions"][idx_1]), axis=-1)
        obs_act_2 = np.concatenate((dataset["observations"][idx_2], dataset["actions"][idx_2]), axis=-1)
        attention_1 = dataset["attention"][idx_1]
        attention_2 = dataset["attention"][idx_2]

    # Build test set for evaluation
    test_feedback_num = 5000
    test_obs_act_1, test_obs_act_2, test_labels, test_binary_labels = consist_test_dataset(
        dataset,
        test_feedback_num,
        traj_total,
        segment_size=config.segment_size,
        threshold=config.threshold,
    )

    dimension = obs_act_1.shape[-1]
    reward_model = RewardModel(config, obs_act_1, obs_act_2, attention_1, attention_2, labels, dimension)
    reward_model.save_test_dataset(test_obs_act_1, test_obs_act_2, test_labels, test_binary_labels)

    # Pretraining if in R-P mode
    if config.mode == "R-P":
        reward_model.pretrain_with_attention(dataset, config)

    # Training
    # wandb_init(config)
    reward_model.train_model(mode=config.mode)
    # wandb.finish()
    reward_model.save_model(config.checkpoints_path)


if __name__ == "__main__":
    train()
