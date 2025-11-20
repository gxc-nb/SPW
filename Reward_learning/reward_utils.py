import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import gym, random, torch, os, uuid
import rich
# import wandb  # wandb can be re-enabled later


def set_seed(seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False):
    """Set random seeds for reproducibility."""
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def get_indices(traj_total, config):
    """Randomly select a trajectory and sample one segment."""
    traj_idx = np.random.choice(traj_total, replace=False)
    idx_st = 500 * traj_idx + np.random.randint(0, 500 - config.segment_size)
    idx = [[j for j in range(idx_st, idx_st + config.segment_size)]]
    return idx


def get_indices_expert(expert_traj_lengths, config):
    """Randomly select a valid expert trajectory and sample one segment (global indices)."""
    valid_indices = [i for i, l in enumerate(expert_traj_lengths) if l > config.segment_size]
    if not valid_indices:
        raise ValueError("No expert trajectories longer than segment_size")

    selected_traj = np.random.choice(valid_indices)
    start_offset = sum(expert_traj_lengths[:selected_traj])
    traj_length = expert_traj_lengths[selected_traj]
    start_idx_in_traj = np.random.randint(0, traj_length - config.segment_size + 1)

    idx = [[start_offset + j for j in range(start_idx_in_traj,
                                            start_idx_in_traj + config.segment_size)]]
    return idx


def consist_test_dataset(dataset, test_feedback_num, traj_total, segment_size, threshold):
    """Build test query dataset."""
    test_traj_idx = np.random.choice(traj_total, 2 * test_feedback_num, replace=True)
    test_idx = [500 * i + np.random.randint(0, 500 - segment_size) for i in test_traj_idx]

    test_idx_st_1 = test_idx[:test_feedback_num]
    test_idx_st_2 = test_idx[test_feedback_num:]

    test_idx_1 = [[j for j in range(i, i + segment_size)] for i in test_idx_st_1]
    test_idx_2 = [[j for j in range(i, i + segment_size)] for i in test_idx_st_2]

    test_labels = obtain_labels(dataset, test_idx_1, test_idx_2,
                                segment_size=segment_size,
                                threshold=threshold,
                                noise=0.0)
    test_binary_labels = obtain_labels(dataset, test_idx_1, test_idx_2,
                                       segment_size=segment_size,
                                       threshold=0,
                                       noise=0.0)
    test_obs_act_1 = np.concatenate((dataset["observations"][test_idx_1],
                                     dataset["actions"][test_idx_1]), axis=-1)
    test_obs_act_2 = np.concatenate((dataset["observations"][test_idx_2],
                                     dataset["actions"][test_idx_2]), axis=-1)

    return test_obs_act_1, test_obs_act_2, test_labels, test_binary_labels


def collect_feedback(dataset, traj_total, config):
    """Collect preference feedback samples by random segment pairing."""
    idx_st_1, idx_st_2, labels = [], [], []
    print("config.feedback_num", config.feedback_num)

    for _ in range(config.feedback_num):
        idx_1 = get_indices(traj_total, config)
        idx_2 = get_indices(traj_total, config)
        idx_st_1.append(idx_1[0][0])
        idx_st_2.append(idx_2[0][0])

        label = obtain_labels(dataset, idx_1, idx_2,
                              segment_size=config.segment_size,
                              threshold=config.threshold,
                              noise=config.noise)
        labels.append(label[0])

    return np.array(idx_st_1), np.array(idx_st_2), np.array(labels)


def obtain_labels(dataset, idx_1, idx_2, segment_size=25, threshold=0.5, noise=0.0):
    """Generate preference labels between two segments."""
    idx_1 = np.array(idx_1)
    idx_2 = np.array(idx_2)

    reward_1 = np.sum(dataset["rewards"][idx_1], axis=1)
    reward_2 = np.sum(dataset["rewards"][idx_2], axis=1)

    labels = np.where(reward_1 < reward_2, 1, 0)
    labels = np.array([[1, 0] if i == 0 else [0, 1] for i in labels]).astype(float)

    equal_labels = np.where(np.abs(reward_1 - reward_2) <= segment_size * threshold, 1, 0)
    labels = np.array([labels[i] if equal_labels[i] == 0 else [0.5, 0.5]
                       for i in range(len(labels))])

    if noise != 0.0:
        p = noise
        for i in range(len(labels)):
            if labels[i][0] == 1:
                if random.random() < p:
                    if random.random() < 0.5:
                        labels[i] = [0, 1]
                    else:
                        labels[i] = [0.5, 0.5]
            elif labels[i][1] == 1:
                if random.random() < p:
                    if random.random() < 0.5:
                        labels[i] = [1, 0]
                    else:
                        labels[i] = [0.5, 0.5]
            else:
                if random.random() < p:
                    if random.random() < 0.5:
                        labels[i] = [0, 1]
                    else:
                        labels[i] = [1, 0]
    return labels


def attention(kd_tree, key, num_k, action_dim, beta, alpha, no_action_dim=False, attention_type="exp"):
    """Compute attention weights using distance queries."""
    distance, _ = kd_tree.query(key, k=[num_k], workers=-1)
    if attention_type == "exp":
        return squashing_func(distance, action_dim, beta, alpha, no_action_dim)
    elif attention_type == "linear":
        return linear_attention(distance, action_dim, alpha, no_action_dim)


def squashing_func(distance, action_dim, beta=0.5, alpha=1.0, no_action_dim=False):
    """Exponential decay attention."""
    if no_action_dim:
        return alpha * np.exp(-beta * distance)
    return alpha * np.exp(-beta * distance / action_dim)


def linear_attention(distance, action_dim, alpha=1.0, no_action_dim=False):
    """Linear attention decay (closer distance = higher weight)."""
    if no_action_dim:
        max_dist = np.max(distance)
        return alpha * (1 - distance / max_dist)
    max_dist = np.max(distance / action_dim)
    return alpha * (1 - (distance / action_dim) / max_dist)


def load_human_preference(config):
    """Load human preference dataset from pickle."""
    import pickle

    save_dir = f"./preference_datasets/{config.env}"
    filename = f"pref_data_fn{config.feedback_num}_s{config.segment_size}_human.pkl"
    load_path = os.path.join(save_dir, filename)

    if not os.path.exists(load_path):
        print(f"Preference dataset not found: {load_path}")
        return None

    with open(load_path, "rb") as f:
        preference_data = pickle.load(f)

    obs_1 = preference_data["observations_1"]
    act_1 = preference_data["actions_1"]
    obs_2 = preference_data["observations_2"]
    act_2 = preference_data["actions_2"]
    labels = preference_data["labels"]
    idx_st_1 = preference_data["idx_st_1"]
    idx_st_2 = preference_data["idx_st_2"]

    obs_act_1 = np.concatenate((obs_1, act_1), axis=-1)
    obs_act_2 = np.concatenate((obs_2, act_2), axis=-1)

    print(f"Loaded preference dataset: {load_path}")
    print(f"Dataset size: {len(labels)} pairs")

    return idx_st_1, idx_st_2, labels, obs_act_1, obs_act_2
