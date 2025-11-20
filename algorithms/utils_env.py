import numpy as np
import torch
import torch.nn.functional as F
import gym
import os
import dmc2gym
import metaworld
import metaworld.envs.mujoco.env_dict as _env_dict
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from gym.wrappers.time_limit import TimeLimit
from rlkit.envs.wrappers import NormalizedBoxEnv
import pickle as pkl


def make_metaworld_env(env_name, seed):
    """Create a MetaWorld environment"""
    env_name = env_name.replace("metaworld_", "")
    if env_name in _env_dict.ALL_V2_ENVIRONMENTS:
        env_cls = _env_dict.ALL_V2_ENVIRONMENTS[env_name]
    else:
        env_cls = _env_dict.ALL_V1_ENVIRONMENTS[env_name]

    env = env_cls()
    env._partially_observable = False
    env._freeze_rand_vec = False
    env._set_task_called = True
    env.seed(seed)
    return TimeLimit(NormalizedBoxEnv(env), env.max_path_length)


def make_dmc_env(env_name, seed):
    """Create a DMC environment"""
    env_name = env_name.replace("dmc_", "")
    domain_name, task_name = env_name.split("-")
    env = dmc2gym.make(domain_name=domain_name.lower(), task_name=task_name.lower(), seed=seed)
    return env


def MetaWorld_dataset(config):
    """Load MetaWorld dataset"""
    base_path = os.path.join(os.getcwd(), "dataset/MetaWorld/")
    env_name = config.env
    base_path += str(env_name.replace("metaworld_", ""))
    dataset = dict()
    for seed in range(3):
        path = base_path + f"/saved_replay_buffer_1000000_seed{seed}.pkl"
        with open(path, "rb") as f:
            load_dataset = pkl.load(f)

        for key in load_dataset.keys():
            load_dataset[key] = load_dataset[key][: int(config.data_quality * 100_000)]
        load_dataset["terminals"] = load_dataset["dones"][: int(config.data_quality * 100_000)]
        load_dataset.pop("dones", None)

        for key in load_dataset.keys():
            if key not in dataset:
                dataset[key] = load_dataset[key]
            else:
                dataset[key] = np.concatenate((dataset[key], load_dataset[key]), axis=0)

    N = dataset["rewards"].shape[0]
    obs_, next_obs_, action_, reward_, done_ = [], [], [], [], []

    dataset["rewards"] = dataset["rewards"].reshape(-1)
    dataset["terminals"] = dataset["terminals"].reshape(-1)

    for i in range(N):
        obs_.append(dataset["observations"][i].astype(np.float32))
        next_obs_.append(dataset["next_observations"][i].astype(np.float32))
        action_.append(dataset["actions"][i].astype(np.float32))
        reward_.append(dataset["rewards"][i].astype(np.float32))
        done_.append(bool(dataset["terminals"][i]))

    return {
        "observations": np.array(obs_),
        "actions": np.array(action_),
        "next_observations": np.array(next_obs_),
        "rewards": np.array(reward_),
        "terminals": np.array(done_),
    }


def DMC_dataset(config):
    """Load DMC dataset"""
    base_path = os.path.join(os.getcwd(), "dataset/DMControl/")
    env_name = config.env.replace("dmc_", "")
    base_path += str(env_name)
    dataset = dict()
    for seed in range(3):
        path = base_path + f"/saved_replay_buffer_1000000_seed{seed}.pkl"
        with open(path, "rb") as f:
            load_dataset = pkl.load(f)

        if "humanoid" in env_name:
            for key in load_dataset.keys():
                load_dataset[key] = load_dataset[key][200000 : int(config.data_quality * 100_000)]
            load_dataset["terminals"] = load_dataset["dones"][0 : int(config.data_quality * 100_000) - 200000]
            load_dataset.pop("dones", None)
        else:
            for key in load_dataset.keys():
                load_dataset[key] = load_dataset[key][0 : int(config.data_quality * 100_000)]
            load_dataset["terminals"] = load_dataset["dones"][0 : int(config.data_quality * 100_000)]
            load_dataset.pop("dones", None)

        for key in load_dataset.keys():
            if key not in dataset:
                dataset[key] = load_dataset[key]
            else:
                dataset[key] = np.concatenate((dataset[key], load_dataset[key]), axis=0)

    N = dataset["rewards"].shape[0]
    obs_, next_obs_, action_, reward_, done_ = [], [], [], [], []

    dataset["rewards"] = dataset["rewards"].reshape(-1)
    dataset["terminals"] = dataset["terminals"].reshape(-1)

    for i in range(N):
        obs_.append(dataset["observations"][i].astype(np.float32))
        next_obs_.append(dataset["next_observations"][i].astype(np.float32))
        action_.append(dataset["actions"][i].astype(np.float32))
        reward_.append(dataset["rewards"][i].astype(np.float32))
        done_.append(bool(dataset["terminals"][i]))

    return {
        "observations": np.array(obs_),
        "actions": np.array(action_),
        "next_observations": np.array(next_obs_),
        "rewards": np.array(reward_),
        "terminals": np.array(done_),
    }


def metaworld_extract_top_episodes(config):
    """Extract top-n MetaWorld episodes (only keep up to first success)."""
    base_path = os.path.join(os.getcwd(), "dataset/MetaWorld/")
    env_name = config.env
    base_path += str(env_name.replace("metaworld_", ""))
    dataset = dict()

    for seed in range(3):
        path = base_path + f"/saved_replay_buffer_1000000_seed{seed}.pkl"
        with open(path, "rb") as f:
            load_dataset = pkl.load(f)
        load_dataset["terminals"] = load_dataset.pop("dones")

        for key in load_dataset.keys():
            if key not in dataset:
                dataset[key] = load_dataset[key]
            else:
                dataset[key] = np.concatenate((dataset[key], load_dataset[key]), axis=0)

    episode_ends = np.where(dataset['terminals'])[0]
    episode_starts = np.concatenate(([0], episode_ends[:-1] + 1))

    successful_episodes = []
    for start, end in zip(episode_starts, episode_ends):
        if 'success' in dataset:
            success_positions = np.where(dataset['success'][start:end+1] > 0.5)[0]
            if len(success_positions) > 0:
                first_success_idx = success_positions[0] + start
                total_reward = np.sum(dataset['rewards'][start:end+1])
                successful_episodes.append((start, first_success_idx, total_reward))

    if not successful_episodes:
        return {
            'observations': np.array([]),
            'actions': np.array([]),
            'next_observations': np.array([]),
            'rewards': np.array([]),
            'success': np.array([]) if 'success' in dataset else None
        }, []

    n = min(config.num_top_episodes, len(successful_episodes))
    successful_episodes.sort(key=lambda x: x[2], reverse=True)
    top_episodes = successful_episodes[:n]

    top_n_dataset = {'observations': [], 'actions': [], 'next_observations': [], 'rewards': []}
    if 'success' in dataset:
        top_n_dataset['success'] = []

    expert_episode_lengths = []

    for start, first_success_idx, _ in top_episodes:
        data_range = slice(start, first_success_idx + 1)
        expert_episode_lengths.append(first_success_idx - start + 1)

        top_n_dataset['observations'].append(dataset['observations'][data_range].astype(np.float32))
        top_n_dataset['actions'].append(dataset['actions'][data_range].astype(np.float32))
        top_n_dataset['next_observations'].append(dataset['next_observations'][data_range].astype(np.float32))
        top_n_dataset['rewards'].append(dataset['rewards'][data_range].astype(np.float32))

        if 'success' in dataset:
            top_n_dataset['success'].append(dataset['success'][data_range].astype(np.float32))

    for key in top_n_dataset:
        if top_n_dataset[key]:
            top_n_dataset[key] = np.concatenate(top_n_dataset[key], axis=0)
        else:
            top_n_dataset[key] = np.array([])

    return top_n_dataset, expert_episode_lengths


def dmc_extract_top_episodes(config):
    """Extract top-n episodes from DMC dataset."""
    base_path = os.path.join(os.getcwd(), "dataset/DMControl/")
    env_name = config.env.replace("dmc_", "")
    base_path += str(env_name)
    dataset = dict()

    for seed in range(3):
        path = base_path + f"/saved_replay_buffer_1000000_seed{seed}.pkl"
        with open(path, "rb") as f:
            load_dataset = pkl.load(f)
        load_dataset["terminals"] = load_dataset.pop("dones")

        if "humanoid" in env_name:
            start_idx = 200000
            for key in load_dataset.keys():
                load_dataset[key] = load_dataset[key][start_idx:]

        for key in load_dataset.keys():
            if key not in dataset:
                dataset[key] = load_dataset[key]
            else:
                dataset[key] = np.concatenate((dataset[key], load_dataset[key]), axis=0)

    episode_ends = np.where(dataset['terminals'])[0]
    episode_starts = np.concatenate(([0], episode_ends[:-1] + 1))

    episode_rewards = []
    episode_indices = []
    for start, end in zip(episode_starts, episode_ends):
        total_reward = np.sum(dataset['rewards'][start:end+1])
        episode_rewards.append(total_reward)
        episode_indices.append((start, end))

    n = config.num_top_episodes
    top_n_indices = np.argsort(episode_rewards)[-n:]

    top_n_dataset = {'observations': [], 'actions': [], 'next_observations': []}
    for idx in top_n_indices:
        start, end = episode_indices[idx]
        top_n_dataset['observations'].append(dataset['observations'][start:end+1].astype(np.float32))
        top_n_dataset['actions'].append(dataset['actions'][start:end+1].astype(np.float32))
        top_n_dataset['next_observations'].append(dataset['next_observations'][start:end+1].astype(np.float32))

    for key in top_n_dataset:
        top_n_dataset[key] = np.concatenate(top_n_dataset[key], axis=0)

    return top_n_dataset
