# 🔍 SPW: Search-based Preference Weighting

This repository implements **SPW (Search-based Preference Weighting)** — a unified single-stage framework for learning reward functions from both demonstrations and preference feedback.

```plaintext
📦 SPW
├── algorithms/                  # RL algorithms
│   ├── BC.py                    # Behavior Cloning (BC)

│   ├── iql.py                   # IQL policy learning

│   └── utils_env.py             # Environment & dataset utilities

├── Reward_learning/             # Reward model components
│   ├── learn_reward.py          # Train reward model
│   ├── reward_model.py          # Reward model architecture
│   └── reward_utils.py          # Reward learning utilities
│
├── configs/                     # YAML config files
│   ├── bc.yaml

│   ├── iql.yaml

│   └── reward.yaml
│
├── dataset/                     # MetaWorld & DMC datasets
│
├── preference_datasets/         # Optional human preference data
│
├── scripts/                     # Example scripts (e.g., run experiments)
│
├── SPW.yml                      # Conda environment file
│
└── README.md


## ⚙️ Installation

Create a conda environment and install dependencies:

```bash
conda env create -f SPW.yml
pip install git+https://github.com/Farama-Foundation/Metaworld.git@master#egg=metaworld
pip install git+https://github.com/denisyarats/dmc2gym.git
```

🚀 Quick Start

1️⃣ Train the Reward Model (SPW mode)

```bash
python Reward_learning/learn_reward.py \
  --config=configs/reward.yaml \
  --env=metaworld_box-close-v2 \
  --mode=SPW \
  --spw_tau=0.7
```

2️⃣ Run IQL with SPW Reward

```bash
python algorithms/iql.py \
  --config=configs/iql.yaml \
  --use_reward_model=True \
  --env=metaworld_box-close-v2 \
  --mode=SPW \
  --spw_tau=0.7
```

Or run the full pipeline with:

```bash
bash scripts/example.sh
```

# 📌 Algorithms

In this repo, we can run the following reward-learning methods:

- **MR** – MLP Reward Model  
- **BC-P** – Behavior Cloning Pretraining  
- **R-P** – Reward Pretraining  
- **RD** – Reward Distribution  
- **D-REX** – Disturbance-based Reward Extrapolation  
- **SPW** – Search-based Preference Weighting  

For other baselines, we experimented with the following repositories:

| Algorithm | URL |
|----------|-----|
| **OPRL** (Offline Preference-based Reward Learning) | https://github.com/danielshin1/oprl |
| **PT** (Preference Transformer) | https://github.com/csmile-1006/PreferenceTransformer |
| **IPL** (Inverse Preference Learning) | https://github.com/jhejna/inverse-preference-learning |
| **LiRE** (Listwise Reward Estimation) | https://github.com/chwoong/LiRE |
