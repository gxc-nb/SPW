#!/bin/bash

# # 切换到分支
# git checkout main || { echo "Git checkout failed"; exit 1; }

env=metaworld_box-close-v2
data_quality=8.0        
epochs=300     
activation=tanh  
threshold=0.5
noise=0.0  
batch_size=512
ensemble_num=3   
ensemble_method=mean
human=True
seed=1
feedback_num=200
segment_size=25
num_top_episodes=1
mode=SPW   # Options: MR / SPW / BC-P / R-P / D-REX / RD

# Extra params
spw_tau=0.7
rd_gamma=20

extra_args=""
# Add mode-specific params
if [ "$mode" == "SPW" ]; then
    extra_args="--spw_tau=$spw_tau"
elif [ "$mode" == "RD" ]; then
    extra_args="--rd_gamma=$rd_gamma"
fi

# GPU setup
export CUDA_VISIBLE_DEVICES=0

# Run BC pretraining first if mode=BC-P
if [ "$mode" == "BC-P" ]; then
    echo "▶ Running BC pretraining first..."
    python3 algorithms/BC.py \
        --config=configs/bc.yaml \
        --env=$env \
        --seed=$seed \
        --num_top_episodes=$num_top_episodes
fi

# Run Reward Learning
python3 Reward_learning/learn_reward.py \
    --config=configs/reward.yaml \
    --env=$env \
    --data_quality=$data_quality \
    --feedback_num=$feedback_num \
    --seed=$seed \
    --mode=$mode \
    --epochs=$epochs \
    --activation=$activation \
    --num_top_episodes=$num_top_episodes \
    --ensemble_num=$ensemble_num \
    --ensemble_method=$ensemble_method \
    --segment_size=$segment_size \
    --threshold=$threshold \
    --noise=$noise \
    --human=$human \
    --batch_size=$batch_size \
    $extra_args \
    --checkpoints_path=reward_model/

# Run IQL
python3 algorithms/iql.py \
    --config=configs/iql.yaml \
    --env=$env \
    --data_quality=$data_quality \
    --feedback_num=$feedback_num \
    --seed=$seed \
    --mode=$mode \
    --epochs=$epochs \
    --activation=$activation \
    --num_top_episodes=$num_top_episodes \
    --ensemble_num=$ensemble_num \
    --ensemble_method=$ensemble_method \
    --segment_size=$segment_size \
    --threshold=$threshold \
    --noise=$noise \
    --human=$human \
    --batch_size=$batch_size \
    --use_reward_model=True \
    $extra_args \
    --max_timesteps=250_000 \
    --eval_freq=5000
