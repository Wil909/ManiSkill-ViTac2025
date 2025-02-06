#!/bin/bash

MODEL_DIR="long_open_lock_2025-01-26_01-35-10.511"
MODEL_NAME="best_model"

python Track_1/scripts/open_lock_sim_evaluation.py \
    --team_name UESTCSCU \
    --model_name TD3PolicyForLongOpenLockPointFlowEnv \
    --policy_file_path "Track_1/training_log/$MODEL_DIR/$MODEL_NAME.zip"