python -m debugpy --wait-for-client --listen localhost:5678 Track_1/scripts/open_lock_sim_evaluation.py \
    --team_name welcome_the_next_team \
    --model_name TD3PolicyForLongOpenLockPointFlowEnv \
    --policy_file_path ManiSkill-ViTac2025_ckpt/ckpt/Track_1/pretrain_weight/lock_opening/best_model.zip