
MODEL_DIR="3shape_1.5mm_2025-01-26_01-22-07.819"
MODEL_NAME="best_model"

python Track_1/scripts/peg_insertion_sim_evaluation.py \
    --team_name UESTCSCU \
    --model_name TD3PolicyForPointFlowEnv \
    --policy_file_path "Track_1/training_log/$MODEL_DIR/$MODEL_NAME.zip"