#!/bin/bash

# We use a global batch size of 8 during GRPO
GLOBAL_BATCH_SIZE=8
BATCH_PER_DEVICE=1
NUM_DEVICES=8
GRAD_ACCUM_STEPS=1

# Path to the Video-R2 SFT model (e.g., the model after SFT stage)
MODEL_NAME="path/to/the/Video-R2-SFT-Model"
# Path to the SFT dataset json: video-r2-grpo-dataset.json
DATA_PATH="path/to/video-r2-grpo-dataset.json"
# Path to the videos (extracted from: https://huggingface.co/datasets/MBZUAI/Video-R2-Dataset/tree/main/videos)
DATA_FOLDER="path/to/videos"
# Path to the subtitles folder (https://huggingface.co/datasets/MBZUAI/Video-R2-Dataset/blob/main/subtitles.tar)
SUBTITLES_FOLDER="path/to/subtitles"
# Path to save the checkpoints and training progress
OUTPUT_DIR="path/to/output_folder"

# We write timestamps and subtitles on the corresponding frames of the video
export WRITE_TIMESTAMPS_ON_FRAMES=True
export WRITE_SUBTITLES_ON_FRAMES=True
# We sample up to 32 frames per video during SFT
export FPS_MAX_FRAMES=32

# LLM Judge Server related exports (e.g., server address and port)
# Qwen/Qwen3-Next-80B-A3B-Instruct should be hosted on the server before running the GRPO stage
export LLM_SERVER_ADDRESS="server-address or ip"
export SERVER_PORT="8011"

# Proxy setup - may not be required in your case
export no_proxy="${LLM_SERVER_ADDRESS},$(hostname),$(hostname -f)"
export NO_PROXY="$no_proxy"

# We use OpenAI Python API to query the hosted LLM
export OPENAI_API_BASE="http://${LLM_SERVER_ADDRESS}:${SERVER_PORT}/v1"
# The model names should be the same as of the hosted model
export SERVED_MODEL="Qwen/Qwen3-Next-80B-A3B-Instruct"


torchrun --nproc-per-node=8 src/train/train_grpo.py \
    --deepspeed scripts/zero3.json \
    --model_id "$MODEL_NAME" \
    --data_path "$DATA_PATH" \
    --image_folder "$DATA_FOLDER" \
    --video_subtitles_folder "$SUBTITLES_FOLDER" \
    --freeze_vision_tower True \
    --freeze_llm False \
    --freeze_merger True \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 1 \
    --num_generations 8 \
    --beta 0.04 \
    --per_device_train_batch_size "$BATCH_PER_DEVICE" \
    --gradient_accumulation_steps "$GRAD_ACCUM_STEPS" \
    --max_completion_length 1024 \
    --max_prompt_length 32768 \
    --reward_llm_judge '' \
    --reward_func_names "accuracy_reward,format_reward,temporal_grounding_sentence_embedding_consistency_reward" \
    --reward_weights 1.0,1.0,1.0 \
    --buffer_seconds 2 \
    --similarity_threshold 0.75 \
    --video_max_pixels $((360 * 420)) \
    --fps 2.0 \
    --learning_rate 1e-6 \
    --remove_unused_columns False \
    --weight_decay 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 False \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 2 \
    --dataloader_num_workers 4
