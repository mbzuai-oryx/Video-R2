#!/bin/bash

# We use a global batch size of 32 during SFT
GLOBAL_BATCH_SIZE=32
BATCH_PER_DEVICE=1
NUM_DEVICES=8
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

# The base model path
MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
# Path to the SFT dataset json: video-r2-sft-dataset.json
DATA_PATH="path/to/video-r2-sft-dataset.json"
# Path to the videos (extracted from: https://huggingface.co/datasets/MBZUAI/Video-R2-Dataset/tree/main/videos)
DATA_FOLDER="path/to/videos"
# Path to the subtitles folder (https://huggingface.co/datasets/MBZUAI/Video-R2-Dataset/blob/main/subtitles.tar)
SUBTITLES_FOLDER="path/to/subtitles"
# Path to save the checkpoints and training progress
OUTPUT_DIR="path/to/output_folder"

# We write timestamps and subtitles on the corresponding frames of the video
export WRITE_TIMESTAMPS_ON_FRAMES=True
export WRITE_SUBTITLES_ON_FRAMES=True
# We sample up to 128 frames per video during SFT
export FPS_MAX_FRAMES=128


torchrun --nproc-per-node=8 src/train/train_sft.py \
  --use_liger False \
  --deepspeed scripts/zero3.json \
  --lora_enable True \
  --use_dora False \
  --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
  --lora_rank 64 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --num_lora_modules -1 \
  --model_id $MODEL_NAME \
  --data_path $DATA_PATH \
  --image_folder $DATA_FOLDER \
  --video_subtitles_folder $SUBTITLES_FOLDER \
  --remove_unused_columns False \
  --freeze_vision_tower True \
  --freeze_llm True \
  --freeze_merger True \
  --bf16 True \
  --fp16 False \
  --disable_flash_attn2 False \
  --output_dir $OUTPUT_DIR \
  --num_train_epochs 1 \
  --per_device_train_batch_size $BATCH_PER_DEVICE \
  --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
  --video_max_pixels $((360 * 420)) \
  --fps 1.0 \
  --learning_rate 1e-5 \
  --merger_lr 1e-5 \
  --vision_lr 2e-6 \
  --weight_decay 0.1 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --tf32 False \
  --gradient_checkpointing True \
  --report_to tensorboard \
  --lazy_preprocess True \
  --save_strategy "no" \
  --dataloader_num_workers 4
