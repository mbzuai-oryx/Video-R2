#!/bin/bash

export DECORD_EOF_RETRY_MAX=204800
export HF_HOME="<Replace with you huggingFace home where the benchmark data would be stored.>"
export HF_TOKEN="<Replace with your HuggingFace token.>"

# Path to the pretrained Video-R2 model
export CHECKPOINTS_PATH="MBZUAI/Video-R2"
# Maximum number of sampled frames per video
export FPS_MAX_FRAMES=128
# Number of available GPUs
export NUM_GPUS=8
# Add the provided lmms_eval into the PYTHONPATH
export PYTHONPATH="./eval/lmms-eval:$PYTHONPATH"
# Output path to store the predictions and results
export OUTPUT_PATH="$CHECKPOINTS_PATH"/eval_think_"$FPS_MAX_FRAMES"

TASKS_THINK=(
    # Generic Benchmarks
    "mvbench_think" "videomme_think" "tempcompass_complete_think" "mlvu_dev_think" "longvideobench_val_v_think"

    # Reasoning-focused Benchmarks
    "videomathqa_mcq_think" "video_mmmu_think" "mmvu_val_think" "vsibench_think" "minerva_think" "scivideobench_think"
)

for TASK_THINK in "${TASKS_THINK[@]}"; do
    echo "🚀 Running TASK (think): $TASK_THINK"
    OUTPUT_PREFIX_THINK=$(echo "$TASK_THINK" | tr ',' '_')
    accelerate launch --num_processes=$NUM_GPUS lmms-eval/lmms_eval/__main__.py \
        --model qwen2_5_vl \
        --model_args=pretrained=$CHECKPOINTS_PATH,max_pixels=151200,min_pixels=100352,max_num_frames=$FPS_MAX_FRAMES,attn_implementation=flash_attention_2,device_map=auto \
        --tasks "$TASK_THINK" \
        --batch_size 1 --log_samples --log_samples_suffix qwen_2_5_vl \
        --output_path $OUTPUT_PATH --datetime_str "$OUTPUT_PREFIX_THINK"
done
