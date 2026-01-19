#!/usr/bin/env bash

PREDICTIONS_DIR="<Path to the directory containing predictions and evaluations>"
OUTPUT_DIR="<Path to the output directory to store the TAC results>"
# We use Qwen/Qwen3-Next-80B-A3B-Instruct LLM to parse answer from reasoning
LLM_PATH="Qwen/Qwen3-Next-80B-A3B-Instruct"

python _run_all_benchmarks.py --input_dir "$PREDICTIONS_DIR" --output_dir "$OUTPUT_DIR" \
  --model "$LLM_PATH" --tp 4 --batch_size -1
