#!/usr/bin/envbash

# The LLM to be served
export MODEL="Qwen/Qwen3-Next-80B-A3B-Instruct"
export HOST_PORT=8011

# Make sure the VLLM is installed.
# For instructions, visit https://docs.vllm.ai/en/latest/getting_started/installation

# Serve the LLM (optionally, you can specify the log file path to log the requests for debug purposes).
vllm serve "$MODEL" \
--port "$HOST_PORT" \
--tensor-parallel-size 4 \
--max-model-len 32768 \
--enable-log-requests \
--enable-log-outputs \
--log-config-file "path/to/the/logs/file.json"
