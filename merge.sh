cd /dev/shm/ye/slime
PYTHONPATH="/root/Megatron-LM:/dev/shm/ye/slime" \
python tools/convert_torch_dist_to_hf_parallel.py \
  --input-dir /dev/shm/Qwen3.5-4B-Thinking_recall_agent/iter_0000199 \
  --output-dir /dev/shm/Qwen3.5-4B-Thinking_recall_agent_hf/iter_0000199 \
  --origin-hf-dir /dev/shm/Qwen3.5-4B \
  --load-max-workers 4