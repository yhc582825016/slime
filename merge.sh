cd /mnt/code/yehangcheng/slime
PYTHONPATH="/mnt/code/yehangcheng/slime" \
python tools/convert_torch_dist_to_hf_parallel.py \
  --input-dir /mnt/code/yehangcheng/checkpoint/General_model/slime/Qwen3.5-9B-Thinking_recall_agent/iter_0000199\
  --output-dir /opt/users/ye/checkpoints/Qwen3.5-9B-Thinking_recall_agent_hf \
  --origin-hf-dir /opt/users/models/Qwen3.5-9B \
  --load-max-workers 4