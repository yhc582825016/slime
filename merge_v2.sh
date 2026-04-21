cd /mnt/code/yehangcheng/slime

source scripts/models/qwen3.5-9B.sh

PYTHONPATH="/root/Megatron-LM:$(pwd)" \
torchrun --nproc_per_node=8 tools/convert_to_hf.py \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint /opt/users/models/Qwen3.5-9B \
  --load /mnt/code/yehangcheng/checkpoint/General_model/slime/Qwen3.5-9B-Thinking_recall_agent \
  --output-dir /mnt/code/yehangcheng/checkpoint/General_model/slime/Qwen3.5-9B-Thinking_recall_agent_hf_merged
