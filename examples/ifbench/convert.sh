cd /dev/shm/ye/slime
source scripts/models/qwen3-4B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint /dev/shm/Qwen3-4B \
  --save /dev/shm/Qwen3-4B_torch_dist