cd /mnt/code/yehangcheng/slime
source /mnt/code/yehangcheng/slime/scripts/models/qwen3.5-35B-A3B.sh
PYTHONPATH=/root/Megatron-LM torchrun --nproc-per-node 8 tools/convert_hf_to_torch_dist.py \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint /opt/users/models/Qwen3.5-35B-A3B \
  --save /opt/users/models/Qwen3.5-35B-A3B_torch_dist