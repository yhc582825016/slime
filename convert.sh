cd /mnt/code/yehangcheng/slime

HF_CKPT=/opt/users/models/Qwen3.5-4B
REF_LOAD=/opt/users/models/Qwen3.5-4B_torch_dist
MEGATRON_PATH=/root/Megatron-LM

source scripts/models/qwen3.5-4B.sh

PYTHONPATH="${MEGATRON_PATH}:$PWD" \
python tools/convert_hf_to_torch_dist.py \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "${HF_CKPT}" \
  --save "${REF_LOAD}"
