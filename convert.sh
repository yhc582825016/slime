cd /mnt/code/yehangcheng/slime
export NCCL_SOCKET_IFNAME=lo
HF_CKPT=/opt/users/ye/checkpoints/qwen3.5-4b-509-9061/checkpoint-9061
REF_LOAD=/opt/users/ye/checkpoints/qwen3.5-4b-509-9061_torch_dist
MEGATRON_PATH=/root/Megatron-LM

source /mnt/code/yehangcheng/slime/scripts/models/qwen3.5-4B.sh

PYTHONPATH="${MEGATRON_PATH}:$PWD" \
python /mnt/code/yehangcheng/slime/tools/convert_hf_to_torch_dist.py \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "${HF_CKPT}" \
  --save "${REF_LOAD}"
