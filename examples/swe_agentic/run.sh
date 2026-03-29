cd /dev/shm/ye/slime
USE_WANDB=1 \
WANDB_PROJECT=slime \
WANDB_GROUP=qwen3.5-4B \
WANDB_RUN_NAME=qwen3.5-4B-swe-agentic \
SWE_AGENTIC_DEBUG_REWARD=1 \
SWE_AGENTIC_DEBUG_REWARD_EVERY=1 \
bash examples/swe_agentic/run-qwen3.5-4B-swe-agentic.sh