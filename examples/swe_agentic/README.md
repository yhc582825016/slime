# SWE Agentic Action-Match Example

This example converts trajectory data (with dynamic tools per sample) into slime prompt-data and scores model outputs by matching the predicted next tool call against `expected_action`.

## 1) Convert dataset

```bash
cd /dev/shm/ye/slime
python examples/swe_agentic/preprocess_swe_agent_data.py \
  --input /dev/shm/gym_data/swe_agent/Nemotron-RL-Agentic-SWE-Pivot-v1/train.jsonl \
  --output /dev/shm/gym_data/swe_agent/Nemotron-RL-Agentic-SWE-Pivot-v1/train_slime.jsonl
```

Optional quick check:

```bash
python examples/swe_agentic/preprocess_swe_agent_data.py \
  --input /dev/shm/gym_data/swe_agent/Nemotron-RL-Agentic-SWE-Pivot-v1/train.jsonl \
  --output /tmp/train_slime_1k.jsonl \
  --limit 1000
```

## 2) Training arguments (key parts)

Use the converted file as prompt data:

```bash
--prompt-data /dev/shm/gym_data/swe_agent/Nemotron-RL-Agentic-SWE-Pivot-v1/train_slime.jsonl
--input-key prompt
--metadata-key metadata
--custom-rm-path examples.swe_agentic.reward_expected_action.reward_func
```

## 3) One-click run script (Qwen3.5-4B)

```bash
cd /dev/shm/ye/slime
bash examples/swe_agentic/run-qwen3.5-4B-swe-agentic.sh
```

Useful overrides:

```bash
HF_CKPT=/dev/shm/Qwen3.5-4B \
MEGATRON_PATH=/root/Megatron-LM \
NUM_GPUS=4 \
TP_SIZE=2 \
ROLLOUT_NUM_GPUS_PER_ENGINE=2 \
PROMPT_DATA=/dev/shm/gym_data/swe_agent/Nemotron-RL-Agentic-SWE-Pivot-v1/train_slime.jsonl \
bash examples/swe_agentic/run-qwen3.5-4B-swe-agentic.sh
```

## Notes

- Dynamic tools are preserved per sample in `metadata.tools`.
- `expected_action` is read from the original sample top-level field.
- Reward checks:
  - `1.0` if both tool `name` and `arguments` match
  - `0.2` if only tool `name` matches
  - `0.0` otherwise
- Parser supports:
  - `<tool_call>{...}</tool_call>`
  - fallback raw JSON object containing `name` and `arguments`

