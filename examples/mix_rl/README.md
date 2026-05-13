# Mix RL Rewards

This example migrates Reasoning360's multi-domain RL reward functions into
`slime/examples/mix_rl/reward_score` and exposes a Slime custom reward adapter.

## Convert Guru Data

Slime has one `label` field and one `metadata` field, while Reasoning360 reward
functions need `data_source`, `reward_model["ground_truth"]`, and `extra_info`.
Convert the parquet data first:

```bash
cd /dev/shm/ye/slime
python -m examples.mix_rl.convert_guru_to_slime \
  --input /dev/shm/ye/rl-data/guru-RL-92k/train \
  --output /dev/shm/ye/rl-data/guru-RL-92k-slime/train/all.jsonl
```

When `--input` is a directory and `--output` ends in `.jsonl`, all nested
parquet files are merged into that one JSONL. If `--output` is a directory, the
converter writes one JSONL per parquet file instead.
Use `--limit 10` for a quick smoke-test conversion.

You can select domains by the file prefix before `__`:

```bash
# Exclude stem__web when no LLM judge is available.
python -m examples.mix_rl.convert_guru_to_slime \
  --input /dev/shm/ye/rl-data/guru-RL-92k/train \
  --output /dev/shm/ye/rl-data/guru-RL-92k-slime/train/no_stem.jsonl \
  --exclude-domain stem

# Only convert selected domains.
python -m examples.mix_rl.convert_guru_to_slime \
  --input /dev/shm/ye/rl-data/guru-RL-92k/train \
  --output /dev/shm/ye/rl-data/guru-RL-92k-slime/train/math_logic_table.jsonl \
  --include-domain math,logic,table
```

## Train With Slime

Use the converted JSONL with:

```bash
--prompt-data /dev/shm/ye/rl-data/guru-RL-92k-slime/train/all.jsonl
--input-key prompt
--label-key reward_model
--metadata-key metadata
--apply-chat-template
--custom-rm-path examples.mix_rl.reward.reward_func
--reward-key score
```

For code rewards, configure `CODER1_EXEC` as needed. The migrated default is
`unsafe_local`, matching the Reasoning360 file. For `stem_web`, set
`STEM_LLM_JUDGE_URL` to an OpenAI-compatible verifier server.

Some domains require optional packages from `requirements.txt`.

## Reasoning Gym

The NeMo Gym `Nemotron-RL-ReasoningGym-v1` JSONL can be used directly with the
same custom reward:

```bash
--prompt-data /dev/shm/ye/rl-data/Nemotron-RL-ReasoningGym-v1/data/train.jsonl
--input-key question
--label-key answer
--metadata-key metadata
--custom-rm-path examples.mix_rl.reward.reward_func
--reward-key score
```

The reward wrapper reads `metadata.source_dataset`, extracts the model answer
with the NeMo Gym order (`<answer>` first, then `\boxed{}`), and calls
`reasoning_gym.get_score_answer_fn(...)`.
