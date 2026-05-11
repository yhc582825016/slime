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
  --input /dev/shm/ye/rl-data/guru-RL-92k/train/math__combined_54.4k.parquet \
  --output /dev/shm/ye/rl-data/guru-RL-92k-slime/train/math__combined_54.4k.jsonl
```

You can also pass a directory to convert all nested parquet files.
Use `--limit 10` for a quick smoke-test conversion.

## Train With Slime

Use the converted JSONL with:

```bash
--prompt-data /dev/shm/ye/rl-data/guru-RL-92k-slime/train/math__combined_54.4k.jsonl
--input-key prompt
--label-key reward_model
--metadata-key metadata
--apply-chat-template
--custom-rm-path examples.mix_rl.reward.reward_func
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
```

The reward wrapper reads `metadata.source_dataset`, extracts the model answer
with the NeMo Gym order (`<answer>` first, then `\boxed{}`), and calls
`reasoning_gym.get_score_answer_fn(...)`.
