# AgenticQwen Virtual Tool RL

This example migrates the AgenticQwen/verl virtual tool-use RL flow into slime.

It supports:

- `agenticqwen_synthetic_data.parquet`
- simulated tool returns from `tool_return_expected_json`
- simulated user follow-up from `task_background`, `test_policy`, and `user_escape_strategy`
- final LLM judge reward using `rubrics`

Quick conversion:

```bash
cd /dev/shm/ye/slime
python examples/agenticqwen/preprocess_agenticqwen_data.py \
  --input /dev/shm/ye/rl-data/AgenticQwen-Data/agenticqwen_synthetic_data.parquet \
  --output-dir examples/agenticqwen/data
```

Main slime hooks:

```bash
--custom-generate-function-path examples.agenticqwen.rollout.generate
--custom-config-path examples/agenticqwen/agenticqwen_config.yaml
--custom-rm-path examples.agenticqwen.reward.reward_func
```

LLM endpoints:

```bash
export MOCK_TOOL_API_BASE=...
export MOCK_TOOL_API_KEY=...
export MOCK_TOOL_MODEL_NAME=...

export MOCK_USER_API_BASE=...
export MOCK_USER_API_KEY=...
export MOCK_USER_MODEL_NAME=...

export REWARD_API_BASE=...
export REWARD_API_KEY=...
export REWARD_MODEL_NAME=...
```

The environment first tries a deterministic lookup against `tool_return_expected_json`.
If a tool call does not exactly match, it falls back to the LLM tool simulator.

