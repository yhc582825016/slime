# IFBench Custom Reward Example

This example shows how to reuse the IFBench checker implementation from `ms-swift` inside `slime` through `--custom-rm-path`.

## What It Adds

- `reward_ms_swift.py`
  - A custom reward function with signature `async def reward_func(args, sample, **kwargs) -> float`
  - Reuses `ms-swift/plugin/IFbench/instructions_registry.py`
  - Returns `1.0` only when all IFBench constraints are satisfied
- `run_qwen3_4B.sh`
  - A minimal training script showing how to wire the custom reward into `slime`

## Expected Dataset Fields

The custom reward reads IFBench metadata from `sample.metadata`. The simplest dataset layout is:

```json
{
  "messages": [{"role": "user", "content": "Your prompt here"}],
  "extra_info": {
    "instruction_id_list": ["keywords:existence"],
    "instruction_kwargs": [{"keywords": ["hello"]}]
  }
}
```

In the example script, this is wired through:

```bash
--input-key messages
--metadata-key extra_info
--apply-chat-template
--custom-rm-path examples.ifbench.reward_ms_swift.reward_func
```

## Notes

- If your IFBench plugin is not under `../ms-swift/plugin/IFbench`, set `MS_SWIFT_IFBENCH_DIR`.
- If your metadata field is named `metadata` instead of `extra_info`, change `--metadata-key`.
- For this reward, `--group-rm` is not needed.
