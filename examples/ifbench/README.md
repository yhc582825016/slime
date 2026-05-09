# IFBench Custom Reward Example

This example shows how to reuse the IFBench checker implementation from `ms-swift` inside `slime` through `--custom-rm-path`.

## What It Adds

- `reward_ms_swift.py`
  - A custom reward function with signature `async def reward_func(args, sample, **kwargs) -> float`
  - Reuses `ms-swift/plugin/IFbench/instructions_registry.py`
  - Returns `1.0` only when all IFBench constraints are satisfied
- `run_qwen3.5-35B-A3B.sh`
  - GRPO training on IFBench with **Qwen3.5-35B-A3B** (MoE): TP=2, EP=8, SGLang EP=8, same reward wiring as before

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

## Evaluation data (`IFBench_test/`)

The `examples/ifbench/IFBench_test/` folder in git is **intentionally almost empty**: it only carries the dataset card (`README.md`) for [allenai/IFBench_test](https://huggingface.co/datasets/allenai/IFBench_test). Parquet shards are **not** checked in; you need to download them once, then convert to Slime format.

From the **slime repo root**:

```bash
# 1) Download official test split (writes under IFBench_test/data/)
pip install -U "huggingface_hub[cli]"
hf download allenai/IFBench_test --repo-type dataset --local-dir examples/ifbench/IFBench_test

# 2) Convert HF layout -> slime eval parquet (messages + extra_info)
python examples/ifbench/convert_ifbench_to_slime.py \
  --input examples/ifbench/IFBench_test/data/train-00000-of-00001.parquet \
  --output examples/ifbench/IFBench_test/ifbench_test_slime.parquet
```

If the shard name differs (e.g. multiple `train-0000*-of-*.parquet` files), point `--input` at the file you have. Training scripts default to `IFBench_test/ifbench_test_slime.parquet` via `IFBENCH_EVAL_DATA`.

## Notes

- If your IFBench plugin is not under `../ms-swift/plugin/IFbench`, set `MS_SWIFT_IFBENCH_DIR`.
- If your metadata field is named `metadata` instead of `extra_info`, change `--metadata-key`.
- For this reward, `--group-rm` is not needed.
