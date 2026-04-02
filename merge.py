import json
import os

from safetensors.torch import load_file
from transformers import AutoModelForImageTextToText, AutoProcessor

base_dir = "/dev/shm/Qwen3.5-4B"
delta_dir = "/dev/shm/Qwen3.5-4B-Thinking_recall_agent_hf/iter_0000199"
out_dir = "/dev/shm/Qwen3.5-4B-Thinking_recall_agent_hf_full/iter_0000199"

os.makedirs(out_dir, exist_ok=True)

# 1) 必须加载多模态生成类（Qwen3_5ForConditionalGeneration）
# 才能同时保留并导出 visual + text + mtp 结构。
model = AutoModelForImageTextToText.from_pretrained(
    base_dir,
    trust_remote_code=True,
    device_map="cpu",
    torch_dtype="auto",
)

# 2) 逐个 shard 覆盖训练后的权重（只覆盖 delta 里有的 key）
with open(os.path.join(delta_dir, "model.safetensors.index.json"), "r") as f:
    index = json.load(f)
weight_map = index["weight_map"]

for shard in sorted(set(weight_map.values())):
    shard_path = os.path.join(delta_dir, shard)
    tensors = load_file(shard_path)
    missing, unexpected = model.load_state_dict(tensors, strict=False)
    if unexpected:
        print(f"[warn] unexpected keys in {shard}: {len(unexpected)}")

# 3) 保存完整 HF（此时 visual/mtp 保留，language 权重已更新）
model.save_pretrained(out_dir, safe_serialization=True, max_shard_size="5GB")

# 4) 保存 processor（会带 tokenizer/image processor 等）
processor = AutoProcessor.from_pretrained(base_dir, trust_remote_code=True)
processor.save_pretrained(out_dir)

print("DONE:", out_dir)