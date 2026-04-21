import json
import os
import shutil

from safetensors.torch import load_file
from transformers import AutoModelForImageTextToText, AutoProcessor

base_dir = "/opt/users/models/Qwen3.5-9B"
delta_dir = "/opt/users/ye/checkpoints/Qwen3.5-9B-Thinking_recall_agent_hf"
out_dir = "/opt/users/ye/checkpoints/Qwen3.5-9B-Thinking_recall_agent_hf_full"

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

# 5) 强制继承基座模型的 tokenizer 相关文件，避免 save_pretrained
# 重新序列化后产生与原始模型不一致的 tokenizer_config.json。
for filename in [
    "tokenizer_config.json",
    "tokenizer.json",
    "chat_template.jinja",
]:
    src = os.path.join(base_dir, filename)
    dst = os.path.join(out_dir, filename)
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"[copy] {filename} <- {base_dir}")

print("DONE:", out_dir)
