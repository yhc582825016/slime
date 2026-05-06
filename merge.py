import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from safetensors.torch import load_file
from transformers import AutoModelForImageTextToText, AutoProcessor


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_REFERENCE_HF_DIR = Path("/dev/shm/Qwen3.5-4B")
DEFAULT_BASE_TORCH_DIST_DIR = Path("/dev/shm/Qwen3.5-4B-Thinking_torch_dist/release")
DEFAULT_INPUT_DIR = Path("/dev/shm/Qwen3.5-4B-Thinking_recall_agent_423/iter_0000049")
DEFAULT_OUTPUT_DIR = Path("/dev/shm/Qwen3.5-4B-Thinking_recall_agent_423_hf_full/iter_0000049")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert a torch_dist checkpoint into a full HF model.")
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help="Input torch_dist checkpoint directory to export.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Final full HF model output directory.",
    )
    parser.add_argument(
        "--base-torch-dist-dir",
        default=str(DEFAULT_BASE_TORCH_DIST_DIR),
        help="Base torch_dist checkpoint directory used to build the merge base.",
    )
    parser.add_argument(
        "--reference-hf-dir",
        default=str(DEFAULT_REFERENCE_HF_DIR),
        help="Reference HF model used to preserve config, processor, and multimodal structure.",
    )
    parser.add_argument(
        "--load-max-workers",
        type=int,
        default=4,
        help="Number of parallel load workers passed to the converter.",
    )
    parser.add_argument(
        "--save-max-workers",
        type=int,
        default=16,
        help="Number of parallel save workers passed to the converter.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite auto-derived intermediate directories if they already exist.",
    )
    return parser.parse_args()


def derive_paths(args):
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    base_torch_dist_dir = Path(args.base_torch_dist_dir)
    reference_hf_dir = Path(args.reference_hf_dir)

    work_root = output_dir.parent.parent if output_dir.parent.name.startswith("iter_") else output_dir.parent
    base_delta_dir = work_root / "_merge_tmp_base_delta"
    base_full_dir = work_root / "_merge_tmp_base_full"
    target_delta_dir = work_root / "_merge_tmp_target_delta"

    return {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "base_torch_dist_dir": base_torch_dist_dir,
        "reference_hf_dir": reference_hf_dir,
        "base_delta_dir": base_delta_dir,
        "base_full_dir": base_full_dir,
        "target_delta_dir": target_delta_dir,
    }


def run_conversion(input_dir: Path, output_dir: Path, origin_hf_dir: Path, force: bool, load_max_workers: int, save_max_workers: int):
    if output_dir.exists() and not force:
        raise FileExistsError(f"Intermediate directory already exists: {output_dir}. Re-run with --force.")

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    extra_pythonpath = ["/root/Megatron-LM", str(SCRIPT_DIR)]
    env["PYTHONPATH"] = ":".join(extra_pythonpath + ([existing_pythonpath] if existing_pythonpath else []))

    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "tools" / "convert_torch_dist_to_hf_parallel.py"),
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(output_dir),
        "--origin-hf-dir",
        str(origin_hf_dir),
        "--load-max-workers",
        str(load_max_workers),
        "--save-max-workers",
        str(save_max_workers),
    ]
    if force:
        cmd.append("--force")

    subprocess.run(cmd, cwd=SCRIPT_DIR, env=env, check=True)


def load_weight_map(delta_dir: Path):
    with (delta_dir / "model.safetensors.index.json").open("r") as f:
        return json.load(f)["weight_map"]


def merge_delta_into_full(base_dir: Path, delta_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    model = AutoModelForImageTextToText.from_pretrained(
        str(base_dir),
        trust_remote_code=True,
        device_map="cpu",
        torch_dtype="auto",
    )

    for shard in sorted(set(load_weight_map(delta_dir).values())):
        tensors = load_file(str(delta_dir / shard))
        _missing, unexpected = model.load_state_dict(tensors, strict=False)
        if unexpected:
            print(f"[warn] unexpected keys in {shard}: {len(unexpected)}")

    model.save_pretrained(str(out_dir), safe_serialization=True, max_shard_size="5GB")
    processor = AutoProcessor.from_pretrained(str(base_dir), trust_remote_code=True)
    processor.save_pretrained(str(out_dir))


def remove_dir_if_exists(path: Path):
    if path.exists():
        shutil.rmtree(path)


def main():
    args = parse_args()
    paths = derive_paths(args)

    if not paths["reference_hf_dir"].exists():
        raise FileNotFoundError(f"Reference HF model directory not found: {paths['reference_hf_dir']}")
    if not paths["base_torch_dist_dir"].exists():
        raise FileNotFoundError(f"Base torch_dist checkpoint directory not found: {paths['base_torch_dist_dir']}")
    if not paths["input_dir"].exists():
        raise FileNotFoundError(f"Input torch_dist checkpoint directory not found: {paths['input_dir']}")

    print("[1/4] Convert base torch_dist checkpoint...")
    run_conversion(
        input_dir=paths["base_torch_dist_dir"],
        output_dir=paths["base_delta_dir"],
        origin_hf_dir=paths["reference_hf_dir"],
        force=args.force,
        load_max_workers=args.load_max_workers,
        save_max_workers=args.save_max_workers,
    )

    print("[2/4] Build base HF model...")
    merge_delta_into_full(paths["reference_hf_dir"], paths["base_delta_dir"], paths["base_full_dir"])

    print("[3/4] Convert target torch_dist checkpoint...")
    run_conversion(
        input_dir=paths["input_dir"],
        output_dir=paths["target_delta_dir"],
        origin_hf_dir=paths["base_full_dir"],
        force=args.force,
        load_max_workers=args.load_max_workers,
        save_max_workers=args.save_max_workers,
    )

    print("[4/4] Build final HF model...")
    merge_delta_into_full(paths["base_full_dir"], paths["target_delta_dir"], paths["output_dir"])

    remove_dir_if_exists(paths["base_delta_dir"])
    remove_dir_if_exists(paths["base_full_dir"])
    remove_dir_if_exists(paths["target_delta_dir"])

    print("DONE:", paths["output_dir"])


if __name__ == "__main__":
    main()
