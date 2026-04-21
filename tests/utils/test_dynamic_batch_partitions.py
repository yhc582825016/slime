from __future__ import annotations

import importlib
import sys
import types

import pytest


def install_runtime_stubs() -> None:
    if "ray" not in sys.modules:
        ray_mod = types.ModuleType("ray")
        ray_mod._private = types.SimpleNamespace(
            services=types.SimpleNamespace(get_node_ip_address=lambda: "127.0.0.1")
        )
        sys.modules["ray"] = ray_mod

    if "wandb" not in sys.modules:
        wandb_mod = types.ModuleType("wandb")
        wandb_mod.run = None
        wandb_mod.log = lambda *args, **kwargs: None
        wandb_mod.finish = lambda *args, **kwargs: None
        sys.modules["wandb"] = wandb_mod

    if "megatron" in sys.modules:
        return

    megatron_mod = types.ModuleType("megatron")
    core_mod = types.ModuleType("megatron.core")
    packed_seq_mod = types.ModuleType("megatron.core.packed_seq_params")

    class PackedSeqParams:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    core_mod.mpu = types.SimpleNamespace(
        get_context_parallel_world_size=lambda: 1,
        get_context_parallel_group=lambda: None,
        get_context_parallel_rank=lambda: 0,
        get_data_parallel_rank=lambda with_context_parallel=False: 0,
        get_data_parallel_world_size=lambda with_context_parallel=False: 1,
        get_data_parallel_src_rank=lambda with_context_parallel=True: 0,
        get_data_parallel_group_gloo=lambda with_context_parallel=True: None,
        get_tensor_model_parallel_rank=lambda: 0,
        get_tensor_model_parallel_world_size=lambda: 1,
        get_virtual_pipeline_model_parallel_world_size=lambda: None,
        is_pipeline_last_stage=lambda: True,
    )
    packed_seq_mod.PackedSeqParams = PackedSeqParams

    sys.modules["megatron"] = megatron_mod
    sys.modules["megatron.core"] = core_mod
    sys.modules["megatron.core.packed_seq_params"] = packed_seq_mod


def load_data_module():
    install_runtime_stubs()
    sys.modules.pop("slime.backends.megatron_utils.data", None)
    return importlib.import_module("slime.backends.megatron_utils.data")


@pytest.mark.unit
def test_capped_partitions_allows_single_oversize_sample() -> None:
    data_module = load_data_module()

    lengths = [400, 100, 100]
    partitions = data_module._get_capped_partitions(lengths, num_partitions=2, max_tokens=300)

    assert partitions == [[0], [1, 2]]
    assert sum(lengths[idx] for idx in partitions[0]) > 300
    assert sum(lengths[idx] for idx in partitions[1]) <= 300


@pytest.mark.unit
def test_capped_partitions_raises_clear_error_when_partition_count_is_too_small() -> None:
    data_module = load_data_module()

    with pytest.raises(AssertionError, match="Unable to partition samples within the token cap"):
        data_module._get_capped_partitions([200, 200, 200], num_partitions=1, max_tokens=300)
