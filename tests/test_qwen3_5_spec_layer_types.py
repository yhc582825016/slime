import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def _stub_module(name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


def _load_qwen3_5_module():
    _stub_module("torch", Tensor=object)
    _stub_module("torch.nn", Module=object)
    _stub_module("torch.nn.functional")
    sys.modules["torch"].nn = sys.modules["torch.nn"]
    sys.modules["torch"].nn.functional = sys.modules["torch.nn.functional"]

    _stub_module("megatron")
    _stub_module("megatron.core")
    _stub_module("megatron.core.models")
    _stub_module("megatron.core.models.gpt")
    _stub_module("megatron.core.models.gpt.gpt_layer_specs", get_gpt_decoder_block_spec=lambda *args, **kwargs: None)
    _stub_module("megatron.core.transformer")
    _stub_module("megatron.core.transformer.spec_utils", ModuleSpec=object)
    _stub_module("megatron.core.transformer.transformer_block", get_num_layers_to_build=lambda *args, **kwargs: 0)
    _stub_module(
        "megatron.core.transformer.transformer_layer",
        get_transformer_layer_offset=lambda *args, **kwargs: 0,
    )

    _stub_module("transformers")
    _stub_module("transformers.activations", ACT2FN={})

    pkg = _stub_module("slime_plugins")
    pkg.__path__ = []
    models_pkg = _stub_module("slime_plugins.models")
    models_pkg.__path__ = []
    _stub_module(
        "slime_plugins.models.hf_attention",
        HuggingfaceAttention=type("HuggingfaceAttention", (), {}),
        _load_hf_config=lambda _path: None,
    )

    module_path = Path(__file__).resolve().parents[1] / "slime_plugins" / "models" / "qwen3_5.py"
    module_name = "slime_plugins.models.test_qwen3_5_module"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_get_qwen3_5_spec_resolves_missing_layer_types():
    module = _load_qwen3_5_module()

    class DummyBlockSpec:
        def __init__(self, num_layers: int):
            self.layer_specs = [
                SimpleNamespace(submodules=SimpleNamespace(self_attention=f"attn_{i}")) for i in range(num_layers)
            ]

    sentinel_attention = object()
    captured_modules = []

    def _module_spec(*, module, params):
        captured_modules.append((module, params))
        return SimpleNamespace(module=module, params=params)

    args = SimpleNamespace(num_experts=0, hf_checkpoint="/tmp/mock")
    config = SimpleNamespace(num_layers=4, moe_layer_freq=None, pipeline_model_parallel_layout=None)

    with (
        patch.object(module, "get_gpt_decoder_block_spec", lambda config, **kwargs: DummyBlockSpec(config.num_layers)),
        patch.object(module, "get_num_layers_to_build", lambda config, vp_stage=None: config.num_layers),
        patch.object(module, "get_transformer_layer_offset", lambda config, vp_stage=None: 0),
        patch.object(
            module,
            "_load_hf_config",
            lambda _path: SimpleNamespace(num_hidden_layers=4, full_attention_interval=2),
        ),
        patch.object(module, "Attention", sentinel_attention),
        patch.object(module, "ModuleSpec", _module_spec),
    ):
        spec = module.get_qwen3_5_spec(args, config, vp_stage=None)

    assert config.moe_layer_freq == [0, 0, 0, 0]
    assert captured_modules == [
        (sentinel_attention, {"args": args}),
        (sentinel_attention, {"args": args}),
    ]
    assert spec.layer_specs[0].submodules.self_attention.module is sentinel_attention
    assert spec.layer_specs[1].submodules.self_attention == "attn_1"
    assert spec.layer_specs[2].submodules.self_attention.module is sentinel_attention
    assert spec.layer_specs[3].submodules.self_attention == "attn_3"
