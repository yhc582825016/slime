import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace


def _load_logging_utils_with_stubs():
    stub_wandb = types.SimpleNamespace(run=None, finish=lambda: None, log=lambda metrics: None)
    sys.modules["wandb"] = stub_wandb

    slime_pkg = types.ModuleType("slime")
    slime_pkg.__path__ = []
    utils_pkg = types.ModuleType("slime.utils")
    utils_pkg.__path__ = []
    sys.modules.setdefault("slime", slime_pkg)
    sys.modules["slime.utils"] = utils_pkg

    stub_wandb_utils = types.ModuleType("slime.utils.wandb_utils")
    stub_wandb_utils.init_wandb_primary = lambda *args, **kwargs: None
    stub_wandb_utils.init_wandb_secondary = lambda *args, **kwargs: None
    stub_wandb_utils.reinit_wandb_primary_with_open_metrics = lambda *args, **kwargs: None
    sys.modules["slime.utils.wandb_utils"] = stub_wandb_utils

    finish_calls = []

    class _StubTensorboardAdapter:
        def __init__(self, args):
            self.args = args

        def finish(self):
            finish_calls.append(self.args.tb_experiment_name)

        def log(self, data, step):
            return None

    stub_tensorboard_utils = types.ModuleType("slime.utils.tensorboard_utils")
    stub_tensorboard_utils._TensorboardAdapter = _StubTensorboardAdapter
    sys.modules["slime.utils.tensorboard_utils"] = stub_tensorboard_utils

    module_path = Path(__file__).resolve().parents[2] / "slime" / "utils" / "logging_utils.py"
    spec = importlib.util.spec_from_file_location("slime.utils.logging_utils_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, finish_calls


def test_finish_tracking_closes_tensorboard_writer():
    module, finish_calls = _load_logging_utils_with_stubs()
    args = SimpleNamespace(use_wandb=False, use_tensorboard=True, tb_experiment_name="recall-agent")

    module.finish_tracking(args)

    assert finish_calls == ["recall-agent"]
