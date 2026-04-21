import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace


def _load_wandb_utils_with_stub(stub_wandb):
    sys.modules["wandb"] = stub_wandb
    module_path = Path(__file__).resolve().parents[2] / "slime" / "utils" / "wandb_utils.py"
    spec = importlib.util.spec_from_file_location("test_wandb_utils_secondary_fallback", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_secondary_wandb_init_failure_disables_wandb_in_current_process():
    class CommError(Exception):
        pass

    finish_calls = []

    stub_wandb = types.SimpleNamespace(
        run=None,
        login=lambda **kwargs: None,
        init=lambda **kwargs: (_ for _ in ()).throw(CommError("resume status timeout")),
        finish=lambda: finish_calls.append(True),
        define_metric=lambda *args, **kwargs: None,
        Settings=lambda **kwargs: types.SimpleNamespace(**kwargs),
        errors=types.SimpleNamespace(CommError=CommError),
    )

    module = _load_wandb_utils_with_stub(stub_wandb)
    args = SimpleNamespace(
        use_wandb=True,
        wandb_run_id="abc123",
        wandb_mode=None,
        wandb_key=None,
        wandb_host=None,
        wandb_team="team",
        wandb_project="project",
        wandb_dir=None,
    )

    module.init_wandb_secondary(args)

    assert args.use_wandb is False
    assert args.wandb_mode == "disabled"
    assert sys.modules["wandb"] is stub_wandb
    assert finish_calls == []
