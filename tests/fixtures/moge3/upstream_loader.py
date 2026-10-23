"""Load unchanged pinned model modules without unrelated VLM dependencies."""

import ast
import dataclasses
import inspect
import pathlib
import sys
import types

REVISION = "abd3b2d7b728cb5372d4e092260da325e5ded209"


def load_upstream(source: pathlib.Path) -> None:
    """Register package paths and execute the original base-config class AST."""
    package = source / "mlx_vlm"
    for name, path in (
        ("mlx_vlm", package),
        ("mlx_vlm.models", package / "models"),
        ("mlx_vlm.models.moge3", package / "models/moge3"),
        ("mlx_vlm.models.dinov2", package / "models/dinov2"),
    ):
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module
    base_path = package / "models/base.py"
    tree = ast.parse(base_path.read_text())
    node = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "BaseModelConfig"
    )
    base = types.ModuleType("mlx_vlm.models.base")
    base.inspect = inspect
    base.dataclass = dataclasses.dataclass
    sys.modules[base.__name__] = base
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(base_path), "exec"), base.__dict__)
