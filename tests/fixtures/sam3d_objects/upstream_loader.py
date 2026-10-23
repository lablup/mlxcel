"""Load unchanged pinned model modules without unrelated VLM dependencies."""

import ast
import dataclasses
import inspect
import pathlib
import sys
import types

REVISION = "abd3b2d7b728cb5372d4e092260da325e5ded209"


def load_upstream(source: pathlib.Path) -> None:
    """Register package paths and execute the original base-config and processor-registration AST."""
    package = source / "mlx_vlm"
    for name, path in (
        ("mlx_vlm", package),
        ("mlx_vlm.models", package / "models"),
    ):
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module
    base_path = package / "models/base.py"
    tree = ast.parse(base_path.read_text())
    nodes = [node for node in tree.body if (isinstance(node, ast.ClassDef) and node.name == "BaseModelConfig") or (isinstance(node, ast.FunctionDef) and node.name == "install_auto_processor_patch")]
    if len(nodes) != 2:
        raise ValueError("Pinned base adapter definitions are missing")
    base = types.ModuleType("mlx_vlm.models.base")
    base.inspect = inspect
    base.dataclass = dataclasses.dataclass
    sys.modules[base.__name__] = base
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(base_path), "exec"), base.__dict__)
