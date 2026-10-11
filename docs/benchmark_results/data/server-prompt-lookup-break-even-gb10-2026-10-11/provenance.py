"""Record the exact same-build acceptance inputs before starting servers."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import sys


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_mlx_pin(root: Path, source_commit: str | None = None) -> str:
    helper = root / "scripts/ci/mlx_pinned_commit.sh"
    env = dict(os.environ)
    env["MLX_PIN_CMAKE_FILE"] = str(root / "src/lib/mlx-cpp/CMakeLists.txt")
    if source_commit is None:
        return subprocess.check_output(["bash", str(helper)], env=env, text=True).strip()
    content = subprocess.check_output(["git", "-C", str(root), "show", f"{source_commit}:src/lib/mlx-cpp/CMakeLists.txt"])
    with tempfile.NamedTemporaryFile(suffix=".cmake") as cmake:
        cmake.write(content)
        cmake.flush()
        env["MLX_PIN_CMAKE_FILE"] = cmake.name
        return subprocess.check_output(["bash", str(helper)], env=env, text=True).strip()


def main() -> None:
    root = Path(os.environ["REPO_ROOT"]).resolve()
    binary = Path(os.environ["SERVER_BIN"]).resolve()
    model = (Path(os.environ["MODELS"]) / "qwen3-1.7b-4bit").resolve()
    config = model / "config.json"
    source_commit = os.environ["BINARY_SOURCE_COMMIT"]
    pin = canonical_mlx_pin(root)
    if canonical_mlx_pin(root, source_commit) != pin:
        raise ValueError("binary source and measurement checkout have different MLX pins")
    output = {
        "measurement_checkout": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
        "binary_source_commit": os.environ["BINARY_SOURCE_COMMIT"],
        "binary": str(binary),
        "binary_sha256": sha256(binary),
        "model": str(model),
        "model_config": json.loads(config.read_text()),
        "model_config_sha256": sha256(config),
        "native_mlx_pin": pin,
        "purpose": sys.argv[1],
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
