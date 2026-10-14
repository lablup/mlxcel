"""Record the executable and argv of the actual running server process."""
import json
from pathlib import Path
import sys

process = Path("/proc") / str(int(sys.argv[1]))
print(json.dumps({
    "binary": str((process / "exe").resolve(strict=True)),
    "argv": (process / "cmdline").read_bytes().rstrip(b"\0").decode().split("\0"),
}))
