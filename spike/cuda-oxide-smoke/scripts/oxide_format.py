#!/usr/bin/env python3
"""Parsers for the cuda-oxide artifact formats used by the smoke package.

Three separate formats are involved and none of them is interchangeable:

- the `.oxart` container embedded in the host binary, which carries payload
  and entry records and is not executable device code;
- PTX source, whose `.entry` signature declares parameter types and alignment;
- a cubin, whose `EIATTR_KPARAM_INFO` records give the final byte layout.

Only the Python standard library is used. This module is a library for
`oxide_artifact.py`; it is not a second entry point.
"""

from __future__ import annotations

import hashlib
import re
import struct
import subprocess
from pathlib import Path

ARTIFACT_MAGIC = b"OXIDEART"
HEADER_BYTES = 32
PAYLOAD_RECORD_BYTES = 24
ENTRY_RECORD_BYTES = 24
PAYLOAD_KIND_NAMES = {0x0100: "ptx", 0x0110: "nvvm-ir", 0x0120: "ltoir", 0x0200: "cubin"}
ENTRY_KIND_NAMES = {1: "kernel", 2: "device-function"}
# kparam space codes reported by cuobjdump -elf.
SPACE_NAMES = {0x0: "value", 0x5: "global"}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run(command: list[str], timeout: float = 300.0) -> subprocess.CompletedProcess:
    """Run an external tool and turn its failures into clean ValueErrors."""
    try:
        return subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired as error:
        raise ValueError(
            f"command timed out after {timeout:g}s: {' '.join(command)}"
        ) from error
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or "").strip().splitlines()
        suffix = f": {detail[-1]}" if detail else ""
        raise ValueError(
            f"command failed with exit {error.returncode}: {' '.join(command)}{suffix}"
        ) from error


# ---------------------------------------------------------------------------
# .oxart container
# ---------------------------------------------------------------------------


def parse_artifact_section(data: bytes) -> list[dict]:
    """Parse every bundle blob in a `.oxart` section."""
    bundles = []
    cursor = 0
    while cursor < len(data) and any(data[cursor:]):
        remaining = data[cursor:]
        if len(remaining) < HEADER_BYTES:
            raise ValueError("truncated .oxart header")
        if remaining[0:8] != ARTIFACT_MAGIC:
            raise ValueError(f"bad .oxart magic {remaining[0:8]!r}")
        version, header_len = struct.unpack_from("<HH", remaining, 8)
        (total_len,) = struct.unpack_from("<I", remaining, 12)
        name_len, target_len, payload_count, entry_count = struct.unpack_from("<HHHH", remaining, 16)
        if header_len != HEADER_BYTES:
            raise ValueError(f"unsupported .oxart header length {header_len}")
        # A zero or undersized length used to leave the cursor where it was:
        # the loop re-entered with the same bytes and grew memory until killed.
        # Reject it before slicing so a malformed blob is a clean error.
        if total_len < HEADER_BYTES:
            raise ValueError(
                f"embedded artifact length {total_len} is smaller than the "
                f"{HEADER_BYTES}-byte header"
            )
        blob = remaining[:total_len]
        if len(blob) != total_len:
            raise ValueError("truncated .oxart blob")
        compile_options = 0
        if version == 2:
            (compile_options,) = struct.unpack_from("<Q", blob, 24)
        elif version != 1:
            raise ValueError(f"unsupported .oxart version {version}")

        offset = HEADER_BYTES
        name = blob[offset : offset + name_len].decode("utf-8")
        offset += name_len
        target = blob[offset : offset + target_len].decode("utf-8")
        offset += target_len

        payloads = []
        records = offset
        for index in range(payload_count):
            record = records + index * PAYLOAD_RECORD_BYTES
            kind, _flags, data_offset, data_len, name_offset, item_name_len = struct.unpack_from(
                "<HHIIIH", blob, record
            )
            payload_name = blob[name_offset : name_offset + item_name_len].decode("utf-8")
            payload = blob[data_offset : data_offset + data_len]
            if len(payload) != data_len:
                raise ValueError(f"payload {payload_name} is truncated")
            payloads.append(
                {
                    "kind": kind,
                    "kind_name": PAYLOAD_KIND_NAMES.get(kind, f"unknown-0x{kind:04x}"),
                    "name": payload_name,
                    "bytes": len(payload),
                    "sha256": sha256(payload),
                    "data": payload,
                }
            )
        offset = records + payload_count * PAYLOAD_RECORD_BYTES

        entries = []
        records = offset
        for index in range(entry_count):
            record = records + index * ENTRY_RECORD_BYTES
            kind, flags = struct.unpack_from("<HH", blob, record)
            symbol_offset, symbol_len = struct.unpack_from("<IH", blob, record + 12)
            entries.append(
                {
                    "kind": kind,
                    "kind_name": ENTRY_KIND_NAMES.get(kind, f"unknown-{kind}"),
                    "symbol": blob[symbol_offset : symbol_offset + symbol_len].decode("utf-8"),
                    "has_metadata": bool(flags & 1),
                }
            )

        bundles.append(
            {
                "version": version,
                "compile_options": compile_options,
                "name": name,
                "target": target,
                "payloads": payloads,
                "entries": entries,
            }
        )
        previous = cursor
        cursor += total_len
        if cursor <= previous:
            # Defensive: the length check above makes this unreachable, but a
            # non-advancing cursor must never turn into an unbounded loop.
            raise ValueError("artifact cursor did not advance")

    if not bundles:
        raise ValueError(".oxart section contains no bundle")
    return bundles


# ---------------------------------------------------------------------------
# PTX
# ---------------------------------------------------------------------------

PTX_PARAM_RE = re.compile(
    r"^\s*\.param\s+(?P<type>\.\w+)"
    r"(?:\s+\.ptr(?:\s+\.align\s+(?P<align>\d+))?)?"
    r"\s+(?P<name>[\w$]+)\s*,?\s*$"
)


def parse_ptx(text: str) -> dict:
    version_match = re.search(r"^\s*\.version\s+(\S+)\s*$", text, re.MULTILINE)
    target_match = re.search(r"^\s*\.target\s+(\S+)\s*$", text, re.MULTILINE)
    entry_match = re.search(r"^\s*(?:\.visible\s+|\.extern\s+)?\.entry\s+([\w$]+)\s*\(", text, re.MULTILINE)
    if not (version_match and target_match and entry_match):
        raise ValueError("PTX is missing .version, .target, or .entry")

    entry = entry_match.group(1)
    open_paren = text.index("(", entry_match.start())
    close_paren = text.index(")", open_paren)
    signature = text[open_paren + 1 : close_paren]
    parameters = []
    for line in signature.splitlines():
        stripped = line.strip().rstrip(",")
        if not stripped:
            continue
        match = PTX_PARAM_RE.match(stripped)
        if not match:
            raise ValueError(f"unparsed PTX parameter line: {stripped!r}")
        parameters.append(
            {
                "ptx_type": match.group("type"),
                "pointer": ".ptr" in stripped,
                "pointee_align_bytes": int(match.group("align") or 0),
                "name": match.group("name"),
            }
        )
    return {
        "version": version_match.group(1),
        "target": target_match.group(1),
        "entry": entry,
        "parameters": parameters,
    }


# ---------------------------------------------------------------------------
# cubin (cuobjdump)
# ---------------------------------------------------------------------------

KPARAM_RE = re.compile(
    r"EIATTR_KPARAM_INFO.*?"
    r"Ordinal\s*:\s*0x([0-9a-fA-F]+).*?"
    r"Offset\s*:\s*0x([0-9a-fA-F]+).*?"
    r"Size\s*:\s*0x([0-9a-fA-F]+).*?"
    r"Pointee's logAlignment\s*:\s*0x([0-9a-fA-F]+).*?"
    r"Space\s*:\s*0x([0-9a-fA-F]+)",
    re.DOTALL,
)


def parse_cubin(cubobjdump: str, cubin: Path) -> dict:
    elf = run([cubobjdump, "-elf", str(cubin)]).stdout
    sass = run([cubobjdump, "-sass", str(cubin)]).stdout
    symbols = run([cubobjdump, "-symbols", str(cubin)]).stdout

    target_match = re.search(r"^\s*\.target\s+(\S+)\s*$", sass, re.MULTILINE)
    if not target_match:
        raise ValueError("cuobjdump -sass did not report a .target")
    entry_symbols = [
        line.split()[-1] for line in symbols.splitlines() if "STT_FUNC" in line and "STO_ENTRY" in line
    ]
    kparams = {}
    for ordinal, offset, size, log_align, space in KPARAM_RE.findall(elf):
        exponent = int(log_align, 16)
        if exponent > 64:
            raise ValueError(
                f"kparam ordinal {int(ordinal, 16)} declares pointee "
                f"logAlignment 0x{log_align}, above the supported bound of 64"
            )
        kparams[int(ordinal, 16)] = {
            "ordinal": int(ordinal, 16),
            "byte_offset": int(offset, 16),
            "size_bytes": int(size, 16),
            "pointee_align_bytes": 1 << exponent,
            "space": SPACE_NAMES.get(int(space, 16), f"0x{int(space, 16):x}"),
        }
    return {
        "target": target_match.group(1),
        "entry_symbols": entry_symbols,
        "kparams": [kparams[key] for key in sorted(kparams)],
    }
