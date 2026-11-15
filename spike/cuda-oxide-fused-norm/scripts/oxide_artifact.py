#!/usr/bin/env python3
# Copyright 2025-2026 Lablup Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Artifact export, manifest validation and launch check for the cuda-oxide
fused-norm package (issue #2832).

This consumes the #2831 artifact contract: the `.oxart` container parser, the
cubin `kparam` reader and the digest helpers are imported from the smoke
package's `oxide_format.py`, which is the one place those formats are decoded.
What differs from the smoke tool is that one PTX module here carries five
`.entry` symbols (one per dtype pair), so there is one hand-written schema and
one generated manifest per entry, all pointing at the same PTX and cubin, and
the launch rule is the row kernel's (`grid = rows`, `block =
fused_norm_threads(dim)`) rather than an elementwise `ceil_div(n, block)`.

Subcommands:
  export       extract PTX from .oxart, assemble the cubin, write one manifest
               per schema in --schema-dir
  validate     check every manifest in --manifest-dir against its schema and
               the committed PTX and cubin
  check-launch check the harness PASS lines against the manifests' launch rule
               and declared totals
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
_smoke_scripts = Path(os.environ.get("CUDA_OXIDE_SMOKE_SCRIPTS", _here.parent.parent / "cuda-oxide-smoke" / "scripts"))
if not (_smoke_scripts / "oxide_format.py").is_file():
    print(f"FAIL oxide_format.py not found under {_smoke_scripts} (set CUDA_OXIDE_SMOKE_SCRIPTS)", file=sys.stderr)
    sys.exit(2)
sys.path.insert(0, str(_smoke_scripts))

from oxide_format import (  # noqa: E402
    ARTIFACT_MAGIC,
    PTX_PARAM_RE,
    parse_artifact_section,
    parse_cubin,
    run,
    sha256,
)

ARTIFACT_STEM = "fused_add_rms_norm"
OXART_NAME = "cuda_oxide_fused_norm.oxart"

# Launch geometry rule of the row kernel, mirrored from `fused_norm_threads`
# in src/main.rs and kernels/fused_norm.cpp.
SIMD_WIDTH = 32
N_READS = 4
MAX_THREADS = 1024

ERRORS: list[str] = []


def fail(message: str) -> None:
    ERRORS.append(message)
    print(f"FAIL {message}", file=sys.stderr)


def fused_norm_threads(dim: int) -> int:
    if dim <= 0:
        return SIMD_WIDTH
    lanes = -(-dim // N_READS)
    threads = -(-lanes // SIMD_WIDTH) * SIMD_WIDTH
    return max(SIMD_WIDTH, min(MAX_THREADS, threads))


def artifact_path(base: Path, relative: str) -> Path:
    """Resolve a manifest artifact path and keep it inside the artifact root."""
    candidate = Path(relative)
    if candidate.is_absolute():
        raise ValueError(f"artifact path {relative!r} is absolute; commit a path relative to the artifact root")
    root = base.resolve()
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"artifact path {relative!r} escapes the artifact root {root}") from error
    return resolved


def parse_ptx_entry(text: str, entry: str) -> dict:
    """Parse the `.version`, `.target` and the signature of one named `.entry`."""
    version_match = re.search(r"^\s*\.version\s+(\S+)\s*$", text, re.MULTILINE)
    target_match = re.search(r"^\s*\.target\s+(\S+)\s*$", text, re.MULTILINE)
    entry_match = re.search(
        r"^\s*(?:\.visible\s+|\.extern\s+)?\.entry\s+" + re.escape(entry) + r"\s*\(", text, re.MULTILINE
    )
    if not (version_match and target_match):
        raise ValueError("PTX is missing .version or .target")
    if not entry_match:
        raise ValueError(f"PTX has no .entry {entry}")
    open_paren = text.index("(", entry_match.start())
    close_paren = text.index(")", open_paren)
    parameters = []
    for line in text[open_paren + 1 : close_paren].splitlines():
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
    entries = re.findall(r"^\s*(?:\.visible\s+|\.extern\s+)?\.entry\s+([\w$]+)\s*\(", text, re.MULTILINE)
    return {
        "version": version_match.group(1),
        "target": target_match.group(1),
        "entry": entry,
        "entries": entries,
        "parameters": parameters,
    }


def parse_cubin_entry(cuobjdump: str, cubin: Path, entry: str) -> dict:
    """`kparam` records of one entry: `cuobjdump -elf` prints every function's
    `.nv.info.<entry>` section, so slice the output to the named section."""
    elf = run([cuobjdump, "-elf", str(cubin)]).stdout
    sass = run([cuobjdump, "-sass", str(cubin)]).stdout
    symbols = run([cuobjdump, "-symbols", str(cubin)]).stdout
    target_match = re.search(r"^\s*\.target\s+(\S+)\s*$", sass, re.MULTILINE)
    if not target_match:
        raise ValueError("cuobjdump -sass did not report a .target")
    entry_symbols = [
        line.split()[-1] for line in symbols.splitlines() if "STT_FUNC" in line and "STO_ENTRY" in line
    ]
    section_header = re.search(r"^\s*\.nv\.info\." + re.escape(entry) + r"\b.*$", elf, re.MULTILINE)
    if not section_header:
        raise ValueError(f"cuobjdump -elf has no .nv.info.{entry} section")
    rest = elf[section_header.end() :]
    next_section = re.search(r"^\s*\.nv\.info\.", rest, re.MULTILINE)
    section = rest[: next_section.start()] if next_section else rest
    from oxide_format import KPARAM_RE, SPACE_NAMES  # noqa: E402

    kparams = {}
    for ordinal, offset, size, log_align, space in KPARAM_RE.findall(section):
        exponent = int(log_align, 16)
        if exponent > 64:
            raise ValueError(f"kparam ordinal {int(ordinal, 16)} declares pointee logAlignment above 64")
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


def build_arguments(schema: dict, ptx_info: dict, cubin_info: dict) -> list[dict]:
    """Join the declared ABI with the emitted PTX and cubin argument layout."""
    if len(schema["arguments"]) != len(ptx_info["parameters"]):
        raise ValueError(
            f"{schema['entry_symbol']}: schema declares {len(schema['arguments'])} arguments, "
            f"PTX entry has {len(ptx_info['parameters'])}"
        )
    arguments = []
    for declared in schema["arguments"]:
        ordinal = declared["ordinal"]
        emitted_param = ptx_info["parameters"][ordinal]
        kparam = next((item for item in cubin_info["kparams"] if item["ordinal"] == ordinal), None)
        if kparam is None:
            raise ValueError(f"cubin has no kparam for ordinal {ordinal}")
        arguments.append(
            {
                **declared,
                "emitted": {
                    "ptx_type": emitted_param["ptx_type"],
                    "ptx_pointer": emitted_param["pointer"],
                    "ptx_pointee_align_bytes": emitted_param["pointee_align_bytes"],
                    "cubin_size_bytes": kparam["size_bytes"],
                    "cubin_byte_offset": kparam["byte_offset"],
                    "cubin_space": kparam["space"],
                    "cubin_pointee_align_bytes": kparam["pointee_align_bytes"],
                },
            }
        )
    return arguments


def check_arguments(entry: str, arguments: list[dict]) -> None:
    """The emitted layout must honor every declared claim."""
    expected_type = {64: ".u64", 32: {"f32": ".f32", "u32": ".u32"}}
    for arg in arguments:
        emitted = arg["emitted"]
        width = arg["width_bits"]
        if emitted["cubin_size_bytes"] * 8 != width:
            fail(f"{entry} arg {arg['name']}: cubin size {emitted['cubin_size_bytes']} bytes != declared {width} bits")
        ptx_type = emitted["ptx_type"]
        want = expected_type[64] if width == 64 else expected_type[32].get(arg["cuda_param_type"])
        if want is not None and ptx_type != want:
            fail(f"{entry} arg {arg['name']}: PTX type {ptx_type} != expected {want}")
        if arg["kind"] == "device_pointer":
            if not emitted["ptx_pointer"]:
                fail(f"{entry} arg {arg['name']}: declared a pointer but the PTX parameter is not .ptr")
            if emitted["cubin_space"] != "global":
                fail(f"{entry} arg {arg['name']}: cubin space {emitted['cubin_space']} is not global")
            if emitted["ptx_pointee_align_bytes"] > arg["allocation_align_bytes"]:
                fail(
                    f"{entry} arg {arg['name']}: compiler assumes {emitted['ptx_pointee_align_bytes']}-byte "
                    f"alignment, stricter than the declared {arg['allocation_align_bytes']}-byte allocation"
                )
        elif emitted["ptx_pointer"]:
            fail(f"{entry} arg {arg['name']}: declared a value but the PTX parameter is .ptr")


def load_schemas(schema_dir: Path) -> list[dict]:
    schemas = [json.loads(p.read_text()) for p in sorted(schema_dir.glob("*.schema.json"))]
    if not schemas:
        raise ValueError(f"no *.schema.json under {schema_dir}")
    return schemas


def export(args: argparse.Namespace) -> int:
    binary = Path(args.binary)
    ptx_source = Path(args.ptx)
    out_dir = Path(args.out_dir)
    build_dir = Path(args.build_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    build_dir.mkdir(parents=True, exist_ok=True)
    schemas = load_schemas(Path(args.schema_dir))

    oxart_path = build_dir / OXART_NAME
    run([args.objcopy, "--dump-section", f".oxart={oxart_path}", str(binary)])
    bundles = parse_artifact_section(oxart_path.read_bytes())
    if len(bundles) != 1:
        raise ValueError(f"expected exactly one .oxart bundle, found {len(bundles)}")
    bundle = bundles[0]
    ptx_payloads = [item for item in bundle["payloads"] if item["kind"] == 0x0100]
    if not ptx_payloads:
        kinds = ", ".join(item["kind_name"] for item in bundle["payloads"])
        raise ValueError(f".oxart carries no PTX payload (kinds: {kinds})")
    payload = ptx_payloads[0]
    if ptx_source.read_bytes() != payload["data"]:
        raise ValueError("the .ptx sidecar and the .oxart PTX payload differ; refusing to choose one")
    target = bundle["target"]
    if target != args.target_arch:
        raise ValueError(f".oxart target {target} does not match requested {args.target_arch}")

    ptx_artifact = out_dir / f"{ARTIFACT_STEM}.{target}.ptx"
    cubin_artifact = out_dir / f"{ARTIFACT_STEM}.{target}.cubin"
    ptx_artifact.write_bytes(payload["data"])
    run([args.ptxas, f"-arch={target}", str(ptx_artifact), "-o", str(cubin_artifact)])
    ptxas_version = next(
        (line.strip() for line in run([args.ptxas, "--version"]).stdout.splitlines() if "release" in line),
        "unknown",
    )
    ptx_text = ptx_artifact.read_text()

    for schema in schemas:
        entry = schema["entry_symbol"]
        ptx_info = parse_ptx_entry(ptx_text, entry)
        cubin_info = parse_cubin_entry(args.cuobjdump, cubin_artifact, entry)
        if entry not in cubin_info["entry_symbols"]:
            raise ValueError(f"cubin entry symbol {entry} not found in {cubin_info['entry_symbols']}")
        arguments = build_arguments(schema, ptx_info, cubin_info)
        check_arguments(entry, arguments)
        manifest = {
            "manifest": "mlxcel.cuda-oxide-fused-norm.manifest/v1",
            "abi_version": schema["abi_version"],
            "entry_symbol": entry,
            "module_entries": ptx_info["entries"],
            "source": schema["source_revision"],
            "target": {
                "device_arch": target,
                "ptx_version": ptx_info["version"],
                "cubin_target": cubin_info["target"],
            },
            "container": {
                "section": ".oxart",
                "magic": ARTIFACT_MAGIC.decode(),
                "format_version": bundle["version"],
                "bundle_name": bundle["name"],
                "target": bundle["target"],
                "compile_options": bundle["compile_options"],
                "payload_kind": payload["kind_name"],
                "payload_name": payload["name"],
                "payload_bytes": payload["bytes"],
                "payload_sha256": payload["sha256"],
                "entries": bundle["entries"],
            },
            "artifacts": {
                "ptx": {
                    "path": str(ptx_artifact.relative_to(out_dir.parent)),
                    "sha256": sha256(ptx_artifact.read_bytes()),
                    "bytes": ptx_artifact.stat().st_size,
                    "format": "PTX",
                },
                "cubin": {
                    "path": str(cubin_artifact.relative_to(out_dir.parent)),
                    "sha256": sha256(cubin_artifact.read_bytes()),
                    "bytes": cubin_artifact.stat().st_size,
                    "format": "cubin",
                    "producer": ptxas_version,
                },
            },
            "arguments": arguments,
            "launch": schema["launch"],
        }
        manifest_path = out_dir / f"{entry}.{target}.manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"exported entry={entry} manifest={manifest_path} args={len(arguments)}")
    if ERRORS:
        return 1
    print(
        f"exported ptx={ptx_artifact} cubin={cubin_artifact} entries={len(schemas)} "
        f"container=version-{bundle['version']} target={target} payload={payload['kind_name']}"
    )
    return 0


def validate(args: argparse.Namespace) -> int:
    schema_dir = Path(args.schema_dir)
    manifest_dir = Path(args.manifest_dir)
    schemas = {s["entry_symbol"]: s for s in load_schemas(schema_dir)}
    manifests = sorted(manifest_dir.glob("*.manifest.json"))
    if len(manifests) != len(schemas):
        fail(f"{len(manifests)} manifests under {manifest_dir} for {len(schemas)} schemas")
    base = manifest_dir.parent
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text())
        entry = manifest["entry_symbol"]
        schema = schemas.get(entry)
        if schema is None:
            fail(f"{manifest_path.name}: no schema for entry {entry}")
            continue
        if manifest["abi_version"] != schema["abi_version"]:
            fail(f"{entry}: manifest abi_version differs from the schema")
        if manifest["source"] != schema["source_revision"]:
            fail(f"{entry}: manifest source revision differs from the schema")
        ptx_path = artifact_path(base, manifest["artifacts"]["ptx"]["path"])
        cubin_path = artifact_path(base, manifest["artifacts"]["cubin"]["path"])
        ptx_bytes = ptx_path.read_bytes()
        cubin_bytes = cubin_path.read_bytes()
        if sha256(ptx_bytes) != manifest["artifacts"]["ptx"]["sha256"]:
            fail(f"{entry}: PTX digest differs from the manifest")
        if sha256(cubin_bytes) != manifest["artifacts"]["cubin"]["sha256"]:
            fail(f"{entry}: cubin digest differs from the manifest")
        if manifest["container"]["payload_sha256"] != manifest["artifacts"]["ptx"]["sha256"]:
            fail(f"{entry}: the committed PTX is not the .oxart payload")
        ptx_info = parse_ptx_entry(ptx_bytes.decode(), entry)
        cubin_info = parse_cubin_entry(args.cuobjdump, cubin_path, entry)
        if entry not in cubin_info["entry_symbols"]:
            fail(f"{entry}: not an entry symbol of the cubin")
        if ptx_info["target"] != manifest["target"]["device_arch"]:
            fail(f"{entry}: PTX target {ptx_info['target']} != manifest {manifest['target']['device_arch']}")
        if cubin_info["target"] != manifest["target"]["cubin_target"]:
            fail(f"{entry}: cubin target {cubin_info['target']} != manifest {manifest['target']['cubin_target']}")
        try:
            rederived = build_arguments(schema, ptx_info, cubin_info)
        except ValueError as error:
            fail(f"{entry}: {error}")
            continue
        for declared, recorded in zip(rederived, manifest["arguments"]):
            for field in ("name", "cuda_param_type", "width_bits", "kind"):
                if declared.get(field) != recorded.get(field):
                    fail(f"{entry} arg {declared['name']}: manifest {field}={recorded.get(field)!r} != schema {declared.get(field)!r}")
            if declared["emitted"] != recorded["emitted"]:
                fail(f"{entry} arg {declared['name']}: emitted layout differs from the manifest")
        check_arguments(entry, rederived)
        if manifest["launch"] != schema["launch"]:
            fail(f"{entry}: manifest launch section differs from the schema")
        print(
            f"manifest valid: entry={entry} target={ptx_info['target']} ptx_version={ptx_info['version']} "
            f"args={len(rederived)} ptx_sha256={sha256(ptx_bytes)[:16]} cubin_sha256={sha256(cubin_bytes)[:16]}"
        )
    return 1 if ERRORS else 0


PASS_RE = re.compile(
    r"^PASS case=(?P<case>\S+) act=(?P<act>\S+) w=(?P<w>\S+) rows=(?P<rows>\d+) dim=(?P<dim>\d+) "
    r"grid=(?P<grid>\d+) block=(?P<block>\d+) elements=(?P<elements>\d+)"
)
RESULT_RE = re.compile(
    r"^RESULT standalone=1 mlx=0 cases_passed=(?P<passed>\d+) cases_total=(?P<total>\d+) "
    r"rejected=(?P<rejected>\d+) elements_checked=(?P<elements>\d+)"
)
ENTRY_FOR_PAIR = {
    ("f32", "f32"): "fused_add_rms_norm_f32",
    ("f16", "f16"): "fused_add_rms_norm_f16",
    ("bf16", "bf16"): "fused_add_rms_norm_bf16",
    ("f16", "f32"): "fused_add_rms_norm_f16_wf32",
    ("bf16", "f32"): "fused_add_rms_norm_bf16_wf32",
}


def check_launch(args: argparse.Namespace) -> int:
    """Verify each harness launch against its entry's manifest launch rule and
    the declared per-entry totals, so a shortened or re-routed run fails."""
    manifests = {}
    for path in sorted(Path(args.manifest_dir).glob("*.manifest.json")):
        manifest = json.loads(path.read_text())
        manifests[manifest["entry_symbol"]] = manifest
    cases = []
    result = None
    rejects = 0
    for line in Path(args.log).read_text().splitlines():
        stripped = line.strip()
        match = PASS_RE.match(stripped)
        if match:
            cases.append({k: (int(v) if v.isdigit() else v) for k, v in match.groupdict().items()})
            continue
        if stripped.startswith("REJECT case="):
            rejects += 1
            continue
        match = RESULT_RE.match(stripped)
        if match:
            result = {k: int(v) for k, v in match.groupdict().items()}
    if not cases:
        fail(f"no PASS lines found in {args.log}; did the harness execute?")
    if result is None:
        fail(f"no RESULT line found in {args.log}")

    per_entry: dict[str, dict[str, int]] = {}
    seen = set()
    for case in cases:
        entry = ENTRY_FOR_PAIR.get((case["act"], case["w"]))
        if entry is None or entry not in manifests:
            fail(f"case {case['case']}: no manifest for dtype pair {case['act']}/{case['w']}")
            continue
        launch = manifests[entry]["launch"]
        key = (case["case"], case["rows"], case["dim"])
        if key in seen:
            fail(f"duplicate case {case['case']}")
        seen.add(key)
        if launch.get("grid_dim_rule") != "rows":
            fail(f"{entry}: manifest grid_dim_rule {launch.get('grid_dim_rule')!r} is not 'rows'")
        if launch.get("block_dim_rule") != "fused_norm_threads(dim)":
            fail(f"{entry}: manifest block_dim_rule {launch.get('block_dim_rule')!r} is not 'fused_norm_threads(dim)'")
        if launch.get("shared_mem_bytes") != 0:
            fail(f"{entry}: manifest requests dynamic shared memory; the harness launches with 0")
        if case["grid"] != case["rows"]:
            fail(f"case {case['case']}: launched grid={case['grid']}, rows={case['rows']}")
        want_block = fused_norm_threads(case["dim"])
        if case["block"] != want_block:
            fail(f"case {case['case']}: launched block={case['block']}, fused_norm_threads({case['dim']})={want_block}")
        if case["block"] % SIMD_WIDTH or case["block"] > MAX_THREADS or case["block"] <= 0:
            fail(f"case {case['case']}: block {case['block']} is outside the kernel's contract")
        if case["elements"] != case["rows"] * case["dim"]:
            fail(f"case {case['case']}: verified {case['elements']} elements for rows*dim={case['rows'] * case['dim']}")
        if case["rows"] <= 0 or case["dim"] <= 0:
            fail(f"case {case['case']}: non-positive shape")
        bucket = per_entry.setdefault(entry, {"cases": 0, "elements": 0})
        bucket["cases"] += 1
        bucket["elements"] += case["elements"]

    total_elements = 0
    for entry, manifest in manifests.items():
        launch = manifest["launch"]
        got = per_entry.get(entry, {"cases": 0, "elements": 0})
        if launch.get("case_count") != got["cases"]:
            fail(f"{entry}: manifest case_count={launch.get('case_count')}, log has {got['cases']} PASS lines")
        if launch.get("elements_total") != got["elements"]:
            fail(f"{entry}: manifest elements_total={launch.get('elements_total')}, PASS lines sum to {got['elements']}")
        total_elements += got["elements"]
    if result is not None:
        if result["total"] != len(cases) or result["passed"] != len(cases):
            fail(f"RESULT reports {result['passed']}/{result['total']} cases, log has {len(cases)} PASS lines")
        if result["elements"] != total_elements:
            fail(f"RESULT reports {result['elements']} elements, PASS lines sum to {total_elements}")
        if result["rejected"] != rejects or rejects == 0:
            fail(f"RESULT reports {result['rejected']} rejected shapes, log has {rejects} REJECT lines")
    if ERRORS:
        return 1
    print(
        f"launch valid: cases={len(cases)} entries={len(per_entry)} grid_rule=rows "
        f"block_rule=fused_norm_threads(dim) elements={total_elements} rejected={rejects}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    exporter = sub.add_parser("export", help="extract PTX from .oxart, assemble the cubin, write one manifest per schema")
    exporter.add_argument("--binary", required=True)
    exporter.add_argument("--ptx", required=True, help=".ptx sidecar the compiler wrote")
    exporter.add_argument("--schema-dir", required=True)
    exporter.add_argument("--out-dir", required=True)
    exporter.add_argument("--build-dir", required=True)
    exporter.add_argument("--target-arch", required=True)
    exporter.add_argument("--objcopy", default=shutil.which("objcopy") or "objcopy")
    exporter.add_argument("--ptxas", default=shutil.which("ptxas") or "ptxas")
    exporter.add_argument("--cuobjdump", default=shutil.which("cuobjdump") or "cuobjdump")
    exporter.set_defaults(func=export)
    validator = sub.add_parser("validate", help="check every manifest against its schema and the committed artifacts")
    validator.add_argument("--manifest-dir", required=True)
    validator.add_argument("--schema-dir", required=True)
    validator.add_argument("--cuobjdump", default=shutil.which("cuobjdump") or "cuobjdump")
    validator.set_defaults(func=validate)
    launcher = sub.add_parser("check-launch", help="check harness launches against the manifests")
    launcher.add_argument("--manifest-dir", required=True)
    launcher.add_argument("--log", required=True)
    launcher.set_defaults(func=check_launch)
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, OSError, KeyError) as error:
        print(f"FAIL {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
