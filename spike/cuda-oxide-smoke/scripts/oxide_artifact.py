#!/usr/bin/env python3
"""Export and validate the cuda-oxide smoke artifacts (issue #2831).

The compiler embeds device code in a `.oxart` container section inside the
host binary. That container is not PTX: it is a versioned bundle with a magic
header, a target string, payload records, and entry records. This tool parses
the container, extracts the PTX payload, assembles a cubin with ptxas, and
writes a manifest. `validate` re-parses the committed PTX and cubin and checks
the manifest against them, including the argument layout the assembler reports.

Only the Python standard library is used. The CUDA and objcopy binaries come
from the environment set by scripts/env.sh. Format parsers live in
`oxide_format.py`.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from oxide_format import (
    ARTIFACT_MAGIC,
    parse_artifact_section,
    parse_cubin,
    parse_ptx,
    run,
    sha256,
)

ERRORS: list[str] = []


def fail(message: str) -> None:
    ERRORS.append(message)
    print(f"FAIL {message}", file=sys.stderr)


def export(args: argparse.Namespace) -> int:
    binary = Path(args.binary)
    ptx_source = Path(args.ptx)
    schema = json.loads(Path(args.schema).read_text())
    out_dir = Path(args.out_dir)
    build_dir = Path(args.build_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    build_dir.mkdir(parents=True, exist_ok=True)

    oxart_path = build_dir / "cuda_oxide_smoke.oxart"
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
    emitted = ptx_source.read_bytes()
    if emitted != payload["data"]:
        raise ValueError("the .ptx sidecar and the .oxart PTX payload differ; refusing to choose one")

    target = bundle["target"]
    if target != args.target_arch:
        raise ValueError(f".oxart target {target} does not match requested {args.target_arch}")

    ptx_artifact = out_dir / f"vecadd.{target}.ptx"
    cubin_artifact = out_dir / f"vecadd.{target}.cubin"
    ptx_artifact.write_bytes(payload["data"])
    run([args.ptxas, f"-arch={target}", str(ptx_artifact), "-o", str(cubin_artifact)])
    ptxas_version = next(
        (line.strip() for line in run([args.ptxas, "--version"]).stdout.splitlines() if "release" in line),
        "unknown",
    )

    ptx_info = parse_ptx(ptx_artifact.read_text())
    cubin_info = parse_cubin(args.cuobjdump, cubin_artifact)
    entry_symbol = schema["entry_symbol"]
    if ptx_info["entry"] != entry_symbol:
        raise ValueError(f"PTX entry {ptx_info['entry']} is not the schema entry {entry_symbol}")
    if entry_symbol not in cubin_info["entry_symbols"]:
        raise ValueError(f"cubin entry symbol {entry_symbol} not found in {cubin_info['entry_symbols']}")

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

    manifest = {
        "manifest": "mlxcel.cuda-oxide-smoke.manifest/v1",
        "abi_version": schema["abi_version"],
        "entry_symbol": entry_symbol,
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
    manifest_path = out_dir / "vecadd.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(
        f"exported ptx={ptx_artifact} cubin={cubin_artifact} manifest={manifest_path}\n"
        f"container=version-{bundle['version']} target={target} payload={payload['kind_name']} "
        f"ptx_version={ptx_info['version']} args={len(arguments)}"
    )
    return 0


def validate(args: argparse.Namespace) -> int:
    manifest = json.loads(Path(args.manifest).read_text())
    schema = json.loads(Path(args.schema).read_text())
    base = Path(args.manifest).parent.parent
    ptx_path = base / manifest["artifacts"]["ptx"]["path"]
    cubin_path = base / manifest["artifacts"]["cubin"]["path"]

    # Source schema and manifest agree on the declared ABI.
    if manifest["abi_version"] != schema["abi_version"]:
        fail(f"manifest abi_version {manifest['abi_version']} != schema {schema['abi_version']}")
    if manifest["entry_symbol"] != schema["entry_symbol"]:
        fail("manifest entry_symbol differs from the schema")
    if manifest["launch"] != schema["launch"]:
        fail("manifest launch constraints differ from the schema")

    # Artifact digests bind the manifest to the committed bytes.
    for name, path in (("ptx", ptx_path), ("cubin", cubin_path)):
        digest = sha256(path.read_bytes())
        if digest != manifest["artifacts"][name]["sha256"]:
            fail(f"{name} sha256 {digest} != manifest {manifest['artifacts'][name]['sha256']}")
        if path.stat().st_size != manifest["artifacts"][name]["bytes"]:
            fail(f"{name} size differs from the manifest")

    ptx_info = parse_ptx(ptx_path.read_text())
    cubin_info = parse_cubin(args.cuobjdump, cubin_path)

    if ptx_info["version"] != manifest["target"]["ptx_version"]:
        fail(f"PTX .version {ptx_info['version']} != manifest {manifest['target']['ptx_version']}")
    if ptx_info["target"] != manifest["target"]["device_arch"]:
        fail(f"PTX .target {ptx_info['target']} != manifest {manifest['target']['device_arch']}")
    if cubin_info["target"] != manifest["target"]["device_arch"]:
        fail(f"cubin target {cubin_info['target']} != manifest {manifest['target']['device_arch']}")
    if ptx_info["entry"] != manifest["entry_symbol"]:
        fail(f"PTX entry {ptx_info['entry']} != manifest {manifest['entry_symbol']}")
    if manifest["entry_symbol"] not in cubin_info["entry_symbols"]:
        fail(f"cubin entry symbols {cubin_info['entry_symbols']} do not contain {manifest['entry_symbol']}")

    expected = len(schema["arguments"])
    if len(ptx_info["parameters"]) != expected:
        fail(f"PTX has {len(ptx_info['parameters'])} parameters, schema declares {expected}")
    if len(cubin_info["kparams"]) != expected:
        fail(f"cubin has {len(cubin_info['kparams'])} kparams, schema declares {expected}")
    if len(manifest["arguments"]) != expected:
        fail(f"manifest has {len(manifest['arguments'])} arguments, schema declares {expected}")

    declared_fields = (
        "ordinal",
        "name",
        "cuda_param_type",
        "width_bits",
        "kind",
        "allocation_align_bytes",
        "length_argument",
    )
    for index, declared in enumerate(schema["arguments"]):
        ordinal = declared["ordinal"]
        if ordinal != index:
            fail(f"schema argument {index} declares ordinal {ordinal}; ordinals must be contiguous from 0")
            continue
        manifest_entry = next(
            (item for item in manifest["arguments"] if item["ordinal"] == ordinal), None
        )
        if manifest_entry is None:
            fail(f"manifest has no argument for ordinal {ordinal}")
            continue
        for field in declared_fields:
            if field in declared and manifest_entry.get(field) != declared[field]:
                fail(
                    f"arg {declared['name']}: manifest {field}={manifest_entry.get(field)!r} "
                    f"!= schema {declared[field]!r}"
                )
        if ordinal >= len(ptx_info["parameters"]):
            continue
        param = ptx_info["parameters"][ordinal]
        if param["ptx_type"] != f".{declared['cuda_param_type']}":
            fail(f"arg {declared['name']}: PTX type {param['ptx_type']} != .{declared['cuda_param_type']}")
        if param["pointer"] != (declared["kind"] == "device_pointer"):
            fail(f"arg {declared['name']}: PTX pointer-ness {param['pointer']} disagrees with schema kind")
        if declared["kind"] == "device_pointer":
            # The compiler may assume less alignment than the CUDA allocator
            # guarantees, but never more; a stricter assumption would be an ABI
            # bug on a buffer with the declared alignment.
            if param["pointee_align_bytes"] > declared["allocation_align_bytes"]:
                fail(
                    f"arg {declared['name']}: PTX assumes {param['pointee_align_bytes']}-byte alignment, "
                    f"above the declared {declared['allocation_align_bytes']}"
                )
        measured = manifest_entry["emitted"]
        if measured["ptx_type"] != param["ptx_type"] or measured["ptx_pointer"] != param["pointer"]:
            fail(f"arg {declared['name']}: manifest PTX measurement is stale")

        kparam = next((item for item in cubin_info["kparams"] if item["ordinal"] == ordinal), None)
        if kparam is None:
            fail(f"arg {declared['name']}: no cubin kparam for ordinal {ordinal}")
            continue
        if kparam["size_bytes"] != declared["width_bits"] // 8:
            fail(
                f"arg {declared['name']}: cubin param size {kparam['size_bytes']} bytes "
                f"!= declared {declared['width_bits'] // 8}"
            )
        if kparam["byte_offset"] != ordinal * (declared["width_bits"] // 8):
            fail(f"arg {declared['name']}: cubin offset {kparam['byte_offset']} is not ordinal*8")
        expected_space = "global" if declared["kind"] == "device_pointer" else "value"
        if kparam["space"] != expected_space:
            fail(f"arg {declared['name']}: cubin space {kparam['space']} != {expected_space}")
        if (
            kparam["byte_offset"] != measured["cubin_byte_offset"]
            or kparam["size_bytes"] != measured["cubin_size_bytes"]
        ):
            fail(f"arg {declared['name']}: manifest cubin measurement is stale")

    if ERRORS:
        print(f"manifest validation failed with {len(ERRORS)} error(s)", file=sys.stderr)
        return 1
    print(
        f"manifest valid: entry={ptx_info['entry']} target={ptx_info['target']} "
        f"ptx_version={ptx_info['version']} args={len(ptx_info['parameters'])} "
        f"ptx_sha256={manifest['artifacts']['ptx']['sha256'][:16]} "
        f"cubin_sha256={manifest['artifacts']['cubin']['sha256'][:16]}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    exporter = sub.add_parser("export", help="extract PTX from .oxart, assemble cubin, write manifest")
    exporter.add_argument("--binary", required=True)
    exporter.add_argument("--ptx", required=True, help=".ptx sidecar the compiler wrote")
    exporter.add_argument("--schema", required=True)
    exporter.add_argument("--out-dir", required=True)
    exporter.add_argument("--build-dir", required=True)
    exporter.add_argument("--target-arch", required=True)
    exporter.add_argument("--objcopy", default=shutil.which("objcopy") or "objcopy")
    exporter.add_argument("--ptxas", default=shutil.which("ptxas") or "ptxas")
    exporter.add_argument("--cuobjdump", default=shutil.which("cuobjdump") or "cuobjdump")
    exporter.set_defaults(func=export)

    validator = sub.add_parser("validate", help="check a manifest against the committed PTX and cubin")
    validator.add_argument("--manifest", required=True)
    validator.add_argument("--schema", required=True)
    validator.add_argument("--cuobjdump", default=shutil.which("cuobjdump") or "cuobjdump")
    validator.set_defaults(func=validate)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
