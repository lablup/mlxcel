"""Fail closed on missing cells or unmet #2264 performance acceptance."""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
import statistics as st
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from provenance import canonical_mlx_pin


def records(path: Path, count: int) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.startswith("{")]
    if len(rows) != count:
        raise ValueError(f"{path}: expected {count} records, got {len(rows)}")
    return rows


def positive(value: float) -> float:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"invalid measurement {value}")
    return value


def counters(rows: list[dict], side: str, *, closed: bool = False, require_drafts: bool = False) -> None:
    for row in rows:
        proposed, accepted, rounds = (row[key] for key in ("draft_n", "draft_n_accepted", "draft_rounds"))
        if any(type(value) is not int or value < 0 for value in (proposed, accepted, rounds)):
            raise ValueError(f"invalid draft counters: {row}")
        if not accepted <= proposed or not rounds <= proposed <= rounds * 7:
            raise ValueError(f"inconsistent draft counters: {row}")
        if (side == "off" or closed) and (proposed or accepted or rounds):
            raise ValueError(f"{side} arm unexpectedly verified: {row}")
        if require_drafts and (not proposed or not accepted):
            raise ValueError(f"copy arm did not exercise verification: {row}")
        if "draft_kind" in row:
            expected = "prompt-lookup" if proposed else None
            if row["draft_kind"] != expected:
                raise ValueError(f"wrong draft kind: {row}")


def verify_arm(directory: Path, label: str, side: str, storage: str | None = None) -> None:
    metadata = json.loads((directory / f"{label}.arm.json").read_text())
    provenance = json.loads((directory / "environment.json").read_text())
    args = metadata["argv"]
    if metadata["binary"] != provenance["binary"]:
        raise ValueError(f"wrong server binary for {label}")
    def option(name: str) -> str | None:
        if args.count(name) > 1:
            raise ValueError(f"duplicate {name} in {label}")
        return args[args.index(name) + 1] if name in args else None
    if option("--draft-kind") != ("prompt-lookup" if side == "on" else None):
        raise ValueError(f"wrong flag arm for {label}")
    if str(Path(option("-m") or "").resolve()) != provenance["model"]:
        raise ValueError(f"wrong model for {label}")
    if option("--decode-storage-backend") != storage:
        raise ValueError(f"wrong storage for {label}")


def rates(rows: list[dict]) -> list[float]:
    for row in rows:
        if row["tokens"] != 256 or row["conc"] != 1:
            raise ValueError(f"incomplete raw completion: {row}")
        if not 0 <= row["draft_n_accepted"] <= row["draft_n"]:
            raise ValueError(f"invalid acceptance counters: {row}")
    return [positive(row["decode_tok_s"][0]) for row in rows]


def copy_gains(directory: Path, *, verify_identity: bool = False) -> list[float]:
    paired = [[] for _ in range(4)]
    for round_id in range(1, 6):
        arms = {}
        for side in ("on", "off"):
            label = f"ab-{side}-r{round_id}"
            if verify_identity:
                verify_arm(directory, label, side)
            rows = records(directory / f"{label}.txt", 8)
            counters(rows, side)
            copies = [row for row in rows if row["style"] == "copy"]
            if [row["rep"] for row in copies] != list(range(4)):
                raise ValueError("missing or reordered copy requests")
            counters(copies, side, require_drafts=side == "on")
            arms[side] = rates(copies)
        for index in range(4):
            paired[index].append(arms["on"][index] / arms["off"][index] - 1)
    return [st.median(values) for values in paired]


def concurrency(path: Path) -> dict[str, float]:
    result = {}
    for block in path.read_text().split("Server:")[1:]:
        style = re.search(r"style (\w+)", block)
        if style is None:
            raise ValueError(f"missing concurrency style in {path}")
        cells = re.findall(r"^\s+2\s+2\s+0\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+([\d.]+)\s*$", block, re.M)
        if len(cells) != 1 or style[1] in result:
            raise ValueError(f"missing or duplicate C=2 result in {path}")
        result[style[1]] = positive(float(cells[0]))
    if set(result) != {"plain", "copy"}:
        raise ValueError(f"incomplete concurrency workloads in {path}")
    return result


def below(value: float, floor: float) -> bool:
    # Tolerate only floating-point representation error at the exact gate.
    return value < floor and not math.isclose(value, floor, rel_tol=0.0, abs_tol=1e-12)


def main() -> None:
    directory = Path(sys.argv[1]).resolve()
    historical = directory.parent / "server-prompt-lookup-gb10-2026-10-11"
    provenance = json.loads((directory / "environment.json").read_text())
    model = provenance["model_config"]
    if (model["model_type"], model["hidden_size"], model["num_hidden_layers"], model["quantization"]["bits"]) != ("qwen3", 2048, 28, 4):
        raise ValueError("acceptance requires Qwen3-1.7B affine 4-bit")
    if provenance["native_mlx_pin"] != canonical_mlx_pin(Path(__file__).resolve().parents[4]):
        raise ValueError("recorded native MLX pin differs from the canonical checkout pin")
    if provenance["purpose"] != "final-acceptance":
        raise ValueError("diagnostic runs are not final acceptance")
    for field, size in [("binary_source_commit", 40), ("measurement_checkout", 40), ("binary_sha256", 64), ("model_config_sha256", 64)]:
        if re.fullmatch(rf"[0-9a-f]{{{size}}}", provenance[field]) is None:
            raise ValueError(f"invalid provenance field {field}")
    if (directory / "binary-after.sha256").read_text().split()[0] != provenance["binary_sha256"]:
        raise ValueError("server binary changed during acceptance")
    failures = []
    deltas = []
    print("(a) chat: aggregate decode tokens / aggregate decode seconds per round")
    for round_id in range(1, 6):
        speed = {}
        for side in ("on", "off"):
            label = f"a2-{side}-r{round_id}"
            verify_arm(directory, label, side)
            rows = records(directory / f"{label}.txt", 8)
            counters(rows, side)
            if [row["q"] for row in rows] != list(range(8)):
                raise ValueError("missing or reordered chat questions")
            if any(row["completion_tokens"] != 256 for row in rows):
                raise ValueError("chat did not emit the requested 256 tokens")
            seconds = sum(positive(row["decode_s"]) for row in rows)
            speed[side] = sum(row["completion_tokens"] - 1 for row in rows) / seconds
        delta = speed["on"] / speed["off"] - 1
        deltas.append(delta)
        print(f"round {round_id}: on={speed['on']:.3f} off={speed['off']:.3f} delta={delta:+.3%}")
    chat_delta = st.median(deltas)
    print(f"median={chat_delta:+.3%}, range={min(deltas):+.3%}..{max(deltas):+.3%}")
    if below(chat_delta, -0.01):
        failures.append("(a) median paired chat regression exceeds 1%")

    print("(b) copy: median paired gain per request; retain 90% of original gain")
    for index, (gain, original) in enumerate(zip(copy_gains(directory, verify_identity=True), copy_gains(historical))):
        floor = 0.9 * original
        print(f"q{index}: gain={gain:+.3%} original={original:+.3%} floor={floor:+.3%}")
        if below(gain, floor):
            failures.append(f"(b) q{index} gain below 90% historical gain")

    print("(c) chat-template C=2: medians and round ranges; no disjoint regression")
    for side in ("on", "off"):
        for round_id in (1, 2, 3):
            verify_arm(directory, f"c-{side}-r{round_id}", side)
    for style in ("plain", "copy"):
        arms = {side: [concurrency(directory / f"c-{side}-r{r}.txt")[style] for r in (1, 2, 3)] for side in ("on", "off")}
        on, off = arms["on"], arms["off"]
        print(f"{style}: on={st.median(on):.3f} ({min(on):.3f}..{max(on):.3f}) off={st.median(off):.3f} ({min(off):.3f}..{max(off):.3f})")
        if max(on) < min(off):
            failures.append(f"(c) {style} on/off ranges show disjoint regression")

    print("(d) long copy: ratios of median decode rates; retain 90% of speedup above 1x")
    for storage, length, original in [("paged", 1024, 2.69), ("paged", 2048, 2.79), ("paged", 4096, 1.42), ("dense", 8192, 2.10), ("paged", 8192, None)]:
        arms = {}
        for side in ("on", "off"):
            values = []
            for round_id in (1, 2, 3):
                if length == 8192:
                    label = f"d-{storage}-{side}-r{round_id}"
                    verify_arm(directory, label, side, storage)
                    rows = records(directory / f"{label}.txt", 6)
                else:
                    label = f"d2-{side}-r{round_id}"
                    verify_arm(directory, label, side, "paged")
                    all_rows = records(directory / f"{label}.txt", 18)
                    index = [1024, 2048, 4096].index(length)
                    rows = all_rows[index * 6:(index + 1) * 6]
                closed = storage == "paged" and length == 8192
                counters(rows, side, closed=closed)
                copies = [row for row in rows if row["style"] == "long-copy"]
                counters(copies, side, closed=closed, require_drafts=side == "on" and not closed)
                if [row["rep"] for row in copies] != [0, 1, 2]:
                    raise ValueError("missing or reordered long-copy requests")
                values.extend(rates(copies))
            arms[side] = st.median(values)
        ratio = arms["on"] / arms["off"]
        floor = 1 + 0.9 * (original - 1) if original else None
        print(f"{storage} {length}: on={arms['on']:.3f} off={arms['off']:.3f} ratio={ratio:.4f} floor={floor}")
        if floor is not None and below(ratio, floor):
            failures.append(f"(d) {storage} {length} gain below 90% historical gain")

    print(json.dumps({"passed": not failures, "failures": failures}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
