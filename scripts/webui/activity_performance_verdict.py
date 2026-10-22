# Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
"""Python port of the Activity decode-overhead verdict (#1925).

The rule and its rationale live in webui/scripts/activity-performance-config.mjs; every constant
lives in webui/scripts/activity-performance-policy.json, which this module reads instead of
keeping its own copy. Both implementations are tested against
webui/scripts/activity-performance-fixtures.json. Arithmetic follows the JavaScript operation
for operation, so the two produce bit-identical medians and interval bounds.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

POLICY_PATH = Path(__file__).resolve().parents[2] / "webui" / "scripts" / "activity-performance-policy.json"
FIXTURE_PATH = POLICY_PATH.with_name("activity-performance-fixtures.json")
POLICY_KEYS = ("budget_percent", "precision_ceiling_percent", "tail_alpha", "min_pairs", "pair_step", "max_pairs")
STATUSES = ("investigate", "within-target", "inconclusive")
# Browser clients per measured mode; with a variable pair count, sample and preflight counts are
# derived from each mode's recorded paired_runs by expected_counts().
CLIENTS_PER_MODE = {"one-visible": 1, "two-visible": 2, "hidden": 1}



def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def sign_test_rank(n: int, alpha: float) -> int:
    """Largest k with P(Binomial(n, 1/2) <= k - 1) <= alpha; 0 when no order statistic qualifies."""
    cumulative = 0
    k = 0
    for j in range(n + 1):
        cumulative += math.comb(n, j)
        if cumulative > alpha * 2**n:
            break
        k = j + 1
    return k


def validate_policy(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AssertionError("Activity verdict policy must be an object")
    if sorted(value) != sorted(POLICY_KEYS):
        raise AssertionError(f"Activity verdict policy must have exactly: {', '.join(POLICY_KEYS)}")
    budget, ceiling, alpha = value["budget_percent"], value["precision_ceiling_percent"], value["tail_alpha"]
    if not all(_is_number(v) for v in (budget, ceiling, alpha)) or not ceiling >= budget or not 0 < alpha < 0.5:
        raise AssertionError("Activity verdict policy thresholds are invalid")
    low, step, high = value["min_pairs"], value["pair_step"], value["max_pairs"]
    if not all(_is_int(v) for v in (low, step, high)) or step < 2 or step % 2 or low % 2 or high > 50 or high < low or (high - low) % step:
        raise AssertionError("Activity verdict policy pair counts are invalid")
    if sign_test_rank(low, alpha) < 1:
        raise AssertionError("Activity verdict policy min_pairs is too small for its tail_alpha")
    return dict(value)


def load_policy(path: Path = POLICY_PATH) -> dict[str, Any]:
    return validate_policy(json.loads(path.read_text(encoding="utf-8")))


def pair_degradation(pair: dict[str, Any]) -> float:
    off, on = pair.get("off"), pair.get("on")
    if not (_is_number(off) and off > 0 and _is_number(on) and on > 0):
        raise AssertionError("Activity pair rates must be finite and positive")
    return (off - on) / off * 100


def median_interval(values: list[float], policy: dict[str, Any]) -> tuple[float, list[float] | None]:
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        raise AssertionError("Activity verdict needs at least one pair")
    median = ordered[(n - 1) // 2] if n % 2 == 1 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2
    k = sign_test_rank(n, policy["tail_alpha"])
    return median, (None if k < 1 else [ordered[k - 1], ordered[n - k]])


def look_verdict(pairs: list[dict[str, Any]], policy: dict[str, Any]) -> dict[str, Any]:
    n = len(pairs)
    median, interval = median_interval([pair_degradation(p) for p in pairs], policy)
    base = {"paired_runs": n, "median_decode_degradation_percent": median, "interval_percent": interval}
    if n < policy["min_pairs"] or n > policy["max_pairs"] or (n - policy["min_pairs"]) % policy["pair_step"]:
        return {**base, "status": "continue"}
    assert interval is not None
    if interval[0] > policy["budget_percent"]:
        return {**base, "status": "investigate"}
    if interval[1] <= policy["precision_ceiling_percent"]:
        return {**base, "status": "within-target"}
    return {**base, "status": "inconclusive" if n >= policy["max_pairs"] else "continue"}


def sequential_verdict(pairs: list[dict[str, Any]], policy: dict[str, Any]) -> dict[str, Any]:
    n = policy["min_pairs"]
    while n <= min(len(pairs), policy["max_pairs"]):
        verdict = look_verdict(pairs[:n], policy)
        if verdict["status"] != "continue":
            if n != len(pairs):
                raise AssertionError(f"Activity sampling continued after the verdict at {n} pairs")
            return verdict
        n += policy["pair_step"]
    raise AssertionError(f"Activity sampling stopped at {len(pairs)} pairs without a verdict")


def _fixed(value: float) -> str:
    # JavaScript's toFixed(2) for the magnitudes seen here; only used in human-readable text.
    return f"{value:.2f}"


def verdict_rule(summary: dict[str, Any], policy: dict[str, Any]) -> str:
    low, high = summary["interval_percent"]
    confidence = round((1 - 2 * policy["tail_alpha"]) * 100)
    head = f"median {_fixed(summary['median_decode_degradation_percent'])}%, {confidence}% interval [{_fixed(low)}%, {_fixed(high)}%] over {summary['paired_runs']} pairs"
    if summary["status"] == "investigate":
        return f"{head}; slowdown: the interval lies wholly above the {policy['budget_percent']}% budget"
    if summary["status"] == "within-target":
        return f"{head}; within target: the interval reaches no further than {policy['precision_ceiling_percent']}% and does not lie above the {policy['budget_percent']}% budget"
    return f"{head}; measurement noise, not a slowdown: after the {policy['max_pairs']}-pair cap the interval still straddles the {policy['budget_percent']}% budget and reaches past {policy['precision_ceiling_percent']}%"


def mode_summary(mode: str, pairs: list[dict[str, Any]], policy: dict[str, Any]) -> dict[str, Any]:
    verdict = sequential_verdict(pairs, policy)
    degradations = [pair_degradation(p) for p in pairs]
    baseline = [p["off"] for p in pairs]
    total = 0.0
    for value in baseline:
        total += value
    mean = total / len(baseline)
    squares = 0.0
    for value in baseline:
        squares += (value - mean) ** 2
    cv = math.sqrt(squares / len(baseline)) / mean * 100
    summary = {
        "mode": mode,
        "status": verdict["status"],
        "paired_runs": verdict["paired_runs"],
        "median_decode_degradation_percent": verdict["median_decode_degradation_percent"],
        "interval_percent": verdict["interval_percent"],
        "baseline_cv_percent": cv,
        "paired_range_percent": [min(degradations), max(degradations)],
    }
    return {**summary, "rule": verdict_rule(summary, policy)}


def pairs_from_samples(samples: list[Any]) -> dict[str, list[dict[str, float]]]:
    """Pair a recorded sample list by adjacency, exactly as pairsFromSamples does in JavaScript."""
    measured = [s for s in samples if not (isinstance(s, dict) and s.get("mode") == "warmup")]
    if len(measured) % 2:
        raise AssertionError("Activity samples do not form whole pairs")
    by_mode: dict[str, list[dict[str, float]]] = {}
    for i in range(0, len(measured), 2):
        first, second = measured[i], measured[i + 1]
        if not (isinstance(first, dict) and isinstance(second, dict)):
            raise AssertionError("Activity samples must be objects")
        off, on = (first, second) if first.get("mode") == "off" else (second, first)
        if off.get("mode") != "off" or on.get("mode") == "off" or not isinstance(on.get("mode"), str) or first.get("pair") != second.get("pair"):
            raise AssertionError(f"Activity samples {i + 1} and {i + 2} are not one off/on pair")
        pairs = by_mode.setdefault(on["mode"], [])
        if not _is_int(on.get("pair")) or on["pair"] != len(pairs) or (on["pair"] % 2 == 0) != (first.get("mode") == "off"):
            raise AssertionError(f"Activity {on['mode']} pair {on.get('pair')} is out of order")
        pairs.append({"off": off.get("tokens_per_second"), "on": on.get("tokens_per_second")})
    return by_mode


def check_summary_consistency(item: dict[str, Any], policy: dict[str, Any]) -> None:
    """Structure and internal consistency of one recorded mode summary, for evidence without samples."""
    mode = item.get("mode")
    status = item.get("status")
    if status not in STATUSES:
        raise AssertionError(f"{mode} status {status!r} is not a final verdict")
    n = item.get("paired_runs")
    if not _is_int(n) or n < policy["min_pairs"] or n > policy["max_pairs"] or (n - policy["min_pairs"]) % policy["pair_step"]:
        raise AssertionError(f"{mode} paired_runs {n!r} is not a look of the verdict policy")
    interval = item.get("interval_percent")
    if not (isinstance(interval, list) and len(interval) == 2 and all(_is_number(v) for v in interval) and interval[0] <= interval[1]):
        raise AssertionError(f"{mode} interval_percent must be two ordered finite numbers")
    median = item.get("median_decode_degradation_percent")
    if not _is_number(median) or not interval[0] <= median <= interval[1]:
        raise AssertionError(f"{mode} median must be finite and inside its interval")
    low, high = interval
    expected = "investigate" if low > policy["budget_percent"] else "within-target" if high <= policy["precision_ceiling_percent"] else "inconclusive"
    if expected == "inconclusive" and n < policy["max_pairs"]:
        raise AssertionError(f"{mode} stopped undecided at {n} pairs, below the {policy['max_pairs']}-pair cap")
    if status != expected:
        raise AssertionError(f"{mode} status {status} contradicts its interval [{low}, {high}] under the verdict policy")


def failure_message(summaries: list[dict[str, Any]]) -> str:
    """One line per failing mode: status, point estimate, interval, pair count and the rule that fired."""
    lines = []
    for item in summaries:
        status = item.get("status")
        if status == "within-target":
            continue
        kind = {"investigate": "slowdown", "inconclusive": "measurement noise"}.get(status, "unknown verdict")
        lines.append(f"{item.get('mode')}: {status} ({kind}): {item.get('rule', 'no rule recorded')}")
    return "; ".join(lines)


def expected_counts(paired_runs: dict[str, int]) -> tuple[dict[str, int], dict[str, int]]:
    """Samples: one warmup, one per pair per mode, and an `off` baseline for every pair. Preflight:
    every clients() call for a mode records each of its clients, once before the warmup and once
    per pair."""
    samples = {"warmup": 1, "off": sum(paired_runs.values()), **paired_runs}
    preflight = {mode: CLIENTS_PER_MODE[mode] * (1 + runs) for mode, runs in paired_runs.items()}
    return samples, preflight
