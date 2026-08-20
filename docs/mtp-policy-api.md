# Adaptive MTP policy API (`/v1/internal/mtp-policy`)

`mlxcel-server` decides per machine whether the singleton (B=1) MTP speculative burst is worth running for the pairing it is serving. It profiles the first few qualifying requests, settles on an enable or decline verdict, and persists that verdict so a restart does not re-profile. See `MLXCEL_MTP_ADAPTIVE` in [Environment variables](environment-variables.md) for how the decision is made, and [Continuous batching](CONTINUOUS_BATCHING.md) for where it sits in the scheduler.

That verdict is the only machine-specific answer to "does MTP help for this pairing here", and it is what a host application wants to show a user. This endpoint is the supported way to read it.

Implementation source map:

| Module | Responsibility |
|--------|----------------|
| `src/server/batch/mtp_policy.rs` | The policy state machine, the persisted hint, and the published `MtpPolicySnapshot`. |
| `src/server/batch/observability.rs` | Holds the last snapshot the worker published. |
| `src/server/routes/mtp_policy.rs` | Route handler and the wire body. |

## Endpoint

| Method | Path | Description |
|--------|------|-------------|
| GET | `/v1/internal/mtp-policy` | Report the adaptive MTP policy state for the running pairing. |

No `/v1`-less alias is mounted.

The endpoint is always mounted, unlike `/props`, `/slots`, and `/metrics`, which are gated behind CLI flags. It answers with a well-formed body in every state, including when no policy is running at all, so a consumer can poll it unconditionally and never has to tell "nothing to report" apart from "this server does not serve this path". It sits behind the same API-key middleware as every other route, and its payload is as coarse as the persisted hint: a verdict, an acceptance rate, a sample count, and the pairing identity. No prompt data, no token ids, nothing request-identifying.

## Response body

```json
{
  "schema_version": 1,
  "state": "settled",
  "reason": null,
  "verdict": "enable",
  "mtp_enabled": true,
  "target": "Gemma4-12B",
  "drafter": "Gemma4-12B-MTP",
  "hardware": "M5-16c",
  "block_size": 4,
  "acceptance_rate": 0.62,
  "samples": 4,
  "samples_required": 4,
  "samples_remaining": null
}
```

| Field | Type | Meaning |
|-------|------|---------|
| `schema_version` | integer | Wire schema version of this body. See [Versioning and compatibility](#versioning-and-compatibility). |
| `state` | string | `"settled"`, `"profiling"`, `"forced"`, or `"unavailable"`. |
| `reason` | string or null | Why the policy is unavailable. Non-null only when `state` is `"unavailable"`. |
| `verdict` | string or null | `"enable"` or `"decline"` once settled, `null` otherwise. Same values as the `verdict` field of the persisted hint. |
| `mtp_enabled` | boolean or null | Whether the B=1 MTP burst runs right now. This is the live gate, not the verdict: it is `true` while profiling, because profiling forces MTP on to collect the sample. |
| `target` | string or null | Served model directory basename. |
| `drafter` | string or null | Draft model directory basename. |
| `hardware` | string or null | Coarse hardware-class label, for example `"M5-16c"`. Apple GPU generation plus GPU core count; non-Apple hosts report `"Unknown-0c"`. |
| `block_size` | integer or null | Draft block size (K) the pairing is keyed on. |
| `acceptance_rate` | number or null | Coarse measured acceptance rate (accepted draft tokens over proposed). Running value while profiling, final value once settled, `null` when nothing was measured. Rounded to two decimals once settled, matching the persisted hint exactly. |
| `samples` | integer | Qualifying samples accumulated so far, or behind the settled verdict. `0` when forced or unavailable. |
| `samples_required` | integer | Qualifying samples a profiling window needs before it settles. |
| `samples_remaining` | integer or null | Qualifying samples still needed. Non-null only while `state` is `"profiling"`, because nothing is pending in any other state. |

The four fields `target`, `drafter`, `hardware`, and `block_size` are the pairing key. A consumer showing a verdict should confirm they match the pairing it is showing, in particular `block_size`: a verdict profiled at one K does not carry to another, and changing `--num-draft-tokens` / `MLXCEL_DRAFT_BLOCK_SIZE` starts a fresh profiling window.

### States

| `state` | Meaning |
|---------|---------|
| `settled` | A verdict is in effect, either measured in this process or restored from a persisted hint. `verdict` says which way. |
| `profiling` | Still accumulating samples. There is no verdict yet; `samples_remaining` says how many qualifying single-request generations are still needed. MTP is forced on meanwhile, so `mtp_enabled` is `true`. |
| `forced` | `MLXCEL_ENABLE_MTP_B1` pinned the decision. Nothing was profiled and nothing was measured, so `verdict`, `acceptance_rate`, and `samples` carry no measurement. `mtp_enabled` still reports what the pin resolved to. |
| `unavailable` | No adaptive policy is running. `reason` says why. |

`forced` is deliberately not folded into `settled`. An operator pin is not a measured verdict, and a consumer that rendered it as "this machine measured MTP as worth it here" would be reporting something nobody measured. Render a pin as a pin.

### Unavailable reasons

| `reason` | Meaning |
|----------|---------|
| `no_mtp_dispatch` | The server has no MTP speculative dispatch: no drafter was supplied, or the drafter resolved to a different speculative kind. The B=1 MTP burst never runs, so `mtp_enabled` is `false`. |
| `adaptive_disabled` | `MLXCEL_MTP_ADAPTIVE` is off. The static per-hardware gate decides instead, and nothing is measured or persisted. `mtp_enabled` reports what that gate decided. |
| `worker_not_ready` | No batch worker has published a policy state yet: the model is still loading, or this server runs a worker variant with no MTP path. `mtp_enabled` is `null`. |

Distinguishing these is the point of the endpoint. Reading the hint files from another process could not: an empty directory meant "still profiling", "no MTP configured", and "the cache root resolved somewhere else" all at once, and the consumer degraded to a blank surface in every case.

## Versioning and compatibility

The body carries `schema_version`, an integer that starts at `1`.

Within one `schema_version`, mlxcel promises:

- Existing fields keep their names, their types, and their meanings.
- The `state`, `reason`, and `verdict` label sets only grow. A new state or a new unavailable reason can appear without a version bump.
- New fields may be added.

A consumer must therefore ignore unknown fields, and must treat an unrecognised `state`, `reason`, or `verdict` label as "no verdict I can render" rather than as an error. A consumer that hard-fails on an unknown label will break on a release that adds one.

`schema_version` is bumped for anything that breaks those promises: a removed or renamed field, a changed field type, a changed meaning, or a narrowed label set. A bump is a breaking change and gets a changelog entry. A consumer that does not recognise the `schema_version` it receives should treat the body as unreadable and show nothing, rather than guessing at the fields.

`schema_version` is independent of the persisted hint's `version` (`HINT_VERSION`). The hint version tracks the on-disk format and the verdict semantics behind it, and it is bumped whenever a stored verdict must be discarded and re-profiled. The two numbers are unrelated and will drift apart.

## Relationship to the persisted hint files

Settled verdicts are still written to `${MLXCEL_CACHE_DIR:-$HOME/.cache/mlxcel}/mtp-policy/<key-hash>.json`, and that behavior is unchanged. The files are how a verdict survives a restart; they are not an interface. `HINT_VERSION`, the subdirectory name, and the hint body are private and can change in a patch release without notice.

This endpoint exposes everything the hint file carries, plus the states a file cannot express (profiling, forced, unavailable) and the pairing's `hardware` label. Read it instead of the files.

## Example: is MTP running, and why

```bash
curl -s http://127.0.0.1:8080/v1/internal/mtp-policy | jq
```

While profiling:

```json
{
  "schema_version": 1,
  "state": "profiling",
  "reason": null,
  "verdict": null,
  "mtp_enabled": true,
  "target": "Gemma4-12B",
  "drafter": "Gemma4-12B-MTP",
  "hardware": "M5-16c",
  "block_size": 4,
  "acceptance_rate": 0.58,
  "samples": 2,
  "samples_required": 4,
  "samples_remaining": 2
}
```

which renders as "measuring: 2 of 4 single-request generations done".

On a server with no drafter:

```json
{
  "schema_version": 1,
  "state": "unavailable",
  "reason": "no_mtp_dispatch",
  "verdict": null,
  "mtp_enabled": false,
  "target": null,
  "drafter": null,
  "hardware": null,
  "block_size": null,
  "acceptance_rate": null,
  "samples": 0,
  "samples_required": 4,
  "samples_remaining": null
}
```

The same state is also included in the `/health` observability snapshot under `observability.mtp_policy`, in a similar but unversioned shape (it spells the state field `status` and omits `schema_version` and `samples_remaining`), for operators already polling that endpoint. `/health` is an operator surface with no stability promise; `/v1/internal/mtp-policy` is the one with the contract above.
