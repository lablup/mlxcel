# Paged context admission diagnostic

This is a pre-acceptance admission diagnostic for #2264, not the final performance acceptance run. It uses the original `m_client.py` long-copy workload with a 4096-token target and a 256-token continuation. Workload text, output length, and historical gain thresholds are unchanged.

The diagnostic began at 2026-10-11 10:15:55 UTC. Its binary was built from `89581b8dd6cbffd315833ee284d7bb9aacc9e307`; the measurement checkout was `25fbee3fbb2e50c2fda4fae6a5e647b663699e2b`. The intervening changes affected comments, tests, and measurement scripts, not production behavior. Binary SHA-256: `7a7b56aabf9d49ea08037acdda9a3cc72ffe3b2ec15d4ee06f2e8f3918d0b13e`. The canonical native MLX pin was `81ba1c6a0e50a9268b931579c2d4f1158b9aab5a`.

The server had the new emission governor but still used `PROMPT_LOOKUP_PAGED_CONTEXT_LIMIT = Some(4096)`. Both arms used paged storage. The running-process executable and arguments are preserved in `diagnostic-context-{off,on}.arm.json`; model and source provenance are in `diagnostic-context-environment.json`. The original output records and server logs are retained alongside this note.

| Request | Actual prompt tokens, both arms | Generated tokens, both arms | Off proposed / accepted / rounds | On proposed / accepted / rounds |
|---|---:|---:|---:|---:|
| 0 | 4099 | 256 | 0 / 0 / 0 | 0 / 0 / 0 |
| 1 | 4101 | 256 | 0 / 0 / 0 | 0 / 0 / 0 |
| 2 | 4102 | 256 | 0 / 0 / 0 | 0 / 0 / 0 |

All three flag-on requests exceed the 4096-total-token cutoff before decoding and perform no verification. The original report's 1.42x copy gain at the nominal 4096 target was measured before that cutoff was installed. Consequently, preserving the historical workload requires admitting its actual tokenized prompt, not changing the prompt to fit the cutoff.

The observed envelope is `4102 + 256 = 4358` total tokens. A 4352-token cap would still cut into the last request. The new cap is 4608, the smallest 512-token boundary covering the observed envelope, adding one such interval to the prior 4096 boundary. It retains a bounded guard and continues to exclude the 8192-token control. This is a narrow admission margin, not evidence that every workload through 4608 is profitable.

Boundary regression coverage checks actual scheduler admission for the observed prompts, the complete 4358-token envelope, the inclusive 4608 boundary, 4609 tokens reached through either prompt or continuation, and 8192 tokens. It checks that dense storage remains exempt. The fixture restores real token state before any further model execution.

The final performance matrix must use a fresh release binary incorporating the cap change and the integrated prerequisite work. Its provenance and raw results must be recorded separately. This diagnostic establishes the cause of exclusion; it does not establish the required chat, copy, concurrency, or long-copy performance gains.
