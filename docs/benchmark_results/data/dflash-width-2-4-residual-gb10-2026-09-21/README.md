# Issue #1935 residual at widths 2 and 4: served arms and round transcripts (GB10, 2026-09-21)

Raw data for the residual PR #1939 left open: on `models/mlx/qwen3.5-4b-4bit` with `models/mlx/qwen3.5-4b-dflash`, greedy served output at verify widths 2 and 4 parts from classic decode at generated token 105.

`identity.txt` is the host, binary and checkpoint identity for every arm here. One binary, one session, three arms, two requests each.

## What is in here

`arms/arm.<tag>.json` is one served arm: the request bodies' completion text, the per-token logprob tokens and values, and the usage block, for both requests against the same server process. `arms/server.<tag>.log` is that arm's server log at `RUST_LOG=info,mlxcel_core::drafter::dflash::round_loop=debug`, which carries the per-round `DFlash round transcript` line the round loop emits.

`prompt_ids.json` and `classic_ids.json` are the 158 prompt ids and the 200 classic generated ids, from `MLXCEL_PRINT_TOKEN_IDS=1 mlxcel generate -m models/mlx/qwen3.5-4b-4bit --no-chat-template --temp 0 --max-tokens 200`. They are what the in-process probe arms take as `MLXCEL_Q35_PROBE_PROMPT` and `MLXCEL_Q35_PROBE_REFERENCE`. Note the prompt file ends in a newline and shell command substitution strips it: passing the prompt without that newline tokenizes to 157 ids, not 158, and produces a different first token, so the CLI reference then matches nothing.

`arms/w4_ids.json` and `arms/w2_ids.json` are the emitted id streams reconstructed from the round transcripts by `harness/algebra.py`.

## Harness

`harness/transcript.py` runs one arm: one server, `--ignore-eos --max-batch-size 1`, n non-streaming `/v1/completions` at temperature 0 with logprobs, against the same process so a cross-request difference would show. Identity only, so its host gate is the foreign-model check alone.

`harness/algebra.py` checks a run's round algebra offline against the transcript lines, with no GPU: that each round's bonus is the previous round's last emitted token, that `accepted` is the longest common prefix of the drafter's proposals and the target's block argmax, and that the emitted tokens are `draft[:accepted] + [target[accepted]]`. Rounds are logged 0-based, which is where a burst boundary is.

`harness/compare.py` reads the arms against the classic null arm: the first differing token index, and whether the logprobs agree before it.
