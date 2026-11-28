# PR #3489: Upgrade jiff to 1.x and honour the strftime_now format and zone (issue #3480)

**Date**: 2026-11-27

**Status**: Merged as `9d8949cd7`. 5 files, +137 / -44. Validated locally on an M3 Ultra; remote CI was cancelled by maintainer decision. The Windows build was not run.

**Risk Level**: Low (one chat-template helper changes output; every other timestamp is byte-identical)

## Executive Summary

#3462 replaced the unmaintained `chrono` with `jiff` 0.2 and kept the existing `strftime_now` behaviour, which ignored its argument and always returned the UTC date as `%d %b %Y`. This PR moves `jiff` to 1.0.3 and makes `strftime_now(format)` format the current time in the host time zone with the format the template passes. Templates that ask for `%Y-%m-%d` or a time of day now get it.

## 1. Problem Statement

- **Format argument ignored.** Chat templates call `strftime_now(fmt)` to put today's date in the system prompt. mlxcel registered a function that discarded the argument and returned `%d %b %Y`. That matches Llama 3.1+ (`strftime_now("%d %b %Y")`), which is why the gap went unnoticed, but a template asking for `%Y-%m-%d` or `%d %B %Y` got `27 Nov 2026` instead of `2026-11-27` or `27 November 2026`.
- **UTC instead of local time.** The Python reference (`datetime.now().strftime(fmt)` in transformers' Jinja environment) formats local time. A host in KST at 08:00 got the previous day's date.
- **jiff 0.2 pinned.** #3462 used 0.2.46, the version already in the cargo cache, although 1.0.3 is the current release line. The user asked to move to the latest so time-zone work would not need a second dependency bump.

## 2. Technical Decisions

### 2.1 jiff feature set

| Feature | Why |
|---|---|
| `std` | `Timestamp::try_from(SystemTime)` and `strtime::format` |
| `tz-system` | `TimeZone::system()` resolves the host zone (`TZ`, `/etc/localtime`, the Windows registry) |
| `tzdb-zoneinfo` | reads `/usr/share/zoneinfo` on Unix, so no database is compiled in |
| `tzdb-bundle-platform` | bundles a database only on Windows, which has no zoneinfo directory |

`default-features = false` keeps `serde`, `logging` and the always-bundled database out. The lockfile lists `jiff-tzdb` and `jiff-tzdb-platform` on every platform, but they are compiled only for Windows targets.

### 2.2 One formatting function with an explicit zone

`strftime_in(t: SystemTime, tz: &TimeZone, format: &str)` holds the logic and `strftime_now(format)` is a one-line wrapper over `SystemTime::now()` and `TimeZone::system()`. Tests pass a fixed instant and a fixed `+09:00` zone, so they do not depend on the host clock or `TZ`. `day_month_year` and `day_month_year_now` had no production callers and were removed; the old `%d %b %Y` assertion now runs through `strftime_in`.

### 2.3 Failure behaviour

| Case | Behaviour | Reason |
|---|---|---|
| Host zone cannot be determined | jiff returns UTC | Same as the old output, and the prompt still renders |
| `SystemTime` outside jiff's range | clamped to `Timestamp::MIN` / `MAX` | `try_from` cannot fail the render |
| Unsupported directive (`%J`) | the format string is returned unchanged, logged once at debug level | A template typo must not abort the request; `Once` keeps a hot path from flooding logs |

Python's `strftime` on glibc and macOS also leaves unknown directives in place, so the fallback errs in the same direction as the reference.

## 3. Change Summary

| File | Change |
|---|---|
| `Cargo.toml`, `Cargo.lock` | jiff 0.2.46 to 1.0.3, jiff-core 0.1.8 to 1.0.2, jiff-static 0.2.46 to 1.0.3, new jiff-tzdb 1.0.0 and jiff-tzdb-platform 1.0.0 (Windows only) |
| `src/server/clock.rs` | `strftime_in`, `strftime_now`; `day_month_year*` removed; module doc reflowed |
| `src/server/chat_template.rs` | `strftime_now` registered with its format argument |
| `src/server/clock_tests.rs` | fixed-instant formats, `+09:00` moves 23:30 UTC to the next day, unsupported directive |

| Commit | Message |
|---|---|
| `2d6beea79` | update(deps): upgrade jiff to 1.0.3 and honour strftime_now format and zone |
| `49afc8b65` | update(server): reflow the clock module doc after the strftime_now change |

## 4. Validation

| Check | Result |
|---|---|
| `cargo test ... server::clock` | 9 passed |
| `cargo test ... server::chat_template` | 167 passed, 4 ignored; new case renders `strftime_now("%Y-%m-%d")` through the template path |
| `cargo clippy -p mlxcel --lib --tests --features metal,accelerate -- -D warnings` | clean |
| `cargo deny check` | advisories, bans, licenses, sources ok, no new exemptions |
| `cargo tree -p mlxcel -e normal -i jiff` | `jiff v1.0.3` |

## 5. Follow-up

- The Windows zone path (`tzdb-bundle-platform` plus the registry lookup) was not built or run.
- The server's other timestamps (Unix values and RFC 3339 `Z` strings) stay in UTC on purpose; only the template helper follows the host zone.
- If a deployment needs a fixed zone regardless of the host, setting `TZ` on the server process is enough; no mlxcel option exists for it.
