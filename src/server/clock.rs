// Copyright 2025-2026 Lablup Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Wall-clock timestamps in the exact shapes the server emits.
//!
//! The server stamps the current time into OpenAI-compatible `created` and
//! `created_at` fields (Unix seconds), `req_` request ids (Unix
//! microseconds), `srv_` instance ids and per-model `last_used` stamps (Unix
//! milliseconds), router-cache staging and quarantine directory names (Unix
//! nanoseconds), RFC 3339 strings in WebUI events, lifecycle operations and
//! runtime snapshots, and the date the `strftime_now` chat-template function
//! returns. Until #3266 each call site asked `chrono` directly; `chrono` is
//! declared unmaintained (RUSTSEC-2026-0543), so the formats live here
//! instead and match what `chrono` produced byte for byte. `clock_tests.rs`
//! pins every one of them at fixed instants, and the same file passed against
//! the `chrono` implementation before the switch.
//!
//! Unix values come from `std::time` alone. Calendar fields come from
//! `time::OffsetDateTime`, and the strings are assembled here rather than
//! through `time`'s format descriptions so that the `chrono` shapes (`+00:00`
//! and a 0/3/6/9-digit fraction from `to_rfc3339`, `Z` and no fraction from
//! `to_rfc3339_opts(Secs, true)`) are spelled out where they can be read.
//! Like `time` without its `large-dates` feature, the calendar formatters
//! cover years -9999 through 9999 and match `chrono` for years 0 through
//! 9999; a system clock does not report anything outside that.

use std::time::{SystemTime, UNIX_EPOCH};

use time::OffsetDateTime;

const NANOS_PER_SECOND: i128 = 1_000_000_000;

/// `%b` in `chrono` (and C `strftime` in the C locale).
const MONTH_ABBREVIATIONS: [&str; 12] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

/// Signed nanoseconds since the Unix epoch. Instants before the epoch are
/// negative, so the floor divisions below round toward the past the way
/// `chrono`'s `timestamp*` accessors do.
fn unix_nanos_wide(t: SystemTime) -> i128 {
    match t.duration_since(UNIX_EPOCH) {
        Ok(after) => after.as_nanos() as i128,
        Err(before) => -(before.duration().as_nanos() as i128),
    }
}

/// Unix seconds, rounded toward the past (`chrono` `timestamp()`).
pub(crate) fn unix_seconds(t: SystemTime) -> i64 {
    unix_nanos_wide(t).div_euclid(NANOS_PER_SECOND) as i64
}

/// Unix milliseconds, rounded toward the past (`chrono` `timestamp_millis()`).
pub(crate) fn unix_millis(t: SystemTime) -> i64 {
    unix_nanos_wide(t).div_euclid(1_000_000) as i64
}

/// Unix microseconds, rounded toward the past (`chrono` `timestamp_micros()`).
pub(crate) fn unix_micros(t: SystemTime) -> i64 {
    unix_nanos_wide(t).div_euclid(1_000) as i64
}

/// Unix nanoseconds, or 0 when the value does not fit an `i64` (before
/// 1677-09-21 or after 2262-04-11), as `chrono`
/// `timestamp_nanos_opt().unwrap_or_default()` returned.
pub(crate) fn unix_nanos_or_zero(t: SystemTime) -> i64 {
    i64::try_from(unix_nanos_wide(t)).unwrap_or_default()
}

/// `YYYY-MM-DDTHH:MM:SS` in UTC, the part both RFC 3339 shapes share.
fn utc_date_time(dt: OffsetDateTime) -> String {
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}",
        dt.year(),
        u8::from(dt.month()),
        dt.day(),
        dt.hour(),
        dt.minute(),
        dt.second()
    )
}

/// RFC 3339 in UTC with whole seconds and a `Z` suffix, sub-second part
/// dropped: `2026-11-26T16:07:00Z` (`chrono`
/// `to_rfc3339_opts(SecondsFormat::Secs, true)`).
pub(crate) fn rfc3339_seconds_z(t: SystemTime) -> String {
    format!("{}Z", utc_date_time(OffsetDateTime::from(t)))
}

/// RFC 3339 in UTC with a `+00:00` offset and the shortest of no, 3, 6 or 9
/// fraction digits that holds the sub-second part exactly:
/// `2026-11-26T16:07:00.123456+00:00` (`chrono` `to_rfc3339()`).
pub(crate) fn rfc3339(t: SystemTime) -> String {
    let dt = OffsetDateTime::from(t);
    let nanos = dt.nanosecond();
    let fraction = if nanos == 0 {
        String::new()
    } else if nanos % 1_000_000 == 0 {
        format!(".{:03}", nanos / 1_000_000)
    } else if nanos % 1_000 == 0 {
        format!(".{:06}", nanos / 1_000)
    } else {
        format!(".{nanos:09}")
    };
    format!("{}{fraction}+00:00", utc_date_time(dt))
}

/// Zero-padded day, English month abbreviation and four-digit year in UTC:
/// `05 Mar 2026` (`chrono` `format("%d %b %Y")`).
pub(crate) fn day_month_year(t: SystemTime) -> String {
    let dt = OffsetDateTime::from(t);
    format!(
        "{:02} {} {:04}",
        dt.day(),
        MONTH_ABBREVIATIONS[usize::from(u8::from(dt.month())) - 1],
        dt.year()
    )
}

/// [`unix_seconds`] of the current time.
///
/// Used by: the OpenAI-compatible `created` / `created_at` / `completed_at`
/// stamps in `types/response.rs`, `types/stream.rs`, `model_provider.rs`,
/// `router_server.rs`, `routes/models.rs`, `routes/responses.rs`,
/// `routes/responses_compact.rs`, `routes/responses_mcp_loop.rs` and
/// `routes/responses_run.rs`, and `started_at_unix` in `state.rs`.
/// Regenerate the exact list with
/// `grep -rln '\bunix_seconds_now(' src --include='*.rs'`.
pub(crate) fn unix_seconds_now() -> i64 {
    unix_seconds(SystemTime::now())
}

/// [`unix_millis`] of the current time.
///
/// Used by: the `srv_{pid}_{millis}` server instance id in
/// `router_lifecycle_ops.rs` and the per-model `last_used` stamps in
/// `router_models.rs`.
pub(crate) fn unix_millis_now() -> i64 {
    unix_millis(SystemTime::now())
}

/// [`unix_micros`] of the current time.
///
/// Used by: the `req_{micros}` request ids built in `app.rs`,
/// `router_server.rs`, `webui/events.rs`, `webui/library.rs` and
/// `webui/security.rs`.
pub(crate) fn unix_micros_now() -> i64 {
    unix_micros(SystemTime::now())
}

/// [`unix_nanos_or_zero`] of the current time.
///
/// Used by: the staging and quarantine directory names in
/// `router_cache/anchored_publish.rs` and `router_cache/anchored_remove.rs`.
pub(crate) fn unix_nanos_now_or_zero() -> i64 {
    unix_nanos_or_zero(SystemTime::now())
}

/// [`rfc3339_seconds_z`] of the current time.
///
/// Used by: `UiEvent::emitted_at` in `router_lifecycle_dto.rs` and the
/// operation `created_at` / `updated_at` stamps in `router_lifecycle_ops.rs`.
pub(crate) fn rfc3339_seconds_z_now() -> String {
    rfc3339_seconds_z(SystemTime::now())
}

/// [`rfc3339`] of the current time.
///
/// Used by: the runtime snapshot `sampled_at` / `measured_at` in
/// `webui/runtime.rs`.
pub(crate) fn rfc3339_now() -> String {
    rfc3339(SystemTime::now())
}

/// [`day_month_year`] of the current time.
///
/// Used by: the `strftime_now` chat-template function in `chat_template.rs`.
pub(crate) fn day_month_year_now() -> String {
    day_month_year(SystemTime::now())
}

/// Whether `s` parses as an RFC 3339 date-time. Test code checks emitted
/// stamps with it.
///
/// Used by: `app.rs` and `router_contract_test_support.rs` (test code only).
#[cfg(test)]
pub(crate) fn is_rfc3339(s: &str) -> bool {
    OffsetDateTime::parse(s, &time::format_description::well_known::Rfc3339).is_ok()
}

#[cfg(test)]
#[path = "clock_tests.rs"]
mod tests;
