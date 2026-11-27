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

//! Every timestamp shape in `clock.rs`, pinned at fixed instants. The
//! expected strings and numbers were produced by `chrono` 0.4.45 (the
//! implementation before #3266), and this file passed unchanged against both
//! that implementation and the `std::time` / `time` one that replaced it.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use jiff::tz::{Offset, TimeZone};

use super::{
    is_rfc3339, rfc3339, rfc3339_seconds_z, strftime_in, unix_micros, unix_millis,
    unix_nanos_or_zero, unix_seconds,
};

/// `secs` whole seconds from the epoch (negative is before it), plus `nanos`.
fn at(secs: i64, nanos: u32) -> SystemTime {
    let whole = if secs >= 0 {
        UNIX_EPOCH + Duration::from_secs(secs.unsigned_abs())
    } else {
        UNIX_EPOCH - Duration::from_secs(secs.unsigned_abs())
    };
    whole + Duration::from_nanos(u64::from(nanos))
}

/// 2026-11-26T16:07:00Z.
const LATE_NOV: i64 = 1_795_709_220;
/// 2026-03-05T09:04:03Z: single-digit day and month.
const MAR_05: i64 = 1_772_701_443;
/// 2028-02-29T23:59:59Z: leap day, last second.
const LEAP_DAY_END: i64 = 1_835_481_599;
/// 2300-01-01T00:00:00Z: past the `i64` nanosecond range.
const Y2300: i64 = 10_413_792_000;
/// 0999-12-31T23:59:59Z: three-digit year, before the `i64` nanosecond range.
const Y0999_END: i64 = -30_610_224_001;

#[test]
fn unix_values_round_toward_the_past() {
    let t = at(LATE_NOV, 999_999_999);
    assert_eq!(unix_seconds(t), 1_795_709_220);
    assert_eq!(unix_millis(t), 1_795_709_220_999);
    assert_eq!(unix_micros(t), 1_795_709_220_999_999);
    assert_eq!(unix_nanos_or_zero(t), 1_795_709_220_999_999_999);

    // 1.5 s before the epoch floors to -2 s, not -1 s.
    let t = at(-2, 500_000_000);
    assert_eq!(unix_seconds(t), -2);
    assert_eq!(unix_millis(t), -1_500);
    assert_eq!(unix_micros(t), -1_500_000);
    assert_eq!(unix_nanos_or_zero(t), -1_500_000_000);

    let t = at(0, 0);
    assert_eq!((unix_seconds(t), unix_millis(t), unix_micros(t)), (0, 0, 0));
    assert_eq!(unix_nanos_or_zero(t), 0);

    let t = at(-1, 999_999_999);
    assert_eq!(
        (unix_seconds(t), unix_millis(t), unix_micros(t)),
        (-1, -1, -1)
    );
    assert_eq!(unix_nanos_or_zero(t), -1);
}

#[test]
fn unix_nanos_outside_the_i64_range_is_zero() {
    assert_eq!(unix_seconds(at(Y2300, 0)), 10_413_792_000);
    assert_eq!(unix_millis(at(Y2300, 0)), 10_413_792_000_000);
    assert_eq!(unix_nanos_or_zero(at(Y2300, 0)), 0);
    assert_eq!(unix_seconds(at(Y0999_END, 0)), -30_610_224_001);
    assert_eq!(unix_nanos_or_zero(at(Y0999_END, 0)), 0);
}

#[test]
fn rfc3339_seconds_z_drops_the_fraction() {
    for (t, expected) in [
        (at(LATE_NOV, 0), "2026-11-26T16:07:00Z"),
        (at(LATE_NOV, 999_999_999), "2026-11-26T16:07:00Z"),
        (at(MAR_05, 123_456_789), "2026-03-05T09:04:03Z"),
        (at(LEAP_DAY_END, 999_999_999), "2028-02-29T23:59:59Z"),
        (at(LEAP_DAY_END + 1, 0), "2028-03-01T00:00:00Z"),
        (at(0, 0), "1970-01-01T00:00:00Z"),
        (at(-2, 500_000_000), "1969-12-31T23:59:58Z"),
        (at(Y2300, 1), "2300-01-01T00:00:00Z"),
        (at(Y0999_END, 0), "0999-12-31T23:59:59Z"),
    ] {
        assert_eq!(rfc3339_seconds_z(t), expected);
    }
}

#[test]
fn rfc3339_uses_plus_zero_offset_and_three_six_or_nine_fraction_digits() {
    for (nanos, expected) in [
        (0, "2026-11-26T16:07:00+00:00"),
        (1, "2026-11-26T16:07:00.000000001+00:00"),
        (1_000, "2026-11-26T16:07:00.000001+00:00"),
        (1_000_000, "2026-11-26T16:07:00.001+00:00"),
        (120_000_000, "2026-11-26T16:07:00.120+00:00"),
        (123_000_000, "2026-11-26T16:07:00.123+00:00"),
        (123_400_000, "2026-11-26T16:07:00.123400+00:00"),
        (123_456_000, "2026-11-26T16:07:00.123456+00:00"),
        (123_456_700, "2026-11-26T16:07:00.123456700+00:00"),
        (123_456_789, "2026-11-26T16:07:00.123456789+00:00"),
        (999_999_999, "2026-11-26T16:07:00.999999999+00:00"),
    ] {
        assert_eq!(rfc3339(at(LATE_NOV, nanos)), expected, "nanos {nanos}");
    }
    for (t, expected) in [
        (at(MAR_05, 0), "2026-03-05T09:04:03+00:00"),
        (at(0, 0), "1970-01-01T00:00:00+00:00"),
        (at(-2, 500_000_000), "1969-12-31T23:59:58.500+00:00"),
        (at(LEAP_DAY_END, 5_000), "2028-02-29T23:59:59.000005+00:00"),
        (at(Y2300, 0), "2300-01-01T00:00:00+00:00"),
        (at(Y0999_END, 0), "0999-12-31T23:59:59+00:00"),
    ] {
        assert_eq!(rfc3339(t), expected);
    }
}

#[test]
fn strftime_day_month_year_pads_the_day_and_abbreviates_the_month() {
    for (t, expected) in [
        (at(LATE_NOV, 0), "26 Nov 2026"),
        (at(MAR_05, 0), "05 Mar 2026"),
        (at(LEAP_DAY_END, 999_999_999), "29 Feb 2028"),
        (at(LEAP_DAY_END + 1, 0), "01 Mar 2028"),
        (at(0, 0), "01 Jan 1970"),
        (at(-2, 500_000_000), "31 Dec 1969"),
        (at(Y2300, 0), "01 Jan 2300"),
        (at(Y0999_END, 0), "31 Dec 0999"),
    ] {
        assert_eq!(strftime_in(t, &TimeZone::UTC, "%d %b %Y"), expected);
    }
    // The 15th of every month of 2026, 00:00:00Z.
    let months: Vec<String> = [
        1_768_435_200,
        1_771_113_600,
        1_773_532_800,
        1_776_211_200,
        1_778_803_200,
        1_781_481_600,
        1_784_073_600,
        1_786_752_000,
        1_789_430_400,
        1_792_022_400,
        1_794_700_800,
        1_797_292_800,
    ]
    .into_iter()
    .map(|secs| strftime_in(at(secs, 0), &TimeZone::UTC, "%d %b %Y"))
    .collect();
    assert_eq!(
        months.join(","),
        "15 Jan 2026,15 Feb 2026,15 Mar 2026,15 Apr 2026,15 May 2026,15 Jun 2026,\
         15 Jul 2026,15 Aug 2026,15 Sep 2026,15 Oct 2026,15 Nov 2026,15 Dec 2026"
    );
}

#[test]
fn strftime_in_applies_the_requested_format() {
    let t = at(MAR_05, 0);
    assert_eq!(strftime_in(t, &TimeZone::UTC, "%d %b %Y"), "05 Mar 2026");
    assert_eq!(strftime_in(t, &TimeZone::UTC, "%Y-%m-%d"), "2026-03-05");
    assert_eq!(
        strftime_in(t, &TimeZone::UTC, "%B %d, %Y"),
        "March 05, 2026"
    );
}

#[test]
fn strftime_in_applies_the_zone() {
    // 2026-03-05T23:30:00Z is already 6 March at +09:00.
    let late = at(1_772_753_400, 0);
    assert_eq!(
        strftime_in(late, &TimeZone::UTC, "%Y-%m-%d %H:%M"),
        "2026-03-05 23:30"
    );
    let tokyo = TimeZone::fixed(Offset::from_seconds(9 * 3600).unwrap());
    assert_eq!(
        strftime_in(late, &tokyo, "%Y-%m-%d %H:%M"),
        "2026-03-06 08:30"
    );
}

#[test]
fn strftime_in_returns_an_unsupported_format_unchanged() {
    let t = at(MAR_05, 0);
    assert_eq!(strftime_in(t, &TimeZone::UTC, "%J"), "%J");
    assert_eq!(strftime_in(t, &TimeZone::UTC, "100%"), "100%");
}

#[test]
fn emitted_stamps_parse_as_rfc3339() {
    for t in [
        at(LATE_NOV, 0),
        at(LATE_NOV, 123_456_789),
        at(MAR_05, 120_000_000),
        at(-2, 500_000_000),
    ] {
        assert!(is_rfc3339(&rfc3339_seconds_z(t)));
        assert!(is_rfc3339(&rfc3339(t)));
    }
    for s in ["2026-11-26T16:07:00.5-05:30", "2026-11-26T16:07:00+09:00"] {
        assert!(is_rfc3339(s), "{s}");
    }
    for s in [
        "",
        "2026-11-26",
        "2026-11-26T16:07:00",
        "26 Nov 2026",
        "1795709220",
        "2026-13-01T00:00:00Z",
        "2026-02-30T00:00:00Z",
        "2026-11-26T24:00:00Z",
        // ISO 8601 / Temporal forms `jiff` parses that RFC 3339 does not
        // allow.
        "2026-11-26T16:07Z",
        "20261126T160700Z",
        "2026-11-26T16:07:00Z[UTC]",
        "2026-11-26T16:07:00,5Z",
        "2026-11-26T16:07:00.Z",
        "2026-11-26T16:07:00+0900",
    ] {
        assert!(!is_rfc3339(s), "{s}");
    }
}
