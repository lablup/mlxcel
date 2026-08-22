// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
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

//! CPython `json.dumps` compatible `tojson` filter for chat templates.
//!
//! Chat templates published on HuggingFace are written against the `transformers`
//! renderer, which is Jinja2 running on CPython. There, `tojson` is
//! `json.dumps` with `ensure_ascii=False` prefilled, so template authors call it
//! with `json.dumps` keyword arguments and expect `json.dumps` output. minijinja's
//! builtin `tojson` does neither:
//!
//! * it rejects every keyword argument except `indent` (`Kwargs::assert_all_used`),
//!   so `tojson(ensure_ascii=False)` and `tojson(sort_keys=true, separators=(",", ":"))`
//!   abort the render and the request degrades to the generic `User:/Assistant:`
//!   fallback prompt, and
//! * it emits `serde_json`'s compact form (`{"a":1,"b":2}`), while a bare
//!   `x | tojson` in CPython emits `{"a": 1, "b": 2}`. That difference is silent:
//!   the prompt still renders, but it tokenizes differently from the prompt the
//!   model provider trained and evaluated against.
//!
//! [`python_tojson`] is registered on the chat-template environment (see
//! `chat_template::configure_environment`), where it shadows the builtin.
//!
//! # Supported arguments
//!
//! | argument | default | notes |
//! |----------|---------|-------|
//! | `ensure_ascii` | `false` | `transformers`' default, not CPython's own `True`. |
//! | `indent` | `None` | positional or keyword; `true` means 2, for minijinja compatibility. |
//! | `separators` | `None` | a two element tuple `(item_separator, key_separator)`. |
//! | `sort_keys` | `false` | sorts object keys recursively by Unicode code point. |
//!
//! With `separators` absent, CPython uses `(", ", ": ")` when `indent` is `None`
//! and `(",", ": ")` when it is set (the trailing space after the comma is
//! redundant once every item is on its own line).
//!
//! # Deliberate differences from minijinja's builtin
//!
//! The builtin additionally escapes `<`, `>`, `&` and `'` as `\uXXXX` so its
//! output is safe to inline into an HTML document. `transformers` does not do
//! that, and a chat prompt is not an HTML document, so applying it here would
//! corrupt every tool schema that contains a comparison operator. This filter
//! does not escape them.
//!
//! The `default=` argument of `json.dumps` is not implemented; no published chat
//! template passes it, and it would require calling back into the template engine.

use std::fmt::Write as _;

use minijinja::value::{Kwargs, Value, ValueKind};
use minijinja::{Error, ErrorKind};

/// Maximum container nesting this filter will serialize.
///
/// A chat template is operator-controlled but the values it serializes are not:
/// `tools` comes straight from the client request body. Recursion here is
/// unbounded otherwise, and minijinja's fuel budget does not help because fuel
/// counts VM instructions, not native stack frames inside one filter call. The
/// limit matches `serde_json`'s own default recursion limit.
const MAX_DEPTH: usize = 128;

/// `json.dumps` options resolved from the filter's arguments.
#[derive(Debug, Clone, PartialEq, Eq)]
struct DumpOptions {
    ensure_ascii: bool,
    indent: Option<usize>,
    item_separator: String,
    key_separator: String,
    sort_keys: bool,
}

impl DumpOptions {
    /// Resolve the options from the positional `indent` argument and the kwargs.
    ///
    /// Named arguments are not bound to declared parameters by minijinja, so
    /// `indent` has to be looked up in the kwargs explicitly when it was not
    /// passed positionally. The builtin filter does the same.
    fn resolve(positional_indent: Option<Value>, kwargs: &Kwargs) -> Result<Self, Error> {
        let ensure_ascii = kwargs.get::<Option<bool>>("ensure_ascii")?.unwrap_or(false);
        let sort_keys = kwargs.get::<Option<bool>>("sort_keys")?.unwrap_or(false);

        let indent_arg = match positional_indent {
            Some(value) => Some(value),
            None => kwargs.get::<Option<Value>>("indent")?,
        };
        let indent = match indent_arg {
            None => None,
            // `tojson(true)` means "pretty print" in minijinja. Keep accepting it
            // so a template written against minijinja does not start failing.
            Some(value) => match bool::try_from(value.clone()) {
                Ok(true) => Some(2),
                Ok(false) => None,
                Err(_) => Some(usize::try_from(value)?),
            },
        };

        let (item_separator, key_separator) = match kwargs.get::<Option<Value>>("separators")? {
            Some(value) => read_separators(&value)?,
            None if indent.is_some() => (",".to_string(), ": ".to_string()),
            None => (", ".to_string(), ": ".to_string()),
        };

        Ok(Self {
            ensure_ascii,
            indent,
            item_separator,
            key_separator,
            sort_keys,
        })
    }
}

/// Read a `separators=(item, key)` argument.
fn read_separators(value: &Value) -> Result<(String, String), Error> {
    let invalid = || {
        Error::new(
            ErrorKind::InvalidOperation,
            "tojson: `separators` must be a two element tuple of strings, e.g. (\",\", \": \")",
        )
    };
    if value.kind() != ValueKind::Seq {
        return Err(invalid());
    }
    let parts: Vec<Value> = value.try_iter()?.collect();
    let [item, key] = parts.as_slice() else {
        return Err(invalid());
    };
    let item = item.as_str().ok_or_else(invalid)?.to_string();
    let key = key.as_str().ok_or_else(invalid)?.to_string();
    Ok((item, key))
}

/// `tojson` filter with CPython `json.dumps` semantics.
///
/// Registered by `chat_template::configure_environment`, which is the single
/// funnel every template-facing callable in mlxcel goes through.
// Used by: server::chat_template::configure_environment
pub fn python_tojson(value: Value, indent: Option<Value>, kwargs: Kwargs) -> Result<Value, Error> {
    let options = DumpOptions::resolve(indent, &kwargs)?;
    // Still reject unknown keyword arguments: a template asking for something we
    // do not implement (`default=`, `cls=`, a typo) must fail loudly rather than
    // silently render output the template author did not ask for.
    kwargs.assert_all_used()?;

    let mut out = String::new();
    write_value(&mut out, &value, &options, 0)?;
    // Chat templates render with autoescape off, but mark the result safe anyway
    // so the filter behaves identically if that ever changes.
    Ok(Value::from_safe_string(out))
}

/// Serialize one value at nesting depth `level`.
fn write_value(
    out: &mut String,
    value: &Value,
    options: &DumpOptions,
    level: usize,
) -> Result<(), Error> {
    if level > MAX_DEPTH {
        return Err(Error::new(
            ErrorKind::InvalidOperation,
            format!("tojson: value nests deeper than {MAX_DEPTH} levels"),
        ));
    }

    match value.kind() {
        // minijinja serializes both as the unit value, and `json.dumps(None)` is
        // `null`. An undefined lookup in a template therefore renders as `null`
        // rather than aborting, matching the builtin filter.
        ValueKind::Undefined | ValueKind::None => out.push_str("null"),
        ValueKind::Bool => out.push_str(if value.is_true() { "true" } else { "false" }),
        ValueKind::Number => write_number(out, value)?,
        ValueKind::String => {
            write_json_string(
                out,
                value.as_str().unwrap_or_default(),
                options.ensure_ascii,
            );
        }
        // A plain object has no sequence or mapping behavior; minijinja's own
        // `Serialize` impl writes its string form, so do the same.
        ValueKind::Plain => write_json_string(out, &value.to_string(), options.ensure_ascii),
        ValueKind::Bytes => {
            let bytes: Vec<Value> = value
                .as_bytes()
                .unwrap_or_default()
                .iter()
                .map(|byte| Value::from(*byte))
                .collect();
            write_array(out, &bytes, options, level)?;
        }
        ValueKind::Seq | ValueKind::Iterable => {
            let items: Vec<Value> = value.try_iter()?.collect();
            write_array(out, &items, options, level)?;
        }
        ValueKind::Map => write_object(out, value, options, level)?,
        other => {
            return Err(Error::new(
                ErrorKind::InvalidOperation,
                format!("tojson: cannot serialize a {other} value"),
            ));
        }
    }
    Ok(())
}

/// Emit the newline plus indentation CPython writes before a container item.
///
/// A no-op when `indent` is `None`, which is what makes the same emitter serve
/// both the compact and the pretty-printed form.
fn write_container_break(out: &mut String, options: &DumpOptions, level: usize) {
    if let Some(width) = options.indent {
        out.push('\n');
        for _ in 0..width * level {
            out.push(' ');
        }
    }
}

fn write_array(
    out: &mut String,
    items: &[Value],
    options: &DumpOptions,
    level: usize,
) -> Result<(), Error> {
    // CPython writes `[]` with no interior newline for an empty list.
    if items.is_empty() {
        out.push_str("[]");
        return Ok(());
    }
    out.push('[');
    for (index, item) in items.iter().enumerate() {
        if index > 0 {
            out.push_str(&options.item_separator);
        }
        write_container_break(out, options, level + 1);
        write_value(out, item, options, level + 1)?;
    }
    write_container_break(out, options, level);
    out.push(']');
    Ok(())
}

fn write_object(
    out: &mut String,
    value: &Value,
    options: &DumpOptions,
    level: usize,
) -> Result<(), Error> {
    // `try_iter` over a map yields its keys in insertion order (minijinja is
    // built with `preserve_order`), which is the order the client sent them in.
    let mut entries: Vec<(String, Value)> = Vec::new();
    for key in value.try_iter()? {
        let item = value.get_item(&key)?;
        entries.push((object_key(&key)?, item));
    }
    if options.sort_keys {
        // Rust compares `str` by UTF-8 bytes, which is code-point order for
        // valid UTF-8 and therefore the same order Python's `sorted` produces.
        // The sort is stable, so duplicate keys keep their relative order.
        entries.sort_by(|left, right| left.0.cmp(&right.0));
    }

    if entries.is_empty() {
        out.push_str("{}");
        return Ok(());
    }
    out.push('{');
    for (index, (key, item)) in entries.iter().enumerate() {
        if index > 0 {
            out.push_str(&options.item_separator);
        }
        write_container_break(out, options, level + 1);
        write_json_string(out, key, options.ensure_ascii);
        out.push_str(&options.key_separator);
        write_value(out, item, options, level + 1)?;
    }
    write_container_break(out, options, level);
    out.push('}');
    Ok(())
}

/// Render a map key as the string CPython would use for it.
///
/// JSON object keys are strings, so `json.dumps` coerces `int`, `float`, `bool`
/// and `None` keys rather than failing (`{1: 2}` becomes `{"1": 2}`).
fn object_key(key: &Value) -> Result<String, Error> {
    Ok(match key.kind() {
        ValueKind::String => key.as_str().unwrap_or_default().to_string(),
        ValueKind::Bool => if key.is_true() { "true" } else { "false" }.to_string(),
        ValueKind::Undefined | ValueKind::None => "null".to_string(),
        ValueKind::Number => {
            let mut rendered = String::new();
            write_number(&mut rendered, key)?;
            rendered
        }
        _ => key.to_string(),
    })
}

/// Write a number the way CPython's JSON encoder writes it.
///
/// The integer / float split matters: Python prints `2` for an `int` and `2.0`
/// for a `float`, and minijinja keeps the two apart (`Value::is_integer`).
fn write_number(out: &mut String, value: &Value) -> Result<(), Error> {
    if value.is_integer() {
        let _ = write!(out, "{value}");
        return Ok(());
    }
    out.push_str(&format_python_float(f64::try_from(value.clone())?));
    Ok(())
}

/// Format an `f64` exactly as CPython's `repr` does.
///
/// This is the one place where "close enough" is not good enough: the output
/// feeds a tokenizer, so a single differing character shifts token ids. Rust's
/// `{}` for `f64` differs from CPython on three separate points, all of which
/// this function corrects:
///
/// 1. Rust prints `1` for `1.0_f64`; CPython prints `1.0`. Every float that
///    happens to be integral is affected.
/// 2. Rust never switches to exponent form; CPython does when the decimal
///    exponent is `<= -4` or `> 16`. So `0.00001` is `1e-05` in Python but
///    `0.00001` in Rust, and `1e16` is `1e+16` in Python but
///    `10000000000000000` in Rust. The `> 16` cutoff (rather than `> 17`) is
///    CPython's, chosen so a 16-digit shortest repr is never padded with digits
///    that are not really there.
/// 3. Rust's `{:e}` writes `1e-5`; CPython always writes a sign and at least two
///    exponent digits, so `1e-05`.
///
/// The shortest round-trip digits themselves come from Rust's `LowerExp`, which
/// like CPython's `_Py_dg_dtoa` in mode 0 produces the shortest decimal that
/// reads back as the same `f64`.
///
/// Non-finite values render bare as `NaN` / `Infinity` / `-Infinity`. That is
/// not valid JSON, but it is what `json.dumps` emits by default and therefore
/// what a template author sees from the reference renderer.
fn format_python_float(value: f64) -> String {
    if value.is_nan() {
        return "NaN".to_string();
    }
    if value.is_infinite() {
        return if value.is_sign_negative() {
            "-Infinity".to_string()
        } else {
            "Infinity".to_string()
        };
    }

    // `{:e}` is always `[-]d[.ddd]e[-]dd`, so both splits below are total.
    let scientific = format!("{value:e}");
    let (sign, unsigned) = match scientific.strip_prefix('-') {
        Some(rest) => ("-", rest),
        None => ("", scientific.as_str()),
    };
    let Some((mantissa, exponent)) = unsigned.split_once('e') else {
        // Unreachable for a finite f64; fall back rather than panic.
        return scientific;
    };
    let Ok(exponent) = exponent.parse::<i32>() else {
        return scientific;
    };
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();

    // `decpt` is CPython's: the value equals `0.<digits> * 10^decpt`.
    let decpt = exponent + 1;

    let mut out = String::with_capacity(digits.len() + 8);
    out.push_str(sign);

    if decpt <= -4 || decpt > 16 {
        out.push_str(&digits[..1]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        let exponent = decpt - 1;
        out.push('e');
        out.push(if exponent < 0 { '-' } else { '+' });
        let magnitude = exponent.unsigned_abs();
        if magnitude < 10 {
            out.push('0');
        }
        let _ = write!(out, "{magnitude}");
    } else if decpt <= 0 {
        out.push_str("0.");
        for _ in 0..-decpt {
            out.push('0');
        }
        out.push_str(&digits);
    } else if decpt as usize >= digits.len() {
        out.push_str(&digits);
        for _ in 0..decpt as usize - digits.len() {
            out.push('0');
        }
        // CPython's `Py_DTSF_ADD_DOT_0`: an integral float still reads as a float.
        out.push_str(".0");
    } else {
        let split = decpt as usize;
        out.push_str(&digits[..split]);
        out.push('.');
        out.push_str(&digits[split..]);
    }
    out
}

/// Write a JSON string literal using CPython's escape table.
///
/// `ensure_ascii=False` escapes `"`, `\` and the C0 controls (with the short
/// forms `\b \f \n \r \t` where CPython has them) and nothing else; note that
/// U+007F is deliberately left alone, as CPython leaves it alone. With
/// `ensure_ascii=True` everything outside `\x20..=\x7e` is escaped as `\uXXXX`
/// with lowercase hex, and anything above the BMP as a UTF-16 surrogate pair.
fn write_json_string(out: &mut String, text: &str, ensure_ascii: bool) {
    out.push('"');
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            _ if (ch as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", ch as u32);
            }
            _ if ensure_ascii && (ch as u32) > 0x7e => {
                let code = ch as u32;
                if code < 0x1_0000 {
                    let _ = write!(out, "\\u{code:04x}");
                } else {
                    let offset = code - 0x1_0000;
                    let high = 0xd800 | (offset >> 10);
                    let low = 0xdc00 | (offset & 0x3ff);
                    let _ = write!(out, "\\u{high:04x}\\u{low:04x}");
                }
            }
            _ => out.push(ch),
        }
    }
    out.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Render `value` through the filter with the given kwargs.
    fn dump(value: Value, kwargs: &[(&str, Value)]) -> String {
        let kwargs: Kwargs = kwargs.iter().map(|(k, v)| (*k, v.clone())).collect();
        python_tojson(value, None, kwargs)
            .expect("filter must succeed")
            .to_string()
    }

    fn float_str(value: f64) -> String {
        format_python_float(value)
    }

    #[test]
    fn python_float_repr_matches_cpython_for_the_threshold_cases() {
        // Every expected value here is `repr(x)` in CPython 3. The three columns
        // Rust's `{}` gets wrong are the `.0` suffix, the exponent thresholds,
        // and the two-digit signed exponent.
        assert_eq!(float_str(1.0), "1.0");
        assert_eq!(float_str(2.5), "2.5");
        assert_eq!(float_str(-0.0), "-0.0");
        assert_eq!(float_str(0.0), "0.0");
        assert_eq!(float_str(-2.5), "-2.5");

        // Lower exponent threshold: switch at decpt <= -4, i.e. below 1e-4.
        assert_eq!(float_str(0.0001), "0.0001");
        assert_eq!(float_str(0.00001), "1e-05");
        assert_eq!(float_str(0.000123), "0.000123");
        assert_eq!(float_str(1.5e-5), "1.5e-05");
        assert_eq!(float_str(1e-323), "1e-323");

        // Upper exponent threshold: switch at decpt > 16, i.e. above 1e15.
        assert_eq!(float_str(1e15), "1000000000000000.0");
        assert_eq!(float_str(1e16), "1e+16");
        assert_eq!(float_str(1.0000000000000002e16), "1.0000000000000002e+16");
        assert_eq!(float_str(1e20), "1e+20");
        assert_eq!(float_str(1.7976931348623157e308), "1.7976931348623157e+308");

        // Shortest round-trip digits, not a fixed precision.
        assert_eq!(float_str(0.1 + 0.2), "0.30000000000000004");
        assert_eq!(float_str(1.0 / 3.0), "0.3333333333333333");

        // `json.dumps` emits these bare, which is not valid JSON but is what the
        // reference renderer produces.
        assert_eq!(float_str(f64::NAN), "NaN");
        assert_eq!(float_str(f64::INFINITY), "Infinity");
        assert_eq!(float_str(f64::NEG_INFINITY), "-Infinity");
    }

    #[test]
    fn every_finite_float_round_trips_through_the_repr() {
        // The formatter reassembles shortest round-trip digits by hand, so the
        // property that actually matters is that the text still parses back to
        // the same bits.
        for value in [
            0.0f64,
            -0.0,
            1.0,
            -1.0,
            2.5,
            1e-5,
            1e-4,
            1e15,
            1e16,
            1e300,
            1e-300,
            f64::MIN_POSITIVE,
            f64::MAX,
            f64::MIN,
            std::f64::consts::PI,
            0.1 + 0.2,
        ] {
            let text = format_python_float(value);
            let parsed: f64 = text.parse().expect("repr must parse back");
            assert_eq!(
                parsed.to_bits(),
                value.to_bits(),
                "{text} did not round-trip for {value:?}"
            );
        }
    }

    #[test]
    fn integers_and_floats_are_kept_apart() {
        assert_eq!(dump(Value::from(2), &[]), "2");
        assert_eq!(dump(Value::from(2.0), &[]), "2.0");
        assert_eq!(dump(Value::from(-7i64), &[]), "-7");
        assert_eq!(dump(Value::from(u64::MAX), &[]), "18446744073709551615");
    }

    #[test]
    fn default_separators_have_the_python_spaces() {
        let value = Value::from_serialize(serde_json::json!({"b": [1, 2], "a": "x"}));
        assert_eq!(dump(value, &[]), r#"{"b": [1, 2], "a": "x"}"#);
    }

    #[test]
    fn compact_separators_and_sort_keys_apply_recursively() {
        let value = Value::from_serialize(serde_json::json!({
            "b": {"z": 1, "y": 2},
            "a": [{"n": 1, "m": 2}]
        }));
        let rendered = dump(
            value,
            &[
                ("sort_keys", Value::from(true)),
                (
                    "separators",
                    Value::from(vec![Value::from(","), Value::from(":")]),
                ),
            ],
        );
        assert_eq!(rendered, r#"{"a":[{"m":2,"n":1}],"b":{"y":2,"z":1}}"#);
    }

    #[test]
    fn ensure_ascii_escapes_non_ascii_and_surrogate_pairs() {
        let value = Value::from("é 😀 \u{7f}");
        assert_eq!(
            dump(value.clone(), &[("ensure_ascii", Value::from(true))]),
            r#""\u00e9 \ud83d\ude00 \u007f""#
        );
        // The default leaves UTF-8 alone, U+007F included.
        assert_eq!(dump(value, &[]), "\"é 😀 \u{7f}\"");
    }

    #[test]
    fn control_characters_use_the_python_escape_table() {
        let value = Value::from("a\nb\tc\rd\u{8}e\u{c}f\u{1}g\"h\\i");
        assert_eq!(dump(value, &[]), r#""a\nb\tc\rd\be\ff\u0001g\"h\\i""#);
    }

    #[test]
    fn html_characters_are_not_escaped() {
        // minijinja's builtin turns these four into \\u003c, \\u003e, \\u0026
        // and \\u0027 so its output can be inlined into an HTML document.
        // `transformers` does not, and a JSON Schema `pattern` full of `<`
        // is a different prompt than the provider tokenized.
        let value = Value::from("a < b && c > d, it's fine");
        assert_eq!(dump(value, &[]), r#""a < b && c > d, it's fine""#);
    }

    #[test]
    fn indent_switches_to_python_item_separators() {
        let value = Value::from_serialize(serde_json::json!({"a": 1, "b": [1, 2]}));
        let rendered = dump(value, &[("indent", Value::from(2))]);
        assert_eq!(
            rendered,
            "{\n  \"a\": 1,\n  \"b\": [\n    1,\n    2\n  ]\n}"
        );
    }

    #[test]
    fn indent_true_means_two_spaces() {
        let value = Value::from_serialize(serde_json::json!({"a": 1}));
        assert_eq!(
            dump(value, &[("indent", Value::from(true))]),
            "{\n  \"a\": 1\n}"
        );
    }

    #[test]
    fn empty_containers_stay_on_one_line_under_indent() {
        let value = Value::from_serialize(serde_json::json!({"a": {}, "b": []}));
        assert_eq!(
            dump(value, &[("indent", Value::from(2))]),
            "{\n  \"a\": {},\n  \"b\": []\n}"
        );
    }

    #[test]
    fn explicit_separators_win_over_the_indent_default() {
        let value = Value::from_serialize(serde_json::json!({"a": 1, "b": 2}));
        let rendered = dump(
            value,
            &[
                ("indent", Value::from(2)),
                (
                    "separators",
                    Value::from(vec![Value::from(","), Value::from(" = ")]),
                ),
            ],
        );
        assert_eq!(rendered, "{\n  \"a\" = 1,\n  \"b\" = 2\n}");
    }

    #[test]
    fn null_and_bool_scalars_match_python() {
        assert_eq!(dump(Value::from(()), &[]), "null");
        assert_eq!(dump(Value::from(true), &[]), "true");
        assert_eq!(dump(Value::from(false), &[]), "false");
        assert_eq!(dump(Value::UNDEFINED, &[]), "null");
    }

    #[test]
    fn non_string_object_keys_are_coerced_like_python() {
        let mut map = std::collections::BTreeMap::new();
        map.insert(Value::from(1), Value::from("a"));
        map.insert(Value::from(true), Value::from("b"));
        let rendered = dump(Value::from(map), &[("sort_keys", Value::from(true))]);
        assert_eq!(rendered, r#"{"1": "a", "true": "b"}"#);
    }

    #[test]
    fn unknown_keyword_arguments_are_still_rejected() {
        let kwargs: Kwargs = [("cls", Value::from("Encoder"))].into_iter().collect();
        let err = python_tojson(Value::from(1), None, kwargs).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::TooManyArguments);
    }

    #[test]
    fn malformed_separators_are_rejected() {
        let kwargs: Kwargs = [("separators", Value::from(vec![Value::from(",")]))]
            .into_iter()
            .collect();
        let err = python_tojson(Value::from(1), None, kwargs).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidOperation);
    }

    #[test]
    fn deeply_nested_values_are_refused_rather_than_overflowing_the_stack() {
        let mut value = Value::from(1);
        for _ in 0..MAX_DEPTH + 5 {
            value = Value::from(vec![value]);
        }
        let err = python_tojson(
            value,
            None,
            Kwargs::from_iter(Vec::<(String, Value)>::new()),
        )
        .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidOperation);
    }

    #[test]
    fn map_insertion_order_is_preserved_by_default() {
        // The `preserve_order` guard: a client-sent JSON Schema must reach the
        // prompt in wire order, not alphabetized.
        let value: serde_json::Value =
            serde_json::from_str(r#"{"pattern":1,"include":2,"a":3}"#).unwrap();
        assert_eq!(
            dump(Value::from_serialize(value), &[]),
            r#"{"pattern": 1, "include": 2, "a": 3}"#
        );
    }
}
