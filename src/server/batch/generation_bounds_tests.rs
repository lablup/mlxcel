//! Unit coverage for the two b10621 generation bounds (#1477).
//!
//! The fixtures are the pinned-binary measurements: a seeded `n_indent: 4`
//! completion of `def f():\n    x = 1\n` stops after `    print("done")\n`
//! because the next line is empty, and `t_max_predict_ms` fires on the first
//! newline generated after the deadline rather than at the deadline itself.

use super::*;

/// Feed a whole generation one piece at a time, as the decode loop does.
fn run(bounds: &mut GenerationBounds, pieces: &[&str]) -> Option<BoundStop> {
    for piece in pieces {
        if let Some(stop) = bounds.observe(piece, piece) {
            return Some(stop);
        }
    }
    None
}

#[test]
fn inert_bounds_never_fire_and_accumulate_nothing() {
    let mut bounds = GenerationBounds::default();
    assert!(!bounds.is_active());
    assert_eq!(run(&mut bounds, &["a\n", "  b\n", "c\n"]), None);
    assert_eq!(bounds.fired(), None);
    assert!(bounds.text.is_empty());
}

#[test]
fn n_indent_stops_on_a_line_that_falls_below_the_requested_indentation() {
    let mut bounds = GenerationBounds::new(4, None);
    // "    a\n" arms has_new_line; the next token advances the cursor past it;
    // the one after that evaluates the dedented "b" line.
    let stop = run(&mut bounds, &["    a\n", "b", "c"]);
    let Some(BoundStop::Indent { keep_bytes }) = stop else {
        panic!("expected an indentation stop, got {stop:?}");
    };
    // Upstream erases from the first character after the leading whitespace of
    // the offending line, which here has none: the cut lands right after the
    // newline, so the kept text is the first line alone.
    assert_eq!(&bounds.text[..keep_bytes], "    a\n");
}

#[test]
fn n_indent_accepts_a_line_that_meets_the_requested_indentation() {
    let mut bounds = GenerationBounds::new(4, None);
    assert_eq!(run(&mut bounds, &["    a\n", "    b", "    c"]), None);
}

#[test]
fn n_indent_treats_an_empty_line_as_zero_indentation() {
    // The measured stop on the pinned binary: generation ended because the line
    // after `    print("done")\n` was empty, not because of a dedented
    // statement.
    let mut bounds = GenerationBounds::new(4, None);
    let stop = run(&mut bounds, &["    print(1)\n", "x", "\n"]);
    assert!(matches!(stop, Some(BoundStop::Indent { .. })));
}

#[test]
fn n_indent_ignores_the_first_line_because_upstream_has_no_cursor_for_it() {
    // `last_nl_pos` starts at 0 and the rule only runs once it is positive, so
    // a prompt continuation that begins unindented is not stopped on its own
    // first line.
    let mut bounds = GenerationBounds::new(4, None);
    assert_eq!(run(&mut bounds, &["nope", " still", "\n"]), None);
}

#[test]
fn t_max_predict_ms_fires_only_on_a_newline_after_the_deadline() {
    let mut bounds = GenerationBounds::new(0, Some(0));
    // No newline yet: the deadline cannot fire however long the request runs.
    // The first observation is also what starts the clock.
    assert_eq!(bounds.observe("tokens ", "tokens "), None);
    std::thread::sleep(std::time::Duration::from_millis(2));
    assert_eq!(bounds.observe("without newlines", "without newlines"), None);
    assert_eq!(bounds.observe("\n", "\n"), Some(BoundStop::Time));
}

#[test]
fn t_max_predict_ms_does_not_fire_before_its_deadline() {
    let mut bounds = GenerationBounds::new(0, Some(60_000));
    assert_eq!(run(&mut bounds, &["a\n", "b\n", "c\n"]), None);
}

#[test]
fn a_fired_bound_is_sticky_and_ignores_later_pieces() {
    let mut bounds = GenerationBounds::new(0, Some(0));
    // The clock starts at the first observed token, upstream's "measured since
    // the first token", so a newline on that very token cannot be late yet.
    assert_eq!(bounds.observe("a", "a"), None);
    std::thread::sleep(std::time::Duration::from_millis(2));
    assert_eq!(bounds.observe("b\n", "b\n"), Some(BoundStop::Time));
    assert_eq!(bounds.observe("c\n", "c\n"), None);
    assert_eq!(bounds.fired(), Some(BoundStop::Time));
}

#[test]
fn the_deadline_arms_on_the_emitted_text_not_the_decoded_text() {
    // While a stop string is being matched upstream sends empty content, so its
    // `has_new_line` does not arm on a newline the client never saw. This first
    // observation also starts the deadline clock, so the sleep belongs after it.
    let mut bounds = GenerationBounds::new(0, Some(0));
    assert_eq!(bounds.observe("a\n", ""), None);
    std::thread::sleep(std::time::Duration::from_millis(2));
    assert_eq!(bounds.observe("b\n", "b\n"), Some(BoundStop::Time));
}
