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

//! Unit tests for slot save/restore persistence (issue #1440): filename
//! validation parity with b10621's `fs_validate_filename`, storage-root
//! confinement including symlink escapes, atomicity, and identity binding.

use super::{SlotPersistError, load, save, validate_filename};

fn temp_root(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "mlxcel-slot-persist-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).expect("create temp root");
    dir
}

#[test]
fn filename_validation_matches_b10621_rules() {
    // Accepted.
    for name in ["cache.bin", "slot-0.state", "한글이름", "a"] {
        assert!(validate_filename(name), "{name} must be accepted");
    }
    // Refused: empty, separators, dot-dot, lookalike codepoints, control
    // characters, Windows-stripped forms, and the illegal character set.
    let refused = [
        "",
        "a/b",
        "a\\b",
        "..",
        "a..b",
        ".",
        "con:",
        "a*",
        "a?",
        "a\"",
        "a<",
        "a>",
        "a|",
        " leading",
        "trailing ",
        "trailing.",
        "nul\u{0}byte",
        "u2215\u{2215}slash",
        "u2216\u{2216}backslash",
        "uFF0E\u{FF0E}period",
        "bom\u{FEFF}",
        "repl\u{FFFD}",
    ];
    for name in refused {
        assert!(!validate_filename(name), "{name:?} must be refused");
    }
    // The 255-byte cap.
    assert!(!validate_filename(&"x".repeat(256)));
    assert!(validate_filename(&"x".repeat(255)));
}

#[test]
fn save_then_load_roundtrips_tokens_and_identity() {
    let root = temp_root("roundtrip");
    let tokens = vec![1, 2, 3, 40000, -1];
    let n_written = save(&root, "state.bin", "model-a", "fp-1", &tokens).expect("save");
    assert!(n_written > 0);
    let (envelope, n_read) = load(&root, "state.bin", "model-a", "fp-1").expect("load");
    assert_eq!(envelope.tokens, tokens);
    assert_eq!(n_read, n_written);
}

#[test]
fn load_rejects_model_and_tokenizer_mismatch() {
    let root = temp_root("mismatch");
    save(&root, "state.bin", "model-a", "fp-1", &[1, 2]).expect("save");
    let by_model = load(&root, "state.bin", "model-b", "fp-1");
    assert!(matches!(by_model, Err(SlotPersistError::Invalid(_))));
    let by_tokenizer = load(&root, "state.bin", "model-a", "fp-2");
    assert!(matches!(by_tokenizer, Err(SlotPersistError::Invalid(_))));
}

#[test]
fn load_rejects_garbage_and_missing_files() {
    let root = temp_root("garbage");
    std::fs::write(root.join("junk.bin"), b"not json").expect("write junk");
    assert!(matches!(
        load(&root, "junk.bin", "m", "fp"),
        Err(SlotPersistError::Invalid(_))
    ));
    assert!(matches!(
        load(&root, "absent.bin", "m", "fp"),
        Err(SlotPersistError::Unreadable(_))
    ));
}

#[test]
fn traversal_filenames_cannot_escape_the_root() {
    let root = temp_root("traversal");
    for name in ["../escape.bin", "..", "a/../../b", "/etc/passwd"] {
        assert!(
            matches!(
                save(&root, name, "m", "fp", &[1]),
                Err(SlotPersistError::InvalidFilename)
            ),
            "{name} must be refused on save"
        );
        assert!(
            matches!(
                load(&root, name, "m", "fp"),
                Err(SlotPersistError::InvalidFilename)
            ),
            "{name} must be refused on load"
        );
    }
}

#[cfg(unix)]
#[test]
fn symlink_inside_the_root_cannot_read_outside_it() {
    let root = temp_root("symlink-read");
    let outside = temp_root("symlink-target");
    // A verbatim slot file planted OUTSIDE the root, reachable only through
    // a symlink inside the root.
    save(&outside, "real.bin", "m", "fp", &[7]).expect("save outside");
    std::os::unix::fs::symlink(outside.join("real.bin"), root.join("link.bin"))
        .expect("plant symlink");
    let result = load(&root, "link.bin", "m", "fp");
    assert!(
        matches!(result, Err(SlotPersistError::InvalidFilename)),
        "symlink escape must be refused, got {result:?}"
    );
}

#[cfg(unix)]
#[test]
fn symlink_inside_the_root_cannot_be_written_through() {
    let root = temp_root("symlink-write");
    let outside = temp_root("symlink-victim");
    let victim = outside.join("victim.bin");
    std::fs::write(&victim, b"original").expect("write victim");
    std::os::unix::fs::symlink(&victim, root.join("link.bin")).expect("plant symlink");
    let result = save(&root, "link.bin", "m", "fp", &[1]);
    assert!(
        matches!(result, Err(SlotPersistError::InvalidFilename)),
        "writing through a symlink must be refused, got {result:?}"
    );
    assert_eq!(std::fs::read(&victim).expect("victim intact"), b"original");
}

#[test]
fn save_is_atomic_no_temp_file_survives() {
    let root = temp_root("atomic");
    save(&root, "state.bin", "m", "fp", &[1, 2, 3]).expect("save");
    let leftovers: Vec<_> = std::fs::read_dir(&root)
        .expect("read root")
        .map(|e| e.expect("entry").file_name().to_string_lossy().into_owned())
        .filter(|name| name != "state.bin")
        .collect();
    assert!(
        leftovers.is_empty(),
        "temporary files must not survive a save: {leftovers:?}"
    );
}

#[test]
fn save_into_missing_root_reports_io_error() {
    let root = std::env::temp_dir().join("mlxcel-slot-persist-definitely-missing-root");
    let _ = std::fs::remove_dir_all(&root);
    assert!(matches!(
        save(&root, "state.bin", "m", "fp", &[1]),
        Err(SlotPersistError::Io(_))
    ));
}
