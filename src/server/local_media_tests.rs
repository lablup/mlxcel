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

//! Command-line media resolution (issue #2173): file reads, video sources,
//! and the Inkling layout selection, none of which needs a checkpoint.

use super::*;

const IDS: InklingPromptTokenIds = InklingPromptTokenIds {
    message_user: 1,
    content_text: 2,
    content_image: 3,
    end_message: 4,
};

#[test]
fn a_raw_prompt_keeps_inkling_on_the_plain_layouts() {
    let layouts =
        inkling_cli_layouts(true, || panic!("a raw prompt needs no part markers")).expect("plain");
    assert_eq!(layouts.video, InklingVideoPromptLayout::Plain);
    assert_eq!(layouts.audio, InklingAudioPromptLayout::Plain);
}

#[test]
fn a_templated_prompt_uses_the_structured_layouts_never_ordered() {
    let layouts = inkling_cli_layouts(false, || Ok(IDS)).expect("structured");
    assert_eq!(layouts.video, InklingVideoPromptLayout::Structured(IDS));
    assert_eq!(layouts.audio, InklingAudioPromptLayout::Structured(IDS));
}

#[test]
fn a_templated_prompt_reports_unresolvable_part_markers() {
    let err = inkling_cli_layouts(false, || anyhow::bail!("no <|message_user|> token"))
        .expect_err("markers missing");
    assert!(err.to_string().contains("<|message_user|>"), "{err}");
}

#[test]
fn only_single_modality_inkling_requests_take_the_cli_path() {
    assert_eq!(
        inkling_cli_media(true, false, true),
        Some(InklingCliMedia::Video)
    );
    assert_eq!(
        inkling_cli_media(true, true, false),
        Some(InklingCliMedia::Audio)
    );
    assert_eq!(inkling_cli_media(true, true, true), None, "server refuses");
    assert_eq!(inkling_cli_media(true, false, false), None, "images only");
    assert_eq!(inkling_cli_media(false, true, false), None);
    assert_eq!(inkling_cli_media(false, false, true), None);
}

#[test]
fn files_are_read_in_order() {
    let dir = tempfile::tempdir().expect("tempdir");
    let first = dir.path().join("a.png");
    let second = dir.path().join("b.png");
    std::fs::write(&first, b"first").expect("write");
    std::fs::write(&second, b"second").expect("write");
    let bytes = read_files(&[first, second], "image").expect("read");
    assert_eq!(bytes, vec![b"first".to_vec(), b"second".to_vec()]);
}

#[test]
fn a_missing_file_is_an_error_naming_its_kind_and_path() {
    let err = read_files(&[PathBuf::from("/nonexistent/clip.wav")], "audio").expect_err("missing");
    let message = err.to_string();
    assert!(message.contains("audio"), "{message}");
    assert!(message.contains("/nonexistent/clip.wav"), "{message}");
}

#[test]
fn videos_are_opened_by_path_at_the_requested_rate() {
    let paths = [PathBuf::from("/v/one.mp4"), PathBuf::from("/v/two.mp4")];
    let videos = resolved_videos(&paths, 1.5);
    assert_eq!(videos.len(), 2);
    for (video, path) in videos.iter().zip(&paths) {
        assert!(matches!(&video.source, VideoSource::Path(p) if p == path));
        assert_eq!(video.fps, Some(1.5));
        assert!(video.temp_guard.is_none());
    }
    assert!(resolved_videos(&[], 2.0).is_empty());
}
