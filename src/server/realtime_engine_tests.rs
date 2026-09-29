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

//! Engine tests against a fake model (no checkpoint).

use std::sync::{Arc, Mutex};

use super::*;
use crate::models::nemotron_voicechat::VoiceChatEvent;

type Log = Arc<Mutex<Vec<String>>>;

struct FakeModel {
    log: Log,
}

struct FakeSession {
    log: Log,
    frames: u64,
    closed: bool,
}

impl RealtimeModel for FakeModel {
    fn open_session(
        &self,
        config: &RealtimeSessionConfig,
    ) -> Result<Box<dyn RealtimeSession + '_>, VoiceChatError> {
        if config.seed == 666 {
            panic!("fake open panic");
        }
        self.log.lock().unwrap().push("open".to_string());
        Ok(Box::new(FakeSession {
            log: self.log.clone(),
            frames: 0,
            closed: false,
        }))
    }
}

impl RealtimeSession for FakeSession {
    fn info(&self) -> RealtimeSessionInfo {
        RealtimeSessionInfo {
            input_sample_rate: 16_000,
            output_sample_rate: 22_050,
            frame_samples: 1280,
        }
    }

    fn push_audio(
        &mut self,
        samples: &[f32],
        _sample_rate: u32,
    ) -> Result<Vec<VoiceChatEvent>, VoiceChatError> {
        if samples.first() == Some(&9.0) {
            panic!("fake push panic");
        }
        self.frames += 1;
        Ok(vec![VoiceChatEvent::Done {
            frame_index: self.frames,
        }])
    }

    fn flush(&mut self, _pad_partial: bool) -> Result<Vec<VoiceChatEvent>, VoiceChatError> {
        self.closed = true;
        self.log.lock().unwrap().push("flush".to_string());
        Ok(vec![VoiceChatEvent::Done {
            frame_index: self.frames,
        }])
    }

    fn cancel(&mut self) -> Vec<VoiceChatEvent> {
        self.closed = true;
        self.log.lock().unwrap().push("cancel".to_string());
        vec![VoiceChatEvent::Cancelled {
            frame_index: self.frames,
        }]
    }

    fn is_closed(&self) -> bool {
        self.closed
    }
}

fn engine() -> (RealtimeVoiceChatEngine, Log) {
    let log: Log = Arc::default();
    let model_log = log.clone();
    let engine = RealtimeVoiceChatEngine::spawn_with_loader("fake", move || {
        Ok(FakeModel { log: model_log })
    })
    .unwrap();
    (engine, log)
}

#[test]
fn second_reserve_fails_until_release() {
    let (engine, _) = engine();
    assert!(!engine.is_reserved());
    assert!(engine.try_reserve("sess_a"));
    assert!(engine.is_reserved());
    assert!(!engine.try_reserve("sess_b"));
    // Releasing someone else's id is a no-op.
    engine.release("sess_b");
    assert!(!engine.try_reserve("sess_b"));
    engine.release("sess_a");
    assert!(!engine.is_reserved());
    assert!(engine.try_reserve("sess_b"));
}

#[tokio::test]
async fn push_on_unopened_session_errors() {
    let (engine, _) = engine();
    let err = engine.push("sess_a", vec![0.0; 1280], 16_000).await;
    assert_eq!(err, Err(RealtimeEngineError::NotActive));

    engine
        .open("sess_a", RealtimeSessionConfig::default())
        .await
        .unwrap();
    // Another connection's id cannot drive the active session.
    let err = engine.flush("sess_b", true).await;
    assert_eq!(err, Err(RealtimeEngineError::NotActive));
    let err = engine
        .open("sess_b", RealtimeSessionConfig::default())
        .await;
    assert_eq!(err, Err(RealtimeEngineError::AlreadyActive));

    let events = engine
        .push("sess_a", vec![0.0; 1280], 16_000)
        .await
        .unwrap();
    assert_eq!(events, vec![ServerEvent::Done { frame_index: 1 }]);
}

#[tokio::test]
async fn close_cancels_unflushed_session() {
    let (engine, log) = engine();
    engine
        .open("sess_a", RealtimeSessionConfig::default())
        .await
        .unwrap();
    engine.close("sess_a").await.unwrap();
    assert_eq!(*log.lock().unwrap(), vec!["open", "cancel"]);
    assert_eq!(
        engine.push("sess_a", Vec::new(), 16_000).await,
        Err(RealtimeEngineError::NotActive)
    );

    // A flushed session is not cancelled again on close.
    log.lock().unwrap().clear();
    engine
        .open("sess_b", RealtimeSessionConfig::default())
        .await
        .unwrap();
    engine.flush("sess_b", true).await.unwrap();
    engine.close("sess_b").await.unwrap();
    assert_eq!(*log.lock().unwrap(), vec!["open", "flush"]);

    // A detached close (dropped socket task) runs before the next open.
    log.lock().unwrap().clear();
    engine
        .open("sess_c", RealtimeSessionConfig::default())
        .await
        .unwrap();
    engine.close_detached("sess_c");
    engine
        .open("sess_d", RealtimeSessionConfig::default())
        .await
        .unwrap();
    assert_eq!(*log.lock().unwrap(), vec!["open", "cancel", "open"]);
}

#[tokio::test]
async fn panicking_session_is_dropped_and_engine_keeps_serving() {
    let (engine, _) = engine();
    engine
        .open("sess_a", RealtimeSessionConfig::default())
        .await
        .unwrap();
    let err = engine.push("sess_a", vec![9.0], 16_000).await.unwrap_err();
    assert!(matches!(err, RealtimeEngineError::Panicked(_)), "{err:?}");
    assert_eq!(
        engine.push("sess_a", vec![0.0], 16_000).await,
        Err(RealtimeEngineError::NotActive)
    );
    let config = RealtimeSessionConfig {
        seed: 666,
        ..RealtimeSessionConfig::default()
    };
    assert!(matches!(
        engine.open("sess_b", config).await,
        Err(RealtimeEngineError::Panicked(_))
    ));
    engine
        .open("sess_c", RealtimeSessionConfig::default())
        .await
        .unwrap();
}

#[test]
fn failed_load_is_reported() {
    let err = RealtimeVoiceChatEngine::spawn_with_loader::<FakeModel, _>("fake", || {
        Err("no weights".to_string())
    })
    .unwrap_err();
    assert!(err.contains("no weights"), "{err}");
}
