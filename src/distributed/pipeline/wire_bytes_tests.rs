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

use serde::{Deserialize, Serialize};

use super::super::activation_transfer::ActivationMessage;
use super::super::remote_service::{RemoteStageCommand, RemoteStageResponse};
use crate::distributed::RequestId;

#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct Probe {
    #[serde(with = "super::base64_bytes")]
    data: Vec<u8>,
    #[serde(with = "super::base64_bytes_opt")]
    mask: Option<Vec<u8>>,
}

#[test]
fn json_round_trips_and_is_compact() {
    let data: Vec<u8> = (0..=255u8).cycle().take(600_000).collect();
    let probe = Probe {
        data: data.clone(),
        mask: Some(vec![1, 0, 1]),
    };
    let json = serde_json::to_vec(&probe).unwrap();
    // Base64 is 4/3 of the input plus the field names; the old array
    // encoding was about 3.6 characters per byte.
    assert!(json.len() < data.len() * 4 / 3 + 64, "{}", json.len());
    let back: Probe = serde_json::from_slice(&json).unwrap();
    assert_eq!(back, probe);

    let none = Probe {
        data: Vec::new(),
        mask: None,
    };
    let json = serde_json::to_string(&none).unwrap();
    assert_eq!(json, r#"{"data":"","mask":null}"#);
    let back: Probe = serde_json::from_str(&json).unwrap();
    assert_eq!(back, none);
}

#[test]
fn invalid_base64_is_a_typed_error() {
    let err = serde_json::from_str::<Probe>(r#"{"data":"not base64!","mask":null}"#).unwrap_err();
    assert!(err.to_string().contains("invalid base64 bytes"), "{err}");
    // The old array encoding is no longer accepted: mixed versions fail loudly.
    assert!(serde_json::from_str::<Probe>(r#"{"data":[1,2,3],"mask":null}"#).is_err());
}

#[test]
fn stage_protocol_messages_round_trip_through_json() {
    let logits: Vec<u8> = (0..4096u32).map(|i| (i % 251) as u8).collect();
    let response = RemoteStageResponse::RunResult {
        stage_index: 1,
        nonce: 7,
        request_id: "req".to_string(),
        micro_batch_id: 0,
        logits: logits.clone(),
    };
    let json = serde_json::to_vec(&response).unwrap();
    assert!(json.len() < logits.len() * 4 / 3 + 128);
    match serde_json::from_slice::<RemoteStageResponse>(&json).unwrap() {
        RemoteStageResponse::RunResult {
            logits: back,
            nonce,
            ..
        } => {
            assert_eq!(back, logits);
            assert_eq!(nonce, 7);
        }
        other => panic!("unexpected {other:?}"),
    }

    let command = RemoteStageCommand::RunEntry {
        request_id: "req".to_string(),
        micro_batch_id: 2,
        input_ids: vec![9, 8, 7],
        attention_mask: None,
    };
    let json = serde_json::to_vec(&command).unwrap();
    match serde_json::from_slice::<RemoteStageCommand>(&json).unwrap() {
        RemoteStageCommand::RunEntry {
            input_ids,
            attention_mask,
            ..
        } => {
            assert_eq!(input_ids, vec![9, 8, 7]);
            assert_eq!(attention_mask, None);
        }
        other => panic!("unexpected {other:?}"),
    }

    let activation = ActivationMessage::forward(
        RequestId::new(),
        0,
        0,
        2,
        vec![5u8; 2048],
        Some(vec![1u8; 16]),
        None,
        16,
    );
    let json = serde_json::to_vec(&activation).unwrap();
    let back: ActivationMessage = serde_json::from_slice(&json).unwrap();
    assert_eq!(back.tensor_data, activation.tensor_data);
    assert_eq!(back.attention_mask, activation.attention_mask);
    assert_eq!(back.position_ids, None);
}
