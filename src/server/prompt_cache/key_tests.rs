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

use super::*;

fn tokens(n: usize) -> Vec<i32> {
    (0..n as i32).collect()
}

// Convenience: build a text-only key (empty multimodal digest).
fn text_key<'a>(
    model_id: &'a str,
    lora_id: Option<&'a str>,
    template_sig: &'a str,
    session_key: Option<&'a str>,
    tokens: &'a [i32],
) -> PromptCacheKey<'a> {
    PromptCacheKey::new_full(
        model_id,
        lora_id,
        template_sig,
        session_key,
        MultimodalDigest::empty(),
        tokens,
    )
}

#[test]
fn digest_is_stable_for_identical_inputs() {
    let toks = tokens(16);
    let a = text_key("m", Some("l"), "tpl", Some("s"), &toks);
    let b = text_key("m", Some("l"), "tpl", Some("s"), &toks);
    assert_eq!(a.digest(), b.digest());
}

#[test]
fn digest_changes_when_model_id_differs() {
    let toks = tokens(16);
    let a = text_key("model-a", None, "tpl", None, &toks);
    let b = text_key("model-b", None, "tpl", None, &toks);
    assert_ne!(a.digest(), b.digest());
}

#[test]
fn digest_changes_when_lora_differs() {
    let toks = tokens(16);
    let a = text_key("m", None, "tpl", None, &toks);
    let b = text_key("m", Some("lora-1"), "tpl", None, &toks);
    assert_ne!(a.digest(), b.digest());
}

#[test]
fn digest_changes_when_template_sig_differs() {
    let toks = tokens(16);
    let a = text_key("m", None, "tpl-a", None, &toks);
    let b = text_key("m", None, "tpl-b", None, &toks);
    assert_ne!(a.digest(), b.digest());
}

#[test]
fn digest_changes_when_session_key_differs() {
    let toks = tokens(16);
    let a = text_key("m", None, "tpl", None, &toks);
    let b = text_key("m", None, "tpl", Some("session-1"), &toks);
    assert_ne!(a.digest(), b.digest());
}

#[test]
fn digest_changes_with_each_extra_token() {
    let a_tokens = tokens(16);
    let b_tokens = tokens(17);
    let a = text_key("m", None, "tpl", None, &a_tokens);
    let b = text_key("m", None, "tpl", None, &b_tokens);
    assert_ne!(a.digest(), b.digest());
}

#[test]
fn prefix_len_saturates_at_token_length() {
    let toks = tokens(8);
    let k =
        PromptCacheKey::new_prefix("m", None, "tpl", None, MultimodalDigest::empty(), &toks, 64);
    assert_eq!(k.effective_prefix_len(), 8);
}

#[test]
fn shorter_prefix_digest_matches_truncated_full_digest() {
    // A prefix-keyed digest over tokens[..8] must equal the full digest of
    // a 8-token input, because only the first `prefix_len` tokens are
    // hashed.
    let full = tokens(16);
    let truncated = tokens(8);

    let a = PromptCacheKey::new_prefix("m", None, "tpl", None, MultimodalDigest::empty(), &full, 8);
    let b = text_key("m", None, "tpl", None, &truncated);
    assert_eq!(a.digest(), b.digest());
}

#[test]
fn length_prefix_prevents_boundary_collisions() {
    // Without length prefixes, ("ab", "c") and ("a", "bc") concatenate to
    // the same string. Length prefixes + domain separator must distinguish
    // them.
    let toks = tokens(8);
    let a = text_key("ab", None, "c", None, &toks);
    let b = text_key("a", None, "bc", None, &toks);
    assert_ne!(a.digest(), b.digest());
}

#[test]
fn hex_roundtrip_length() {
    let toks = tokens(4);
    let d = text_key("m", None, "tpl", None, &toks).digest();
    assert_eq!(d.to_hex().len(), 64);
    assert_eq!(d.short_hex().len(), 16);
}

#[test]
fn digest_treats_none_and_empty_string_identically() {
    // The length-prefixed wire format emits `[len=0][]` for both, so this
    // is intentional. Callers that need to distinguish "no session" from
    // "empty session id" must normalize upstream.
    let toks = tokens(16);
    let a = text_key("m", None, "tpl", None, &toks);
    let b = text_key("m", Some(""), "tpl", Some(""), &toks);
    assert_eq!(a.digest(), b.digest());
}

#[test]
fn token_order_matters() {
    let forward: Vec<i32> = (0..16).collect();
    let reversed: Vec<i32> = (0..16).rev().collect();
    let a = text_key("m", None, "tpl", None, &forward);
    let b = text_key("m", None, "tpl", None, &reversed);
    assert_ne!(a.digest(), b.digest());
}

// ---------------------------------------------------------------------------
// session-key resolution
// ---------------------------------------------------------------------------

#[test]
fn resolve_session_key_prefers_prompt_cache_key() {
    let got = resolve_session_key(Some("pck"), Some("user-1"));
    assert_eq!(got, "pck");
}

#[test]
fn resolve_session_key_falls_back_to_user() {
    let got = resolve_session_key(None, Some("user-1"));
    assert_eq!(got, "user-1");
}

#[test]
fn resolve_session_key_uses_anonymous_sentinel_when_both_absent() {
    let got = resolve_session_key(None, None);
    assert_eq!(got, ANONYMOUS_SESSION_SENTINEL);
    // The sentinel is non-empty so it distinguishes from a `None` session.
    assert!(!got.is_empty());
}

#[test]
fn resolve_session_key_treats_empty_prompt_cache_key_as_absent() {
    let got = resolve_session_key(Some(""), Some("user-1"));
    assert_eq!(got, "user-1");
}

#[test]
fn resolve_session_key_treats_empty_user_as_absent() {
    let got = resolve_session_key(None, Some(""));
    assert_eq!(got, ANONYMOUS_SESSION_SENTINEL);
}

#[test]
fn resolve_session_key_all_empty_returns_sentinel() {
    let got = resolve_session_key(Some(""), Some(""));
    assert_eq!(got, ANONYMOUS_SESSION_SENTINEL);
}

// ---------------------------------------------------------------------------
// template signature
// ---------------------------------------------------------------------------

fn sample_tool(name: &str) -> Tool {
    Tool {
        tool_type: "function".to_string(),
        function: crate::server::types::request::FunctionDefinition {
            name: name.to_string(),
            description: Some(format!("{name} description")),
            parameters: Some(serde_json::json!({"type": "object"})),
        },
    }
}

#[test]
fn template_sig_is_stable_for_identical_inputs() {
    let kw = ChatTemplateKwargs::from_json_str(r#"{"preserve_thinking": true}"#).unwrap();
    let a = template_sig("tpl", &kw, None, None);
    let b = template_sig("tpl", &kw, None, None);
    assert_eq!(a, b);
    assert_eq!(a.len(), 64, "must be 64-char hex");
}

#[test]
fn template_sig_changes_when_template_source_changes() {
    let kw = ChatTemplateKwargs::new();
    let a = template_sig("template-a", &kw, None, None);
    let b = template_sig("template-b", &kw, None, None);
    assert_ne!(a, b);
}

#[test]
fn template_sig_changes_when_kwargs_change() {
    let kw_a = ChatTemplateKwargs::from_json_str(r#"{"preserve_thinking": true}"#).unwrap();
    let kw_b = ChatTemplateKwargs::from_json_str(r#"{"preserve_thinking": false}"#).unwrap();
    let a = template_sig("tpl", &kw_a, None, None);
    let b = template_sig("tpl", &kw_b, None, None);
    assert_ne!(a, b);
}

#[test]
fn template_sig_canonicalizes_kwargs_key_order() {
    // {"a":1,"b":2} and {"b":2,"a":1} — canonicalization must collapse
    // these to the same digest so map-insertion-order drift is absorbed.
    let kw_a = ChatTemplateKwargs::from_json_str(r#"{"a": 1, "b": 2}"#).unwrap();
    let kw_b = ChatTemplateKwargs::from_json_str(r#"{"b": 2, "a": 1}"#).unwrap();
    let a = template_sig("tpl", &kw_a, None, None);
    let b = template_sig("tpl", &kw_b, None, None);
    assert_eq!(a, b, "kwargs key order must not affect the signature");
}

#[test]
fn template_sig_changes_when_tool_choice_mode_changes() {
    let kw = ChatTemplateKwargs::new();
    let a = template_sig(
        "tpl",
        &kw,
        Some(&ToolChoice::Mode("auto".to_string())),
        None,
    );
    let b = template_sig(
        "tpl",
        &kw,
        Some(&ToolChoice::Mode("none".to_string())),
        None,
    );
    assert_ne!(a, b);
}

#[test]
fn template_sig_distinguishes_absent_from_auto_tool_choice() {
    let kw = ChatTemplateKwargs::new();
    let absent = template_sig("tpl", &kw, None, None);
    let auto = template_sig(
        "tpl",
        &kw,
        Some(&ToolChoice::Mode("auto".to_string())),
        None,
    );
    assert_ne!(absent, auto);
}

#[test]
fn template_sig_changes_when_tools_added() {
    let kw = ChatTemplateKwargs::new();
    let tools_a: Vec<Tool> = vec![];
    let tools_b = vec![sample_tool("get_weather")];
    let a = template_sig("tpl", &kw, None, Some(&tools_a));
    let b = template_sig("tpl", &kw, None, Some(&tools_b));
    assert_ne!(a, b);
}

#[test]
fn template_sig_changes_when_tool_removed() {
    let kw = ChatTemplateKwargs::new();
    let two = vec![sample_tool("get_weather"), sample_tool("send_email")];
    let one = vec![sample_tool("get_weather")];
    let a = template_sig("tpl", &kw, None, Some(&two));
    let b = template_sig("tpl", &kw, None, Some(&one));
    assert_ne!(a, b);
}

#[test]
fn template_sig_changes_when_tools_reordered() {
    // tools_digest is order-preserving by design; reordering must change
    // the signature.
    let kw = ChatTemplateKwargs::new();
    let forward = vec![sample_tool("a"), sample_tool("b")];
    let reversed = vec![sample_tool("b"), sample_tool("a")];
    let a = template_sig("tpl", &kw, None, Some(&forward));
    let b = template_sig("tpl", &kw, None, Some(&reversed));
    assert_ne!(
        a, b,
        "tool reordering must invalidate the template signature"
    );
}

#[test]
fn template_sig_treats_empty_tools_and_none_tools_identically() {
    // Both map to the same "no tools" marker in the digest.
    let kw = ChatTemplateKwargs::new();
    let empty: Vec<Tool> = vec![];
    let a = template_sig("tpl", &kw, None, None);
    let b = template_sig("tpl", &kw, None, Some(&empty));
    assert_eq!(a, b);
}

#[test]
fn tools_digest_changes_with_parameters() {
    // The function.parameters JSON Schema participates in the digest.
    let mut tool_a = sample_tool("weather");
    let mut tool_b = sample_tool("weather");
    tool_a.function.parameters =
        Some(serde_json::json!({"type": "object", "properties": {"a": {"type": "string"}}}));
    tool_b.function.parameters =
        Some(serde_json::json!({"type": "object", "properties": {"b": {"type": "string"}}}));
    let a = tools_digest(Some(&[tool_a]));
    let b = tools_digest(Some(&[tool_b]));
    assert_ne!(a, b);
}

#[test]
fn tools_digest_canonicalizes_parameter_key_order() {
    // Re-ordering JSON object keys inside parameters must not change the digest.
    let mut tool_a = sample_tool("fn");
    let mut tool_b = sample_tool("fn");
    tool_a.function.parameters = Some(serde_json::json!({"a": 1, "b": 2}));
    tool_b.function.parameters = Some(serde_json::json!({"b": 2, "a": 1}));
    let a = tools_digest(Some(&[tool_a]));
    let b = tools_digest(Some(&[tool_b]));
    assert_eq!(a, b);
}

#[test]
fn template_sig_changes_with_tool_name() {
    // Two tools with the same type and description but different names must
    // still yield different signatures.
    let kw = ChatTemplateKwargs::new();
    let a = template_sig("tpl", &kw, None, Some(&[sample_tool("alpha")]));
    let b = template_sig("tpl", &kw, None, Some(&[sample_tool("beta")]));
    assert_ne!(a, b);
}

#[test]
fn template_sig_fits_into_prompt_cache_key() {
    // Smoke: signature hex can slot straight into the cache key's
    // `template_sig` field.
    let kw = ChatTemplateKwargs::new();
    let sig = template_sig("tpl", &kw, None, None);
    let toks = tokens(4);
    let key = text_key("m", None, sig.as_str(), None, &toks);
    assert_eq!(key.template_sig, sig.as_str());
    // Computes a digest without panicking.
    let _ = key.digest();
}

// ---------------------------------------------------------------------------
// multimodal digest
// ---------------------------------------------------------------------------

/// Fake image bytes for testing — a minimal 1×1 PNG header stub.
fn fake_image(seed: u8) -> Vec<u8> {
    vec![0x89, 0x50, 0x4E, 0x47, seed, 0x0A]
}

/// Fake audio bytes for testing.
fn fake_audio(seed: u8) -> Vec<u8> {
    vec![0x52, 0x49, 0x46, 0x46, seed, 0x00]
}

#[test]
fn multimodal_digest_empty_is_stable() {
    // Two calls with no payloads produce the same digest.
    let a = multimodal_digest(&[], &[]);
    let b = multimodal_digest(&[], &[]);
    assert_eq!(a, b);
}

#[test]
fn multimodal_digest_empty_matches_empty_helper() {
    // MultimodalDigest::empty() is the canonical "no content" sentinel and
    // must equal multimodal_digest(&[], &[]).
    assert_eq!(MultimodalDigest::empty(), multimodal_digest(&[], &[]));
}

#[test]
fn same_text_different_images_produce_different_cache_keys() {
    // Acceptance criterion 1: two requests with same text but different
    // images must produce different bucket digests.
    let toks = tokens(16);
    let img_a = fake_image(1);
    let img_b = fake_image(2);

    let mm_a = multimodal_digest(&[img_a.as_slice()], &[]);
    let mm_b = multimodal_digest(&[img_b.as_slice()], &[]);

    let key_a = PromptCacheKey::new_full("m", None, "tpl", None, mm_a, &toks);
    let key_b = PromptCacheKey::new_full("m", None, "tpl", None, mm_b, &toks);
    assert_ne!(key_a.digest(), key_b.digest());
}

#[test]
fn same_text_same_image_bytes_produce_same_cache_key() {
    // Acceptance criterion 2: two requests with identical image bytes must
    // map to the same bucket regardless of how the bytes were delivered
    // (URL, base64, file path).
    let toks = tokens(16);
    let img = fake_image(42);

    let mm_a = multimodal_digest(&[img.as_slice()], &[]);
    let mm_b = multimodal_digest(&[img.as_slice()], &[]);
    assert_eq!(mm_a, mm_b);

    let key_a = PromptCacheKey::new_full("m", None, "tpl", None, mm_a, &toks);
    let key_b = PromptCacheKey::new_full("m", None, "tpl", None, mm_b, &toks);
    assert_eq!(key_a.digest(), key_b.digest());
}

#[test]
fn same_text_same_image_different_audio_produces_different_key() {
    // Adding audio to a request that already has an image must change the
    // cache key even if the image is identical.
    let toks = tokens(16);
    let img = fake_image(7);
    let aud = fake_audio(3);

    let mm_image_only = multimodal_digest(&[img.as_slice()], &[]);
    let mm_image_audio = multimodal_digest(&[img.as_slice()], &[aud.as_slice()]);

    let key_image_only = PromptCacheKey::new_full("m", None, "tpl", None, mm_image_only, &toks);
    let key_image_audio = PromptCacheKey::new_full("m", None, "tpl", None, mm_image_audio, &toks);
    assert_ne!(key_image_only.digest(), key_image_audio.digest());
}

#[test]
fn image_order_sensitivity_swapping_changes_key() {
    // The digest is order-preserving: swapping two images must change the
    // cache key because the image placeholder token positions in the LLM
    // token stream are determined by image order in the request.
    let toks = tokens(16);
    let img_a = fake_image(10);
    let img_b = fake_image(20);

    let mm_ab = multimodal_digest(&[img_a.as_slice(), img_b.as_slice()], &[]);
    let mm_ba = multimodal_digest(&[img_b.as_slice(), img_a.as_slice()], &[]);

    assert_ne!(mm_ab, mm_ba, "swapping image order must change the digest");

    let key_ab = PromptCacheKey::new_full("m", None, "tpl", None, mm_ab, &toks);
    let key_ba = PromptCacheKey::new_full("m", None, "tpl", None, mm_ba, &toks);
    assert_ne!(key_ab.digest(), key_ba.digest());
}

#[test]
fn multimodal_digest_from_vecs_matches_direct() {
    // The Vec<Vec<u8>> convenience wrapper must produce the same digest as
    // the slice-reference form.
    let img = fake_image(5);
    let aud = fake_audio(9);

    let via_slices = multimodal_digest(&[img.as_slice()], &[aud.as_slice()]);
    let via_vecs =
        multimodal_digest_from_vecs(std::slice::from_ref(&img), std::slice::from_ref(&aud));
    assert_eq!(via_slices, via_vecs);
}

#[test]
fn text_only_key_uses_empty_multimodal_digest() {
    // A text-only key must equal one that explicitly passes the empty digest,
    // confirming the sentinel is the correct default.
    let toks = tokens(8);
    let a = text_key("m", None, "tpl", None, &toks);
    let b = PromptCacheKey::new_full("m", None, "tpl", None, MultimodalDigest::empty(), &toks);
    assert_eq!(a.digest(), b.digest());
}

#[test]
fn multimodal_digest_does_not_collide_with_empty() {
    // A key with actual image content must differ from the empty-digest key,
    // even if everything else is identical.
    let toks = tokens(16);
    let img = fake_image(99);

    let mm_img = multimodal_digest(&[img.as_slice()], &[]);
    let key_img = PromptCacheKey::new_full("m", None, "tpl", None, mm_img, &toks);
    let key_empty = text_key("m", None, "tpl", None, &toks);
    assert_ne!(key_img.digest(), key_empty.digest());
}

#[test]
fn hex_representation_of_multimodal_digest_is_64_chars() {
    let d = MultimodalDigest::empty();
    assert_eq!(d.to_hex().len(), 64);
}

// ---------------------------------------------------------------------------
// integration: multimodal multi-turn prefix stability
// ---------------------------------------------------------------------------

#[test]
fn multimodal_multi_turn_prefix_stability() {
    // Simulate a multi-turn conversation where the same image appears in the
    // system/first-turn message and the user adds new text each turn.
    //
    // The text tokens grow each turn; the prompt-cache key for the shared
    // image + system prefix should remain stable (same mm_digest + same
    // leading token prefix) so the cache can reuse it.

    let shared_img = fake_image(55);
    let mm = multimodal_digest(&[shared_img.as_slice()], &[]);

    // Turn 1: 16 text tokens (system + image placeholder + first user msg).
    let turn1_tokens: Vec<i32> = (0..16).collect();
    // Turn 2: 24 tokens (turn1 + assistant reply + second user msg tokens).
    let turn2_tokens: Vec<i32> = (0..24).collect();
    // Turn 3: 32 tokens.
    let turn3_tokens: Vec<i32> = (0..32).collect();

    // For each turn, compute a prefix-scoped key over the first 16 tokens
    // (the shared prefix) and confirm it stays the same across turns.
    let prefix_key_turn1 =
        PromptCacheKey::new_prefix("m", None, "tpl", None, mm, &turn1_tokens, 16);
    let prefix_key_turn2 =
        PromptCacheKey::new_prefix("m", None, "tpl", None, mm, &turn2_tokens, 16);
    let prefix_key_turn3 =
        PromptCacheKey::new_prefix("m", None, "tpl", None, mm, &turn3_tokens, 16);

    assert_eq!(
        prefix_key_turn1.digest(),
        prefix_key_turn2.digest(),
        "shared prefix digest must be stable turn 1→2"
    );
    assert_eq!(
        prefix_key_turn2.digest(),
        prefix_key_turn3.digest(),
        "shared prefix digest must be stable turn 2→3"
    );

    // The full-turn keys must differ because the token sequences differ.
    let full_key_turn1 = PromptCacheKey::new_full("m", None, "tpl", None, mm, &turn1_tokens);
    let full_key_turn2 = PromptCacheKey::new_full("m", None, "tpl", None, mm, &turn2_tokens);
    assert_ne!(
        full_key_turn1.digest(),
        full_key_turn2.digest(),
        "full-turn keys must differ when new tokens are added"
    );
}
