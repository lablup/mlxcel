# 기술 보고서: 이슈 #2122 - /v1/responses 입력에서 type 없는 item과 output_text part 허용

**작성일**: 2026-10-06

**상태**: GB10에서 구현 및 검증 완료, 머지 대기.

**언어**: Rust (Responses 요청 타입, 입력 translator, 테스트), Markdown

**위험도**: 낮음 (이전에 422나 400이던 입력만 새로 허용한다. type이 있는 입력은 기존과 똑같이 파싱되고 변환된다)

## 요약

클라이언트가 2번째 턴에서 `response.output`을 다시 보낼 때 `{"role":"assistant","content":[{"type":"output_text","text":"Paris."}]}` 형태로, item에 `type` 없이 보낸다. `/v1/responses`는 422 `data did not match any variant of untagged enum ResponseInput`을 반환했다. 서로 독립된 두 결함이 원인이며, 하나만 고쳐서는 이 요청이 여전히 실패한다.

## 1. `type`이 없는 item

`ResponseInputItem`이 `#[serde(tag = "type")]`였기 때문에 EasyInputMessage는 어떤 variant와도 맞지 않았다. 이제 `serde_json::Value`를 거쳐 역직렬화한다. `type`이 없고 `role`이 있는 객체는 `message`로 읽고, 그 외 type 없는 형태는 기존의 `type` 누락 오류를 유지한다. serde는 같은 타입의 수동 구현에서 derive 구현을 호출할 수 없으므로, tagged 파싱은 비공개 미러 enum(`TaggedInputItem`)으로 옮겼다.

## 2. output_text 및 refusal part

두 part는 `ResponseInputPart::Unknown`으로 역직렬화된 뒤 `ContentPart::try_from`에서 거부되었다. `ResponseInputPart`에 `OutputText { text }`와 `Refusal { refusal }`을 추가하고 `ContentPart::Text`로 변환한다. translator는 user, system, developer 메시지에서 이 part를 기존의 "input part type '...' is not supported by this server" 오류로 거부하므로 assistant 턴만 가질 수 있다. `ResponseInputContent::as_text`, 저장소 크기 추정, function-call-output 변환도 새 variant를 처리한다.

## 검증

- `responses_input_parts_tests`: type 없는 message가 `Message`로 파싱되고 `{"content":"x"}`는 여전히 실패한다. type이 있는 assistant `output_text`와 `refusal`은 assistant 텍스트로 변환된다. user, system, developer의 `output_text`는 오류다.
- `reasoning_echo::tests::endpoint::responses_accepts_untyped_items_and_output_text_parts`: 3개 메시지 요청이 200을 반환하고 렌더된 프롬프트에 assistant 턴으로 `Paris.`가 들어 있다.
- GB10, `jamba-v0.1-4bit` (AI21-Jamba-Reasoning-3B): 같은 요청이 200과 독일/베를린 답변을 반환한다.
