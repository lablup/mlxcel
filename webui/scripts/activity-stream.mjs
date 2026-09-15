// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
/* global TextDecoder */
// Acceptance-only SSE validation. Keep at most one bounded event, never output text.
export async function validateChatStream(reader) {
  const decoder = new TextDecoder('utf-8', { fatal: true });
  let buffer = '';
  let finished = false;
  let terminated = false;
  let events = 0;
  const limit = 1024 * 1024;
  function event(frame) {
    const lines = frame.split(/\r?\n/);
    const data = lines.filter((line) => line.startsWith('data:')).map((line) => line.slice(5).replace(/^ /, '')).join('\n');
    if (lines.some((line) => /^event: ?error$/.test(line))) throw new Error('Chat stream reported an error event.');
    if (!data) return;
    if (terminated) throw new Error('Chat stream sent data after [DONE].');
    if (data === '[DONE]') {
      if (!finished) throw new Error('Chat stream ended without a successful finish reason.');
      terminated = true;
      return;
    }
    let value;
    try { value = JSON.parse(data); } catch { throw new Error('Chat stream contained malformed JSON.'); }
    if (value === null || typeof value !== 'object' || Array.isArray(value) || 'error' in value) throw new Error('Chat stream reported an invalid or error payload.');
    if (!Array.isArray(value.choices)) throw new Error('Chat stream omitted choices.');
    for (const choice of value.choices) {
      if (finished) throw new Error('Chat stream sent a choice after its finish reason.');
      if (choice === null || typeof choice !== 'object' || !Number.isInteger(choice.index) || choice.index !== 0) throw new Error('Chat stream contained an invalid choice.');
      if (choice.finish_reason != null) {
        if (!['stop', 'length', 'tool_calls', 'function_call'].includes(choice.finish_reason)) throw new Error('Chat stream reported an unsuccessful finish reason.');
        finished = true;
      }
    }
    events += 1;
  }
  function consume(text) {
    // Check between frames too: a large read must not bypass the event bound.
    for (const part of text.split(/(\n)/)) {
      buffer += part;
      if (buffer.length > limit) throw new Error('Chat stream event exceeded the acceptance bound.');
      if (/\r?\n\r?\n$/.test(buffer)) { event(buffer); buffer = ''; }
    }
  }
  try {
    for (;;) {
      const chunk = await reader.read();
      if (chunk.done) break;
      consume(decoder.decode(chunk.value, { stream: true }));
    }
    consume(decoder.decode());
    if (buffer.trim() !== '' || !terminated) throw new Error('Chat stream was truncated or omitted [DONE].');
    return { complete: true, events };
  } catch (error) {
    // Decoder/transport errors may include arbitrary details; never retain them.
    if (error instanceof Error && error.message.startsWith('Chat stream ')) throw error;
    // eslint-disable-next-line preserve-caught-error -- Transport causes can disclose private request details.
    throw new Error('Chat stream transport or UTF-8 decoding failed.');
  }
}
