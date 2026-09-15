// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
/* global TextEncoder */
import test from 'node:test';
import assert from 'node:assert/strict';
import { validateChatStream } from './activity-stream.mjs';
const encode = (value) => new TextEncoder().encode(value);
const frame = (value) => `data: ${JSON.stringify(value)}\n\n`;
const choice = (finish_reason = null) => ({ choices: [{ index: 0, delta: { content: 'private output 한글' }, finish_reason }] });
const done = 'data: [DONE]\n\n';
async function parse(value, width = 7) {
  const bytes = encode(value); let offset = 0;
  return validateChatStream({ read: async () => { const value = bytes.slice(offset, offset += width); return { done: value.length === 0, value }; } });
}
test('accepts split UTF-8, CRLF, comments, usage and explicit completion without retaining text', async () => {
  const result = await parse((': heartbeat\n\n' + frame(choice()) + frame(choice('length')) + frame({ choices: [], usage: { completion_tokens: 256 } }) + done).replaceAll('\n', '\r\n'), 1);
  assert.deepEqual(result, { complete: true, events: 3 });
  assert.ok(!JSON.stringify(result).includes('private'));
});
for (const [name, value] of [
  ['error payload', frame(choice()) + frame({ error: { message: 'private server failure' } }) + done],
  ['named error', 'event: error\ndata: {}\n\n'],
  ['malformed JSON', 'data: {secret\n\n'],
  ['missing finish', frame(choice()) + done],
  ['missing DONE', frame(choice('stop'))],
  ['truncated final event', frame(choice('stop')) + 'data: [DONE]'],
  ['delta after finish', frame(choice('stop')) + frame(choice()) + done],
  ['duplicate finish', frame(choice('stop')) + frame(choice('stop')) + done],
  ['data after DONE', frame(choice('stop')) + done + frame(choice())],
  ['invalid choice', frame({ choices: [null] }) + done],
  ['error finish', frame(choice('error')) + done],
  ['oversized frame', 'data: ' + 'x'.repeat(1024 * 1024) + '\n\n'],
]) test(`rejects ${name} without disclosing payload`, async () => {
  await assert.rejects(parse(value, 65536), (error) => error.message.startsWith('Chat stream ') && !error.message.includes('private') && !error.message.includes('secret'));
});
test('rejects invalid UTF-8 with a fixed safe error', async () => {
  let sent = false;
  await assert.rejects(validateChatStream({ read: async () => ({ done: sent, value: sent ? undefined : (sent = true, new Uint8Array([255])) }) }), /UTF-8 decoding failed/);
});
