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

// Shared pure policy tests run in the same offline acceptance gate.
import { performanceMode, assertGeometry, performanceCompletion } from './activity-performance-config.mjs';
test('performance default remains headed and includes actual hidden mode', () => {
  assert.deepEqual(performanceMode(), { name: 'full', headless: false, modes: ['one-visible', 'two-visible', 'hidden'] });
  assert.throws(() => performanceMode('headless'), /WEBUI_PERF_MODE/);
});
test('zero or nonfinite viewport fails before inference', () => {
  const valid = { innerWidth: 700, innerHeight: 900, visualWidth: 700, visualHeight: 900 };
  assert.doesNotThrow(() => assertGeometry(valid));
  for (const key of Object.keys(valid)) for (const value of [0, -1, NaN, Infinity, undefined]) assert.throws(() => assertGeometry({ ...valid, [key]: value }), /no usable/);
});
test('headed visible-only keeps the hidden acceptance unclaimed', () => {
  const mode = performanceMode('visible-only-headed');
  assert.equal(mode.headless, false);
  assert.deepEqual(mode.modes, ['one-visible', 'two-visible']);
  for (const status of ['within-target', 'investigate']) {
    const result = performanceCompletion(mode, [{ status }]);
    assert.equal(result.status, 'incomplete');
    assert.equal(result.hidden_native, 'not-run');
    assert.match(result.limitation, /native hidden acceptance is not run/);
  }
  assert.throws(() => performanceMode('visible-only'), /WEBUI_PERF_MODE/);
});
test('visible-only diagnostic never reports complete acceptance', () => {
  const mode = performanceMode('visible-only-headless');
  assert.deepEqual(mode.modes, ['one-visible', 'two-visible']);
  for (const status of ['within-target', 'investigate']) {
    const result = performanceCompletion(mode, [{ status }]);
    assert.equal(result.status, 'incomplete');
    assert.equal(result.hidden_native, 'not-run');
  }
  assert.equal(performanceCompletion(performanceMode(), [{ status: 'investigate' }]).status, 'investigate');
});
