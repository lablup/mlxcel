// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
/* global process, URL, fetch, setTimeout, AbortSignal */
// Root-owned actual long-request observation; never logs prompt/output/key.
import { readFile, writeFile } from 'node:fs/promises';
const base = new URL(process.env.WEBUI_PERF_BASE ?? 'http://127.0.0.1:8080/');
if (!['127.0.0.1', '[::1]', 'localhost'].includes(base.hostname) || base.username || base.password || base.search || base.hash) throw new Error('Use a credential-free loopback API base.');
if (!base.pathname.endsWith('/')) base.pathname += '/';
const key = (await readFile(required('WEBUI_PERF_KEY_FILE'), 'utf8')).trim();
const prompt = await readFile(required('WEBUI_PERF_PROMPT_FILE'), 'utf8');
const model = required('WEBUI_PERF_INFERENCE_MODEL');
const modelId = required('WEBUI_PERF_MODEL_ID');
const output = required('WEBUI_SLOTS_OUTPUT');
const headers = { Authorization: `Bearer ${key}`, 'Content-Type': 'application/json' };
const samples = [];
let done = false;
let failure = null;
const url = (path, query = {}) => { const value = new URL(path, base); for (const [k, v] of Object.entries(query)) value.searchParams.set(k, v); return value; };
async function json(path, query) { const response = await fetch(url(path, query), { headers, signal: AbortSignal.timeout(10000) }); if (!response.ok) throw new Error(`Observation ${path} failed ${response.status}`); return response.json(); }
// Plain slots are projected to a whitelist; debug params, prompt and generated
// payloads can never enter this evidence file.
const slots = (items) => items.map((slot) => ({ id: slot.id, n_ctx: slot.n_ctx, processing: slot.is_processing, context_tokens: slot.n_prompt_tokens ?? null, decoded_tokens: slot.next_token?.[0]?.n_decoded ?? null }));
const request = (async () => {
  try {
    const response = await fetch(url('v1/chat/completions', { autoload: 'false' }), { method: 'POST', headers, body: JSON.stringify({ model, messages: [{ role: 'user', content: prompt }], max_tokens: 256, stream: true, temperature: 0, seed: 42 }), signal: AbortSignal.timeout(600000) });
    if (!response.ok || !response.body) throw new Error(`Real chat request failed ${response.status}`);
    const reader = response.body.getReader();
    try { while (!(await reader.read()).done) { /* drain, never retain generated text */ } } finally { reader.releaseLock(); }
  } catch (error) { failure = error instanceof Error ? error.message : 'request failed'; } finally { done = true; }
})();
try {
  while (!done) {
    const before = slots(await json('slots', { model, autoload: 'false' }));
    const runtime = await json('ui-api/v1/runtime', { model_id: modelId, autoload: 'false' });
    const after = slots(await json('slots', { model, autoload: 'false' }));
    samples.push({ received_at: new Date().toISOString(), before, runtime: runtime.slots, after });
    if (samples.length > 300) throw new Error('Long request exceeded bounded ten-minute observation window.');
    await writeFile(output, JSON.stringify({ status: 'in-progress', samples }, null, 2));
    await new Promise((resolve) => setTimeout(resolve, 2000));
  }
  await request;
  if (failure) throw new Error(failure);
  if (!samples.some((sample) => sample.runtime.items.some((slot) => slot.processing))) throw new Error('No active slot observed; use a sufficiently long actual prompt, do not mark this accepted.');
  if (!samples.some((sample) => sample.runtime.items.some((slot) => slot.decoded_tokens > 0))) throw new Error('No decode observation captured.');
  await writeFile(output, JSON.stringify({ status: 'captured-requires-comparison', source: 'actual chat stream with sequential native-before/runtime/native-after samples; compare within each bracket, not byte-identical concurrent claims', samples }, null, 2));
} catch (error) { await writeFile(output, JSON.stringify({ status: 'failed', error: error instanceof Error ? error.message.replaceAll(key, '[redacted]') : 'unknown', samples }, null, 2)); throw error; }
function required(name) { const value = process.env[name]; if (!value) throw new Error(`${name} is required`); return value; }
