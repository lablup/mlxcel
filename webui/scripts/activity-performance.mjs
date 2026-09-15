// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
/* global URL, process, setTimeout, fetch, document, AbortSignal, window */
// Root-owned real acceptance. Never run alongside unrelated GPU inference/tests.
import { readFile, writeFile } from 'node:fs/promises';
import { chromium } from '@playwright/test';
import { performanceMode, assertGeometry, performanceCompletion } from './activity-performance-config.mjs';

const base = new URL(process.env.WEBUI_PERF_BASE ?? 'http://127.0.0.1:8080/');
if (!['127.0.0.1', '[::1]', 'localhost'].includes(base.hostname) || base.username || base.password || base.search || base.hash) throw new Error('Use a credential-free loopback API base (including any API prefix).');
if (!base.pathname.endsWith('/')) base.pathname += '/';
const key = (await readFile(required('WEBUI_PERF_KEY_FILE'), 'utf8')).trim();
const model = required('WEBUI_PERF_INFERENCE_MODEL');
const modelId = required('WEBUI_PERF_MODEL_ID');
const output = required('WEBUI_PERF_OUTPUT');
const prompt = process.env.WEBUI_PERF_PROMPT_FILE ? await readFile(process.env.WEBUI_PERF_PROMPT_FILE, 'utf8') : 'Explain the difference between a process and a thread, in detail. '.repeat(100);
const modeConfig = performanceMode(process.env.WEBUI_PERF_MODE);
const browser = await chromium.launch({ headless: modeConfig.headless });
const preflight = [];
const results = [];
let uiRequests = 0;
let contexts = [];
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const api = (path) => new URL(path, base).href;
async function closeClients() { await Promise.all(contexts.map((context) => context.close())); contexts = []; }
async function clients(mode) {
  await closeClients();
  if (mode === 'off') return;
  for (let i = 0; i < (mode === 'two-visible' ? 2 : 1); i++) {
    const context = await browser.newContext({ viewport: { width: 700, height: 900 } }); contexts.push(context);
    const page = await context.newPage();
    // Headless visible diagnostics use the default shell, without native windows.
    let cdp = null;
    let windowId;
    if (!modeConfig.headless) {
      cdp = await context.newCDPSession(page);
      ({ windowId } = await cdp.send('Browser.getWindowForTarget'));
      await cdp.send('Browser.setWindowBounds', { windowId, bounds: { windowState: 'normal', left: 20 + i * 710, top: 20, width: 700, height: 1000 } });
    }
    page.on('request', (request) => { if (request.url().includes('/ui-api/')) uiRequests++; });
    await page.goto(api('webui/#activity'));
    const geometry = await page.evaluate(() => ({ innerWidth: window.innerWidth, innerHeight: window.innerHeight, visualWidth: window.visualViewport?.width ?? 0, visualHeight: window.visualViewport?.height ?? 0 }));
    preflight.push({ mode, client: i, geometry });
    assertGeometry(geometry);
    await page.locator('input[type="password"]').fill(key);
    await page.locator('form button[type="submit"]').click();
    await page.getByTestId('activity-page').waitFor();
    const picker = page.getByRole('combobox');
    await picker.click();
    // Labels come from the same canonical catalog; never interpolate a repo into paths.
    const catalog = await fetch(api('ui-api/v1/catalog?limit=200'), { headers: { Authorization: `Bearer ${key}` } }).then((r) => r.json());
    const entry = catalog.items.find((item) => item.identity.id === modelId);
    if (!entry) throw new Error('Selected model must be in the first canonical 200-entry page for this bounded harness.');
    await page.getByRole('option', { name: entry.identity.display_name, exact: true }).click();
    if (mode === 'hidden') {
      if (cdp === null) throw new Error('Native hidden acceptance requires a headed browser.');
      await cdp.send('Browser.setWindowBounds', { windowId, bounds: { windowState: 'minimized' } });
    }
    await sleep(500);
    if ((await page.evaluate(() => document.hidden)) !== (mode === 'hidden')) throw new Error(`Actual document visibility did not match ${mode}; do not substitute a synthetic event.`);
  }
  // Recheck every client after creating all windows, not just the newest one.
  for (const context of contexts) {
    for (const page of context.pages()) {
      if ((await page.evaluate(() => document.hidden)) !== (mode === 'hidden')) throw new Error(`Actual concurrent visibility did not match ${mode}.`);
    }
  }
  await sleep(2500);
  if (mode === 'hidden') {
    const before = uiRequests; await sleep(4500);
    if (uiRequests !== before) throw new Error('Hidden Activity continued sending observation requests.');
  }
}
async function request(mode, pair) {
  const response = await fetch(api('completion?autoload=false'), { method: 'POST', headers: { Authorization: `Bearer ${key}`, 'Content-Type': 'application/json' }, body: JSON.stringify({ model, prompt, n_predict: 256, seed: 42, temperature: 0, cache_prompt: false, stream: false }), signal: AbortSignal.timeout(600000) });
  if (!response.ok) throw new Error(`Native completion failed (${response.status}); no response body logged.`);
  const body = await response.json();
  const rate = body.timings?.predicted_per_second;
  if (!(typeof rate === 'number' && Number.isFinite(rate) && rate > 0)) throw new Error('No valid backend predicted_per_second; do not replace with wall-clock or word-count inference.');
  const sample = { mode, pair, tokens_per_second: rate, predicted_tokens: body.timings.predicted_n, predicted_ms: body.timings.predicted_ms };
  results.push(sample); process.stdout.write(`${JSON.stringify(sample)}\n`);
  await writeFile(output, JSON.stringify({ status: 'in-progress', scope: modeConfig.name, hidden_native: modeConfig.headless ? 'not-run' : 'pending', preflight, samples: results }, null, 2));
  return rate;
}
function median(values) { const sorted = [...values].sort((a, b) => a - b); return sorted[Math.floor(sorted.length / 2)]; }
try {
  // Validate geometry and actual visibility for every requested condition before
  // even the GPU warmup. Never synthesize visibility or bypass failed clicks.
  for (const condition of modeConfig.modes) await clients(condition);
  await clients('off'); await request('warmup', -1);
  const summaries = [];
  for (const mode of modeConfig.modes) {
    const paired = []; const baseline = [];
    for (let pair = 0; pair < 5; pair++) {
      const rates = {};
      for (const condition of pair % 2 === 0 ? ['off', mode] : [mode, 'off']) { await clients(condition); rates[condition] = await request(condition, pair); }
      baseline.push(rates.off); paired.push((rates.off - rates[mode]) / rates.off * 100);
    }
    const mean = baseline.reduce((a, b) => a + b, 0) / baseline.length;
    const cv = Math.sqrt(baseline.reduce((sum, value) => sum + (value - mean) ** 2, 0) / baseline.length) / mean * 100;
    const degradation = median(paired);
    summaries.push({ mode, paired_runs: 5, median_decode_degradation_percent: degradation, baseline_cv_percent: cv, paired_range_percent: [Math.min(...paired), Math.max(...paired)], status: degradation > 2 || cv > 5 ? 'investigate' : 'within-target' });
  }
  await writeFile(output, JSON.stringify({ ...performanceCompletion(modeConfig, summaries), preflight, source: 'native completion backend timings.predicted_per_second; no browser or word-count estimate', samples: results, summaries }, null, 2));
} catch (error) {
  await writeFile(output, JSON.stringify({ status: 'failed', scope: modeConfig.name, hidden_native: modeConfig.headless ? 'not-run' : 'not-completed', preflight, samples: results, error: error instanceof Error ? error.message.replaceAll(key, '[redacted]') : 'unknown' }, null, 2));
  throw error;
} finally { await closeClients(); await browser.close(); }
function required(name) { const value = process.env[name]; if (!value) throw new Error(`${name} is required`); return value; }
