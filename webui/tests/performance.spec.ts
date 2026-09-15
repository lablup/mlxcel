// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import { writeFile } from 'node:fs/promises';
import { cpus, hostname, platform, release, totalmem } from 'node:os';
import { expect, test, type Locator, type Page, type TestInfo } from '@playwright/test';
import type { CatalogEntry } from '../src/api/types';
import { bootProduct, productVariants, submitSessionKey } from './browser-fixtures';
import { loadValidator } from './models-fixtures';
import { bootstrapResponse, catalogEntry, catalogPage, operationPage, runtimeFor } from './performance-fixtures';

type Evidence = Record<string, unknown>;
type Validator = Awaited<ReturnType<typeof loadValidator>>;
type BrowserPaintPredicate =
  | { kind: 'visible-test-id'; testId: string }
  | { kind: 'table-row-count'; testId: string; rowCount: number; textIncludes?: string }
  | { kind: 'transcript-reset'; composerSelector: string };
type BrowserPaintResult<T> = { elapsedMs: number; state: T; instrumentation: Record<string, unknown> };
const CONSOLE_NUMERIC_EVIDENCE_KEYS = new Set([
  'usable_wall_ms',
  'feedback_event_to_activity_visible_paint_ms',
  'search_event_to_filtered_table_paint_ms',
  'post_render_event_to_empty_composer_paint_ms',
  'event_to_predicate_ms',
  'predicate_to_paint_ms',
  'before_action_two_raf_calibration_ms',
  'driver_assertion_wall_ms',
  'feedback_budget_ms',
  'budget_ms',
]);

function collectConsoleMetrics(value: unknown, output: Record<string, number>, prefix = ''): void {
  if (typeof value !== 'object' || value === null) return;
  for (const [key, item] of Object.entries(value)) {
    const label = prefix ? `${prefix}.${key}` : key;
    if (typeof item === 'number' && Number.isFinite(item) && CONSOLE_NUMERIC_EVIDENCE_KEYS.has(key)) output[label] = item;
    else if (typeof item === 'object' && item !== null) collectConsoleMetrics(item, output, label);
  }
}

async function writeEvidence(testInfo: TestInfo, name: string, page: Page, evidence: Evidence): Promise<void> {
  const browser = await page.evaluate(() => ({
    userAgent: navigator.userAgent,
    language: navigator.language,
    hardwareConcurrency: navigator.hardwareConcurrency,
    devicePixelRatio: window.devicePixelRatio,
    viewport: { width: window.innerWidth, height: window.innerHeight },
  }));
  const path = testInfo.outputPath(name);
  const payload = {
    measured_at: new Date().toISOString(),
    host: { hostname: hostname(), platform: platform(), release: release(), cpus: cpus().length, totalmem_bytes: totalmem() },
    project: testInfo.project.name,
    // Retry index of the attempt that produced these numbers: 0 is the first run. A non-zero
    // value means an earlier attempt exceeded a budget and its own evidence file was kept.
    attempt: testInfo.retry,
    browser,
    ...evidence,
  };
  const metrics: Record<string, number> = {};
  collectConsoleMetrics(evidence, metrics);
  console.log(JSON.stringify({ kind: 'webui-performance-evidence', name, project: testInfo.project.name, attempt: testInfo.retry, metrics }));
  await writeFile(path, JSON.stringify(payload, null, 2), { mode: 0o600, flag: 'wx' });
  await testInfo.attach(name, { path, contentType: 'application/json' });
}

async function installPerformanceApi(page: Page, options: { catalog: readonly CatalogEntry[]; slowCatalog?: boolean; chatEntry?: CatalogEntry }): Promise<{ releaseCatalog: () => void; validated: string[] }> {
  const validator = await loadValidator();
  const validated: string[] = [];
  let released = !options.slowCatalog;
  let releaseCatalog = (): void => { released = true; };
  let releasePromise = Promise.resolve();
  if (options.slowCatalog) releasePromise = new Promise<void>((resolve) => { releaseCatalog = () => { released = true; resolve(); }; });
  const validate = (schema: Parameters<Validator['validateAgainstSchema']>[0], body: unknown): void => {
    validator.validateAgainstSchema(schema, body);
    validated.push(schema);
  };
  await page.route('**/*', async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    const path = url.pathname;
    if (!path.startsWith('/ui-api/v1/') && path !== '/v1/chat/completions') {
      await route.continue();
      return;
    }
    if (path === '/ui-api/v1/bootstrap') {
      const body = bootstrapResponse();
      validate('BootstrapResponse', body);
      await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(body) });
      return;
    }
    if (path === '/ui-api/v1/catalog') {
      if (!released) await releasePromise;
      const limit = Math.min(Math.max(Number(url.searchParams.get('limit') ?? '200') || 200, 1), 200);
      const cursor = url.searchParams.get('cursor');
      const offset = cursor === null ? 0 : Number(cursor.replace(/^cursor_/, ''));
      const body = catalogPage(options.catalog, Number.isFinite(offset) ? offset : 0, limit, 3000, options.catalog.length);
      validate('CatalogListResponse', body);
      await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(body) });
      return;
    }
    if (path === '/ui-api/v1/operations') {
      const body = operationPage(3000);
      validate('OperationsListResponse', body);
      await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(body) });
      return;
    }
    if (path === '/ui-api/v1/runtime') {
      const entry = options.chatEntry ?? options.catalog[0];
      const body = runtimeFor(entry, 3000);
      validate('RuntimeSnapshot', body);
      await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(body) });
      return;
    }
    if (path === '/ui-api/v1/events') {
      await route.fulfill({ status: 200, contentType: 'text/event-stream', body: 'data: [DONE]\n\n' });
      return;
    }
    if (path === '/v1/chat/completions') {
      const content = 'token '.repeat(10_000);
      const frames = [
        { choices: [{ index: 0, delta: { content }, finish_reason: null }] },
        { choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] },
        { choices: [], usage: { prompt_tokens: 4, completion_tokens: 10_000, total_tokens: 10_004 } },
      ];
      await route.fulfill({ status: 200, contentType: 'text/event-stream', body: `${frames.map((frame) => `data: ${JSON.stringify(frame)}\n\n`).join('')}data: [DONE]\n\n` });
      return;
    }
    await route.fulfill({ status: 404, contentType: 'application/json', body: JSON.stringify({ error: { code: 'not_found', message: `Unexpected ${path}`, retryable: false }, request_id: 'req_perf_missing' }) });
  });
  return { releaseCatalog, validated };
}

async function actionToSettledPaintMs<T>(locator: Locator, action: () => Promise<void>, predicate: BrowserPaintPredicate, assertState: () => Promise<T>, eventName: 'click' | 'input' = 'click'): Promise<BrowserPaintResult<T>> {
  const page = locator.page();
  const twoRafCalibrationMs = await page.evaluate(() => new Promise<number>((resolve) => {
    const started = performance.now();
    requestAnimationFrame(() => requestAnimationFrame(() => resolve(performance.now() - started)));
  }));
  await locator.evaluate((element, options) => {
    type Predicate = typeof options.predicate;
    type TimedResult = { elapsedMs: number; predicateElapsedMs: number; paintAfterPredicateMs: number; browserState: Record<string, unknown>; instrumentation: Record<string, unknown> };
    type TimedWindow = Window & { __mlxcelPerfActionToPaint?: { promise: Promise<TimedResult>; cancel: (reason: string) => void } };
    const target = window as TimedWindow;
    const visible = (candidate: Element | null): candidate is HTMLElement => {
      if (!(candidate instanceof HTMLElement)) return false;
      const style = window.getComputedStyle(candidate);
      const rect = candidate.getBoundingClientRect();
      return !candidate.hidden && style.display !== 'none' && style.visibility !== 'hidden' && rect.width > 0 && rect.height > 0;
    };
    const byTestId = (testId: string): Element | null => document.querySelector(`[data-testid="${testId}"]`);
    const evaluatePredicate = (current: Predicate): Record<string, unknown> | null => {
      if (current.kind === 'visible-test-id') return visible(byTestId(current.testId)) ? { [`${current.testId.replace(/-/gu, '_')}_visible`]: true } : null;
      if (current.kind === 'table-row-count') {
        const table = byTestId(current.testId);
        const rows = Array.from(table?.querySelectorAll('tbody tr') ?? []);
        if (rows.length !== current.rowCount) return null;
        const requiredText = current.textIncludes;
        if (requiredText && !rows.some((row) => row.textContent?.includes(requiredText))) return null;
        return { filtered_rows: rows.length };
      }
      const textarea = document.querySelector(current.composerSelector);
      const transcript = document.querySelector('section[aria-label="Conversation transcript"]');
      const hasTurns = (transcript?.querySelectorAll('article.chat-turn').length ?? 0) > 0;
      const emptyTextVisible = visible(transcript?.querySelector('.chat-transcript p') ?? null) && transcript?.textContent?.includes('No messages yet.') === true;
      if (textarea instanceof HTMLTextAreaElement && !textarea.disabled && textarea.value === '' && !hasTurns && emptyTextVisible) return { composer_empty: true, transcript_empty: true };
      return null;
    };
    if (evaluatePredicate(options.predicate) !== null) throw new Error(`Performance predicate ${options.predicate.kind} was already true before the action.`);
    let cancelMeasurement: (reason: string) => void = () => {};
    const promise = new Promise<TimedResult>((resolve, reject) => {
      let observer: MutationObserver | null = null;
      let raf = 0;
      let timeout: number | null = null;
      let phase: 'waiting-event' | 'checking' | 'painting' | 'done' = 'waiting-event';
      let removeListener: (() => void) | null = null;
      const cleanup = (): void => {
        phase = 'done';
        observer?.disconnect();
        if (raf) cancelAnimationFrame(raf);
        if (timeout !== null) window.clearTimeout(timeout);
        removeListener?.();
      };
      const fail = (error: Error): void => {
        cleanup();
        reject(error);
      };
      cancelMeasurement = (reason: string): void => fail(new Error(reason));
      const settleAfterPaint = (started: number, deadline: number, browserState: Record<string, unknown>): void => {
        phase = 'painting';
        observer?.disconnect();
        const predicateAt = performance.now();
        const assertInsideDeadline = (): boolean => {
          if (performance.now() <= deadline) return true;
          fail(new Error(`Timed out waiting for ${options.predicate.kind} paint.`));
          return false;
        };
        raf = requestAnimationFrame(() => {
          if (phase === 'done' || !assertInsideDeadline()) return;
          raf = requestAnimationFrame(() => {
            if (phase === 'done' || !assertInsideDeadline()) return;
            const elapsedMs = performance.now() - started;
            cleanup();
            resolve({
              elapsedMs,
              predicateElapsedMs: predicateAt - started,
              paintAfterPredicateMs: elapsedMs - (predicateAt - started),
              browserState,
              instrumentation: {
                clock: 'performance.now',
                start: 'capturing DOM event listener',
                end: 'semantic DOM predicate satisfied, then two requestAnimationFrame paints before the same deadline',
                excludes: 'Playwright assertion polling and driver IPC after the timed browser predicate',
              },
            });
          });
        });
      };
      const begin = (): void => {
        if (phase === 'done') return;
        phase = 'checking';
        if (timeout !== null) window.clearTimeout(timeout);
        const started = performance.now();
        const deadline = started + options.timeoutMs;
        timeout = window.setTimeout(() => fail(new Error(`Timed out waiting for ${options.predicate.kind} before painted semantic state.`)), options.timeoutMs);
        const check = (): void => {
          if (phase !== 'checking') return;
          try {
            const browserState = evaluatePredicate(options.predicate);
            if (browserState !== null) { settleAfterPaint(started, deadline, browserState); return; }
            if (performance.now() > deadline) {
              fail(new Error(`Timed out waiting for ${options.predicate.kind} before paint.`));
            }
          } catch (error) {
            fail(error instanceof Error ? error : new Error(String(error)));
          }
        };
        observer = new MutationObserver(check);
        observer.observe(document.documentElement, { subtree: true, childList: true, attributes: true, characterData: true });
        const tick = (): void => { check(); if (phase === 'checking') raf = requestAnimationFrame(tick); };
        tick();
      };
      element.addEventListener(options.eventName, begin, { once: true, capture: true });
      removeListener = () => element.removeEventListener(options.eventName, begin, { capture: true });
      timeout = window.setTimeout(() => fail(new Error(`Timed out waiting for ${options.eventName} event for ${options.predicate.kind}.`)), options.timeoutMs);
    });
    promise.catch(() => undefined);
    target.__mlxcelPerfActionToPaint = { promise, cancel: cancelMeasurement };
  }, { eventName, predicate, timeoutMs: 1000 });
  try {
    await action();
  } catch (error) {
    await page.evaluate(async () => {
      const target = window as Window & { __mlxcelPerfActionToPaint?: { promise: Promise<unknown>; cancel: (reason: string) => void } };
      const pending = target.__mlxcelPerfActionToPaint;
      delete target.__mlxcelPerfActionToPaint;
      if (pending) {
        pending.cancel('Playwright action failed before the browser measurement completed.');
        try { await pending.promise; } catch { /* consume the intentional cancellation */ }
      }
    });
    throw error;
  }
  const browserResult = await page.evaluate(() => {
    const target = window as Window & { __mlxcelPerfActionToPaint?: { promise: Promise<{ elapsedMs: number; predicateElapsedMs: number; paintAfterPredicateMs: number; browserState: Record<string, unknown>; instrumentation: Record<string, unknown> }>; cancel: (reason: string) => void } };
    const pending = target.__mlxcelPerfActionToPaint?.promise;
    delete target.__mlxcelPerfActionToPaint;
    if (!pending) throw new Error('Performance action listener was not installed.');
    return pending;
  });
  const driverStarted = process.hrtime.bigint();
  const state = await assertState();
  const driverAssertionMs = Number(process.hrtime.bigint() - driverStarted) / 1_000_000;
  return { elapsedMs: browserResult.elapsedMs, state, instrumentation: { ...browserResult.instrumentation, event_to_predicate_ms: browserResult.predicateElapsedMs, predicate_to_paint_ms: browserResult.paintAfterPredicateMs, before_action_two_raf_calibration_ms: twoRafCalibrationMs, browser_state: browserResult.browserState, driver_assertion_wall_ms: driverAssertionMs } };
}

test('cold product shell is usable within the frontend budget', async ({ page }, testInfo) => {
  await installPerformanceApi(page, { catalog: [catalogEntry(0)] });
  await page.setViewportSize({ width: 1440, height: 900 });
  const started = process.hrtime.bigint();
  await page.goto('/#models');
  await expect(page.getByTestId('auth-login')).toBeVisible();
  const usableMs = Number(process.hrtime.bigint() - started) / 1_000_000;
  const navigation = await page.evaluate(() => performance.getEntriesByType('navigation').map((entry) => {
    const nav = entry as PerformanceNavigationTiming;
    return { dom_content_loaded_ms: nav.domContentLoadedEventEnd, load_event_ms: nav.loadEventEnd, response_end_ms: nav.responseEnd };
  }));
  await writeEvidence(testInfo, 'cold-usable-evidence.json', page, { dataset: 'Vite-served frontend shell with UI API route fixtures; no Rust server, model load, or inference backend work', usable_wall_ms: usableMs, budget_ms: 2000, navigation });
  expect(usableMs).toBeLessThanOrEqual(2000);
});

test('navigation feedback and status updates stay responsive without layout shift', async ({ page }, testInfo) => {
  const api = await installPerformanceApi(page, { catalog: [catalogEntry(0)], slowCatalog: true });
  await bootProduct(page, { ...productVariants[0], signedIn: true });
  const layoutShift = await page.evaluate(() => {
    const win = window as Window & { __mlxcelCls?: number; __mlxcelClsSupported?: boolean };
    win.__mlxcelCls = 0;
    win.__mlxcelClsSupported = typeof PerformanceObserver !== 'undefined' && (PerformanceObserver.supportedEntryTypes ?? []).includes('layout-shift');
    if (win.__mlxcelClsSupported) {
      new PerformanceObserver((list) => {
        for (const entry of list.getEntries() as Array<PerformanceEntry & { value?: number; hadRecentInput?: boolean }>) {
          if (!entry.hadRecentInput) win.__mlxcelCls = (win.__mlxcelCls ?? 0) + (entry.value ?? 0);
        }
      }).observe({ type: 'layout-shift', buffered: true });
    }
    return win.__mlxcelClsSupported;
  });
  await submitSessionKey(page, 'performance-key');
  await expect(page.getByTestId('connection-authenticated-detail')).toContainText(/catalog pending|카탈로그 대기/i);
  api.releaseCatalog();
  await expect(page.getByTestId('models-table')).toBeVisible();
  const activityNav = page.locator('[data-testid="nav-activity"]:visible').first();
  const activityTiming = await actionToSettledPaintMs(activityNav, () => activityNav.click(), { kind: 'visible-test-id', testId: 'activity-page' }, async () => {
    await expect(page.getByTestId('activity-page')).toBeVisible();
    return { activity_visible: true };
  });
  const feedbackMs = activityTiming.elapsedMs;
  const cls = await page.evaluate(() => (window as Window & { __mlxcelCls?: number }).__mlxcelCls ?? 0);
  await writeEvidence(testInfo, 'feedback-status-layout-evidence.json', page, { dataset: 'slow catalog status transition plus browser event-to-semantic-DOM-to-paint Activity navigation; route fixtures only', feedback_event_to_activity_visible_paint_ms: feedbackMs, feedback_end_state: activityTiming.state, feedback_instrumentation: activityTiming.instrumentation, feedback_budget_ms: 100, cumulative_layout_shift: layoutShift ? cls : null, layout_shift_status: layoutShift ? 'measured' : 'not-run-unsupported', validated_schemas: api.validated });
  expect(feedbackMs).toBeLessThanOrEqual(100);
  if (layoutShift) expect(cls).toBeLessThanOrEqual(0.001);
});

test('1000-entry catalog remains searchable and bounded', async ({ page }, testInfo) => {
  const entries = Array.from({ length: 1000 }, (_, index) => catalogEntry(index));
  const api = await installPerformanceApi(page, { catalog: entries });
  await bootProduct(page, { ...productVariants[0], signedIn: true });
  const started = process.hrtime.bigint();
  await submitSessionKey(page, 'catalog-scale-key');
  await expect(page.getByTestId('models-table')).toBeVisible();
  await expect(page.locator('[data-testid="models-table"] tbody tr')).toHaveCount(25);
  const loadedMs = Number(process.hrtime.bigint() - started) / 1_000_000;
  const search = page.getByTestId('models-search');
  const searchTiming = await actionToSettledPaintMs(search, () => search.fill('0999'), { kind: 'table-row-count', testId: 'models-table', rowCount: 1, textIncludes: '0999' }, async () => {
    await expect(page.locator('[data-testid="models-table"] tbody tr')).toHaveCount(1);
    return { filtered_rows: 1 };
  }, 'input');
  const searchMs = searchTiming.elapsedMs;
  await writeEvidence(testInfo, 'catalog-1000-evidence.json', page, { dataset: '1000 schema-validated catalog entries over five 200-entry canonical pages', load_to_table_wall_ms: loadedMs, search_event_to_filtered_table_paint_ms: searchMs, search_end_state: searchTiming.state, search_instrumentation: searchTiming.instrumentation, feedback_budget_ms: 100, rendered_rows: 25, pages_expected: 5, validated_schemas: api.validated });
  expect(searchMs).toBeLessThanOrEqual(100);
});

test('10000-token transcript keeps post-render controls responsive', async ({ page }, testInfo) => {
  const ready = catalogEntry(0, 'ready');
  const api = await installPerformanceApi(page, { catalog: [ready], chatEntry: ready });
  await bootProduct(page, { ...productVariants[0], signedIn: true });
  await submitSessionKey(page, 'transcript-scale-key');
  await expect(page.getByTestId('models-table')).toBeVisible();
  const chatNav = page.locator('[data-testid="nav-chat"]:visible').first();
  await chatNav.click();
  await page.getByRole('combobox', { name: 'Model for next turn' }).click();
  await page.getByRole('option', { name: /Catalog performance model 0000 · ready/i }).click();
  const composer = page.getByRole('textbox', { name: 'Message', exact: true });
  await composer.fill('Produce a long deterministic transcript');
  const renderStarted = process.hrtime.bigint();
  await page.getByRole('button', { name: 'Send', exact: true }).click();
  await expect(page.getByRole('status').filter({ hasText: 'Response complete.' })).toBeVisible();
  const renderMs = Number(process.hrtime.bigint() - renderStarted) / 1_000_000;
  await expect(page.locator('section[aria-label="Conversation transcript"] article.chat-turn')).not.toHaveCount(0);
  const newConversation = page.getByRole('button', { name: 'New conversation', exact: true });
  const conversationTiming = await actionToSettledPaintMs(newConversation, () => newConversation.click(), { kind: 'transcript-reset', composerSelector: 'textarea[aria-label="Message"]' }, async () => {
    await expect(composer).toBeEnabled();
    await expect(composer).toHaveValue('');
    await expect(page.locator('section[aria-label="Conversation transcript"] article.chat-turn')).toHaveCount(0);
    await expect(page.locator('section[aria-label="Conversation transcript"]')).toContainText('No messages yet.');
    return { composer_empty: true, transcript_empty: true };
  });
  const feedbackMs = conversationTiming.elapsedMs;
  await writeEvidence(testInfo, 'transcript-10000-evidence.json', page, { dataset: '10000 repeated token streamed transcript fixture; no real inference backend', render_wall_ms: renderMs, post_render_event_to_empty_composer_paint_ms: feedbackMs, post_render_end_state: conversationTiming.state, post_render_instrumentation: conversationTiming.instrumentation, feedback_budget_ms: 100, validated_schemas: api.validated });
  expect(feedbackMs).toBeLessThanOrEqual(100);
});
