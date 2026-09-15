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

async function writeEvidence(testInfo: TestInfo, name: string, page: Page, evidence: Evidence): Promise<void> {
  const browser = await page.evaluate(() => ({
    userAgent: navigator.userAgent,
    language: navigator.language,
    hardwareConcurrency: navigator.hardwareConcurrency,
    devicePixelRatio: window.devicePixelRatio,
    viewport: { width: window.innerWidth, height: window.innerHeight },
  }));
  const path = testInfo.outputPath(name);
  await writeFile(path, JSON.stringify({
    measured_at: new Date().toISOString(),
    host: { hostname: hostname(), platform: platform(), release: release(), cpus: cpus().length, totalmem_bytes: totalmem() },
    project: testInfo.project.name,
    browser,
    ...evidence,
  }, null, 2), { mode: 0o600, flag: 'wx' });
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

async function actionToSettledPaintMs<T>(locator: Locator, action: () => Promise<void>, waitForState: () => Promise<T>, eventName: 'click' | 'input' = 'click'): Promise<{ elapsedMs: number; state: T }> {
  await locator.evaluate((element, event) => {
    const target = window as Window & { __mlxcelPerfStart?: Promise<number> };
    target.__mlxcelPerfStart = new Promise<number>((resolve) => {
      element.addEventListener(event, () => resolve(performance.now()), { once: true, capture: true });
    });
  }, eventName);
  await action();
  const startTime = await locator.page().evaluate(() => {
    const target = window as Window & { __mlxcelPerfStart?: Promise<number> };
    if (!target.__mlxcelPerfStart) throw new Error('Performance action listener was not installed.');
    return target.__mlxcelPerfStart;
  });
  const state = await waitForState();
  const elapsedMs = await locator.page().evaluate((started) => new Promise<number>((resolve) => {
    requestAnimationFrame(() => requestAnimationFrame(() => resolve(performance.now() - started)));
  }), startTime);
  return { elapsedMs, state };
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
  const activityTiming = await actionToSettledPaintMs(activityNav, () => activityNav.click(), async () => {
    await expect(page.getByTestId('activity-page')).toBeVisible();
    return { activity_visible: true };
  });
  const feedbackMs = activityTiming.elapsedMs;
  const cls = await page.evaluate(() => (window as Window & { __mlxcelCls?: number }).__mlxcelCls ?? 0);
  await writeEvidence(testInfo, 'feedback-status-layout-evidence.json', page, { dataset: 'slow catalog status transition plus browser event-to-paint Activity navigation; route fixtures only', feedback_event_to_activity_visible_paint_ms: feedbackMs, feedback_end_state: activityTiming.state, feedback_budget_ms: 100, cumulative_layout_shift: layoutShift ? cls : null, layout_shift_status: layoutShift ? 'measured' : 'not-run-unsupported', validated_schemas: api.validated });
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
  const searchTiming = await actionToSettledPaintMs(search, () => search.fill('0999'), async () => {
    await expect(page.locator('[data-testid="models-table"] tbody tr')).toHaveCount(1);
    return { filtered_rows: 1 };
  }, 'input');
  const searchMs = searchTiming.elapsedMs;
  await writeEvidence(testInfo, 'catalog-1000-evidence.json', page, { dataset: '1000 schema-validated catalog entries over five 200-entry canonical pages', load_to_table_wall_ms: loadedMs, search_event_to_filtered_table_paint_ms: searchMs, search_end_state: searchTiming.state, feedback_budget_ms: 100, rendered_rows: 25, pages_expected: 5, validated_schemas: api.validated });
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
  const newConversation = page.getByRole('button', { name: 'New conversation', exact: true });
  const conversationTiming = await actionToSettledPaintMs(newConversation, () => newConversation.click(), async () => {
    await expect(composer).toBeEnabled();
    await expect(composer).toHaveValue('');
    return { composer_empty: true };
  });
  const feedbackMs = conversationTiming.elapsedMs;
  await writeEvidence(testInfo, 'transcript-10000-evidence.json', page, { dataset: '10000 repeated token streamed transcript fixture; no real inference backend', render_wall_ms: renderMs, post_render_event_to_empty_composer_paint_ms: feedbackMs, post_render_end_state: conversationTiming.state, feedback_budget_ms: 100, validated_schemas: api.validated });
  expect(feedbackMs).toBeLessThanOrEqual(100);
});
