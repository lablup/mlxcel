// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import { cpus, hostname, platform, release, totalmem } from 'node:os';
import { expect, test, type Page, type TestInfo } from '@playwright/test';
import catalogFixture from '../../tests/fixtures/webui/examples/catalog.page.json' with { type: 'json' };
import bootstrapFixture from '../../tests/fixtures/webui/examples/bootstrap.model-free.json' with { type: 'json' };
import runtimeFixture from '../../tests/fixtures/webui/examples/runtime.snapshot.json' with { type: 'json' };
import type { CatalogEntry, CatalogListResponse } from '../src/api/types';
import { bootProduct, installMockApi, productVariants, submitSessionKey } from './browser-fixtures';
import { model } from './models-fixtures';

type Evidence = Record<string, unknown>;

async function attachEvidence(testInfo: TestInfo, name: string, page: Page, evidence: Evidence): Promise<void> {
  const browser = await page.evaluate(() => ({
    userAgent: navigator.userAgent,
    language: navigator.language,
    hardwareConcurrency: navigator.hardwareConcurrency,
    devicePixelRatio: window.devicePixelRatio,
    viewport: { width: window.innerWidth, height: window.innerHeight },
  }));
  await testInfo.attach(name, {
    body: JSON.stringify({
      measured_at: new Date().toISOString(),
      host: { hostname: hostname(), platform: platform(), release: release(), cpus: cpus().length, totalmem_bytes: totalmem() },
      project: testInfo.project.name,
      browser,
      ...evidence,
    }, null, 2),
    contentType: 'application/json',
  });
}

function largeCatalog(size: number): CatalogListResponse {
  const base = model();
  const items: CatalogEntry[] = Array.from({ length: size }, (_, index) => ({
    ...base,
    identity: {
      ...base.identity,
      id: `perf_catalog_${String(index).padStart(4, '0')}`,
      display_name: `Catalog performance model ${String(index).padStart(4, '0')}`,
      revision: base.identity.revision + index,
    },
  }));
  return {
    ...(structuredClone(catalogFixture) as CatalogListResponse),
    items,
    pagination: { limit: 200, next_cursor: null, total_known: size },
    server_instance_id: bootstrapFixture.server.server_instance_id,
    snapshot_sequence: 1000 + size,
  };
}

async function installLargeCatalog(page: Page, size: number): Promise<void> {
  await page.route('**/ui-api/v1/catalog*', async (route) => {
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(largeCatalog(size)) });
  });
}

async function installReadyChat(page: Page): Promise<void> {
  const base = model();
  const ready: CatalogEntry = {
    ...base,
    lifecycle: { ...base.lifecycle, state: 'ready', worker_exit_observed: false },
    capabilities: [{ task: 'chat', phase: 'provider_ready', available: true, reason: null }],
  };
  await page.route('**/ui-api/v1/catalog*', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        ...(structuredClone(catalogFixture) as CatalogListResponse),
        items: [ready],
        server_instance_id: bootstrapFixture.server.server_instance_id,
        snapshot_sequence: 2000,
      }),
    });
  });
  await page.route('**/ui-api/v1/runtime?**', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ ...runtimeFixture, server_instance_id: bootstrapFixture.server.server_instance_id, model_id: ready.identity.id, revision: ready.identity.revision, snapshot_sequence: 2000 }),
    });
  });
  await page.route('**/v1/chat/completions?autoload=false', async (route) => {
    const content = 'token '.repeat(10_000);
    const frames = [
      { choices: [{ index: 0, delta: { content }, finish_reason: null }] },
      { choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] },
      { choices: [], usage: { prompt_tokens: 4, completion_tokens: 10_000, total_tokens: 10_004 } },
    ];
    await route.fulfill({ status: 200, contentType: 'text/event-stream', body: `${frames.map((frame) => `data: ${JSON.stringify(frame)}\n\n`).join('')}data: [DONE]\n\n` });
  });
}

async function performanceNow(page: Page): Promise<number> {
  return page.evaluate(() => performance.now());
}

test('cold product shell is usable within the frontend budget', async ({ page }, testInfo) => {
  await installMockApi(page, 'happy');
  await page.setViewportSize({ width: 1440, height: 900 });
  const started = await performanceNow(page);
  await page.goto('/#models');
  await expect(page.getByTestId('auth-login')).toBeVisible();
  const usableMs = (await performanceNow(page)) - started;
  await attachEvidence(testInfo, 'cold-usable-evidence.json', page, { dataset: 'model-free public shell, backend excluded by route fixtures', usable_ms: usableMs, budget_ms: 2000 });
  expect(usableMs).toBeLessThanOrEqual(2000);
});

test('navigation feedback and status updates stay responsive without layout shift', async ({ page }, testInfo) => {
  const mock = await installMockApi(page, 'slow-catalog');
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
  mock.releaseCatalog();
  await expect(page.getByTestId('models-table')).toBeVisible();
  const before = await performanceNow(page);
  await page.getByTestId('nav-activity').click();
  await expect(page.getByTestId('activity-page')).toBeVisible();
  const feedbackMs = (await performanceNow(page)) - before;
  const cls = await page.evaluate(() => (window as Window & { __mlxcelCls?: number }).__mlxcelCls ?? 0);
  await attachEvidence(testInfo, 'feedback-status-layout-evidence.json', page, { dataset: 'slow catalog status transition plus client-side Activity navigation', feedback_ms: feedbackMs, feedback_budget_ms: 100, cumulative_layout_shift: cls, layout_shift_supported: layoutShift });
  expect(feedbackMs).toBeLessThanOrEqual(100);
  if (layoutShift) expect(cls).toBeLessThanOrEqual(0.001);
});

test('1000-entry catalog remains searchable and bounded', async ({ page }, testInfo) => {
  await installMockApi(page, 'happy');
  await installLargeCatalog(page, 1000);
  await bootProduct(page, { ...productVariants[0], signedIn: true });
  const started = await performanceNow(page);
  await submitSessionKey(page, 'catalog-scale-key');
  await expect(page.getByTestId('models-table')).toBeVisible();
  await expect(page.locator('[data-testid="models-table"] tbody tr')).toHaveCount(25);
  const loadedMs = (await performanceNow(page)) - started;
  const searchStarted = await performanceNow(page);
  await page.getByTestId('models-search').fill('0999');
  await expect(page.locator('[data-testid="models-table"] tbody tr')).toHaveCount(1);
  const searchMs = (await performanceNow(page)) - searchStarted;
  await attachEvidence(testInfo, 'catalog-1000-evidence.json', page, { dataset: '1000 synthetic catalog entries from canonical CatalogEntry fixture', load_to_table_ms: loadedMs, search_feedback_ms: searchMs, feedback_budget_ms: 100, rendered_rows: 25 });
  expect(searchMs).toBeLessThanOrEqual(100);
});

test('10000-token transcript keeps post-render controls responsive', async ({ page }, testInfo) => {
  await installMockApi(page, 'happy');
  await installReadyChat(page);
  await bootProduct(page, { ...productVariants[0], signedIn: true });
  await submitSessionKey(page, 'transcript-scale-key');
  await expect(page.getByTestId('models-table')).toBeVisible();
  await page.getByTestId('nav-chat').click();
  await page.getByRole('combobox', { name: 'Model for next turn' }).click();
  await page.getByRole('option', { name: /alpha · ready/i }).click();
  const composer = page.getByRole('textbox', { name: 'Message', exact: true });
  await composer.fill('Produce a long deterministic transcript');
  const renderStarted = await performanceNow(page);
  await page.getByRole('button', { name: 'Send', exact: true }).click();
  await expect(page.getByRole('status').filter({ hasText: 'Response complete.' })).toBeVisible();
  const renderMs = (await performanceNow(page)) - renderStarted;
  const feedbackStarted = await performanceNow(page);
  await page.getByRole('button', { name: 'New conversation', exact: true }).click();
  await expect(composer).toBeEnabled();
  const feedbackMs = (await performanceNow(page)) - feedbackStarted;
  await attachEvidence(testInfo, 'transcript-10000-evidence.json', page, { dataset: '10000 repeated token streamed transcript fixture, backend excluded', render_ms: renderMs, post_render_feedback_ms: feedbackMs, feedback_budget_ms: 100 });
  expect(feedbackMs).toBeLessThanOrEqual(100);
});
