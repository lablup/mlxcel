// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import { expect, test, type Page } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
import { model, bootstrap, loadValidator } from './models-fixtures';
import type { CatalogEntry, ModelActionRequest, Operation } from '../src/api/types';

// Deterministic API composition tests; these do not claim real download/inference acceptance.
async function installLibrary(page: Page, initial: CatalogEntry[] = []) {
  const { validateAgainstSchema } = await loadValidator();
  let entries = initial;
  let sequence = 10;
  let operation: Operation | null = null;
  const posts: { path: string; body: Record<string, unknown> }[] = [];
  const makeOperation = (kind: Operation['kind'], target: Operation['target']): Operation => ({
    operation_id: `op_${++sequence}`,
    kind,
    target,
    state: 'running',
    created_at: '2026-09-14T00:00:00Z',
    updated_at: '2026-09-14T00:00:00Z',
    idempotency_scope: 'server_instance',
    progress: { completed_bytes: 0, total_bytes: null, indeterminate: true },
    result: null,
    error: null,
    cancellable: kind === 'download',
    cancel_reason: null,
  });
  await page.route('**/ui-api/v1/**', async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    const path = url.pathname;
    const json = (body: unknown, status = 200) => {
      const schema =
        status >= 400
          ? 'ErrorEnvelope'
          : request.method() === 'POST'
            ? 'OperationAccepted'
            : path.endsWith('/bootstrap')
              ? 'BootstrapResponse'
              : path.endsWith('/catalog')
                ? 'CatalogListResponse'
                : path.endsWith('/operations')
                  ? 'OperationsListResponse'
                  : 'RuntimeSnapshot';
      validateAgainstSchema(schema, body);
      return route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) });
    };
    if (request.method() === 'POST') {
      const body = request.postDataJSON() as Record<string, unknown>;
      posts.push({ path, body });
      if (path.endsWith('/downloads'))
        operation = makeOperation('download', {
          target_kind: 'download',
          repo_id: String(body.repo_id),
          revision: null,
        });
      else if (path.endsWith('/model-actions')) {
        const action = body as unknown as ModelActionRequest;
        operation = makeOperation(action.action === 'load' ? 'model_load' : 'model_unload', {
          target_kind: 'model',
          model_id: action.model_id,
          requested_revision: action.expected_revision,
        });
        entries = entries.map((entry) =>
          entry.identity.id === action.model_id
            ? {
                ...entry,
                identity: { ...entry.identity, revision: entry.identity.revision + 1 },
                lifecycle: { ...entry.lifecycle, state: action.action === 'load' ? 'loading' : 'draining', busy: true },
              }
            : entry,
        );
      } else if (path.endsWith('/model-removals')) {
        operation = makeOperation('model_removal', {
          target_kind: 'model',
          model_id: String(body.model_id),
          requested_revision: Number(body.expected_revision),
        });
      } else if (path.endsWith('/cancel') && operation)
        operation = { ...operation, state: 'cancelling', cancellable: false };
      else if (path.endsWith('/refresh'))
        operation = makeOperation('catalog_refresh', { target_kind: 'catalog', scope: 'full' });
      if (!operation)
        return json(
          {
            error: { code: 'invalid_request', message: 'Unexpected POST', retryable: false },
            request_id: 'req_unknown',
          },
          400,
        );
      return json({ operation_id: operation.operation_id, state: operation.state, idempotent_replay: false }, 202);
    }
    if (path.endsWith('/bootstrap')) return json(bootstrap);
    if (path.endsWith('/catalog'))
      return json({
        schema_version: 'webui.ui-api.v1',
        items: entries,
        pagination: { limit: 200, next_cursor: null, total_known: entries.length },
        server_instance_id: bootstrap.server.server_instance_id,
        snapshot_sequence: sequence,
      });
    if (path.endsWith('/operations'))
      return json({
        items: operation ? [operation] : [],
        pagination: { limit: 200, next_cursor: null, total_known: operation ? 1 : 0 },
        server_instance_id: bootstrap.server.server_instance_id,
        snapshot_sequence: sequence,
      });
    if (path.endsWith('/runtime')) {
      const entry = entries.find((item) => item.identity.id === url.searchParams.get('model_id'));
      if (!entry)
        return json(
          {
            error: { code: 'not_found', message: 'Selected model no longer exists', retryable: false },
            request_id: 'req_removed',
          },
          404,
        );
      return json({
        schema_version: 'webui.ui-api.v1',
        server_instance_id: bootstrap.server.server_instance_id,
        model_id: url.searchParams.get('model_id'),
        revision: entry.identity.revision,
        measurements: {},
        settings: { scope: 'next_load_profile', effective: {}, overridden_by_cli: [], partial_errors: [] },
        snapshot_sequence: sequence,
      });
    }
    // Shared SSE parser consumes this mock transport sentinel before UiEvent JSON validation.
    if (path.endsWith('/events'))
      return route.fulfill({ status: 200, contentType: 'text/event-stream', body: 'data: [DONE]\n\n' });
    return json(
      { error: { code: 'not_found', message: 'Unknown operation', retryable: false }, request_id: 'req_missing' },
      404,
    );
  });
  return {
    posts,
    finish: (kind: 'download' | 'load' | 'unload' | 'delete') => {
      if (!operation) throw new Error('No accepted operation');
      sequence += 1;
      if (kind === 'download') entries = [model()];
      else if (kind === 'delete') entries = [];
      else
        entries = entries.map((entry) => ({
          ...entry,
          identity: { ...entry.identity, revision: entry.identity.revision + 1 },
          lifecycle: {
            ...entry.lifecycle,
            state: kind === 'load' ? 'ready' : 'unloaded',
            busy: false,
            worker_exit_observed: kind === 'unload',
          },
          capabilities: [
            { task: 'chat', phase: kind === 'load' ? 'provider_ready' : 'pre_load', available: true, reason: null },
          ],
        }));
      operation = { ...operation, state: 'succeeded', cancellable: false };
    },
  };
}
async function login(page: Page): Promise<void> {
  await page.goto('/#models');
  await page.getByLabel('Session key').fill('fixture-session-key');
  await page.getByRole('button', { name: 'Connect', exact: true }).click();
  await expect(page.getByTestId('models-table')).toBeVisible();
}
async function refresh(page: Page): Promise<void> {
  await page.getByRole('button', { name: 'Refresh server state', exact: true }).first().click();
}

test('library journey observes operations before load/chat/unload/cache removal', async ({ page }) => {
  const api = await installLibrary(page);
  await login(page);
  await expect(page.getByTestId('models-empty')).toBeVisible();
  await page.getByTestId('models-add').click();
  await page.getByTestId('models-repo').fill('mlx-community/SmolLM-135M-Instruct-4bit');
  await expect(page.getByTestId('models-download-submit')).toBeDisabled();
  await page.getByTestId('models-public-repo').check();
  await page.getByTestId('models-download-submit').click();
  await expect.poll(() => api.posts.length).toBe(1);
  await refresh(page);
  await expect(page.getByRole('progressbar')).not.toHaveAttribute('aria-valuenow');
  api.finish('download');
  await refresh(page);
  await page.getByRole('button', { name: `Inspect ${model().identity.display_name}`, exact: true }).click();
  await expect(page.getByTestId('models-load')).toBeEnabled();
  await page.getByTestId('models-load').click();
  await expect.poll(() => api.posts.length).toBe(2);
  await refresh(page);
  await expect(page.getByTestId('models-use-chat')).toBeDisabled();
  api.finish('load');
  await refresh(page);
  await expect(page.getByTestId('models-use-chat')).toBeEnabled();
  await page.getByTestId('models-use-chat').click();
  await expect(page).toHaveURL(/#chat$/);
  await page.getByRole('link', { name: 'Models', exact: true }).first().click();
  await page.getByTestId('models-unload').click();
  await expect(page.getByTestId('models-confirm')).toContainText('Files remain on disk');
  await page.getByTestId('models-confirm-submit').click();
  await expect.poll(() => api.posts.length).toBe(3);
  api.finish('unload');
  await refresh(page);
  await expect(page.getByTestId('models-delete')).toBeEnabled();
  await page.getByTestId('models-delete').click();
  await page.getByTestId('models-confirm-name').fill(model().identity.display_name);
  await page.getByTestId('models-confirm-submit').click();
  await expect.poll(() => api.posts.length).toBe(4);
  api.finish('delete');
  await refresh(page);
  await expect(page.getByTestId('models-empty')).toBeVisible();
  await expect(page.getByTestId('models-pending')).toHaveCount(0);
  await expect(page.getByTestId('connection-error-title')).toHaveCount(0);
  expect(api.posts.map((item) => item.path)).toEqual([
    '/ui-api/v1/downloads',
    '/ui-api/v1/model-actions',
    '/ui-api/v1/model-actions',
    '/ui-api/v1/model-removals',
  ]);
});

for (const variant of [
  { width: 1440, theme: 'light' },
  { width: 390, theme: 'dark' },
]) {
  test(`Models ${variant.width} ${variant.theme}: bounded inventory, keyboard inspection, no search POST, a11y`, async ({
    page,
  }) => {
    await page.setViewportSize({ width: variant.width, height: 900 });
    await page.addInitScript(
      (theme) =>
        localStorage.setItem(
          'mlxcel.webui.appearance',
          JSON.stringify({ theme, material: 'opaque', locale: 'en', reduceMotion: true }),
        ),
      variant.theme,
    );
    const entries = Array.from({ length: 120 }, (_, index) => ({
      ...model(),
      identity: {
        ...model().identity,
        id: `mdl_${String(index).padStart(43, '0')}`,
        display_name: `긴 체크포인트 模型 ${index} — long readable identity without truncating meaning`,
      },
    }));
    const api = await installLibrary(page, entries);
    await login(page);
    await expect(page.locator('[data-testid="models-table"] tbody tr')).toHaveCount(25);
    const inspect = page.getByRole('button', { name: /^Inspect / }).first();
    await inspect.focus();
    await page.keyboard.press('Enter');
    await expect(page.getByRole('complementary', { name: 'Model details' })).toBeVisible();
    await page.getByTestId('models-search').fill('模型 119');
    await expect(page.locator('[data-testid="models-table"] tbody tr')).toHaveCount(1);
    expect(api.posts).toEqual([]);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
    expect((await new AxeBuilder({ page }).analyze()).violations).toEqual([]);
  });
}
