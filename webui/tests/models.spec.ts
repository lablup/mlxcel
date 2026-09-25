// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import { expect, test, type Page } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
import { expectAxeClean, expectSafeLayout } from './browser-assertions';
import { model, runtime, bootstrap, loadValidator } from './models-fixtures';
import type { CatalogEntry, ModelActionRequest, Operation } from '../src/api/types';
import { isolate } from '../src/features/models/labels';

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
      return json(runtime(entry, sequence));
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
  await page.getByRole('button', { name: `Inspect ${isolate(model().identity.display_name)}`, exact: true }).click();
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
  await page.getByTestId('models-confirm-name').fill(model().identity.id);
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
        display_name: `긴-체크포인트-模型-${index}-long_readable_identity_without_truncating_meaning`,
      },
    }));
    const api = await installLibrary(page, entries);
    await login(page);
    await expect(page.locator('[data-testid="models-table"] tbody tr')).toHaveCount(25);
    const inspect = page.getByRole('button', { name: /^Inspect / }).first();
    await inspect.focus();
    await page.keyboard.press('Enter');
    if (variant.width >= 1100) await expect(page.getByRole('complementary', { name: 'Model details' })).toBeVisible();
    else {
      // Below 1100 px the inspector is a modal drawer: a dialog with the same name, closed with Escape.
      const drawer = page.getByRole('dialog', { name: 'Model details' });
      await expect(drawer).toBeVisible();
      await expect(page.getByRole('complementary', { name: 'Model details' })).toHaveCount(0);
      await expect(drawer.getByRole('heading', { level: 3 })).toHaveText(/-0-long_readable/);
      await page.keyboard.press('Escape');
      await expect(drawer).toBeHidden();
      await expect(inspect).toBeFocused();
    }
    await page.getByTestId('models-search').fill('模型-119');
    await expect(page.locator('[data-testid="models-table"] tbody tr')).toHaveCount(1);
    expect(api.posts).toEqual([]);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
    expect((await new AxeBuilder({ page }).analyze()).violations).toEqual([]);
  });
}

// Whole-row activation: a user who never finds the Inspect button still reaches the inspector.
test.describe('Models row activation', () => {
  const entries = (): CatalogEntry[] =>
    ['alpha', 'bravo', 'charlie'].map((name, index) => ({
      ...model(),
      identity: {
        ...model().identity,
        id: `mdl_${String(index + 1).padStart(43, '0')}`,
        display_name: `row-${name}-checkpoint`,
      },
    }));
  const bodyRows = (page: Page) => page.locator('[data-testid="models-table"] tbody tr');
  // State stays visible at every width the list reaches here, including beside the inspector pane.
  const stateCell = (page: Page, index: number) => bodyRows(page).nth(index).locator('td.models-col-state');
  const inspector = (page: Page) => page.getByRole('complementary', { name: 'Model details' });
  const inspected = (page: Page) => inspector(page).getByRole('heading', { level: 3 });
  const inspect = (page: Page, entry: CatalogEntry) =>
    page.getByRole('button', { name: `Inspect ${isolate(entry.identity.display_name)}`, exact: true });

  test('clicking a non-name cell opens that entry and reaches Load, Use in Chat and Unload', async ({ page }) => {
    const catalog = entries();
    const api = await installLibrary(page, catalog);
    await login(page);
    await expect(bodyRows(page)).toHaveCount(3);
    await expect(inspector(page)).toHaveCount(0);
    await stateCell(page, 1).click();
    await expect(inspector(page)).toBeVisible();
    await expect(inspected(page)).toHaveText(catalog[1].identity.display_name);
    await expect(bodyRows(page).nth(1)).toHaveClass(/models-selected/);
    await expect(bodyRows(page).nth(0)).not.toHaveClass(/models-selected/);
    expect(api.posts).toEqual([]);
    await bodyRows(page).nth(1).locator('td.models-col-name').click();
    await expect(inspected(page)).toHaveText(catalog[1].identity.display_name);
    expect(api.posts).toEqual([]);
    await page.getByTestId('models-load').click();
    await expect.poll(() => api.posts.length).toBe(1);
    expect(api.posts[0].body).toMatchObject({ action: 'load', model_id: catalog[1].identity.id });
    api.finish('load');
    await refresh(page);
    await expect(page.getByTestId('models-use-chat')).toBeEnabled();
    await expect(page.getByTestId('models-unload')).toBeEnabled();
  });

  test('keyboard Tab walks each row\'s controls with the row outlined, and Space or Enter on Inspect opens it', async ({ page }) => {
    const catalog = entries();
    const api = await installLibrary(page, catalog);
    await login(page);
    const outline = (index: number) =>
      bodyRows(page)
        .nth(index)
        .evaluate((row) => {
          const style = window.getComputedStyle(row);
          return { style: style.outlineStyle, width: Number.parseFloat(style.outlineWidth) };
        });
    const row = (index: number) => bodyRows(page).nth(index);
    await inspect(page, catalog[0]).focus();
    // Row order: Load, Inspect, Delete; the next row begins with its Load.
    await page.keyboard.press('Tab');
    await expect(row(0).getByTestId('models-row-delete')).toBeFocused();
    await page.keyboard.press('Tab');
    await expect(row(1).getByTestId('models-row-load')).toBeFocused();
    const focused = await outline(1);
    expect(focused.style).not.toBe('none');
    expect(focused.width).toBeGreaterThan(0);
    expect((await outline(0)).style).toBe('none');
    await page.keyboard.press('Tab');
    await expect(inspect(page, catalog[1])).toBeFocused();
    await page.keyboard.press('Space');
    await expect(inspected(page)).toHaveText(catalog[1].identity.display_name);
    for (let step = 0; step < 3; step += 1) await page.keyboard.press('Tab');
    await expect(inspect(page, catalog[2])).toBeFocused();
    expect((await outline(2)).style).not.toBe('none');
    await page.keyboard.press('Enter');
    await expect(inspected(page)).toHaveText(catalog[2].identity.display_name);
    await expect(bodyRows(page).nth(2)).toHaveClass(/models-selected/);
    expect(api.posts).toEqual([]);
  });

  test('dragging to select text inside a cell does not activate the row', async ({ page }) => {
    const catalog = entries();
    await installLibrary(page, catalog);
    await login(page);
    await inspect(page, catalog[0]).click();
    await expect(inspected(page)).toHaveText(catalog[0].identity.display_name);
    const name = bodyRows(page).nth(2).locator('td.models-col-name .truncate');
    await name.scrollIntoViewIfNeeded();
    const line = await name.evaluate((cell) => {
      const range = document.createRange();
      range.selectNodeContents(cell);
      const first = range.getClientRects()[0];
      return { left: first.left, right: first.right, middle: first.top + first.height / 2 };
    });
    await page.mouse.move(line.left + 1, line.middle);
    await page.mouse.down();
    await page.mouse.move(line.right - 1, line.middle, { steps: 8 });
    await page.mouse.up();
    expect((await page.evaluate(() => window.getSelection()?.toString() ?? '')).trim()).not.toBe('');
    await page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));
    await expect(inspected(page)).toHaveText(catalog[0].identity.display_name);
    await expect(bodyRows(page).nth(2)).not.toHaveClass(/models-selected/);
    // Control: with the selection gone, a click on the same point activates the row, so the drag did land inside it.
    await page.evaluate(() => window.getSelection()?.removeAllRanges());
    await page.mouse.click(line.left + 1, line.middle);
    await expect(inspected(page)).toHaveText(catalog[2].identity.display_name);
  });

  test('rows keep table semantics, each entry keeps one Inspect button, and the page stays axe clean', async ({ page }) => {
    const catalog = entries();
    await installLibrary(page, catalog);
    await login(page);
    for (let index = 0; index < catalog.length; index += 1) {
      expect(await bodyRows(page).nth(index).getAttribute('role')).toBeNull();
      expect(await bodyRows(page).nth(index).getAttribute('tabindex')).toBeNull();
    }
    await expect(page.getByTestId('models-table').getByRole('row')).toHaveCount(catalog.length + 1);
    await expect(page.getByTestId('models-table').getByRole('button', { name: /^Inspect / })).toHaveCount(catalog.length);
    for (const entry of catalog) {
      // The router-real harness locator form; it must resolve to exactly one element.
      const escaped = entry.identity.display_name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      await expect(page.getByRole('button', { name: new RegExp(`Inspect .*${escaped}`) })).toHaveCount(1);
    }
    await stateCell(page, 0).click();
    await expect(inspected(page)).toHaveText(catalog[0].identity.display_name);
    expect((await new AxeBuilder({ page }).analyze()).violations).toEqual([]);
  });

  // Row activation makes a selected row routine, so its content must stay readable on the selection
  // fill in every theme, including high contrast, where the fill is 30 percent focus color.
  test('a selected row, and a hovered one, stay contrast-clean in every theme with and without high contrast', async ({ page }) => {
    const catalog = entries();
    await installLibrary(page, catalog);
    const failures: string[] = [];
    for (const themeFamily of ['mlxcel', 'glass'])
      for (const colorScheme of ['light', 'dark'])
        for (const highContrast of ['off', 'on']) {
          const appearance = { themeFamily, colorScheme, highContrast, material: 'opaque', locale: 'en', reduceMotion: true };
          // Store the appearance on the app origin, then boot a fresh document so it is read at startup.
          await page.goto('/#models');
          await page.evaluate((value) => localStorage.setItem('mlxcel.webui.appearance', value), JSON.stringify(appearance));
          await page.goto('about:blank');
          await login(page);
          await expect(page.locator('html')).toHaveAttribute('data-theme', `${themeFamily}-${colorScheme}`);
          await stateCell(page, 0).click();
          await expect(bodyRows(page).nth(0)).toHaveClass(/models-selected/);
          await stateCell(page, 1).hover();
          const results = await new AxeBuilder({ page }).include('[data-testid="models-table"]').withRules(['color-contrast']).analyze();
          for (const violation of results.violations)
            for (const node of violation.nodes) failures.push(`${themeFamily}-${colorScheme} high-contrast ${highContrast}: ${node.target.join(' ')} ${node.failureSummary ?? ''}`);
        }
    expect(failures).toEqual([]);
  });
});

// #1918: the row is the unit of work. The primary path (load, chat, unload) never needs the inspector.
test.describe('Models row actions', () => {
  const catalog = (): CatalogEntry[] =>
    ['alpha', 'bravo', 'charlie'].map((name, index) => ({
      ...model(),
      identity: { ...model().identity, id: `mdl_${String(index + 1).padStart(43, '0')}`, display_name: `${name}-4bit` },
    }));
  const row = (page: Page, name: string) => page.locator('[data-testid="models-table"] tbody tr').filter({ has: page.getByTitle(name, { exact: true }) });

  test('load, Use in Chat and unload run from the row buttons alone', async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 900 });
    const entries = catalog();
    const api = await installLibrary(page, entries);
    await login(page);
    const bravo = () => row(page, 'bravo-4bit');
    await bravo().getByTestId('models-row-load').click();
    await expect.poll(() => api.posts.length).toBe(1);
    expect(api.posts[0].body).toMatchObject({ action: 'load', model_id: entries[1].identity.id });
    // Nothing was selected: the inspector never opened on the way.
    await expect(page.getByRole('complementary', { name: 'Model details' })).toHaveCount(0);
    await refresh(page);
    await expect(bravo().getByTestId('models-row-load')).toBeDisabled();
    api.finish('load');
    await refresh(page);
    await expect(bravo().getByTestId('models-row-load')).toHaveCount(0);
    await expect(page.getByRole('complementary', { name: 'Model details' })).toHaveCount(0);
    await bravo().getByTestId('models-row-chat').click();
    await expect(page).toHaveURL(/#chat$/);
    await page.getByRole('link', { name: 'Models', exact: true }).first().click();
    await bravo().getByTestId('models-row-unload').click();
    await expect(page.getByTestId('models-confirm')).toContainText('Files remain on disk');
    await page.getByTestId('models-confirm-submit').click();
    await expect.poll(() => api.posts.length).toBe(2);
    expect(api.posts[1].body).toMatchObject({ action: 'unload', model_id: entries[1].identity.id });
    api.finish('unload');
    await refresh(page);
    await expect(bravo().getByTestId('models-row-load')).toBeEnabled();
    expect(api.posts.map((item) => item.path)).toEqual(['/ui-api/v1/model-actions', '/ui-api/v1/model-actions']);
  });

  test('at 1440x900 at least 15 rows of a 120-entry library are in view without scrolling', async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 900 });
    const entries = Array.from({ length: 120 }, (_, index) => ({
      ...model(),
      identity: {
        ...model().identity,
        id: `mdl_${String(index).padStart(43, '0')}`,
        display_name: `긴-체크포인트-模型-${index}-long_readable_identity_without_truncating_meaning`,
      },
    }));
    await installLibrary(page, entries);
    await login(page);
    await expect(page.locator('[data-testid="models-table"] tbody tr')).toHaveCount(25);
    const inView = await page.evaluate(() =>
      [...document.querySelectorAll('[data-testid="models-table"] tbody tr')].filter((item) => {
        const rect = item.getBoundingClientRect();
        return rect.top >= 0 && rect.bottom <= window.innerHeight;
      }).length,
    );
    expect(inView).toBeGreaterThanOrEqual(15);
    // A long name stays one line: truncated, with the full name in its title and the Inspect button's name.
    const name = page.locator('[data-testid="models-table"] tbody tr').first().locator('.truncate');
    await expect(name).toHaveAttribute('title', entries[0].identity.display_name);
    expect(await name.evaluate((element) => element.scrollWidth > element.clientWidth)).toBe(true);
  });

  test('below 1100 px a confirmation opened from the inspector drawer returns focus to the drawer', async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 });
    const entries = catalog();
    entries[1] = {
      ...entries[1],
      lifecycle: { ...entries[1].lifecycle, state: 'ready' },
      capabilities: [{ task: 'chat', phase: 'provider_ready', available: true, reason: null }],
    };
    const api = await installLibrary(page, entries);
    await login(page);
    await page.getByRole('button', { name: `Inspect ${isolate('alpha-4bit')}`, exact: true }).click();
    const drawer = page.getByRole('dialog', { name: 'Model details' });
    await expect(drawer).toBeVisible();
    await drawer.getByTestId('models-delete').click();
    await expect(page.getByTestId('models-confirm')).toBeVisible();
    await page.getByTestId('models-confirm').getByRole('button', { name: 'Cancel', exact: true }).click();
    await expect(page.getByTestId('models-confirm')).toHaveCount(0);
    await expect(drawer.getByTestId('models-delete')).toBeFocused();
    await drawer.getByRole('button', { name: 'Close', exact: true }).click();
    await expect(drawer).toBeHidden();
    // A confirmed Unload disables the control that opened it; focus stays in the drawer, not on the page behind it.
    await page.getByRole('button', { name: `Inspect ${isolate('bravo-4bit')}`, exact: true }).click();
    await expect(drawer.getByRole('heading', { level: 3 })).toHaveText('bravo-4bit');
    await drawer.getByTestId('models-unload').click();
    await page.getByTestId('models-confirm-submit').click();
    await expect.poll(() => api.posts.length).toBe(1);
    await expect(drawer.getByRole('button', { name: 'Close', exact: true })).toBeFocused();
    await expectSafeLayout(page);
    // WCAG rules only: the shared Drawer renders <aside role="dialog">, which axe's best-practice
    // aria-allowed-role flags on the navigation sheet as well.
    await expectAxeClean(page);
  });

  test('below 1100 px Escape in a drawer confirmation closes only the confirmation, and Tab reaches the Details disclosure', async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 });
    await installLibrary(page, catalog());
    await login(page);
    await page.getByRole('button', { name: `Inspect ${isolate('alpha-4bit')}`, exact: true }).click();
    const drawer = page.getByRole('dialog', { name: 'Model details' });
    await expect(drawer).toBeVisible();
    const remove = drawer.getByTestId('models-delete');
    await remove.click();
    await expect(page.getByTestId('models-confirm')).toBeVisible();
    // The native confirmation owns this Escape: it closes itself and hands focus back to the
    // drawer's Delete, and the drawer beneath it stays open.
    await page.keyboard.press('Escape');
    await expect(page.getByTestId('models-confirm')).toHaveCount(0);
    await expect(drawer).toBeVisible();
    await expect(drawer).toHaveClass(/(?:^|\s)drawer--open(?:\s|$)/);
    await expect(remove).toBeFocused();
    // The Details disclosure holds the ids, reasons, roots and docs link, which nothing else on
    // the page shows below 1100 px, so it and everything inside it must be in the Tab cycle.
    await page.keyboard.press('Tab');
    await expect(drawer.locator('summary')).toBeFocused();
    await expect(drawer.locator('summary')).toHaveText('Details');
    await page.keyboard.press('Enter');
    await expect(drawer.getByTestId('models-details')).toHaveJSProperty('open', true);
    // The body mounts from the async toggle event, a task after `open` flips; wait for it.
    await expect(drawer.getByRole('link', { name: 'API task documentation', exact: true })).toBeVisible();
    const reached: string[] = [];
    for (let step = 0; step < 12 && reached.at(-1) !== 'Close'; step += 1) {
      await page.keyboard.press('Tab');
      const focus = await page.evaluate(() => {
        const active = document.activeElement;
        return { inDrawer: Boolean(active?.closest('[data-testid="models-inspector-drawer"]')), name: active?.getAttribute('aria-label') ?? active?.textContent?.trim() ?? '' };
      });
      expect(focus.inDrawer, `Tab ${step + 1} left the drawer`).toBe(true);
      reached.push(focus.name);
    }
    expect(reached).toContain('Copy command');
    // The docs link is the drawer's last stop: Tab from it wraps to Close, and Shift+Tab comes back.
    expect(reached.slice(-2)).toEqual(['API task documentation', 'Close']);
    await expect(drawer.getByRole('button', { name: 'Close', exact: true })).toBeFocused();
    await page.keyboard.press('Shift+Tab');
    await expect(drawer.getByRole('link', { name: 'API task documentation', exact: true })).toBeFocused();
    await expectAxeClean(page);
  });

  // A download row has no inspector to recover what the narrow list hides, so its name cell carries
  // the progress and state that the Size and State columns hold at wider widths.
  test('at 390 px a download row still shows its progress, its state and Cancel', async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 });
    const api = await installLibrary(page, []);
    await login(page);
    await page.getByTestId('models-add').click();
    await page.getByTestId('models-repo').fill('mlx-community/SmolLM-135M-Instruct-4bit');
    await page.getByTestId('models-public-repo').check();
    await page.getByTestId('models-download-submit').click();
    await expect.poll(() => api.posts.length).toBe(1);
    await refresh(page);
    const download = page.locator('[data-testid="models-table"] tbody tr.models-download-row');
    await expect(download).toHaveCount(1);
    await expect(download.locator('td.models-col-size')).toBeHidden();
    await expect(download.locator('td.models-col-state')).toBeHidden();
    await expect(download.getByRole('progressbar', { name: 'Download progress', exact: true })).toBeVisible();
    await expect(download.getByText('Downloading', { exact: true }).filter({ visible: true })).toBeVisible();
    await expect(download.getByRole('button', { name: 'Cancel download', exact: true })).toBeVisible();
    await expectSafeLayout(page);
    await expectAxeClean(page);
  });

  test('an empty library at 390 px has no unreachable scroll region and offers the configured roots', async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 });
    await installLibrary(page, []);
    await login(page);
    await expect(page.getByTestId('models-empty')).toBeVisible();
    expect((await new AxeBuilder({ page }).analyze()).violations).toEqual([]);
    await page.getByTestId('models-roots').click();
    await expect(page.getByTestId('models-roots-dialog')).toContainText('--models-dir /path/to/models');
  });
});
