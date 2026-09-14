import { test, expect } from '@playwright/test';
import { readFileSync } from 'node:fs';
import { bootProduct, installMockApi, loginWithMockApi, productVariants } from './browser-fixtures';
import { expectAxeClean, expectSafeLayout } from './browser-assertions';

function fixture(name: string): Record<string, unknown> { const data = JSON.parse(readFileSync(new URL(`../../tests/fixtures/webui/examples/${name}`, import.meta.url), 'utf8')) as Record<string, unknown>; delete data.$schemaName; return data; }
for (const width of [390, 1440]) {
  test(`scoped Settings selects a ready model without autoload at ${width}`, async ({ page }) => {
    const calls: { path: string; method: string; autoload: string | null }[] = [];
    page.on('request', (request) => { const url = new URL(request.url()); calls.push({ path: url.pathname, method: request.method(), autoload: url.searchParams.get('autoload') }); });
    await installMockApi(page, 'happy');
    const bootstrap = fixture('bootstrap.model-free.json');
    bootstrap.features = [...new Set([...bootstrap.features as string[], 'settings'])];
    (bootstrap.server as Record<string, unknown>).mode = 'router_pool';
    const catalog = fixture('catalog.page.json');
    const entry = (catalog.items as { identity: { id: string; display_name: string; revision: number }; capabilities: Record<string, unknown>[]; lifecycle: Record<string, unknown> }[])[0];
    entry.lifecycle = { ...entry.lifecycle, state: 'ready', worker_exit_observed: false };
    catalog.server_instance_id = (bootstrap.server as {server_instance_id: string}).server_instance_id;
    const runtime = fixture('runtime.snapshot.json'); runtime.model_id = entry.identity.id; runtime.server_instance_id = catalog.server_instance_id; runtime.revision = entry.identity.revision; runtime.snapshot_sequence = catalog.snapshot_sequence;
    // The production shared client validates each complete canonical response before UI rendering.
    await page.route('**/ui-api/v1/**', async (route) => {
      const path = new URL(route.request().url()).pathname;
      const body = path.endsWith('/bootstrap') ? bootstrap : path.endsWith('/catalog') ? catalog : path.endsWith('/runtime') ? runtime : null;
      if (body === null) await route.fallback(); else await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(body) });
    });
    await page.route('**/props?*', (route) => route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ default_generation_settings: { n_ctx: 2048 }, total_slots: 1, kv_cache_mode: 'fp16', geometry: { kv_unified: false } }) }));
    await page.route('**/settings?*', (route) => route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ schema: [{ name: 'default_temperature', type: 'float', default: 1, mutable: true, allowed: null, help: 'Temperature for future requests' }], current: { default_temperature: 1 }, fingerprint: 'a'.repeat(64) }) }));
    await bootProduct(page, { ...productVariants[0], width, height: 900 });
    await loginWithMockApi(page);
    await page.evaluate(() => { window.location.hash = '#settings'; });
    await expect(page.getByRole('heading', { name: 'Generation · next request' })).toBeVisible();
    expect(calls.some((call) => call.path === '/props' || call.path === '/settings')).toBe(false);
    await page.getByTestId('settings-model-selector').getByRole('combobox').click();
    await page.getByRole('option', { name: `${entry.identity.display_name} · ready`, exact: true }).click();
    await expect(page.getByLabel('Per-slot context tokens (/props n_ctx)', { exact: true })).toHaveValue('2048');
    await expect(page.getByRole('heading', { name: 'Loaded model · live server values' })).toBeVisible();
    await expect(page.getByLabel('default_temperature', { exact: true })).toHaveValue('1');
    const observation = calls.filter((call) => call.path === '/props' || call.path === '/settings' || call.path === '/ui-api/v1/runtime');
    expect(observation.length).toBeGreaterThanOrEqual(2); expect(observation.every((call) => call.autoload === 'false' && call.method === 'GET')).toBe(true);
    expect(calls.some((call) => call.path === '/ui-api/v1/model-actions' || call.path.startsWith('/v1/'))).toBe(false);
    await page.getByRole('button', { name: 'Reset request defaults…', exact: true }).click();
    await expect(page.getByRole('dialog')).toBeVisible(); await page.keyboard.press('Escape');
    await expect(page.getByRole('dialog')).toBeHidden();
    await expect(page.getByRole('button', { name: 'Reset request defaults…', exact: true })).toBeFocused();
    await expectAxeClean(page); await expectSafeLayout(page);
    await page.getByTestId('toolbar-logout').click();
    await expect(page.getByRole('heading', { name: 'Next load · browser profile' })).toHaveCount(0);
    await expect(page.getByRole('heading', { name: 'Generation · next request' })).toBeVisible();
  });
}
