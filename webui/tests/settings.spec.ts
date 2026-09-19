import { test, expect, type Page } from '@playwright/test';
import { readFileSync } from 'node:fs';
import { bootProduct, installMockApi, loginWithMockApi, productVariants } from './browser-fixtures';
import { expectAxeClean, expectSafeLayout } from './browser-assertions';
import { fixtureString, localizedPair } from './strings-fixture';

function fixture(name: string): Record<string, unknown> { const data = JSON.parse(readFileSync(new URL(`../../tests/fixtures/webui/examples/${name}`, import.meta.url), 'utf8')) as Record<string, unknown>; delete data.$schemaName; return data; }
const tab = (page: Page, name: string) => page.getByRole('tab', { name, exact: true });

// A ready model whose /settings carries one of each schema kind, with the f32 widening the server really sends.
async function installReadyModel(page: Page): Promise<{ calls: { path: string; method: string; autoload: string | null }[]; entry: { identity: { id: string; display_name: string } } }> {
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
  const schema = [
    { name: 'default_temperature', type: 'float', default: 0.800000011920929, mutable: true, allowed: null, help: 'Temperature for future requests' },
    { name: 'default_seed', type: 'int_or_null', default: null, mutable: true, allowed: null, help: 'Random seed used when a request omits one; null is random.' },
    { name: 'diffusion_sampler', type: 'str', default: 'entropy-bound', mutable: true, allowed: ['entropy-bound', 'confidence-threshold'], help: 'Default sampler' },
    { name: 'default_dry_sequence_breakers', type: 'array', default: ['\n'], mutable: true, allowed: null, help: 'Breakers' },
    { name: 'lang_bias_config', type: 'object_or_null', default: null, mutable: true, allowed: null, help: 'Language bias' },
    { name: 'n_parallel', type: 'int', default: 1, mutable: false, allowed: null, help: 'Startup server configuration.', reason: 'scheduler or channel geometry is fixed at startup; restart required' },
    { name: 'kv_unified', type: 'bool', default: false, mutable: false, allowed: null, help: 'Startup server configuration.', reason: 'scheduler or channel geometry is fixed at startup; restart required' },
    { name: 'model_alias', type: 'str_or_null', default: null, mutable: false, allowed: null, help: 'Startup server configuration.', reason: 'fixed for the loaded model provider; restart required' },
  ];
  const current = { default_temperature: 0.800000011920929, default_seed: null, diffusion_sampler: 'entropy-bound', default_dry_sequence_breakers: ['\n'], lang_bias_config: null, n_parallel: 1, kv_unified: false, model_alias: null };
  await page.route('**/settings?*', (route) => route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ schema, current, fingerprint: 'a'.repeat(64) }) }));
  return { calls, entry };
}

for (const width of [390, 1440]) {
  test(`scoped Settings selects a ready model without autoload at ${width}`, async ({ page }) => {
    const { calls, entry } = await installReadyModel(page);
    await bootProduct(page, { ...productVariants[0], width, height: 900 });
    await loginWithMockApi(page);
    await page.evaluate(() => { window.location.hash = '#settings'; });
    await expect(tab(page, 'Appearance')).toHaveAttribute('aria-selected', 'true');
    await expect(page.getByTestId('settings-appearance')).toHaveText(fixtureString('en', 'settings.browser_only'));
    await expectAxeClean(page); await expectSafeLayout(page);
    await tab(page, 'Model').click();
    await expect(page).toHaveURL(/#settings\/model$/);
    expect(calls.some((call) => call.path === '/props' || call.path === '/settings')).toBe(false);
    await page.getByTestId('settings-model-selector').getByRole('combobox').click();
    await page.getByRole('option', { name: `${entry.identity.display_name} · ${fixtureString('en', 'models.status.ready')}`, exact: true }).click();
    await expect(page.getByRole('heading', { name: 'Loaded model · live server values' })).toBeVisible();
    // Typed controls: an f32 float reads as typed, an enum is a Select, a null value is an unset toggle, never the text null.
    await expect(page.getByLabel('Temperature', { exact: true })).toHaveValue('0.8');
    await expect(page.getByTestId('setting-diffusion_sampler').getByRole('combobox')).toBeVisible();
    await expect(page.getByTestId('setting-default_seed-unset')).toBeChecked();
    await expect(page.getByTestId('setting-default_seed')).toHaveValue('');
    await expect(page.getByTestId('setting-default_seed')).toBeDisabled();
    expect(await page.locator('main').evaluate((element) => /\bnull\b/.test((element as HTMLElement).innerText) || [...element.querySelectorAll('input, textarea')].some((field) => /\bnull\b/.test((field as HTMLInputElement).value)))).toBe(false);
    await expectAxeClean(page); await expectSafeLayout(page);
    await tab(page, 'Server').click();
    await expect(page.getByTestId('settings-context-n-ctx')).toHaveText('2048');
    await expect(page.getByTestId('settings-startup-values')).toContainText('n_parallel');
    await expectAxeClean(page); await expectSafeLayout(page);
    await tab(page, 'Requests').click();
    await expect(page.getByTestId('settings-requests-server')).toHaveText(fixtureString('en', 'settings.requests.server_from', { model: entry.identity.display_name }));
    await expect(page.getByTestId('settings-request-temperature').locator('xpath=..')).toContainText('Next request: 0.8 (server default)');
    await page.getByLabel('Temperature', { exact: true }).fill('0.5');
    await page.getByRole('button', { name: 'Save session defaults', exact: true }).click();
    await expect(page.getByTestId('settings-request-temperature').locator('xpath=..')).toContainText('Next request: 0.5 (session default)');
    const observation = calls.filter((call) => call.path === '/props' || call.path === '/settings' || call.path === '/ui-api/v1/runtime');
    expect(observation.length).toBeGreaterThanOrEqual(2); expect(observation.every((call) => call.autoload === 'false' && call.method === 'GET')).toBe(true);
    expect(calls.some((call) => call.path === '/ui-api/v1/model-actions' || call.path.startsWith('/v1/'))).toBe(false);
    await page.getByRole('button', { name: 'Reset request defaults…', exact: true }).click();
    await expect(page.getByRole('dialog')).toBeVisible(); await page.keyboard.press('Escape');
    await expect(page.getByRole('dialog')).toBeHidden();
    await expect(page.getByRole('button', { name: 'Reset request defaults…', exact: true })).toBeFocused();
    await expectAxeClean(page); await expectSafeLayout(page);
    await page.getByTestId('toolbar-logout').click();
    await expect(page.getByRole('heading', { name: 'Generation · next request' })).toBeVisible();
    await tab(page, 'Model').click();
    await expect(page.getByRole('heading', { name: 'Next load · browser profile' })).toHaveCount(0);
    await expect(page.getByText(fixtureString('en', 'settings.server.unavailable.title'), { exact: true })).toBeVisible();
  });
}

test('#settings/server opens the Server tab directly and Back leaves Settings', async ({ page }) => {
  await installReadyModel(page);
  await bootProduct(page, productVariants[1]);
  await loginWithMockApi(page);
  await page.evaluate(() => { window.location.hash = '#settings/server'; });
  await expect(tab(page, 'Server')).toHaveAttribute('aria-selected', 'true');
  await expect(page.getByTestId('settings-title')).toBeVisible();
  for (const name of ['Model', 'Requests', 'Appearance']) {
    await tab(page, name).click();
    await expect(tab(page, name)).toHaveAttribute('aria-selected', 'true');
  }
  await expect(page).toHaveURL(/#settings\/appearance$/);
  await page.goBack();
  await expect(page).toHaveURL(/#models$/);
  await expect(page.getByTestId('settings-title')).toHaveCount(0);
});

test('Settings tabs follow the arrow keys', async ({ page }) => {
  await installMockApi(page, 'happy');
  await bootProduct(page, productVariants[1]);
  await page.evaluate(() => { window.location.hash = '#settings'; });
  await tab(page, 'Appearance').focus();
  await page.keyboard.press('ArrowRight');
  await expect(tab(page, 'Requests')).toBeFocused();
  await expect(tab(page, 'Requests')).toHaveAttribute('aria-selected', 'true');
  await expect(page).toHaveURL(/#settings\/requests$/);
  await page.keyboard.press('End');
  await expect(tab(page, 'Server')).toBeFocused();
  await expect(page).toHaveURL(/#settings\/server$/);
});

test('Settings renders catalog copy in Korean', async ({ page }) => {
  await installMockApi(page, 'happy');
  await bootProduct(page, productVariants[3]);
  await loginWithMockApi(page);
  await page.evaluate(() => { window.location.hash = '#settings/requests'; });
  await expect(page.getByRole('heading', { name: localizedPair('ko', 'settings.generation.title').shown, exact: true })).toBeVisible();
  for (const key of ['settings.generation.title', 'settings.generation.save', 'settings.section.requests.description', 'settings.section.appearance', 'settings.section.server']) {
    const { shown, hidden } = localizedPair('ko', key);
    await expect(page.getByText(shown, { exact: true }).first()).toBeVisible();
    await expect(page.getByText(hidden, { exact: true })).toHaveCount(0);
  }
  await page.getByRole('button', { name: fixtureString('ko', 'settings.generation.reset_open'), exact: true }).click();
  await expect(page.getByRole('dialog')).toBeVisible();
  await expect(page.getByRole('dialog').getByTestId('dialog-close')).toHaveAccessibleName(fixtureString('ko', 'common.close'));
  await page.keyboard.press('Escape');
  await expect(page.getByRole('dialog')).toBeHidden();
  await tab(page, fixtureString('ko', 'settings.section.model')).click();
  const profile = localizedPair('ko', 'settings.profile.title');
  await expect(page.getByRole('heading', { name: profile.shown, exact: true })).toBeVisible();
  await expect(page.getByText(profile.hidden, { exact: true })).toHaveCount(0);
  await expectAxeClean(page); await expectSafeLayout(page);
});
