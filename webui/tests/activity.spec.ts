import { expect, test, type Locator, type Page } from '@playwright/test';
import type { MeasuredValue, RuntimeSnapshot } from '../src/api/types';
import { bootProduct, installMockApi, loginWithMockApi, productVariants, readyCatalog, type ApiCall } from './browser-fixtures';
import { expectAxeClean, expectSafeLayout } from './browser-assertions';
import { fixtureString, localizedPair, type FixtureLocale } from './strings-fixture';
import { runtimeForFirstCatalogEntry } from './ui-common-helpers';

for (const width of [1440, 390]) {
  test(`Activity shares controls and exposes honest empty observation at ${width}`, async ({ page }) => {
    const mock = await installMockApi(page);
    await bootProduct(page, { ...productVariants[0], width, appearance: { ...productVariants[0].appearance, theme: width === 390 ? 'dark' : 'light', locale: 'en' } });
    await loginWithMockApi(page);
    await page.evaluate(() => { location.hash = '#activity'; });
    await expect(page.getByTestId('activity-page')).toBeVisible();
    await expect(page.getByRole('heading', { name: 'Activity', exact: true })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Export sanitized diagnostics' })).toBeVisible();
    await expect(page.getByRole('combobox')).toHaveCount(1);
    await expect(page.locator('.page-header__actions').getByRole('combobox')).toBeVisible();
    await expect(page.getByRole('button', { name: /recent history/i })).toHaveCount(0);
    await expectAxeClean(page);
    await expectSafeLayout(page);
    expect(mock.calls.every((call) => call.method === 'GET')).toBe(true);
    await page.getByTestId('toolbar-logout').click();
    await expect(page.getByTestId('auth-login')).toBeVisible();
    await expect(page.getByTestId('activity-page')).toHaveCount(0);
  });
}

test('Activity renders catalog copy in Korean', async ({ page }) => {
  await installMockApi(page);
  await bootProduct(page, productVariants[3]);
  await loginWithMockApi(page);
  await page.evaluate(() => { location.hash = '#activity'; });
  await expect(page.getByTestId('activity-page')).toBeVisible();
  await expect(page.getByRole('heading', { name: localizedPair('ko', 'activity.title').shown, exact: true })).toBeVisible();
  await expect(page.getByTestId('activity-intro')).toContainText(localizedPair('ko', 'activity.intro').shown);
  for (const key of ['activity.export', 'activity.operations', 'activity.kind.model_load', 'activity.state.running', 'activity.target.unknown_model', 'activity.details']) {
    const { shown, hidden } = localizedPair('ko', key);
    await expect(page.getByText(shown, { exact: true }).first()).toBeVisible();
    await expect(page.getByText(hidden, { exact: true })).toHaveCount(0);
  }
  await expectSafeLayout(page);
});

// A loaded model with --metrics and --slots after one chat: four idle slots with a known
// request context, the counters the server publishes, and the twelve it cannot.
function statusRuntime(slotCount = 4): { modelName: string; body: RuntimeSnapshot } {
  const at = '2026-09-15T00:00:00Z';
  const counter = (value: number, unit: string): MeasuredValue => ({ value, unit, scope: 'model', measured_at: at, reason: 'Metrics route completions' });
  const missing = (unit: string, scope: MeasuredValue['scope'], reason: string): MeasuredValue => ({ value: null, unit, scope, measured_at: null, reason });
  const { modelName, body } = runtimeForFirstCatalogEntry({
    active_requests: counter(0, 'requests'),
    queued_requests: counter(0, 'requests'),
    completed_requests_total: counter(1, 'requests'),
    completion_tokens_total: counter(42, 'tokens'),
    generation_time_ms_total: counter(812, 'ms'),
    prompt_cache_bytes: counter(0, 'bytes'),
    prompt_cache_entries: counter(0, 'entries'),
    decode_tokens_total: missing('tokens', 'model', 'no completed decode timing sample has been published by this provider'),
    decode_time_us_total: missing('us', 'model', 'no completed decode timing sample has been published by this provider'),
    gpu_utilization: missing('percent', 'unknown', 'not measured by mlxcel'),
    ttft: missing('ms', 'model', 'request-send to first output token timing is not available in runtime counters'),
    decode_rate: missing('tokens/s', 'model', 'no atomic interval sample'),
    process_resident_bytes: missing('bytes', 'server', 'process-wide resident memory has no existing cheap CPU snapshot'),
    allocator_active_bytes: missing('bytes', 'server', 'process-wide allocator active memory has no existing cheap CPU snapshot'),
    allocator_cache_bytes: missing('bytes', 'server', 'process-wide allocator cache memory has no existing cheap CPU snapshot'),
    allocator_peak_bytes: missing('bytes', 'server', 'process-wide allocator peak memory has no existing cheap CPU snapshot'),
    device_total_bytes: missing('bytes', 'unknown', 'device total memory has no existing cheap CPU snapshot'),
    model_weights_bytes: missing('bytes', 'model', 'resident weights estimate is unavailable'),
    kv_cache_bytes: missing('bytes', 'model', 'no cheap live KV byte snapshot'),
  });
  const items = Array.from({ length: slotCount }, (_, id) => ({ id, processing: false, prompt_tokens: null, cached_prompt_tokens: null, decoded_tokens: null }));
  return { modelName, body: { ...body, slots: { available: true, reason: null, measured_at: at, configured_parallelism: 4, effective_parallelism: null, request_context_tokens: 40960, shared_pool_context_tokens: null, items } } };
}

const idleOccupancy = (locale: FixtureLocale): string => fixtureString(locale, 'activity.slot_progress', { used: '0', total: '40,960' });

async function openStatusPage(page: Page, width: number, locale: FixtureLocale = 'en', slotCount = 4): Promise<ApiCall[]> {
  const mock = await installMockApi(page, 'happy', { catalog: readyCatalog });
  const { modelName, body } = statusRuntime(slotCount);
  await page.route('**/ui-api/v1/runtime*', (route) => route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(body) }));
  await bootProduct(page, { ...productVariants[1], width, height: width === 1440 ? 900 : 844, appearance: { ...productVariants[1].appearance, theme: width === 390 ? 'dark' : 'light', locale } });
  await loginWithMockApi(page);
  await page.evaluate(() => { location.hash = '#activity'; });
  await page.getByTestId('activity-page').getByRole('combobox').click();
  await page.getByRole('option', { name: modelName, exact: true }).click();
  // Wait for measured values, not labels: the waiting tiles already show the labels.
  await expect(page.getByTestId('runtime-summary')).toContainText(fixtureString(locale, 'activity.unit.tokens', { value: '42' }));
  await expect(page.getByTestId('activity-slot-table').getByText(idleOccupancy(locale), { exact: true })).toHaveCount(slotCount);
  return mock.calls;
}

// Polled: a focus ring may still be transitioning in when focus lands.
async function expectVisibleFocus(locator: Locator): Promise<void> {
  await expect(locator).toBeFocused();
  await expect.poll(() => locator.evaluate((element) => {
    const style = window.getComputedStyle(element);
    return (style.outlineStyle !== 'none' && Number.parseFloat(style.outlineWidth) > 0) || style.boxShadow !== 'none' ? 'visible' : `${element.outerHTML.slice(0, 100)} outline=${style.outlineStyle} ${style.outlineWidth} shadow=${style.boxShadow} focus-visible=${element.matches(':focus-visible')}`;
  })).toBe('visible');
}

test('Activity fits one 1440x900 screen with four idle slots and one operation', async ({ page }) => {
  await openStatusPage(page, 1440);
  await expect(page.getByTestId('toolbar-loaded-chip')).toHaveCount(1);
  await expect(page.locator('.activity-operation')).toHaveCount(1);
  // Measure the tallest steady state: the active-request sparkline appears from the second sample.
  await expect(page.locator('[data-testid="runtime-summary"] .activity-sparkline')).toBeVisible({ timeout: 10_000 });
  const height = await page.evaluate(() => document.documentElement.scrollHeight);
  // scrollHeight never reads below the viewport; the page's own bottom edge shows the margin left.
  const contentBottom = await page.getByTestId('activity-page').evaluate((element) => Math.ceil(element.getBoundingClientRect().bottom + window.scrollY));
  console.log(`WEBUI_ACTIVITY_SCROLL_HEIGHT ${JSON.stringify({ width: 1440, height: 900, scrollHeight: height, contentBottom })}`);
  expect(height).toBeLessThanOrEqual(900);
});

for (const width of [1440, 390]) {
  test(`Activity status page is keyboard ordered, focus visible, axe clean and reflowed at ${width}`, async ({ page }) => {
    const calls = await openStatusPage(page, width);
    const summary = page.locator('details.activity-measurement-details > summary');
    await expect(summary.locator('.ds-badge')).toHaveText(fixtureString('en', 'activity.unavailable_badge', { count: '12' }));
    await expectAxeClean(page);
    await expectSafeLayout(page);
    // The table fits the column and four slots fit the box: it does not scroll either way, so its
    // scroller is no region and no tab stop.
    const scroller = page.locator('.activity-slot-scroll');
    expect(await scroller.evaluate((element) => element.scrollWidth - element.clientWidth)).toBeLessThanOrEqual(1);
    await expect(scroller).not.toHaveAttribute('tabindex');
    await expect(scroller).not.toHaveAttribute('role');
    // Keyboard only from here: a pointer left resting on a control would hover it.
    await page.mouse.move(0, 0);
    await page.getByTestId('activity-page').getByRole('combobox').focus();
    const order = [
      page.getByRole('button', { name: fixtureString('en', 'activity.refresh'), exact: true }),
      page.getByRole('button', { name: fixtureString('en', 'activity.export'), exact: true }),
      summary,
      page.getByRole('button', { name: fixtureString('en', 'activity.session_help'), exact: true }),
      page.locator('.activity-operation__details > summary'),
    ];
    for (const target of order) {
      await page.keyboard.press('Tab');
      await expectVisibleFocus(target);
    }
    // The retention note describes the help control; the badge sits inside the summary and opens the list.
    await expect(page.getByRole('button', { name: fixtureString('en', 'activity.session_help'), exact: true })).toHaveAccessibleDescription(fixtureString('en', 'activity.session'));
    await summary.locator('.ds-badge').click();
    await expect(page.locator('details.activity-measurement-details')).toHaveAttribute('open', '');
    await expect(page.locator('.activity-measurements li')).toHaveCount(19);
    await expect(page.locator('details.activity-measurement-details')).toContainText('not measured by mlxcel');
    await expectAxeClean(page);
    await expectSafeLayout(page);
    expect(calls.every((call) => call.method === 'GET')).toBe(true);
  });
}

test('Activity scrolls more than eight slots inside the named table region, not the page', async ({ page }) => {
  await openStatusPage(page, 1440, 'en', 32);
  const region = page.getByRole('region', { name: fixtureString('en', 'activity.slots'), exact: true });
  const box = await region.evaluate((element) => ({ scroll: element.scrollHeight, client: element.clientHeight, rows: element.querySelectorAll('tbody tr').length }));
  expect(box.rows).toBe(32);
  expect(box.scroll).toBeGreaterThan(box.client);
  // Eight rows stay in view before the region scrolls.
  const rowHeight = await region.locator('tbody tr').first().evaluate((row) => row.getBoundingClientRect().height);
  expect(box.client).toBeGreaterThanOrEqual(8 * rowHeight);
  await expect(region).toHaveAttribute('tabindex', '0');
  await expect(region).toHaveAttribute('aria-labelledby', /.+/);
  await expectAxeClean(page);
  await expectSafeLayout(page);
});

test('Activity status page reads in Korean without English copy', async ({ page }) => {
  await openStatusPage(page, 390, 'ko');
  const tiles = page.getByTestId('runtime-summary').locator('.activity-metric');
  await expect(tiles).toHaveCount(4);
  for (const [index, key] of ['activity.metric.active_requests', 'activity.metric.completed_requests_total', 'activity.metric.completion_tokens_total', 'activity.metric.queued_requests'].entries()) await expect(tiles.nth(index)).toContainText(fixtureString('ko', key));
  await expect(page.locator('details.activity-measurement-details > summary')).toHaveText(`${fixtureString('ko', 'activity.metric_details')} ${fixtureString('ko', 'activity.unavailable_badge', { count: '12' })}`);
  for (const key of ['activity.slots', 'activity.idle', 'activity.kind.model_load', 'activity.state.running', 'activity.occupancy']) {
    const { shown, hidden } = localizedPair('ko', key);
    await expect(page.getByText(shown, { exact: true }).first()).toBeVisible();
    await expect(page.getByText(hidden, { exact: true })).toHaveCount(0);
  }
  await expect(page.getByTestId('activity-page')).not.toContainText('N/A');
  await expect(page.getByTestId('activity-page')).not.toContainText(/\brequests\b|\btokens\b|\bunknown\b/);
  await expectSafeLayout(page);
});
