import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { AxeBuilder } from '@axe-core/playwright';
import { expect, test, type Locator, type Page, type Route } from '@playwright/test';

type GalleryTab = 'controls' | 'states' | 'data';
type Variant = { name: string; width: number; height: number; appearance: Record<string, unknown>; tab: GalleryTab; openDrawer?: boolean; textScale?: '200' };
type ProductVariant = { name: string; width: number; height: number; appearance: Record<string, unknown>; signedIn: boolean };
type MockMode = 'happy' | 'bad-key' | 'malformed-bootstrap' | 'sync-401' | 'slow-catalog';
interface ApiCall { readonly url: string; readonly method: string; readonly auth: string; readonly body: string }


const bootstrapFixture = loadFixture<typeof import('../../tests/fixtures/webui/examples/bootstrap.model-free.json')>('../../tests/fixtures/webui/examples/bootstrap.model-free.json');
const catalogFixture = loadFixture<typeof import('../../tests/fixtures/webui/examples/catalog.page.json')>('../../tests/fixtures/webui/examples/catalog.page.json');
const operationsFixture = loadFixture<typeof import('../../tests/fixtures/webui/examples/operations.list.json')>('../../tests/fixtures/webui/examples/operations.list.json');

function loadFixture<T>(relativePath: string): T {
  return JSON.parse(readFileSync(fileURLToPath(new URL(relativePath, import.meta.url)), 'utf8')) as T;
}

const variants: Variant[] = [
  { name: '1440-light-gallery-controls', width: 1440, height: 900, tab: 'controls', appearance: { theme: 'light', material: 'glass', locale: 'en', glassIntensity: 35, highContrast: 'system' } },
  { name: '1440-light-gallery-data', width: 1440, height: 900, tab: 'data', appearance: { theme: 'light', material: 'opaque', locale: 'en', glassIntensity: 0, highContrast: 'system' } },
  { name: '1440-dark-gallery-controls', width: 1440, height: 900, tab: 'controls', appearance: { theme: 'dark', material: 'glass', locale: 'en', glassIntensity: 45, highContrast: 'system' } },
  { name: '1440-dark-gallery-data', width: 1440, height: 900, tab: 'data', appearance: { theme: 'dark', material: 'opaque', locale: 'en', glassIntensity: 0, highContrast: 'system' } },
  { name: '1024-tinted-gallery-data-inspector', width: 1024, height: 768, tab: 'data', appearance: { theme: 'light', material: 'tinted', locale: 'ko', glassIntensity: 45, highContrast: 'system' } },
  { name: '390-opaque-gallery-controls-drawer-cjk', width: 390, height: 844, tab: 'controls', openDrawer: true, appearance: { theme: 'light', material: 'opaque', locale: 'ko', glassIntensity: 0, highContrast: 'system' } },
  { name: '1440-highcontrast-gallery-states', width: 1440, height: 900, tab: 'states', appearance: { theme: 'light', material: 'opaque', highContrast: 'on', locale: 'en', glassIntensity: 0 } },
  { name: '390-dark-opaque-textscale200-gallery-controls', width: 390, height: 844, tab: 'controls', appearance: { theme: 'dark', material: 'opaque', reduceTransparency: true, reduceMotion: true, locale: 'ko', glassIntensity: 100, highContrast: 'off' }, textScale: '200' },
];

const productVariants: ProductVariant[] = [
  { name: '1440-light-product-login', width: 1440, height: 900, signedIn: false, appearance: { theme: 'light', material: 'glass', locale: 'en', glassIntensity: 35, highContrast: 'system' } },
  { name: '1440-light-product-signed-in', width: 1440, height: 900, signedIn: true, appearance: { theme: 'light', material: 'glass', locale: 'en', glassIntensity: 35, highContrast: 'system' } },
  { name: '390-dark-product-login', width: 390, height: 844, signedIn: false, appearance: { theme: 'dark', material: 'glass', reduceTransparency: false, reduceMotion: true, locale: 'ko', glassIntensity: 45, highContrast: 'system' } },
  { name: '390-dark-product-signed-in', width: 390, height: 844, signedIn: true, appearance: { theme: 'dark', material: 'glass', reduceTransparency: false, reduceMotion: true, locale: 'ko', glassIntensity: 45, highContrast: 'system' } },
];

function withoutSchemaName<T>(value: T): T {
  const copy = structuredClone(value);
  if (typeof copy === 'object' && copy !== null && '$schemaName' in copy) delete (copy as Record<string, unknown>).$schemaName;
  return copy;
}

function makeBootstrap(): typeof bootstrapFixture {
  return withoutSchemaName(bootstrapFixture);
}

function makeCatalog(): typeof catalogFixture {
  const page = withoutSchemaName(catalogFixture);
  page.server_instance_id = bootstrapFixture.server.server_instance_id;
  return page;
}

function makeOperations(): typeof operationsFixture {
  const page = withoutSchemaName(operationsFixture);
  page.server_instance_id = bootstrapFixture.server.server_instance_id;
  return page;
}

async function fulfillJson(route: Route, body: unknown, status = 200): Promise<void> {
  await route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) });
}

async function installAbortRecorder(page: Page): Promise<void> {
  await page.addInitScript(() => {
    const target = window as Window & { __webuiAbortLog?: string[] };
    target.__webuiAbortLog = [];
    const originalFetch = window.fetch.bind(window);
    window.fetch = ((input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === 'string' ? input : input instanceof Request ? input.url : String(input);
      const signal = init?.signal ?? (input instanceof Request ? input.signal : undefined);
      if (url.includes('/ui-api/v1/') && signal) signal.addEventListener('abort', () => target.__webuiAbortLog?.push(url), { once: true });
      return originalFetch(input, init);
    }) as typeof window.fetch;
  });
}

async function readAbortLog(page: Page): Promise<string[]> {
  return page.evaluate(() => (window as Window & { __webuiAbortLog?: string[] }).__webuiAbortLog ?? []);
}

async function browserStorageDump(page: Page): Promise<string> {
  return page.evaluate(() => JSON.stringify({
    localStorage: Object.fromEntries(Array.from({ length: localStorage.length }, (_, index) => {
      const key = localStorage.key(index) ?? '';
      return [key, localStorage.getItem(key)];
    })),
    sessionStorage: Object.fromEntries(Array.from({ length: sessionStorage.length }, (_, index) => {
      const key = sessionStorage.key(index) ?? '';
      return [key, sessionStorage.getItem(key)];
    })),
  }));
}

async function installMockApi(page: Page, mode: MockMode = 'happy'): Promise<{ calls: ApiCall[]; releaseCatalog: () => void; catalogFulfillFailures: string[] }> {
  const calls: ApiCall[] = [];
  const catalogFulfillFailures: string[] = [];
  let releaseCatalog = (): void => undefined;
  let slowCatalogReleased = false;
  const slowCatalog = new Promise<void>((resolve) => { releaseCatalog = () => { slowCatalogReleased = true; resolve(); }; });
  await page.route('**/*', async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    const path = url.pathname;
    if (!path.startsWith('/ui-api/v1/') && path !== '/v1/chat/completions' && path !== '/v1/responses') {
      await route.continue();
      return;
    }
    calls.push({ url: `${path}${url.search}`, method: request.method(), auth: request.headers().authorization ?? '', body: request.postData() ?? '' });
    if (path === '/v1/chat/completions' || path === '/v1/responses') {
      await fulfillJson(route, { error: { code: 'unexpected_inference', message: 'Inference endpoints must not be called while browsing.', retryable: false }, request_id: 'req_unexpected_inference' }, 500);
      return;
    }
    if (path === '/ui-api/v1/bootstrap') {
      if (mode === 'bad-key') await fulfillJson(route, { error: { code: 'unauthorized', message: 'bad key', retryable: false }, request_id: 'req_bad_key' }, 401);
      else if (mode === 'malformed-bootstrap') await fulfillJson(route, { schema_version: 'wrong', server: { server_instance_id: 'srv_bad' } });
      else await fulfillJson(route, makeBootstrap());
      return;
    }
    if (path === '/ui-api/v1/catalog') {
      if (mode === 'slow-catalog' && !slowCatalogReleased) await slowCatalog;
      try {
        await fulfillJson(route, makeCatalog());
      } catch (error) {
        catalogFulfillFailures.push(error instanceof Error ? error.message : String(error));
      }
      return;
    }
    if (path === '/ui-api/v1/operations') {
      if (mode === 'sync-401') await fulfillJson(route, { error: { code: 'unauthorized', message: 'stale session', retryable: false }, request_id: 'req_sync_401' }, 401);
      else await fulfillJson(route, makeOperations());
      return;
    }
    if (path === '/ui-api/v1/events') {
      await route.fulfill({ status: 200, contentType: 'text/event-stream', body: 'data: [DONE]\n\n' });
      return;
    }
    await fulfillJson(route, { error: { code: 'not_found', message: `Unexpected ${path}`, retryable: false }, request_id: 'req_unexpected' }, 404);
  });
  return { calls, releaseCatalog, catalogFulfillFailures };
}

async function bootGallery(page: Page, variant: Variant): Promise<void> {
  await page.setViewportSize({ width: variant.width, height: variant.height });
  await page.addInitScript((appearance) => localStorage.setItem('mlxcel.webui.appearance', JSON.stringify(appearance)), variant.appearance);
  await page.goto('/#gallery');
  await expect(page.getByTestId('app-title')).toBeVisible();
  if (typeof variant.appearance.theme === 'string') await expect(page.locator('html')).toHaveAttribute('data-theme', variant.appearance.theme);
  if (variant.textScale) await page.evaluate((scale) => { document.documentElement.dataset.testTextScale = scale; }, variant.textScale);
  await selectGalleryTab(page, variant.tab);
  if (variant.openDrawer) await page.getByRole('button', { name: /navigation|내비게이션/i }).click();
}


async function bootProduct(page: Page, variant: ProductVariant): Promise<void> {
  await page.setViewportSize({ width: variant.width, height: variant.height });
  await page.addInitScript((appearance) => localStorage.setItem('mlxcel.webui.appearance', JSON.stringify(appearance)), variant.appearance);
  await page.goto('/#models');
  await expect(page.getByTestId('app-title')).toBeVisible();
  if (typeof variant.appearance.theme === 'string') await expect(page.locator('html')).toHaveAttribute('data-theme', variant.appearance.theme);
}

async function submitSessionKey(page: Page, token: string): Promise<void> {
  await page.getByLabel(/Session key|세션 키/i).fill(token);
  await page.getByRole('button', { name: /Connect|연결/i }).click();
}

async function loginWithMockApi(page: Page, token = 'good-key'): Promise<void> {
  await submitSessionKey(page, token);
  await expect(page.getByTestId('connection-authenticated-detail')).toContainText(/catalog 1|카탈로그 1/i);
  await expect(page.getByTestId('connection-authenticated-detail')).toContainText(/operations 1|작업 1/i);
}

async function gotoGalleryWithoutReload(page: Page): Promise<void> {
  await page.evaluate(() => { window.history.replaceState(null, '', '#gallery'); window.dispatchEvent(new HashChangeEvent('hashchange')); });
  await expect(page.getByTestId('gallery-title')).toBeVisible();
}

async function selectGalleryTab(page: Page, tab: GalleryTab): Promise<void> {
  const names: Record<GalleryTab, RegExp> = { controls: /Controls|컨트롤/i, states: /States|상태/i, data: /Data display|데이터 표시/i };
  await page.getByRole('tab', { name: names[tab] }).click();
  await expect(page.getByRole('tab', { name: names[tab] })).toHaveAttribute('aria-selected', 'true');
}


async function settleAnimationFrame(page: Page): Promise<void> {
  await page.evaluate(() => new Promise<void>((resolve) => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))));
}

async function expectDataTableColumnsVisible(page: Page): Promise<void> {
  const metrics = await page.locator('.ds-table').evaluate((table) => {
    const element = table as HTMLElement;
    const headers = Array.from(element.querySelectorAll<HTMLElement>('th')).map((header) => header.getBoundingClientRect());
    return {
      overflow: element.scrollWidth - element.clientWidth,
      widths: headers.map((header) => header.width),
    };
  });
  expect(metrics.overflow).toBeLessThanOrEqual(1);
  expect(metrics.widths.length).toBe(3);
  for (const width of metrics.widths) expect(width).toBeGreaterThan(88);
}

async function pressQuestionShortcut(page: Page): Promise<void> {
  await page.evaluate(() => {
    const target = document.activeElement instanceof HTMLElement ? document.activeElement : document.body;
    target.dispatchEvent(new KeyboardEvent('keydown', { key: '?', bubbles: true, cancelable: true }));
  });
}

async function expectAxeClean(page: Page): Promise<void> {
  const results = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'wcag22aa']).analyze();
  expect(results.violations).toEqual([]);
}

async function expectNoOverflowOrInlineStyles(page: Page): Promise<void> {
  const result = await page.evaluate(() => ({
    overflow: document.documentElement.scrollWidth - document.documentElement.clientWidth,
    inlineStyleCount: document.querySelectorAll('[style]').length,
    smallTargets: Array.from(document.querySelectorAll<HTMLElement>('button, a[href], input, select, textarea')).filter((element) => {
      const rect = element.getBoundingClientRect();
      const style = window.getComputedStyle(element);
      if ((rect.width === 0 && rect.height === 0) || style.visibility === 'hidden' || style.display === 'none') return false;
      return rect.width < 24 || rect.height < 24;
    }).map((element) => element.outerHTML.slice(0, 80)),
  }));
  expect(result.overflow).toBeLessThanOrEqual(1);
  expect(result.inlineStyleCount).toBe(0);
  expect(result.smallTargets).toEqual([]);
}


async function expectCompactToolbarHitTargets(page: Page): Promise<void> {
  const targets = await page.evaluate(() => Array.from(document.querySelectorAll<HTMLElement>('.app-toolbar .ds-icon-button')).filter((element) => {
    const rect = element.getBoundingClientRect();
    const style = window.getComputedStyle(element);
    return rect.width > 0 && rect.height > 0 && style.visibility !== 'hidden' && style.display !== 'none';
  }).map((element) => {
    const rect = element.getBoundingClientRect();
    return { label: element.getAttribute('aria-label') ?? element.textContent?.trim() ?? element.tagName, width: rect.width, height: rect.height };
  }));
  expect(targets.length).toBeGreaterThanOrEqual(3);
  for (const target of targets) {
    expect(target.width, target.label).toBeGreaterThanOrEqual(44);
    expect(target.height, target.label).toBeGreaterThanOrEqual(44);
  }
}

async function expectTextScalePanelsReflow(page: Page): Promise<void> {
  const selectors = [
    '.app-main',
    '.app-toolbar',
    '.app-content-grid',
    '.app-content',
    '.screen-stack',
    '.screen-heading',
    '.screen-heading h1',
    '.screen-heading p',
    '.ds-tabs',
    '.ds-tabs [role="tablist"]',
    '.ds-tabs [role="tab"]',
    '.gallery-grid',
    '.surface-card',
    '.surface-card h2',
    '.surface-card p',
    '.control-row',
    '.ds-button',
    '.ds-field',
    '.ds-field > span',
    '.ds-field small',
  ];
  const result = await page.evaluate((panelSelectors) => {
    const visible = (element: HTMLElement): boolean => {
      const rect = element.getBoundingClientRect();
      const style = window.getComputedStyle(element);
      return rect.width > 0 && rect.height > 0 && style.visibility !== 'hidden' && style.display !== 'none';
    };
    const failures: string[] = [];
    for (const selector of panelSelectors) {
      for (const element of Array.from(document.querySelectorAll<HTMLElement>(selector))) {
        if (!visible(element)) continue;
        const delta = Math.ceil(element.scrollWidth - element.clientWidth);
        if (delta > 1) failures.push(`${selector} overflow=${delta} text=${element.textContent?.trim().slice(0, 80) ?? ''}`);
      }
    }
    return {
      documentOverflow: Math.ceil(document.documentElement.scrollWidth - document.documentElement.clientWidth),
      bodyOverflow: Math.ceil(document.body.scrollWidth - document.body.clientWidth),
      failures,
    };
  }, selectors);
  expect(result.documentOverflow, JSON.stringify(result.failures)).toBeLessThanOrEqual(1);
  expect(result.bodyOverflow, JSON.stringify(result.failures)).toBeLessThanOrEqual(1);
  expect(result.failures).toEqual([]);
}

async function expectLocatorWithinViewportX(locator: Locator): Promise<void> {
  await locator.scrollIntoViewIfNeeded();
  await expect(locator).toBeVisible();
  const box = await locator.boundingBox();
  expect(box).not.toBeNull();
  if (!box) return;
  const viewport = locator.page().viewportSize();
  expect(viewport).not.toBeNull();
  if (!viewport) return;
  expect(Math.floor(box.x)).toBeGreaterThanOrEqual(0);
  expect(Math.ceil(box.x + box.width)).toBeLessThanOrEqual(viewport.width + 1);
}

async function expectTextScaleLabelsReachable(page: Page): Promise<void> {
  await expectTextScalePanelsReflow(page);
  const reachable = [
    page.getByRole('tab', { name: /컨트롤/i }),
    page.getByRole('tab', { name: /상태/i }),
    page.getByRole('tab', { name: /데이터 표시/i }),
    page.getByRole('button', { name: /기본/i }),
    page.getByRole('button', { name: /보조/i }),
    page.getByRole('button', { name: /위험/i }),
    page.getByText('저장소 ID', { exact: true }),
    page.getByText('네이티브 select 콤보박스', { exact: true }),
    page.getByRole('combobox', { name: /네이티브 select 콤보박스/i }),
  ];
  for (const locator of reachable) await expectLocatorWithinViewportX(locator);
}

test.describe('design system gallery and shell', () => {
  for (const variant of variants) {
    test(`renders and compares ${variant.name}`, async ({ page }) => {
      await bootGallery(page, variant);
      await expectAxeClean(page);
      await expectNoOverflowOrInlineStyles(page);
      if (variant.width > 960) await expect(page.getByTestId('toolbar-menu')).toBeHidden();
      if (variant.width <= 560) await expectCompactToolbarHitTargets(page);
      if (variant.tab === 'data' && variant.width >= 1024) await expectDataTableColumnsVisible(page);
      if (variant.textScale === '200') {
        const fontSize = await page.evaluate(() => Number.parseFloat(window.getComputedStyle(document.documentElement).fontSize));
        expect(fontSize).toBeGreaterThanOrEqual(32);
        await expectTextScaleLabelsReachable(page);
        await page.evaluate(() => window.scrollTo(0, 0));
        await settleAnimationFrame(page);
      }
      await expect(page).toHaveScreenshot(`${variant.name}.png`, { animations: 'disabled', maxDiffPixelRatio: 0.005, threshold: 0.2 });
    });
  }


  for (const variant of productVariants) {
    test(`renders and compares ${variant.name}`, async ({ page }) => {
      await installMockApi(page, 'happy');
      await bootProduct(page, variant);
      await expectAxeClean(page);
      await expectNoOverflowOrInlineStyles(page);
      if (variant.signedIn) await loginWithMockApi(page);
      else await expect(page.getByTestId('auth-login')).toBeVisible();
      if (variant.width <= 560) await expectCompactToolbarHitTargets(page);
      await settleAnimationFrame(page);
      await expect(page).toHaveScreenshot(`${variant.name}.png`, { animations: 'disabled', maxDiffPixelRatio: 0.005, threshold: 0.2 });
    });
  }

  test('covers product login, provider routing, all-route logout, and token containment', async ({ page }) => {
    const mock = await installMockApi(page, 'happy');
    await bootProduct(page, productVariants[0]);
    expect(mock.calls).toEqual([]);
    await expect(page.getByTestId('auth-login')).toBeVisible();
    await loginWithMockApi(page, 'good-key');
    await expect.poll(() => mock.calls.some((call) => call.url.startsWith('/ui-api/v1/events'))).toBe(true);
    const snapshotCalls = mock.calls.filter((call) => call.url.startsWith('/ui-api/v1/bootstrap') || call.url.startsWith('/ui-api/v1/catalog') || call.url.startsWith('/ui-api/v1/operations') || call.url.startsWith('/ui-api/v1/events'));
    expect(snapshotCalls.map((call) => call.auth)).toEqual(snapshotCalls.map(() => 'Bearer good-key'));
    const routedPaths = snapshotCalls.map((call) => call.url);
    expect(routedPaths.every((url) => url.startsWith('/ui-api/v1/'))).toBe(true);
    expect(routedPaths.join('\n')).toContain('/ui-api/v1/bootstrap');
    expect(routedPaths.join('\n')).toContain('/ui-api/v1/catalog');
    expect(routedPaths.join('\n')).toContain('/ui-api/v1/operations');
    expect(routedPaths.join('\n')).toContain('/ui-api/v1/events');
    expect(routedPaths.join(' ')).not.toContain('good-key');
    expect(snapshotCalls.map((call) => call.body).join(' ')).not.toContain('good-key');
    await expect(page.locator('body')).not.toContainText('good-key');
    expect(await browserStorageDump(page)).not.toContain('good-key');
    await expect(page.getByTestId('toolbar-logout')).toBeVisible();
    await page.getByRole('link', { name: 'Settings' }).first().click();
    await expect(page.getByTestId('toolbar-logout')).toBeVisible();
    await gotoGalleryWithoutReload(page);
    await expect(page.getByTestId('toolbar-logout')).toBeVisible();
    await page.getByTestId('toolbar-logout').click();
    await expect(page.getByTestId('toolbar-logout')).toHaveCount(0);
  });

  test('covers bad-key, 401 purge, inflight abort, and schema-mismatch reload', async ({ page }) => {
    const badKey = await installMockApi(page, 'bad-key');
    await bootProduct(page, productVariants[0]);
    await submitSessionKey(page, 'bad-key-token');
    await expect(page.locator('body')).toContainText('The session key was rejected');
    expect(badKey.calls.map((call) => call.url).join(' ')).not.toContain('bad-key-token');
    expect(badKey.calls.map((call) => call.body).join(' ')).not.toContain('bad-key-token');
    expect(await browserStorageDump(page)).not.toContain('bad-key-token');
    await expect(page.locator('body')).not.toContainText('bad-key-token');

    await page.unroute('**/*');
    const unauthorized = await installMockApi(page, 'sync-401');
    await page.reload();
    await submitSessionKey(page, 'good-key');
    await expect.poll(() => unauthorized.calls.some((call) => call.url.startsWith('/ui-api/v1/operations') && call.auth === 'Bearer good-key')).toBe(true);
    await expect(page.getByTestId('auth-login')).toBeVisible();
    await expect(page.getByTestId('connection-ready').first()).toHaveText('Shell loaded; local API not connected');

    await page.unroute('**/*');
    await installAbortRecorder(page);
    const slow = await installMockApi(page, 'slow-catalog');
    await page.reload();
    await submitSessionKey(page, 'good-key');
    await expect(page.getByTestId('connection-authenticated-detail')).toContainText('catalog pending');
    await page.getByTestId('toolbar-logout').click();
    slow.releaseCatalog();
    await expect(page.getByTestId('auth-login')).toBeVisible();
    await expect.poll(async () => (await readAbortLog(page)).some((url) => url.includes('/ui-api/v1/catalog'))).toBe(true);

    await page.unroute('**/*');
    await installMockApi(page, 'malformed-bootstrap');
    await page.reload();
    await submitSessionKey(page, 'good-key');
    await expect(page.locator('body')).toContainText('UI schema mismatch');
    await Promise.all([page.waitForLoadState('domcontentloaded'), page.getByRole('button', { name: /Reload|새로고침/i }).click()]);
    await expect(page.getByTestId('auth-login')).toBeVisible();
  });


  test('keeps compact reflow on production selectors without masking overflow', async ({ page }) => {
    const productionCss = [
      readFileSync(fileURLToPath(new URL('../src/styles.css', import.meta.url)), 'utf8'),
      readFileSync(fileURLToPath(new URL('../src/design-system/components.css', import.meta.url)), 'utf8'),
    ].join('\n');
    expect(productionCss).not.toContain('data-test-text-scale');
    await bootGallery(page, { name: '390-production-compact-gallery-controls', width: 390, height: 844, tab: 'controls', appearance: { theme: 'dark', material: 'opaque', reduceTransparency: true, reduceMotion: true, locale: 'ko', glassIntensity: 100, highContrast: 'off' } });
    await expectNoOverflowOrInlineStyles(page);
    await expectTextScalePanelsReflow(page);
    const primary = page.getByRole('button', { name: /기본/i });
    await primary.focus();
    await expectLocatorWithinViewportX(primary);
    await expectCompactToolbarHitTargets(page);
  });

  test('keeps production routes honest while gallery stays a direct artifact route', async ({ page }) => {
    await page.addInitScript(() => localStorage.setItem('mlxcel.webui.appearance', JSON.stringify({ theme: 'light', material: 'glass', locale: 'en', highContrast: 'system' })));
    await page.goto('/#models');
    await expect(page.getByTestId('connection-prompt-body')).toBeVisible();
    await expect(page.getByText('No local models yet')).toHaveCount(0);
    await expect(page.getByText('Unloaded')).toHaveCount(0);
    await expect(page.getByTestId('nav-gallery')).toHaveCount(0);
    await page.getByTestId('toolbar-command').click();
    await expect(page.getByRole('button', { name: 'Gallery' })).toHaveCount(0);
    await page.goto('/#gallery');
    await expect(page.getByTestId('gallery-title')).toBeVisible();
  });

  test('covers keyboard controls, focus containment, and modal shortcut suppression', async ({ page }) => {
    await bootGallery(page, variants[0]);
    const select = page.getByRole('combobox', { name: /Native select/i });
    await select.focus();
    await page.keyboard.press('ArrowDown');
    await expect(select).toHaveValue('unloaded');
    await selectGalleryTab(page, 'controls');
    await page.getByRole('tab', { name: /Controls/i }).focus();
    await page.keyboard.press('End');
    await expect(page.getByRole('tabpanel')).toContainText('Model rows');
    await page.keyboard.press('Home');
    await expect(page.getByRole('tabpanel')).toContainText('Buttons and fields');
    const opener = page.getByRole('button', { name: /Open dialog/i });
    await opener.focus();
    await opener.click();
    const dialog = page.locator('dialog[open]');
    await expect(dialog).toHaveAttribute('data-testid', 'gallery-dialog');
    const closeButton = dialog.getByTestId('dialog-close');
    const deleteTokenField = dialog.getByLabel(/Type DELETE/i);
    await expect(closeButton).toBeFocused();
    await page.keyboard.press('Tab');
    await expect(deleteTokenField).toBeFocused();
    await page.keyboard.press('Shift+Tab');
    await expect(closeButton).toBeFocused();
    await page.keyboard.press('Escape');
    await expect(page.getByTestId('gallery-dialog')).toBeHidden();
    await expect(opener).toBeFocused();
    await page.keyboard.press('Control+K');
    await expect(page.getByTestId('command-dialog')).toBeVisible();
    await pressQuestionShortcut(page);
    await expect(page.getByTestId('help-dialog')).toBeHidden();
    await page.keyboard.press('Escape');
    await expect(page.getByTestId('command-dialog')).toBeHidden();
    await expect(opener).toBeFocused();
    await settleAnimationFrame(page);
    await expect(opener).toBeFocused();
    await pressQuestionShortcut(page);
    await expect(page.getByTestId('help-dialog')).toBeVisible();
    await page.keyboard.press('Control+K');
    await expect(page.getByTestId('command-dialog')).toBeHidden();
  });

  test('covers system contrast, reduced motion, visibility pause, and glass fallback', async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 900 });
    const client = await page.context().newCDPSession(page);
    await client.send('Emulation.setEmulatedMedia', { features: [{ name: 'prefers-contrast', value: 'more' }, { name: 'prefers-reduced-motion', value: 'reduce' }] });
    await page.goto('/#gallery');
    await page.evaluate(() => localStorage.setItem('mlxcel.webui.appearance', JSON.stringify({ theme: 'system', material: 'glass', locale: 'en', highContrast: 'system', glassIntensity: 35 })));
    await page.reload();
    await expect(page.getByTestId('gallery-title')).toBeVisible();
    await expect(page.locator('html')).toHaveAttribute('data-high-contrast', 'system');
    await expect(page.locator('.surface-card').first()).toHaveCSS('border-top-width', '2px');
    await page.evaluate(() => localStorage.setItem('mlxcel.webui.appearance', JSON.stringify({ theme: 'light', material: 'glass', locale: 'en', highContrast: 'off', glassIntensity: 35 })));
    await page.reload();
    await expect(page.locator('html')).toHaveAttribute('data-high-contrast', 'off');
    await expect(page.locator('.surface-card').first()).toHaveCSS('border-top-width', '1px');
    await selectGalleryTab(page, 'states');
    await expect(page.locator('.ds-status[data-state="loading"]')).toHaveCSS('animation-name', 'none');
    await page.evaluate(() => { Object.defineProperty(document, 'hidden', { value: true, configurable: true }); document.dispatchEvent(new Event('visibilitychange')); });
    await expect(page.locator('.ds-status[data-state="loading"]')).toHaveCSS('animation-name', 'none');
  });

  test('forces unsupported backdrop detection while glass is requested', async ({ page }) => {
    await page.addInitScript(() => {
      const original = CSS.supports.bind(CSS);
      CSS.supports = ((query: string) => query.includes('backdrop-filter') ? false : original(query)) as typeof CSS.supports;
    });
    await bootGallery(page, { name: 'fallback', width: 1024, height: 768, tab: 'controls', appearance: { theme: 'light', material: 'glass', locale: 'en', glassIntensity: 70, highContrast: 'system' } });
    await expect(page.locator('html')).toHaveAttribute('data-backdrop-filter', 'unsupported');
    await expect(page.locator('html')).toHaveAttribute('data-material', 'glass');
    await expect(page.locator('.material-glass').first()).toHaveCSS('backdrop-filter', /blur\(0px\)|none/);
  });
});
