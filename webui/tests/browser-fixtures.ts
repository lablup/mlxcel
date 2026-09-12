import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { expect, type Page, type Route } from '@playwright/test';

export type GalleryTab = 'controls' | 'states' | 'data';
export type Variant = { name: string; width: number; height: number; appearance: Record<string, unknown>; tab: GalleryTab; openDrawer?: boolean; textScale?: '200' };
export type ProductVariant = { name: string; width: number; height: number; appearance: Record<string, unknown>; signedIn: boolean };
type MockMode = 'happy' | 'bad-key' | 'malformed-bootstrap' | 'sync-401' | 'slow-catalog';
export interface ApiCall { readonly url: string; readonly method: string; readonly auth: string; readonly body: string }

const bootstrapFixture = loadFixture<typeof import('../../tests/fixtures/webui/examples/bootstrap.model-free.json')>('../../tests/fixtures/webui/examples/bootstrap.model-free.json');
const catalogFixture = loadFixture<typeof import('../../tests/fixtures/webui/examples/catalog.page.json')>('../../tests/fixtures/webui/examples/catalog.page.json');
const operationsFixture = loadFixture<typeof import('../../tests/fixtures/webui/examples/operations.list.json')>('../../tests/fixtures/webui/examples/operations.list.json');

function loadFixture<T>(relativePath: string): T {
  return JSON.parse(readFileSync(fileURLToPath(new URL(relativePath, import.meta.url)), 'utf8')) as T;
}

export function loadProductionCss(): string {
  return [
    readFileSync(fileURLToPath(new URL('../src/styles.css', import.meta.url)), 'utf8'),
    readFileSync(fileURLToPath(new URL('../src/design-system/components.css', import.meta.url)), 'utf8'),
  ].join('\n');
}

export const variants: Variant[] = [
  { name: '1440-light-gallery-controls', width: 1440, height: 900, tab: 'controls', appearance: { theme: 'light', material: 'glass', locale: 'en', glassIntensity: 35, highContrast: 'system' } },
  { name: '1440-light-gallery-data', width: 1440, height: 900, tab: 'data', appearance: { theme: 'light', material: 'opaque', locale: 'en', glassIntensity: 0, highContrast: 'system' } },
  { name: '1440-dark-gallery-controls', width: 1440, height: 900, tab: 'controls', appearance: { theme: 'dark', material: 'glass', locale: 'en', glassIntensity: 45, highContrast: 'system' } },
  { name: '1440-dark-gallery-data', width: 1440, height: 900, tab: 'data', appearance: { theme: 'dark', material: 'opaque', locale: 'en', glassIntensity: 0, highContrast: 'system' } },
  { name: '1024-tinted-gallery-data-inspector', width: 1024, height: 768, tab: 'data', appearance: { theme: 'light', material: 'tinted', locale: 'ko', glassIntensity: 45, highContrast: 'system' } },
  { name: '390-opaque-gallery-controls-drawer-cjk', width: 390, height: 844, tab: 'controls', openDrawer: true, appearance: { theme: 'light', material: 'opaque', locale: 'ko', glassIntensity: 0, highContrast: 'system' } },
  { name: '1440-highcontrast-gallery-states', width: 1440, height: 900, tab: 'states', appearance: { theme: 'light', material: 'opaque', highContrast: 'on', locale: 'en', glassIntensity: 0 } },
  { name: '390-dark-opaque-textscale200-gallery-controls', width: 390, height: 844, tab: 'controls', appearance: { theme: 'dark', material: 'opaque', reduceTransparency: true, reduceMotion: true, locale: 'ko', glassIntensity: 100, highContrast: 'off' }, textScale: '200' },
];

export const productVariants: ProductVariant[] = [
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

export async function installAbortRecorder(page: Page): Promise<void> {
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

export async function readAbortLog(page: Page): Promise<string[]> {
  return page.evaluate(() => (window as Window & { __webuiAbortLog?: string[] }).__webuiAbortLog ?? []);
}

export async function browserStorageDump(page: Page): Promise<string> {
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

export async function installMockApi(page: Page, mode: MockMode = 'happy'): Promise<{ calls: ApiCall[]; releaseCatalog: () => void; catalogFulfillFailures: string[] }> {
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

export async function bootGallery(page: Page, variant: Variant): Promise<void> {
  await page.setViewportSize({ width: variant.width, height: variant.height });
  await page.addInitScript((appearance) => localStorage.setItem('mlxcel.webui.appearance', JSON.stringify(appearance)), variant.appearance);
  await page.goto('/#gallery');
  await expect(page.getByTestId('app-title')).toBeVisible();
  if (typeof variant.appearance.theme === 'string') await expect(page.locator('html')).toHaveAttribute('data-theme', variant.appearance.theme);
  if (variant.textScale) await page.evaluate((scale) => { document.documentElement.dataset.testTextScale = scale; }, variant.textScale);
  await selectGalleryTab(page, variant.tab);
  if (variant.openDrawer) await page.getByRole('button', { name: /navigation|내비게이션/i }).click();
}

export async function bootProduct(page: Page, variant: ProductVariant): Promise<void> {
  await page.setViewportSize({ width: variant.width, height: variant.height });
  await page.addInitScript((appearance) => localStorage.setItem('mlxcel.webui.appearance', JSON.stringify(appearance)), variant.appearance);
  await page.goto('/#models');
  await expect(page.getByTestId('app-title')).toBeVisible();
  if (typeof variant.appearance.theme === 'string') await expect(page.locator('html')).toHaveAttribute('data-theme', variant.appearance.theme);
}

export async function submitSessionKey(page: Page, token: string): Promise<void> {
  await page.getByLabel(/Session key|세션 키/i).fill(token);
  await page.getByRole('button', { name: /Connect|연결/i }).click();
}

export async function loginWithMockApi(page: Page, token = 'good-key'): Promise<void> {
  await submitSessionKey(page, token);
  await expect(page.getByTestId('connection-authenticated-detail')).toContainText(/catalog 1|카탈로그 1/i);
  await expect(page.getByTestId('connection-authenticated-detail')).toContainText(/operations 1|작업 1/i);
}

export async function gotoGalleryWithoutReload(page: Page): Promise<void> {
  await page.evaluate(() => { window.history.replaceState(null, '', '#gallery'); window.dispatchEvent(new HashChangeEvent('hashchange')); });
  await expect(page.getByTestId('gallery-title')).toBeVisible();
}

export async function selectGalleryTab(page: Page, tab: GalleryTab): Promise<void> {
  const names: Record<GalleryTab, RegExp> = { controls: /Controls|컨트롤/i, states: /States|상태/i, data: /Data display|데이터 표시/i };
  await page.getByRole('tab', { name: names[tab] }).click();
  await expect(page.getByRole('tab', { name: names[tab] })).toHaveAttribute('aria-selected', 'true');
}

export async function settleAnimationFrame(page: Page): Promise<void> {
  await page.evaluate(() => new Promise<void>((resolve) => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))));
}
