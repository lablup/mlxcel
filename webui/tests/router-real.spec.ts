// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import { mkdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { expect, test, type APIRequestContext, type Page } from '@playwright/test';
import { expectAxeClean, expectSafeLayout } from './browser-assertions';

type Operation = { operation_id: string; kind: string; state: string; result?: unknown; error?: unknown };
type CatalogItem = { identity: { id: string; display_name: string; inference_id: string; revision: number }; lifecycle: { state: string; worker_exit_observed: boolean; active_requests: number }; capabilities: Array<{ task: string; phase: string; available: boolean }> };

const target = new URL(process.env.MLXCEL_WEBUI_ROUTER_URL ?? '');
const artifacts = process.env.MLXCEL_WEBUI_ROUTER_ARTIFACTS ?? '';
const keyPath = process.env.MLXCEL_WEBUI_ROUTER_KEY_FILE ?? '';
const apiBase = target.pathname.slice(0, -'/webui/'.length);
const token = readFileSync(keyPath, 'utf8').trim();
const auth = { Authorization: `Bearer ${token}` };

function saveArtifact(name: string, value: unknown): void {
  mkdirSync(artifacts, { recursive: true });
  writeFileSync(join(artifacts, name), typeof value === 'string' ? value : JSON.stringify(value, null, 2), { mode: 0o600 });
}

async function apiJson<T>(request: APIRequestContext, method: 'get' | 'post' | 'delete', path: string, body?: unknown): Promise<T> {
  const response = await request[method](`${target.origin}${apiBase}${path}`, { headers: auth, data: body });
  expect(response.ok(), `${method.toUpperCase()} ${path}: ${response.status()} ${await response.text()}`).toBe(true);
  return (await response.json()) as T;
}

async function waitOperation(request: APIRequestContext, operationId: string): Promise<Operation> {
  for (let attempt = 0; attempt < 200; attempt += 1) {
    const operation = await apiJson<Operation>(request, 'get', `/ui-api/v1/operations/${operationId}`);
    if (['succeeded', 'failed', 'cancelled'].includes(operation.state)) {
      expect(operation.state, JSON.stringify(operation)).toBe('succeeded');
      return operation;
    }
    await new Promise(resolve => setTimeout(resolve, 25));
  }
  throw new Error(`operation ${operationId} did not reach a terminal state`);
}


async function waitOperationKind(request: APIRequestContext, kind: string): Promise<string> {
  for (let attempt = 0; attempt < 200; attempt += 1) {
    const operations = (await apiJson<{ items: Operation[] }>(request, 'get', '/ui-api/v1/operations')).items;
    const operation = operations.find(item => item.kind === kind);
    if (operation) return operation.operation_id;
    await new Promise(resolve => setTimeout(resolve, 25));
  }
  throw new Error(`operation kind ${kind} was not observed`);
}


function requireCatalogItem(item: CatalogItem | undefined, label: string): CatalogItem {
  if (!item) throw new Error(`${label} catalog entry not found`);
  return item;
}

async function catalog(request: APIRequestContext): Promise<CatalogItem[]> {
  const body = await apiJson<{ items: CatalogItem[] }>(request, 'get', '/ui-api/v1/catalog?limit=200');
  return body.items;
}

async function login(page: Page): Promise<void> {
  await page.goto(`${target.href}#models`);
  await page.getByLabel(/Session key|세션 키/i).fill(token);
  await page.getByRole('button', { name: /Connect|연결/i }).click();
  await expect(page.getByTestId('models-table')).toBeVisible();
}

async function refresh(page: Page): Promise<void> {
  await page.getByRole('button', { name: 'Refresh server state', exact: true }).first().click();
}

test('secured Rust router drives download, load, chat, unload, remove and security cases', async ({ page, request }) => {
  expect(statSync(keyPath).mode & 0o777).toBe(0o600);
  const external: string[] = [];
  page.on('request', req => { const url = new URL(req.url()); if (!['data:', 'blob:'].includes(url.protocol) && url.origin !== target.origin) external.push(req.url()); });
  await page.addInitScript(() => { const violations: string[] = []; Object.defineProperty(window, '__routerCspViolations', { value: violations }); document.addEventListener('securitypolicyviolation', event => violations.push(`${event.effectiveDirective}: ${event.blockedURI}`)); });

  const shell = await page.goto(`${target.href}#models`);
  expect(shell?.ok()).toBe(true);
  const csp = shell?.headers()['content-security-policy'];
  expect(csp).toMatch(/(?:^|;)\s*style-src 'self'(?:;|$)/);
  expect(csp).not.toContain("'unsafe-inline'");
  expect(csp).not.toContain("'unsafe-eval'");
  await login(page);
  await expectSafeLayout(page); await expectAxeClean(page);

  await page.getByTestId('models-add').click();
  await page.getByTestId('models-repo').fill('mlx-community/Router-Harness-Fake-4bit');
  await page.getByTestId('models-public-repo').check();
  await page.getByTestId('models-download-submit').click();
  const acceptedDownload = await waitOperationKind(request, 'download');
  await waitOperation(request, acceptedDownload);
  await refresh(page);
  await expect.poll(async () => (await catalog(request)).length).toBeGreaterThan(1);

  const downloaded = requireCatalogItem((await catalog(request)).find(item => item.identity.display_name.includes('Router-Harness-Fake')) ?? (await catalog(request)).find(item => item.identity.id.includes('Router-Harness-Fake')), 'downloaded fake');
  await page.getByRole('button', { name: new RegExp(`Inspect .*${downloaded.identity.display_name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}`) }).click();
  await page.getByTestId('models-load').click();
  const loadAccepted = await waitOperationKind(request, 'model_load');
  await waitOperation(request, loadAccepted);
  await refresh(page);
  await expect.poll(async () => (await catalog(request)).find(item => item.identity.id === downloaded.identity.id)?.capabilities.some(cap => cap.task === 'chat' && cap.phase === 'provider_ready' && cap.available)).toBe(true);

  const ready = requireCatalogItem((await catalog(request)).find(item => item.identity.id === downloaded.identity.id), 'ready fake');
  await page.goto(`${target.href}#chat`);
  await page.getByRole('combobox', { name: 'Model for next turn' }).click();
  await page.getByRole('option', { name: new RegExp(ready.identity.display_name) }).click();
  await page.getByRole('textbox', { name: 'Message', exact: true }).fill('Say hello from the router harness.');
  await page.getByRole('button', { name: 'Send', exact: true }).click();
  await expect(page.locator('.chat-turn header')).toContainText('complete', { timeout: 30000 });
  await expect(page.locator('.chat-markdown')).toContainText(/router harness/i);

  const unloadAccepted = await apiJson<{ operation_id: string }>(request, 'post', '/ui-api/v1/model-actions', { model_id: ready.identity.id, action: 'unload', expected_revision: ready.identity.revision, idempotency_key: 'router-real-unload' });
  await waitOperation(request, unloadAccepted.operation_id);
  await expect.poll(async () => (await catalog(request)).find(item => item.identity.id === ready.identity.id)?.lifecycle.worker_exit_observed).toBe(true);
  const unloaded = requireCatalogItem((await catalog(request)).find(item => item.identity.id === ready.identity.id), 'unloaded fake');
  const removeAccepted = await apiJson<{ operation_id: string }>(request, 'post', '/ui-api/v1/model-removals', { model_id: unloaded.identity.id, expected_revision: unloaded.identity.revision, confirm_model_id: unloaded.identity.id, idempotency_key: 'router-real-remove' });
  await waitOperation(request, removeAccepted.operation_id);
  await expect.poll(async () => (await catalog(request)).some(item => item.identity.id === ready.identity.id)).toBe(false);

  const unauth = await request.get(`${target.origin}${apiBase}/ui-api/v1/catalog`);
  expect(unauth.status()).toBe(401);
  const hostile = await request.post(`${target.origin}${apiBase}/ui-api/v1/model-actions`, { headers: { ...auth, Host: 'foreign.invalid', Origin: 'http://foreign.invalid', 'Sec-Fetch-Site': 'cross-site' }, data: {} });
  expect(hostile.status()).toBe(403);
  const legacyHostile = await request.post(`${target.origin}${apiBase}/v1/chat/completions`, { headers: { ...auth, Host: 'foreign.invalid', Origin: 'http://foreign.invalid', 'Sec-Fetch-Site': 'cross-site' }, data: {} });
  expect(legacyHostile.status()).toBe(403);
  const missingAsset = await request.get(`${target.origin}${apiBase}/webui/assets/not-present.js`);
  expect(missingAsset.status()).toBe(404);

  const violations = await page.evaluate(() => Reflect.get(window, '__routerCspViolations'));
  expect(violations).toEqual([]); expect(external).toEqual([]);
  await expectSafeLayout(page); await expectAxeClean(page);
  saveArtifact('router-real-evidence.json', { csp, external, violations, final_catalog_size: (await catalog(request)).length, api_base: apiBase, note: 'Fake model/downloader leaves only; secured Rust router, embedded bundle, auth, CSP and lifecycle routes were real.' });
});
