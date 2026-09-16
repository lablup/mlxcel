// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import { writeFile } from 'node:fs/promises';
import { expect, test, type Page, type TestInfo } from '@playwright/test';
import { expectAxeClean, expectSafeLayout } from './browser-assertions';
import { bootProduct, browserStorageDump, installMockApi, loginWithMockApi } from './browser-fixtures';


async function attachJsonEvidence(testInfo: TestInfo, name: string, value: Record<string, unknown>): Promise<void> {
  const path = testInfo.outputPath(name);
  await writeFile(path, JSON.stringify(value, null, 2), { mode: 0o600, flag: 'wx' });
  await testInfo.attach(name, { path, contentType: 'application/json' });
}

const engineVariants = [
  {
    name: '390-ko-dark-opaque-reduced-motion',
    width: 390,
    height: 844,
    appearance: { theme: 'dark', material: 'opaque', locale: 'ko', reduceMotion: true, highContrast: 'off', glassIntensity: 0 },
  },
  {
    name: '1024-ko-light-tinted-high-contrast',
    width: 1024,
    height: 768,
    appearance: { theme: 'light', material: 'tinted', locale: 'ko', reduceMotion: false, highContrast: 'on', glassIntensity: 45 },
  },
  {
    name: '1440-en-light-glass-system-contrast',
    width: 1440,
    height: 900,
    appearance: { theme: 'light', material: 'glass', locale: 'en', reduceMotion: false, highContrast: 'system', glassIntensity: 35 },
  },
] as const;

function visibleByTestId(page: Page, testId: string) {
  return page.locator(`[data-testid="${testId}"]:visible`).first();
}

async function openNavigationIfNeeded(page: Page, testId: string): Promise<void> {
  if ((await visibleByTestId(page, testId).count()) > 0) return;
  await page.getByRole('button', { name: /navigation|내비게이션/i }).click();
  await expect(visibleByTestId(page, testId)).toBeVisible();
}

async function navigate(page: Page, testId: string, expectedHash: RegExp): Promise<void> {
  await openNavigationIfNeeded(page, testId);
  await visibleByTestId(page, testId).click();
  await expect(page).toHaveURL(expectedHash);
}

for (const variant of engineVariants) {
  test(`cross-engine product surfaces stay accessible and keyboard-usable: ${variant.name}`, async ({ page }, testInfo) => {
    const mock = await installMockApi(page, 'happy');
    await bootProduct(page, { ...variant, signedIn: true });
    const token = `engine-${testInfo.project.name}-${variant.width}`;
    await loginWithMockApi(page, token);

    await expect(page.getByTestId('models-table')).toBeVisible();
    await page.keyboard.press('Tab');
    await expect(page.locator(':focus')).toBeVisible();
    await expectAxeClean(page);
    await expectSafeLayout(page);

    await navigate(page, 'nav-chat', /#chat$/);
    const composer = page.getByRole('textbox', { name: /Message|메시지/i, exact: true });
    await expect(composer).toBeVisible();
    await composer.fill('안녕하세요 cross-engine composition guard');
    const callsBeforeIme = mock.calls.length;
    await composer.dispatchEvent('compositionstart');
    await composer.dispatchEvent('keydown', { key: 'Enter', bubbles: true, cancelable: true, isComposing: true });
    await composer.dispatchEvent('compositionend');
    const imeCalls = mock.calls.slice(callsBeforeIme);
    expect(imeCalls.filter((call) => call.url.startsWith('/v1/chat/completions') || call.url.startsWith('/v1/responses') || call.method !== 'GET')).toEqual([]);
    await expect(composer).toHaveValue('안녕하세요 cross-engine composition guard');
    await expectAxeClean(page);
    await expectSafeLayout(page);

    await navigate(page, 'nav-settings', /#settings$/);
    await expect(page.getByRole('heading', { name: /Generation · next request|생성 · 다음 요청/i })).toBeVisible();
    await page.getByRole('button', { name: /Reset request defaults|요청 기본값 초기화/i }).focus();
    await page.keyboard.press('Enter');
    await expect(page.getByRole('dialog')).toBeVisible();
    await page.keyboard.press('Escape');
    await expect(page.getByRole('dialog')).toBeHidden();
    await expectAxeClean(page);
    await expectSafeLayout(page);

    await navigate(page, 'nav-activity', /#activity$/);
    await expect(page.getByTestId('activity-page')).toBeVisible();
    await expect(page.getByRole('button', { name: /Export sanitized diagnostics|진단 내보내기/i })).toBeVisible();
    await expectAxeClean(page);
    await expectSafeLayout(page);

    const apiCalls = mock.calls.filter((call) => call.url.startsWith('/ui-api/v1/'));
    expect(apiCalls.length).toBeGreaterThan(0);
    expect(apiCalls.every((call) => call.auth === `Bearer ${token}`)).toBe(true);
    expect(mock.calls.map((call) => call.url).join(' ')).not.toContain(token);
    expect(mock.calls.map((call) => call.body).join(' ')).not.toContain(token);
    await expect(page.locator('body')).not.toContainText(token);
    expect(await browserStorageDump(page)).not.toContain(token);
    await attachJsonEvidence(testInfo, 'engine-surface-evidence.json', { project: testInfo.project.name, variant: variant.name, viewport: { width: variant.width, height: variant.height }, exercised: ['models', 'chat-composition-event-guard', 'settings-dialog-keyboard', 'activity'], api_calls: mock.calls.length });
  });
}
