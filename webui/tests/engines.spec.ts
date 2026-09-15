// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import { expect, test, type Page } from '@playwright/test';
import { expectAxeClean, expectSafeLayout } from './browser-assertions';
import { bootProduct, installMockApi, loginWithMockApi } from './browser-fixtures';

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

async function openNavigationIfNeeded(page: Page, testId: string): Promise<void> {
  const target = page.getByTestId(testId);
  if (await target.isVisible()) return;
  await page.getByRole('button', { name: /navigation|내비게이션/i }).click();
  await expect(target).toBeVisible();
}

async function navigate(page: Page, testId: string, expectedHash: RegExp): Promise<void> {
  await openNavigationIfNeeded(page, testId);
  await page.getByTestId(testId).click();
  await expect(page).toHaveURL(expectedHash);
}

for (const variant of engineVariants) {
  test(`cross-engine product surfaces stay accessible and keyboard-usable: ${variant.name}`, async ({ page }, testInfo) => {
    const mock = await installMockApi(page, 'happy');
    await bootProduct(page, { ...variant, signedIn: true });
    await loginWithMockApi(page, `engine-${testInfo.project.name}-${variant.width}`);

    await expect(page.getByTestId('models-table')).toBeVisible();
    await page.keyboard.press('Tab');
    await expect(page.locator(':focus')).toBeVisible();
    await expectAxeClean(page);
    await expectSafeLayout(page);

    await navigate(page, 'nav-chat', /#chat$/);
    const composer = page.getByRole('textbox', { name: /Message|메시지/i, exact: true });
    await expect(composer).toBeVisible();
    await composer.fill('안녕하세요 cross-engine IME');
    const callsBeforeIme = mock.calls.length;
    await composer.dispatchEvent('compositionstart');
    await composer.dispatchEvent('keydown', { key: 'Enter', bubbles: true, cancelable: true, isComposing: true });
    await composer.dispatchEvent('compositionend');
    expect(mock.calls).toHaveLength(callsBeforeIme);
    await expect(composer).toHaveValue('안녕하세요 cross-engine IME');
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

    expect(mock.calls.map((call) => call.auth).join(' ')).not.toContain(`engine-${testInfo.project.name}-${variant.width}`);
    await testInfo.attach('engine-surface-evidence.json', {
      body: JSON.stringify({ project: testInfo.project.name, variant: variant.name, viewport: { width: variant.width, height: variant.height }, exercised: ['models', 'chat-ime', 'settings-dialog-keyboard', 'activity'], api_calls: mock.calls.length }),
      contentType: 'application/json',
    });
  });
}
