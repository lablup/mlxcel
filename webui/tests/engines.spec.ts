// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import { expect, test } from '@playwright/test';
import { expectAxeClean, expectSafeLayout } from './browser-assertions';
import { bootGallery, bootProduct, installMockApi, loginWithMockApi, productVariants, selectGalleryTab, variants } from './browser-fixtures';

test('gallery controls and states are usable without Chromium-only CDP or pixel baselines', async ({ page }, testInfo) => {
  const variant = variants.find(item => item.name.includes('1024')) ?? variants[0];
  await bootGallery(page, variant);
  await expect(page.getByTestId('gallery-title')).toBeVisible();
  await expectAxeClean(page);
  await expectSafeLayout(page);
  await selectGalleryTab(page, 'states');
  const heading = page.locator('[data-testid="gallery-empty"] .empty-state__title');
  await expect(heading).toBeVisible();
  await expect(heading).toHaveAttribute('role', 'heading');
  await expect(heading).toHaveAttribute('aria-level', '2');
  await expect(page.getByRole('progressbar')).toHaveCount(2);
  await testInfo.attach('engine-gallery-dom.json', { body: JSON.stringify({ project: testInfo.project.name, heading: await heading.innerText() }), contentType: 'application/json' });
});

test('product shell login, navigation, and model table survive each browser engine', async ({ page }, testInfo) => {
  const mock = await installMockApi(page, 'happy');
  await bootProduct(page, productVariants.find(item => item.signedIn) ?? productVariants[0]);
  await loginWithMockApi(page, `engine-${testInfo.project.name}-key`);
  await expect(page.getByTestId('models-table')).toBeVisible();
  await page.getByRole('link', { name: 'Chat' }).first().click();
  await expect(page.getByRole('textbox', { name: 'Message', exact: true })).toBeVisible();
  await page.getByRole('link', { name: 'Activity' }).first().click();
  await expect(page.getByText('All measurements and sources')).toBeVisible();
  await expectAxeClean(page);
  await expectSafeLayout(page);
  expect(mock.calls.map(call => call.auth).join(' ')).not.toContain(`engine-${testInfo.project.name}-key`);
});
