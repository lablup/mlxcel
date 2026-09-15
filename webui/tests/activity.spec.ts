import { expect, test } from '@playwright/test';
import { bootProduct, installMockApi, loginWithMockApi, productVariants } from './browser-fixtures';
import { expectAxeClean, expectSafeLayout } from './browser-assertions';

for (const width of [1440, 390]) {
  test(`Activity shares controls and exposes honest empty observation at ${width}`, async ({ page }) => {
    const mock = await installMockApi(page);
    await bootProduct(page, { ...productVariants[0], width, appearance: { ...productVariants[0].appearance, theme: width === 390 ? 'dark' : 'light', locale: 'en' } });
    await loginWithMockApi(page);
    await page.evaluate(() => { location.hash = '#activity'; });
    await expect(page.getByTestId('activity-page')).toBeVisible();
    await expect(page.getByRole('heading', { name: 'Activity', exact: true })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Export sanitized diagnostics' })).toBeVisible();
    await expect(page.getByRole('combobox')).toBeVisible();
    await expectAxeClean(page);
    await expectSafeLayout(page);
    await page.getByRole('button', { name: 'Show recent history' }).click();
    await expect(page.getByText(/Five-minute in-memory history/)).toBeVisible();
    await expect(page.locator('.activity-chart')).toHaveCount(0);
    expect(mock.calls.every((call) => call.method === 'GET')).toBe(true);
    await page.getByTestId('toolbar-logout').click();
    await expect(page.getByTestId('auth-login')).toBeVisible();
    await expect(page.getByTestId('activity-page')).toHaveCount(0);
  });
}
