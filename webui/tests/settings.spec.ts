import { test, expect } from '@playwright/test';
import { bootProduct, installMockApi, loginWithMockApi, productVariants } from './browser-fixtures';
import { expectAxeClean, expectSafeLayout } from './browser-assertions';

for (const width of [390, 1440]) {
  test(`scoped Settings remains keyboard-accessible at ${width}`, async ({ page }) => {
    await installMockApi(page, 'happy');
    await bootProduct(page, { ...productVariants[0], width, height: 900 });
    await loginWithMockApi(page);
    await page.evaluate(() => { window.location.hash = '#settings'; });
    await expect(page.getByRole('heading', { name: 'Generation · next request' })).toBeVisible();
    await expect(page.getByRole('heading', { name: 'Next load · browser profile' })).toBeVisible();
    await page.getByRole('button', { name: 'Reset request defaults…', exact: true }).click();
    await expect(page.getByRole('dialog')).toBeVisible();
    await page.keyboard.press('Escape');
    await expect(page.getByRole('dialog')).toBeHidden();
    await expect(page.getByRole('button', { name: 'Reset request defaults…', exact: true })).toBeFocused();
    await expectAxeClean(page);
    await expectSafeLayout(page);
    await page.getByTestId('toolbar-logout').click();
    await expect(page.getByRole('heading', { name: 'Next load · browser profile' })).toHaveCount(0);
    await expect(page.getByRole('heading', { name: 'Generation · next request' })).toBeVisible();
  });
}
