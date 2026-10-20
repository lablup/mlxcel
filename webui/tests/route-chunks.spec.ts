import { test, expect } from '@playwright/test';
import { bootProduct, installMockApi, loginWithMockApi, productVariants } from './browser-fixtures';
import { fixtureString } from './strings-fixture';

const settingsChunk = /\/assets\/settings-screen-[^/]+\.js$/;

test('a route chunk that fails to load keeps the shell and recovers on reload', async ({ page }) => {
  await installMockApi(page, 'happy');
  await page.route(settingsChunk, (route) => route.abort());
  await bootProduct(page, productVariants[1]);
  await loginWithMockApi(page);

  await page.evaluate(() => { window.location.hash = '#settings'; });
  await expect(page.getByTestId('route-load-failed-title')).toBeVisible();
  await expect(page.getByTestId('route-load-failed-title')).toContainText(fixtureString('en', 'route.load_failed.title'));
  await expect(page.getByTestId('app-title')).toBeVisible();
  await expect(page.getByRole('navigation', { name: /Primary navigation/i })).toBeVisible();

  // A reload fetches the chunk again; the browser keeps the failed import until then.
  await page.unroute(settingsChunk);
  await page.getByTestId('route-load-failed-title').getByRole('button', { name: fixtureString('en', 'common.reload') }).click();
  await expect(page.getByTestId('route-load-failed-title')).toHaveCount(0);
  await expect(page.getByRole('tab', { name: 'Appearance', exact: true })).toBeVisible();
});
