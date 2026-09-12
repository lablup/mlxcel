import { expect, test } from '@playwright/test';

test('serves the bundled shell without external asset origins', async ({ page }) => {
  const externalRequests: string[] = [];
  page.on('request', (request) => {
    const url = new URL(request.url());
    if (url.origin !== 'http://127.0.0.1:4173') {
      externalRequests.push(request.url());
    }
  });
  await page.goto('/#models');
  await expect(page.getByRole('heading', { name: /mlxcel WebUI loads/ })).toBeVisible();
  await expect(page.getByRole('link', { name: 'Models' })).toHaveAttribute('aria-current', 'page');
  expect(externalRequests).toEqual([]);
});
