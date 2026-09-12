import { expect, test } from '@playwright/test';
import { expectAxeClean, expectCompactToolbarHitTargets, expectDataTableColumnsVisible, expectLocatorWithinViewportX, expectNoOverflowOrInlineStyles, expectTextScaleLabelsReachable, expectTextScalePanelsReflow, pressQuestionShortcut, reportFontDiagnostics } from './browser-assertions';
import { bootGallery, bootProduct, browserStorageDump, gotoGalleryWithoutReload, installAbortRecorder, installMockApi, loadProductionCss, loginWithMockApi, productVariants, readAbortLog, selectGalleryTab, settleAnimationFrame, submitSessionKey, variants } from './browser-fixtures';

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
    const productionCss = loadProductionCss();
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

  if (process.env.MLXCEL_WEBUI_FONT_DIAGNOSTICS === '1') {
    test('reports browser font diagnostics', async ({ page }) => {
      await bootGallery(page, variants[0]);
      await reportFontDiagnostics(page, variants[0].name);
      await page.goto('about:blank');
      await bootProduct(page, productVariants[2]);
      await reportFontDiagnostics(page, productVariants[2].name);
    });
  }
});
