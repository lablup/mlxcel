// Copyright 2025-2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import { expect, test } from '@playwright/test';
import { expectAxeClean, expectEmptyStateHeading, expectSafeLayout } from './browser-assertions';

for (const theme of ['light', 'dark'] as const) {
  test(`renders common controls under actual served CSP without external requests (${theme})`, async ({ page }) => {
    const target = process.env.MLXCEL_WEBUI_CSP_URL;
    if (!target) throw new Error('Use playwright.csp.config.ts with a real secured URL.');
    const url = new URL(target); url.hash = 'gallery';
    const external: string[] = [];
    page.on('request', (request) => { if (new URL(request.url()).origin !== url.origin) external.push(request.url()); });
    await page.setViewportSize({ width: theme === 'light' ? 1440 : 390, height: 900 });
    await page.addInitScript((theme) => {
      localStorage.setItem('mlxcel.webui.appearance', JSON.stringify({ theme, locale: 'en', material: 'glass', reduceMotion: true }));
      const violations: string[] = [];
      Object.defineProperty(window, '__webuiCspViolations', { value: violations });
      document.addEventListener('securitypolicyviolation', (event) => violations.push(`${event.effectiveDirective}: ${event.blockedURI}`));
    }, theme);
    const response = await page.goto(url.href);
    expect(response?.ok()).toBe(true);
    const csp = response?.headers()['content-security-policy'];
    expect(csp).toBeDefined();
    expect(csp).toMatch(/(?:^|;)\s*style-src 'self'(?:;|$)/);
    expect(csp).not.toContain("'unsafe-inline'");
    expect(csp).not.toContain("'unsafe-eval'");
    await expect(page.getByTestId('gallery-title')).toBeVisible();
    const select = page.getByRole('combobox', { name: 'Shared select combobox' });
    for (let attempt = 0; attempt < 2; attempt += 1) {
      await select.click();
      const popup = page.locator('.select__dropdown--portal');
      await expect(popup).toBeVisible();
      const triggerBox = await select.boundingBox(); const popupBox = await popup.boundingBox();
      if (!triggerBox || !popupBox) throw new Error('Missing actual popup geometry');
      expect(Math.abs(popupBox.x - triggerBox.x)).toBeLessThan(2);
      expect(Math.abs(popupBox.y - (triggerBox.y + triggerBox.height + 4))).toBeLessThan(2);
      expect(Math.abs(popupBox.width - triggerBox.width)).toBeLessThan(2);
      await expectSafeLayout(page); await expectAxeClean(page);
      await page.keyboard.press('Escape'); await expect(select).toBeFocused();
    }
    if (theme === 'dark') {
      // 390px: the shared Drawer writes its width through CSSOM under the served CSP.
      await page.getByRole('button', { name: 'Open navigation', exact: true }).click();
      const drawer = page.getByRole('dialog', { name: 'Primary navigation' });
      await expect(drawer.getByRole('button', { name: 'Close', exact: true })).toBeFocused();
      await expectSafeLayout(page); await expectAxeClean(page);
      await page.keyboard.press('Escape'); await expect(drawer).toBeHidden();
    }
    // The shared Tooltip positions its body-portalled content through CSSOM top/left.
    const tooltipTrigger = page.getByRole('button', { name: 'Hover or focus', exact: true });
    await tooltipTrigger.focus(); await expect(page.getByRole('tooltip')).toBeVisible();
    await tooltipTrigger.hover(); await expect(page.getByRole('tooltip')).toBeVisible();
    await expectSafeLayout(page);
    await page.keyboard.press('Escape'); await expect(page.getByRole('tooltip')).toHaveCount(0);
    await page.getByRole('tab', { name: 'States', exact: true }).click();
    await expectEmptyStateHeading(page);
    // The loading StatCard's decorative skeletons take their size through CSSOM under the served CSP; the layout and violation checks below cover them.
    await expect(page.locator('.ds-stat-card .skeleton[aria-hidden="true"]').first()).toBeVisible();
    const determinate = page.locator('[role="progressbar"][aria-valuenow]').first();
    const ratio = await determinate.evaluate((element) => {
      const fill = element.querySelector('.progress-bar__fill');
      if (!fill) throw new Error('Missing common fill');
      return { actual: fill.getBoundingClientRect().width / element.clientWidth * 100, expected: Number(element.getAttribute('aria-valuenow')) };
    });
    expect(Math.abs(ratio.actual - ratio.expected)).toBeLessThan(1);
    await expectSafeLayout(page); await expectAxeClean(page);
    await expect(page.locator('.progress-bar--indeterminate .progress-bar__fill')).toHaveCSS('animation-name', 'none');
    const violations = await page.evaluate(() => Reflect.get(window, '__webuiCspViolations'));
    expect(violations).toEqual([]); expect(external).toEqual([]);
  });
}
