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
      // The Selector's list is an Astryx popover in the top layer, anchored under its trigger.
      const popup = page.locator('.astryx-selector-popup').filter({ visible: true });
      await expect(popup).toBeVisible();
      // Anchored to the Selector's control box (the trigger button plus its chevron).
      const anchor = select.locator('xpath=ancestor::div[contains(concat(" ", normalize-space(@class), " "), " astryx-selector ")][1]');
      const triggerBox = await anchor.boundingBox(); const popupBox = await popup.boundingBox();
      if (!triggerBox || !popupBox) throw new Error('Missing actual popup geometry');
      expect(Math.abs(popupBox.x - triggerBox.x)).toBeLessThan(2);
      // The list opens at its trigger: just under it, or over it with the selected option in the
      // trigger's place. Without its anchor positioning it would sit in the viewport corner.
      expect(popupBox.y).toBeLessThanOrEqual(triggerBox.y + triggerBox.height + 6);
      expect(popupBox.y + popupBox.height).toBeGreaterThanOrEqual(triggerBox.y);
      expect(Math.abs(popupBox.width - triggerBox.width)).toBeLessThan(2);
      await expectSafeLayout(page); await expectAxeClean(page);
      await page.keyboard.press('Escape'); await expect(select).toBeFocused();
    }
    if (theme === 'dark') {
      // 390px: AppShell's navigation drawer (MobileNav) writes its sheet width and the <html>
      // scroll lock through CSSOM under the served CSP.
      await page.getByRole('button', { name: 'Open navigation', exact: true }).click();
      const drawer = page.getByRole('dialog', { name: 'Primary navigation' });
      await expect(drawer.getByRole('button', { name: 'Close navigation', exact: true })).toBeVisible();
      expect(await page.evaluate(() => Boolean(document.activeElement?.closest('dialog.astryx-mobile-nav')))).toBe(true);
      await expectSafeLayout(page); await expectAxeClean(page);
      await page.keyboard.press('Escape'); await expect(drawer).toBeHidden();
    }
    // The shared Tooltip is an Astryx popover placed with CSS anchor positioning (its slide rules
    // come from the linked astryx-layer-slide-rules.css, never an inserted <style>). Keyboard focus shows it.
    const tooltipTrigger = page.getByRole('button', { name: 'Hover or focus', exact: true });
    await select.focus(); await page.keyboard.press('Tab');
    await expect(tooltipTrigger).toBeFocused(); await expect(page.getByRole('tooltip')).toBeVisible();
    await tooltipTrigger.hover(); await expect(page.getByRole('tooltip')).toBeVisible();
    await expectSafeLayout(page);
    await page.keyboard.press('Escape'); await expect(page.getByRole('tooltip')).toHaveCount(0);
    await page.getByRole('tab', { name: 'States', exact: true }).click();
    await expectEmptyStateHeading(page);
    // The loading StatCard's decorative skeletons take their size through CSSOM under the served CSP; the layout and violation checks below cover them.
    await expect(page.locator('.ds-stat-card .astryx-skeleton[aria-hidden="true"]').first()).toBeVisible();
    const determinate = page.locator('[role="progressbar"][aria-valuenow]').first();
    const ratio = await determinate.evaluate((element) => {
      const fill = element.querySelector('.astryx-progress-bar-fill');
      if (!fill) throw new Error('Missing common fill');
      return { actual: fill.getBoundingClientRect().width / element.clientWidth * 100, expected: Number(element.getAttribute('aria-valuenow')) };
    });
    expect(Math.abs(ratio.actual - ratio.expected)).toBeLessThan(1);
    await expectSafeLayout(page); await expectAxeClean(page);
    await expect(page.locator('.ds-progress .astryx-progress-bar-fill:not([style])').first()).toHaveCSS('animation-name', 'none');
    const violations = await page.evaluate(() => Reflect.get(window, '__webuiCspViolations'));
    expect(violations).toEqual([]); expect(external).toEqual([]);
    // Nothing inserted a <style> element (the CSP would have blocked it), and Astryx found the
    // shipped slide rules under the id it looks for.
    expect(await page.locator('style').count()).toBe(0);
    await expect(page.locator('link#astryx-layer-slide-rules[rel="stylesheet"]')).toHaveCount(1);
  });
}
