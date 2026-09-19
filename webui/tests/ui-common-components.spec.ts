// Copyright 2025-2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import { expect, test } from '@playwright/test';
import { expectSafeLayout } from './browser-assertions';
import { bootGallery, variants } from './browser-fixtures';
import { shownTooltips, text } from './ui-common-helpers';

// Each case asserts the product behavior first and the shared ui-common DOM
// last, so a run against the pre-adoption code fails on the last assertion.
test.describe('shared Tooltip', () => {
  test('keyboard and pointer users get the description without an extra tab stop', async ({ page }) => {
    await bootGallery(page, variants[0]);
    const tooltip = text('gallery.tooltip');
    const trigger = page.getByRole('button', { name: text('gallery.hover_focus'), exact: true });
    await expect.poll(() => shownTooltips(page)).toEqual([]);
    await page.getByRole('combobox', { name: text('gallery.select.native') }).focus();
    await page.keyboard.press('Tab');
    await expect(trigger).toBeFocused();
    await expect(trigger).toHaveAccessibleDescription(tooltip);
    await expect.poll(() => shownTooltips(page)).toEqual([tooltip]);
    await expectSafeLayout(page);
    await page.getByRole('button', { name: text('gallery.dialog.open'), exact: true }).focus();
    await expect.poll(() => shownTooltips(page)).toEqual([]);
    await trigger.hover();
    await expect.poll(() => shownTooltips(page)).toEqual([tooltip]);
    await page.mouse.move(0, 0);
    await expect.poll(() => shownTooltips(page)).toEqual([]);
    await expect(page.locator('.tooltip__wrapper').filter({ has: trigger })).toHaveCount(1);
  });

  test('Escape dismisses a keyboard-opened tooltip and keeps focus (WCAG 1.4.13)', async ({ page }) => {
    await bootGallery(page, variants[0]);
    const trigger = page.getByRole('button', { name: text('gallery.hover_focus'), exact: true });
    await trigger.focus();
    await expect.poll(() => shownTooltips(page)).toEqual([text('gallery.tooltip')]);
    await page.keyboard.press('Escape');
    await expect.poll(() => shownTooltips(page)).toEqual([]);
    await expect(trigger).toBeFocused();
    await expect(page.locator('.tooltip__wrapper').filter({ has: trigger })).toHaveCount(1);
  });
});

test.describe('shared ErrorState', () => {
  test('the gallery warning stays a compact start-aligned alert with its action and high-contrast border', async ({ page }) => {
    await bootGallery(page, variants[6]);
    const title = text('gallery.states.load_failed');
    const banner = page.locator('.ds-banner').filter({ hasText: title });
    await expect(banner).toHaveAttribute('role', 'alert');
    await expect(banner.getByRole('button', { name: text('common.retry'), exact: true })).toBeVisible();
    await expect(banner).toHaveCSS('border-top-width', '2px');
    await expect(page.getByRole('heading', { name: title })).toHaveCount(0);
    // The banner stretches to its grid row, so measure where the title sits instead:
    // a compact banner starts at its padding, not centered in a 200px block.
    const layout = await banner.evaluate((element, label) => {
      const heading = Array.from(element.querySelectorAll<HTMLElement>('*')).find((node) => node.childElementCount === 0 && node.textContent === label);
      if (!heading) return { top: -1, left: -1, align: 'missing' };
      const box = element.getBoundingClientRect();
      const title = heading.getBoundingClientRect();
      return { top: title.top - box.top, left: title.left - box.left, align: window.getComputedStyle(heading).textAlign };
    }, title);
    expect(layout.top).toBeGreaterThanOrEqual(0);
    expect(layout.top).toBeLessThan(32);
    expect(layout.left).toBeLessThan(32);
    expect(['start', 'left']).toContain(layout.align);
    await expect(banner.locator('.error-state')).toHaveCount(1);
  });
});
