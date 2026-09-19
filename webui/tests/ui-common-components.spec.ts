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
