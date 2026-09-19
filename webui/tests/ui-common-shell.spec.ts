// Copyright 2025-2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import { expect, test, type Locator, type Page } from '@playwright/test';
import { expectAxeClean, expectLocatorWithinViewportX, expectSafeLayout } from './browser-assertions';
import { bootGallery, bootProduct, productVariants, settleAnimationFrame, type Variant } from './browser-fixtures';
import { horizontalOverflow, text } from './ui-common-helpers';

// Each case asserts the product behavior first and the shared ui-common DOM
// last, so a run against the pre-adoption code fails on the last assertion.
const compact: Variant = { name: 'drawer-390', width: 390, height: 844, tab: 'controls', appearance: { theme: 'light', material: 'opaque', locale: 'ko', glassIntensity: 0, highContrast: 'system' } };

async function canTakeFocus(locator: Locator): Promise<boolean> {
  return locator.evaluate((element: HTMLElement) => { element.focus(); return document.activeElement === element; });
}
async function tabNeverEntersPanel(page: Page, presses: number): Promise<void> {
  for (let step = 0; step < presses; step += 1) {
    await page.keyboard.press('Tab');
    expect(await page.evaluate(() => Boolean(document.activeElement?.closest('[data-testid="mobile-nav-sheet"]')))).toBe(false);
  }
}
// Measure only after the panel's own transitions finish (none on the native sheet).
async function settled(locator: Locator): Promise<void> {
  await locator.evaluate((element) => Promise.all(element.getAnimations().map((animation) => animation.finished)));
}
async function expectSharedDrawer(page: Page): Promise<void> {
  const panel = page.getByTestId('mobile-nav-sheet');
  await expect(panel).toHaveJSProperty('tagName', 'ASIDE');
  await expect(panel).toHaveClass(/(?:^|\s)drawer(?:\s|$)/);
  await expect(panel).toHaveAttribute('aria-modal', 'true');
}

test.describe('shared Drawer', () => {
  test('390 off-canvas navigation keeps focus entry, wrap, Escape and route restoration', async ({ page }) => {
    await bootGallery(page, compact);
    const menu = page.getByTestId('toolbar-menu');
    const panel = page.getByTestId('mobile-nav-sheet');
    expect(await horizontalOverflow(page)).toBeLessThanOrEqual(1);
    expect(await canTakeFocus(panel.getByTestId('nav-models'))).toBe(false);
    await menu.focus();
    await tabNeverEntersPanel(page, 6);
    await menu.focus();
    await page.keyboard.press('Enter');
    const dialog = page.getByRole('dialog', { name: text('nav.primary', 'ko') });
    await expect(dialog).toBeVisible();
    const close = dialog.getByRole('button', { name: text('common.close', 'ko'), exact: true });
    await expect(close).toBeFocused();
    await expect(dialog.getByRole('button', { name: 'Close', exact: true })).toHaveCount(0);
    await settled(dialog);
    const box = await dialog.boundingBox();
    if (!box) throw new Error('Missing drawer geometry');
    expect(box.x).toBeGreaterThanOrEqual(0);
    expect(box.x).toBeLessThanOrEqual(16);
    expect(box.width).toBeLessThanOrEqual(320 + 1);
    expect(box.x + box.width).toBeLessThanOrEqual(390 - 32 + 16);
    expect(await horizontalOverflow(page)).toBeLessThanOrEqual(1);
    await expectAxeClean(page);
    await expectSafeLayout(page);
    const last = dialog.getByTestId('nav-settings');
    await last.focus();
    await page.keyboard.press('Tab');
    await expect(close).toBeFocused();
    await page.keyboard.press('Shift+Tab');
    await expect(last).toBeFocused();
    await page.keyboard.press('Escape');
    await expect(dialog).toBeHidden();
    await expect(menu).toBeFocused();
    // The native sheet restored focus before React committed the close; give it a frame.
    await settleAnimationFrame(page);
    await page.keyboard.press('Enter');
    await expect(close).toBeFocused();
    await dialog.getByTestId('nav-activity').focus();
    await page.keyboard.press('Enter');
    await expect(page).toHaveURL(/#activity$/);
    await expect(dialog).toBeHidden();
    await expect(menu).toBeFocused();
    expect(await canTakeFocus(panel.getByTestId('nav-models'))).toBe(false);
    await expectSharedDrawer(page);
  });

  test('390 open drawer reflows at the 200 percent text scale without horizontal scroll', async ({ page }) => {
    await bootGallery(page, { ...compact, textScale: '200' });
    await page.getByTestId('toolbar-menu').click();
    const dialog = page.getByRole('dialog', { name: text('nav.primary', 'ko') });
    await expect(dialog.getByRole('button', { name: text('common.close', 'ko'), exact: true })).toBeFocused();
    await settled(dialog);
    expect(await horizontalOverflow(page)).toBeLessThanOrEqual(1);
    for (const id of ['nav-models', 'nav-chat', 'nav-activity', 'nav-settings']) await expectLocatorWithinViewportX(dialog.getByTestId(id));
    expect(await horizontalOverflow(page)).toBeLessThanOrEqual(1);
    await page.keyboard.press('Escape');
    await expect(dialog).toBeHidden();
    expect(await horizontalOverflow(page)).toBeLessThanOrEqual(1);
    await expectSharedDrawer(page);
  });

  test('1440 keeps the closed drawer out of the tab order', async ({ page }) => {
    await bootGallery(page, { ...compact, width: 1440, height: 900, appearance: { ...compact.appearance, locale: 'en' } });
    await expect(page.getByTestId('toolbar-menu')).toBeHidden();
    const panel = page.getByTestId('mobile-nav-sheet');
    for (const id of ['nav-models', 'nav-settings']) expect(await canTakeFocus(panel.getByTestId(id))).toBe(false);
    await page.locator('.desktop-sidebar .brand-mark').focus();
    await tabNeverEntersPanel(page, 8);
    await expectSharedDrawer(page);
  });

  test('the in-app reduce-motion setting turns the drawer transition off without the OS preference', async ({ page }) => {
    await bootGallery(page, { ...compact, appearance: { ...compact.appearance, reduceMotion: true } });
    expect(await page.evaluate(() => window.matchMedia('(prefers-reduced-motion: reduce)').matches)).toBe(false);
    await expect(page.locator('html')).toHaveAttribute('data-reduce-motion', 'true');
    await page.getByTestId('toolbar-menu').click();
    const dialog = page.getByRole('dialog', { name: text('nav.primary', 'ko') });
    await expect(dialog).toBeVisible();
    const durations = await dialog.evaluate((element) => window.getComputedStyle(element).transitionDuration.split(',').map((value) => Number.parseFloat(value)));
    for (const seconds of durations) expect(seconds).toBeLessThanOrEqual(0.001);
    await expectSharedDrawer(page);
  });

  test('widening past 960 px while open closes the drawer and keeps focus on a visible control', async ({ page }) => {
    await bootProduct(page, productVariants[2]);
    await page.getByTestId('toolbar-menu').click();
    const dialog = page.getByRole('dialog', { name: text('nav.primary', 'ko') });
    await dialog.getByTestId('nav-chat').focus();
    await page.setViewportSize({ width: 1440, height: 900 });
    await expect(dialog).toBeHidden();
    await expect(page.locator('.desktop-sidebar a[aria-current="page"]')).toBeFocused();
    await expectSharedDrawer(page);
  });
});
