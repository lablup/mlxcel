// Copyright 2025-2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import { expect, test, type Locator, type Page } from '@playwright/test';
import { expectAxeClean, expectLocatorWithinViewportX, expectSafeLayout, expectTextScalePanelsReflow } from './browser-assertions';
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

  test('390 open drawer keeps keyboard focus and Escape inside it after a pointer press on its heading', async ({ page }) => {
    await bootGallery(page, compact);
    const menu = page.getByTestId('toolbar-menu');
    await menu.click();
    const dialog = page.getByRole('dialog', { name: text('nav.primary', 'ko') });
    await expect(dialog.getByRole('button', { name: text('common.close', 'ko'), exact: true })).toBeFocused();
    await settled(dialog);
    // Pressing the non-focusable heading leaves focus on <body>. The native modal sheet
    // kept the page behind it inert and closed on Escape wherever focus was.
    const heading = dialog.getByRole('heading', { name: text('nav.primary', 'ko') });
    await heading.click();
    const focusInDrawer = (): Promise<boolean> => page.evaluate(() => Boolean(document.activeElement?.closest('[data-testid="mobile-nav-sheet"]')));
    for (let step = 0; step < 3; step += 1) {
      await page.keyboard.press('Shift+Tab');
      expect(await page.evaluate(() => document.activeElement === document.body) || await focusInDrawer()).toBe(true);
    }
    expect(await focusInDrawer()).toBe(true);
    await heading.click();
    await page.keyboard.press('Escape');
    await expect(dialog).toBeHidden();
    await expect(menu).toBeFocused();
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

// The route content must fill the grid's content box exactly: no second padding
// layer at 390 px and no width cap below 1400 px.
async function columnMetrics(page: Page): Promise<{ left: number; right: number; column: number; route: number; document: number; width: number }> {
  return page.evaluate(() => {
    const grid = document.querySelector<HTMLElement>('.app-content-grid');
    const column = document.querySelector<HTMLElement>('.app-content');
    const route = column?.firstElementChild;
    if (!grid || !column || !(route instanceof HTMLElement)) throw new Error('Missing shell content column');
    const style = window.getComputedStyle(grid);
    const box = grid.getBoundingClientRect();
    const inner = column.getBoundingClientRect();
    // Measure the route content, not the column box: a second padding layer sits inside the column.
    const content = route.getBoundingClientRect();
    return {
      left: content.left - (box.left + Number.parseFloat(style.paddingLeft)),
      right: box.right - Number.parseFloat(style.paddingRight) - content.right,
      column: column.scrollWidth - column.clientWidth,
      route: route.scrollWidth - route.clientWidth,
      document: document.documentElement.scrollWidth - document.documentElement.clientWidth,
      width: inner.width,
    };
  });
}

test.describe('shared PageLayout', () => {
  for (const textScale of [undefined, '200'] as const) {
    test(`390 content column fills the grid without double padding${textScale ? ' at the 200 percent text scale' : ''}`, async ({ page }) => {
      await bootGallery(page, { ...compact, textScale });
      const metrics = await columnMetrics(page);
      expect(Math.abs(metrics.left)).toBeLessThanOrEqual(1);
      expect(Math.abs(metrics.right)).toBeLessThanOrEqual(1);
      expect(metrics.column).toBeLessThanOrEqual(1);
      expect(metrics.route).toBeLessThanOrEqual(1);
      expect(metrics.document).toBeLessThanOrEqual(1);
      await expectTextScalePanelsReflow(page);
      await expect(page.locator('.app-content.page-layout.page-layout--wide')).toHaveCount(1);
    });
  }

  test('1440 keeps the content column at the full grid width', async ({ page }) => {
    await bootGallery(page, { ...compact, width: 1440, height: 900, appearance: { ...compact.appearance, locale: 'en' } });
    const metrics = await columnMetrics(page);
    expect(Math.abs(metrics.left)).toBeLessThanOrEqual(1);
    expect(Math.abs(metrics.right)).toBeLessThanOrEqual(1);
    expect(metrics.width).toBeLessThan(1400);
    expect(metrics.document).toBeLessThanOrEqual(1);
    await expect(page.locator('.app-content.page-layout.page-layout--wide')).toHaveCount(1);
  });
});

async function expectHeaderReflows(page: Page, ids: string[]): Promise<void> {
  for (const id of ids) {
    const element = page.getByTestId(id);
    await expectLocatorWithinViewportX(element);
    expect(await element.evaluate((node) => node.scrollWidth - node.clientWidth)).toBeLessThanOrEqual(1);
  }
  await expect(page.getByRole('heading', { level: 1 })).toHaveCount(1);
  // The product's bold page title, not the package's light 300 weight.
  expect(await page.getByTestId(ids[0]).evaluate((node) => Number(window.getComputedStyle(node).fontWeight))).toBeGreaterThanOrEqual(600);
  expect(await horizontalOverflow(page)).toBeLessThanOrEqual(1);
}

test.describe('shared PageHeader', () => {
  test('390 gallery header keeps its test ids and reflows at the 200 percent text scale', async ({ page }) => {
    await bootGallery(page, { ...compact, textScale: '200' });
    await expect(page.getByTestId('gallery-title')).toHaveText(text('gallery.title', 'ko'));
    await expectHeaderReflows(page, ['gallery-title', 'gallery-subtitle']);
    await expectTextScalePanelsReflow(page);
    await expect(page.getByTestId('gallery-title')).toHaveClass(/page-header__title/);
  });

  test('390 settings header keeps its test ids and reflows at the 200 percent text scale', async ({ page }) => {
    await bootGallery(page, { ...compact, textScale: '200' });
    await page.evaluate(() => { window.location.hash = '#settings'; });
    await expect(page.getByTestId('settings-title')).toHaveText(text('settings.title', 'ko'));
    await expectHeaderReflows(page, ['settings-title', 'settings-appearance']);
    await expect(page.getByTestId('settings-title')).toHaveClass(/page-header__title/);
  });

  test('390 signed-out connection surface keeps its title and prompt', async ({ page }) => {
    await bootProduct(page, productVariants[2]);
    await expect(page.getByTestId('models-title')).toHaveText(text('models.title', 'ko'));
    await expectHeaderReflows(page, ['models-title', 'connection-prompt-body']);
    await expectAxeClean(page);
    await expect(page.getByTestId('models-title')).toHaveClass(/page-header__title/);
  });
});
