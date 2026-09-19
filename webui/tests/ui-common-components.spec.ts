// Copyright 2025-2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import { AxeBuilder } from '@axe-core/playwright';
import { expect, test, type Page } from '@playwright/test';
import { expectAxeClean, expectSafeLayout } from './browser-assertions';
import { bootGallery, bootProduct, installMockApi, loginWithMockApi, productVariants, selectGalleryTab, submitSessionKey, variants } from './browser-fixtures';
import { horizontalOverflow, runtimeForFirstCatalogEntry, shownTooltips, text } from './ui-common-helpers';

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

test.describe('shared BaseCard', () => {
  test('gallery cards keep article semantics, no hover lift, and a named keyboard-focusable data region', async ({ page }) => {
    await bootGallery(page, variants[0]);
    const articles = page.getByRole('article');
    await expect(articles).toHaveCount(2);
    for (const heading of [text('gallery.controls.title'), text('gallery.overlays.title')]) await expect(articles.filter({ has: page.getByRole('heading', { name: heading, level: 2 }) })).toHaveCount(1);
    const card = page.locator('.surface-card').first();
    await card.hover();
    await expect(card).toHaveCSS('transform', 'none');
    await selectGalleryTab(page, 'data');
    const dataCard = page.locator('.surface-card.table-card');
    await expect(dataCard).toHaveAccessibleName(text('gallery.sample.caption'));
    await expect(dataCard).toHaveAttribute('tabindex', '0');
    await page.getByRole('tab', { name: text('gallery.tab.data') }).focus();
    for (let step = 0; step < 6 && !(await dataCard.evaluate((element) => element === document.activeElement)); step += 1) await page.keyboard.press('Tab');
    await expect(dataCard).toBeFocused();
    expect(await dataCard.evaluate((element) => window.getComputedStyle(element).outlineStyle)).not.toBe('none');
    await expectAxeClean(page);
    await expect(page.locator('.surface-card.base-card')).toHaveCount(1);
  });
});

async function openActivityRuntime(page: Page, width: number): Promise<void> {
  await installMockApi(page, 'happy');
  const at = '2026-09-15T00:00:00Z';
  const { modelName, body } = runtimeForFirstCatalogEntry({
    active_requests: { value: 2, unit: 'requests', scope: 'model', measured_at: at, reason: 'Authoritative route completions' },
    queued_requests: { value: 0, unit: 'requests', scope: 'model', measured_at: at, reason: null },
    completed_requests_total: { value: 1234567, unit: 'requests', scope: 'model', measured_at: at, reason: null },
    completion_tokens_total: { value: 98765432, unit: 'tokens', scope: 'model', measured_at: at, reason: null },
  });
  await page.route('**/ui-api/v1/runtime*', (route) => route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(body) }));
  await bootProduct(page, { ...productVariants[0], width, appearance: { ...productVariants[0].appearance, locale: 'en' } });
  await loginWithMockApi(page);
  await page.evaluate(() => { window.location.hash = '#activity'; });
  await page.getByTestId('activity-page').getByRole('combobox').click();
  await page.getByRole('option', { name: modelName, exact: true }).click();
  await expect(page.getByTestId('runtime-summary')).toContainText('Total completion tokens');
}

test.describe('shared StatCard', () => {
  for (const width of [390, 1440]) {
    test(`Activity runtime tiles show full labels and values without hover lift at ${width}`, async ({ page }) => {
      await openActivityRuntime(page, width);
      const tiles = page.getByTestId('runtime-summary').locator('.activity-metric');
      await expect(tiles).toHaveCount(4);
      const report = await tiles.evaluateAll((elements) => elements.map((tile) => {
        const leaves = Array.from(tile.querySelectorAll<HTMLElement>('*')).filter((node) => node.childElementCount === 0 && (node.textContent ?? '').trim() !== '');
        return {
          text: tile.textContent ?? '',
          clipped: leaves.filter((node) => node.scrollWidth > node.clientWidth + 1).map((node) => node.textContent),
          transformed: leaves.filter((node) => window.getComputedStyle(node).textTransform !== 'none').map((node) => node.textContent),
        };
      }));
      for (const [index, [label, value]] of [['Active requests', '2 requests'], ['Queued requests', '0 requests'], ['Total completed requests', '1,234,567 requests'], ['Total completion tokens', '98,765,432 tokens']].entries()) {
        expect(report[index].text).toContain(label);
        expect(report[index].text).toContain(value);
        expect(report[index].clipped).toEqual([]);
        expect(report[index].transformed).toEqual([]);
      }
      await tiles.first().hover();
      await expect(tiles.first()).toHaveCSS('transform', 'none');
      expect(await horizontalOverflow(page)).toBeLessThanOrEqual(1);
      await expectAxeClean(page);
      await expect(page.getByTestId('runtime-summary').locator('.activity-metric.stat-card')).toHaveCount(4);
    });
  }
});

test.describe('shared Skeleton', () => {
  for (const width of [390, 1440]) {
    test(`Models shows one visible, localized waiting status while the catalog is pending at ${width}`, async ({ page }) => {
      await installMockApi(page, 'slow-catalog');
      await bootProduct(page, { ...productVariants[0], width, appearance: { ...productVariants[0].appearance, locale: 'en' } });
      await submitSessionKey(page, 'good-key');
      const table = page.getByTestId('models-table');
      const status = table.getByRole('status');
      await expect(status).toHaveCount(1);
      await expect(status).toHaveText(text('models.library.waiting'));
      await expect(status.getByText(text('models.library.waiting'), { exact: true })).toBeVisible();
      await expect(page.getByText('Loading', { exact: true })).toHaveCount(0);
      expect(await horizontalOverflow(page)).toBeLessThanOrEqual(1);
      // Pre-existing on main and independent of the loading content: at 390px the
      // loading/empty table scrolls horizontally with nothing focusable inside. Tolerate
      // exactly that node so the waiting state cannot add anything else.
      const results = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'wcag22aa']).analyze();
      const known = (id: string, html: string): boolean => width === 390 && id === 'scrollable-region-focusable' && html.startsWith('<div class="data-table ds-common-table" data-testid="models-table">');
      expect(results.violations.map((violation) => ({ id: violation.id, nodes: violation.nodes.map((node) => node.html).filter((html) => !known(violation.id, html)) })).filter((violation) => violation.nodes.length > 0)).toEqual([]);
      await expectSafeLayout(page);
      await expect(status.locator('.skeleton').first()).toBeVisible();
    });
  }
});
