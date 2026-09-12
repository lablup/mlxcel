import { AxeBuilder } from '@axe-core/playwright';
import { expect, type Locator, type Page } from '@playwright/test';

export async function expectDataTableColumnsVisible(page: Page): Promise<void> {
  const metrics = await page.locator('.ds-table').evaluate((table) => {
    const element = table as HTMLElement;
    const headers = Array.from(element.querySelectorAll<HTMLElement>('th')).map((header) => header.getBoundingClientRect());
    return {
      overflow: element.scrollWidth - element.clientWidth,
      widths: headers.map((header) => header.width),
    };
  });
  expect(metrics.overflow).toBeLessThanOrEqual(1);
  expect(metrics.widths.length).toBe(3);
  for (const width of metrics.widths) expect(width).toBeGreaterThan(88);
}

export async function pressQuestionShortcut(page: Page): Promise<void> {
  await page.evaluate(() => {
    const target = document.activeElement instanceof HTMLElement ? document.activeElement : document.body;
    target.dispatchEvent(new KeyboardEvent('keydown', { key: '?', bubbles: true, cancelable: true }));
  });
}

export async function expectAxeClean(page: Page): Promise<void> {
  const results = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'wcag22aa']).analyze();
  expect(results.violations).toEqual([]);
}

export async function expectNoOverflowOrInlineStyles(page: Page): Promise<void> {
  const result = await page.evaluate(() => ({
    overflow: document.documentElement.scrollWidth - document.documentElement.clientWidth,
    inlineStyleCount: document.querySelectorAll('[style]').length,
    smallTargets: Array.from(document.querySelectorAll<HTMLElement>('button, a[href], input, select, textarea')).filter((element) => {
      const rect = element.getBoundingClientRect();
      const style = window.getComputedStyle(element);
      if ((rect.width === 0 && rect.height === 0) || style.visibility === 'hidden' || style.display === 'none') return false;
      return rect.width < 24 || rect.height < 24;
    }).map((element) => element.outerHTML.slice(0, 80)),
  }));
  expect(result.overflow).toBeLessThanOrEqual(1);
  expect(result.inlineStyleCount).toBe(0);
  expect(result.smallTargets).toEqual([]);
}

export async function expectCompactToolbarHitTargets(page: Page): Promise<void> {
  const targets = await page.evaluate(() => Array.from(document.querySelectorAll<HTMLElement>('.app-toolbar .ds-icon-button')).filter((element) => {
    const rect = element.getBoundingClientRect();
    const style = window.getComputedStyle(element);
    return rect.width > 0 && rect.height > 0 && style.visibility !== 'hidden' && style.display !== 'none';
  }).map((element) => {
    const rect = element.getBoundingClientRect();
    return { label: element.getAttribute('aria-label') ?? element.textContent?.trim() ?? element.tagName, width: rect.width, height: rect.height };
  }));
  expect(targets.length).toBeGreaterThanOrEqual(3);
  for (const target of targets) {
    expect(target.width, target.label).toBeGreaterThanOrEqual(44);
    expect(target.height, target.label).toBeGreaterThanOrEqual(44);
  }
}

export async function expectTextScalePanelsReflow(page: Page): Promise<void> {
  const selectors = [
    '.app-main',
    '.app-toolbar',
    '.app-content-grid',
    '.app-content',
    '.screen-stack',
    '.screen-heading',
    '.screen-heading h1',
    '.screen-heading p',
    '.ds-tabs',
    '.ds-tabs [role="tablist"]',
    '.ds-tabs [role="tab"]',
    '.gallery-grid',
    '.surface-card',
    '.surface-card h2',
    '.surface-card p',
    '.control-row',
    '.ds-button',
    '.ds-field',
    '.ds-field > span',
    '.ds-field small',
  ];
  const result = await page.evaluate((panelSelectors) => {
    const visible = (element: HTMLElement): boolean => {
      const rect = element.getBoundingClientRect();
      const style = window.getComputedStyle(element);
      return rect.width > 0 && rect.height > 0 && style.visibility !== 'hidden' && style.display !== 'none';
    };
    const failures: string[] = [];
    for (const selector of panelSelectors) {
      for (const element of Array.from(document.querySelectorAll<HTMLElement>(selector))) {
        if (!visible(element)) continue;
        const delta = Math.ceil(element.scrollWidth - element.clientWidth);
        if (delta > 1) failures.push(`${selector} overflow=${delta} text=${element.textContent?.trim().slice(0, 80) ?? ''}`);
      }
    }
    return {
      documentOverflow: Math.ceil(document.documentElement.scrollWidth - document.documentElement.clientWidth),
      bodyOverflow: Math.ceil(document.body.scrollWidth - document.body.clientWidth),
      failures,
    };
  }, selectors);
  expect(result.documentOverflow, JSON.stringify(result.failures)).toBeLessThanOrEqual(1);
  expect(result.bodyOverflow, JSON.stringify(result.failures)).toBeLessThanOrEqual(1);
  expect(result.failures).toEqual([]);
}

export async function expectLocatorWithinViewportX(locator: Locator): Promise<void> {
  await locator.scrollIntoViewIfNeeded();
  await expect(locator).toBeVisible();
  const box = await locator.boundingBox();
  expect(box).not.toBeNull();
  if (!box) return;
  const viewport = locator.page().viewportSize();
  expect(viewport).not.toBeNull();
  if (!viewport) return;
  expect(Math.floor(box.x)).toBeGreaterThanOrEqual(0);
  expect(Math.ceil(box.x + box.width)).toBeLessThanOrEqual(viewport.width + 1);
}

export async function expectTextScaleLabelsReachable(page: Page): Promise<void> {
  await expectTextScalePanelsReflow(page);
  const reachable = [
    page.getByRole('tab', { name: /컨트롤/i }),
    page.getByRole('tab', { name: /상태/i }),
    page.getByRole('tab', { name: /데이터 표시/i }),
    page.getByRole('button', { name: /기본/i }),
    page.getByRole('button', { name: /보조/i }),
    page.getByRole('button', { name: /위험/i }),
    page.getByText('저장소 ID', { exact: true }),
    page.getByText('네이티브 select 콤보박스', { exact: true }),
    page.getByRole('combobox', { name: /네이티브 select 콤보박스/i }),
  ];
  for (const locator of reachable) await expectLocatorWithinViewportX(locator);
}
