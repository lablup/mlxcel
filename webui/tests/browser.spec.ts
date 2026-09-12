import { AxeBuilder } from '@axe-core/playwright';
import { expect, test, type Page } from '@playwright/test';

type GalleryTab = 'controls' | 'states' | 'data';
type Variant = { name: string; width: number; height: number; appearance: Record<string, unknown>; tab: GalleryTab; openDrawer?: boolean; textScale?: '200' };

const variants: Variant[] = [
  { name: '1440-light-gallery-controls', width: 1440, height: 900, tab: 'controls', appearance: { theme: 'light', material: 'glass', locale: 'en', glassIntensity: 35, highContrast: 'system' } },
  { name: '1440-light-gallery-data', width: 1440, height: 900, tab: 'data', appearance: { theme: 'light', material: 'opaque', locale: 'en', glassIntensity: 0, highContrast: 'system' } },
  { name: '1440-dark-gallery-controls', width: 1440, height: 900, tab: 'controls', appearance: { theme: 'dark', material: 'glass', locale: 'en', glassIntensity: 45, highContrast: 'system' } },
  { name: '1440-dark-gallery-data', width: 1440, height: 900, tab: 'data', appearance: { theme: 'dark', material: 'opaque', locale: 'en', glassIntensity: 0, highContrast: 'system' } },
  { name: '1024-tinted-gallery-data-inspector', width: 1024, height: 768, tab: 'data', appearance: { theme: 'light', material: 'tinted', locale: 'ko', glassIntensity: 45, highContrast: 'system' } },
  { name: '390-opaque-gallery-controls-drawer-cjk', width: 390, height: 844, tab: 'controls', openDrawer: true, appearance: { theme: 'light', material: 'opaque', locale: 'ko', glassIntensity: 0, highContrast: 'system' } },
  { name: '1440-highcontrast-gallery-states', width: 1440, height: 900, tab: 'states', appearance: { theme: 'light', material: 'opaque', highContrast: 'on', locale: 'en', glassIntensity: 0 } },
  { name: '390-dark-opaque-textscale200-gallery-controls', width: 390, height: 844, tab: 'controls', appearance: { theme: 'dark', material: 'opaque', reduceTransparency: true, reduceMotion: true, locale: 'ko', glassIntensity: 100, highContrast: 'off' }, textScale: '200' },
];

async function bootGallery(page: Page, variant: Variant): Promise<void> {
  await page.setViewportSize({ width: variant.width, height: variant.height });
  await page.addInitScript((appearance) => localStorage.setItem('mlxcel.webui.appearance', JSON.stringify(appearance)), variant.appearance);
  await page.goto('/#gallery');
  await expect(page.getByTestId('app-title')).toBeVisible();
  if (typeof variant.appearance.theme === 'string') await expect(page.locator('html')).toHaveAttribute('data-theme', variant.appearance.theme);
  if (variant.textScale) await page.evaluate((scale) => { document.documentElement.dataset.testTextScale = scale; }, variant.textScale);
  await selectGalleryTab(page, variant.tab);
  if (variant.openDrawer) await page.getByRole('button', { name: /navigation|내비게이션/i }).click();
}

async function selectGalleryTab(page: Page, tab: GalleryTab): Promise<void> {
  const names: Record<GalleryTab, RegExp> = { controls: /Controls|컨트롤/i, states: /States|상태/i, data: /Data display|데이터 표시/i };
  await page.getByRole('tab', { name: names[tab] }).click();
  await expect(page.getByRole('tab', { name: names[tab] })).toHaveAttribute('aria-selected', 'true');
}


async function settleAnimationFrame(page: Page): Promise<void> {
  await page.evaluate(() => new Promise<void>((resolve) => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))));
}

async function expectDataTableColumnsVisible(page: Page): Promise<void> {
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

async function pressQuestionShortcut(page: Page): Promise<void> {
  await page.evaluate(() => {
    const target = document.activeElement instanceof HTMLElement ? document.activeElement : document.body;
    target.dispatchEvent(new KeyboardEvent('keydown', { key: '?', bubbles: true, cancelable: true }));
  });
}

async function expectAxeClean(page: Page): Promise<void> {
  const results = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'wcag22aa']).analyze();
  expect(results.violations).toEqual([]);
}

async function expectNoOverflowOrInlineStyles(page: Page): Promise<void> {
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

test.describe('design system gallery and shell', () => {
  for (const variant of variants) {
    test(`renders and compares ${variant.name}`, async ({ page }) => {
      await bootGallery(page, variant);
      await expectAxeClean(page);
      await expectNoOverflowOrInlineStyles(page);
      if (variant.width > 960) await expect(page.getByTestId('toolbar-menu')).toBeHidden();
      if (variant.tab === 'data' && variant.width >= 1024) await expectDataTableColumnsVisible(page);
      if (variant.textScale === '200') {
        const fontSize = await page.evaluate(() => Number.parseFloat(window.getComputedStyle(document.documentElement).fontSize));
        expect(fontSize).toBeGreaterThanOrEqual(32);
      }
      await expect(page).toHaveScreenshot(`${variant.name}.png`, { animations: 'disabled', maxDiffPixelRatio: 0.005, threshold: 0.2 });
    });
  }

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
});
