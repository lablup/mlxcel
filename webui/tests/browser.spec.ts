import { AxeBuilder } from '@axe-core/playwright';
import { expect, test, type Page } from '@playwright/test';

type Variant = { name: string; width: number; height: number; hash: string; appearance: Record<string, unknown>; textScale?: '200' };

const variants: Variant[] = [
  { name: '390-light-tinted-cjk', width: 390, height: 844, hash: 'gallery', appearance: { theme: 'light', material: 'tinted', locale: 'ko', glassIntensity: 35, highContrast: 'system' } },
  { name: '1024-dark-glass', width: 1024, height: 768, hash: 'models', appearance: { theme: 'dark', material: 'glass', locale: 'en', glassIntensity: 55, highContrast: 'system' } },
  { name: '1440-light-opaque-highcontrast', width: 1440, height: 900, hash: 'settings', appearance: { theme: 'light', material: 'opaque', highContrast: 'on', locale: 'en', glassIntensity: 0 } },
  { name: '390-dark-opaque-textscale200', width: 390, height: 844, hash: 'activity', appearance: { theme: 'dark', material: 'opaque', reduceTransparency: true, reduceMotion: true, locale: 'ko', glassIntensity: 100, highContrast: 'off' }, textScale: '200' },
];

async function boot(page: Page, variant: Variant): Promise<void> {
  await page.setViewportSize({ width: variant.width, height: variant.height });
  await page.addInitScript((appearance) => localStorage.setItem('mlxcel.webui.appearance', JSON.stringify(appearance)), variant.appearance);
  await page.goto(`/#${variant.hash}`);
  await expect(page.getByTestId('app-title')).toBeVisible();
  if (typeof variant.appearance.theme === 'string') await expect(page.locator('html')).toHaveAttribute('data-theme', variant.appearance.theme);
  if (variant.textScale) await page.evaluate((scale) => { document.documentElement.dataset.testTextScale = scale; }, variant.textScale);
}

async function expectAxeClean(page: Page): Promise<void> {
  const results = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'wcag22aa']).analyze();
  const serious = results.violations.filter((violation) => violation.impact === 'serious' || violation.impact === 'critical');
  expect(serious).toEqual([]);
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
      await boot(page, variant);
      await expectAxeClean(page);
      await expectNoOverflowOrInlineStyles(page);
      if (variant.textScale === '200') {
        const fontSize = await page.evaluate(() => Number.parseFloat(window.getComputedStyle(document.documentElement).fontSize));
        expect(fontSize).toBeGreaterThanOrEqual(32);
      }
      await expect(page).toHaveScreenshot(`${variant.name}.png`, { animations: 'disabled', maxDiffPixelRatio: 0.12, threshold: 0.2 });
    });
  }

  test('uses one accessible mobile sheet and restores focus', async ({ page }) => {
    await boot(page, variants[0]);
    const menu = page.getByRole('button', { name: /navigation|내비게이션/i });
    await menu.focus();
    await menu.click();
    await expect(page.getByTestId('mobile-nav-sheet')).toBeVisible();
    await page.keyboard.press('Escape');
    await expect(page.getByTestId('mobile-nav-sheet')).toBeHidden();
    await expect(menu).toBeFocused();
  });

  test('covers native select keyboard, tabs, fallback materials and offline origin', async ({ page }) => {
    const blocked: string[] = [];
    await page.route('**/*', async (route) => {
      const url = new URL(route.request().url());
      if (url.origin === 'http://127.0.0.1:4173') return route.continue();
      blocked.push(url.href);
      return route.abort();
    });
    await page.addInitScript(() => {
      const original = CSS.supports.bind(CSS);
      CSS.supports = ((query: string) => query.includes('backdrop-filter') ? false : original(query)) as typeof CSS.supports;
    });
    await boot(page, variants[2]);
    expect(blocked).toEqual([]);
    await expect(page.locator('.material-glass').first()).toHaveCSS('backdrop-filter', /blur\(0px\)|none/);
    await page.getByTestId('settings-locale').focus();
    await page.keyboard.press('ArrowDown');
    await page.keyboard.press('Enter');
    await expect(page.getByTestId('settings-locale')).toBeFocused();
    await page.goto('/#gallery');
    const tab = page.getByRole('tab', { name: /States|상태/i });
    await tab.focus();
    await page.keyboard.press('ArrowRight');
    await expect(page.getByRole('tabpanel')).toContainText(/Model rows|모델 행/);
  });
});
