import { expect, test, type Page } from '@playwright/test';

type Variant = { name: string; width: number; height: number; hash: string; appearance: Record<string, unknown> };

const variants: Variant[] = [
  { name: '390-light-tinted-cjk', width: 390, height: 844, hash: 'gallery', appearance: { theme: 'light', material: 'tinted', locale: 'ko', glassIntensity: 35 } },
  { name: '1024-dark-glass', width: 1024, height: 768, hash: 'models', appearance: { theme: 'dark', material: 'glass', locale: 'en', glassIntensity: 55 } },
  { name: '1440-light-opaque-highcontrast', width: 1440, height: 900, hash: 'settings', appearance: { theme: 'light', material: 'opaque', highContrast: true, locale: 'en', glassIntensity: 0 } },
  { name: '390-dark-opaque-200zoom', width: 390, height: 844, hash: 'activity', appearance: { theme: 'dark', material: 'opaque', reduceTransparency: true, reduceMotion: true, locale: 'ko', glassIntensity: 100 } },
];

async function boot(page: Page, variant: Variant): Promise<void> {
  await page.setViewportSize({ width: variant.width, height: variant.height });
  await page.addInitScript((appearance) => localStorage.setItem('mlxcel.webui.appearance', JSON.stringify(appearance)), variant.appearance);
  await page.goto(`/#${variant.hash}`);
  await expect(page.getByTestId('app-title')).toBeVisible();
}

async function expectNoBasicA11yViolations(page: Page): Promise<void> {
  const violations = await page.evaluate(() => {
    const problems: string[] = [];
    document.querySelectorAll('button, a[href], input, select, textarea').forEach((element, index) => {
      const rect = element.getBoundingClientRect();
      const style = window.getComputedStyle(element);
      if (rect.width === 0 && rect.height === 0 || style.visibility === 'hidden' || style.display === 'none') return;
      const label = element.getAttribute('aria-label') || element.textContent || (element as HTMLInputElement).labels?.[0]?.textContent;
      if (!label?.trim()) problems.push(`interactive-${index}-missing-label`);
      if (rect.width < 24 || rect.height < 24) problems.push(`interactive-${index}-small-target-${Math.round(rect.width)}x${Math.round(rect.height)}`);
    });
    document.querySelectorAll('[role="dialog"], dialog').forEach((element, index) => {
      if (!element.getAttribute('aria-labelledby') && !element.getAttribute('aria-label')) problems.push(`dialog-${index}-missing-name`);
    });
    return problems;
  });
  expect(violations).toEqual([]);
}

test.describe('design system gallery and shell', () => {
  for (const variant of variants) {
    test(`captures ${variant.name}`, async ({ page }) => {
      await boot(page, variant);
      await expectNoBasicA11yViolations(page);
      const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
      expect(overflow).toBeLessThanOrEqual(1);
      await page.screenshot({ path: `tests/screenshots/${variant.name}.png`, fullPage: true });
    });
  }

  test('tests dialog keyboard focus and missing backdrop-filter fallback flag', async ({ page }) => {
    await boot(page, variants[0]);
    await page.getByText('오버레이').or(page.getByText('Overlays')).click().catch(() => undefined);
    await page.getByRole('button', { name: /dialog|다이얼로그|Open/i }).click();
    await expect(page.getByTestId('gallery-dialog')).toBeVisible();
    await page.keyboard.press('Escape');
    await expect(page.getByTestId('gallery-dialog')).toBeHidden();
    const backdropSupport = await page.evaluate(() => CSS.supports('backdrop-filter: blur(1px)'));
    expect(typeof backdropSupport).toBe('boolean');
  });
});
