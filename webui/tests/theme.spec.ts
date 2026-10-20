// Copyright 2025-2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import { expect, test, type Page } from '@playwright/test';
import { expectAxeClean, expectSafeLayout } from './browser-assertions';
import { bootGallery, installMockApi } from './browser-fixtures';

const STORAGE_KEY = 'mlxcel.webui.appearance';

// The color each family and scheme paints on <html> before anything mounts (--color-background-body).
const CANVAS: Record<string, string> = {
  'mlxcel-light': 'rgb(243, 244, 246)',
  'mlxcel-dark': 'rgb(14, 17, 23)',
  'glass-light': 'rgb(230, 235, 243)',
  'glass-dark': 'rgb(11, 13, 18)',
};
const ASTRYX_THEME: Record<string, string> = { mlxcel: 'mlxcel', glass: 'mlxcel-glass' };

/** The root theme attributes, in the order <html> carries them. */
async function rootTheme(page: Page): Promise<{ theme: string | null; astryxTheme: string | null; family: string | null }> {
  return page.evaluate(() => ({
    theme: document.documentElement.getAttribute('data-theme'),
    astryxTheme: document.documentElement.getAttribute('data-astryx-theme'),
    family: document.documentElement.getAttribute('data-theme-family'),
  }));
}

async function expectRootTheme(page: Page, family: string, scheme: string): Promise<void> {
  await expect(page.locator('html')).toHaveAttribute('data-theme', scheme);
  await expect(page.locator('html')).toHaveAttribute('data-theme-family', family);
  expect(await rootTheme(page)).toEqual({ theme: scheme, astryxTheme: ASTRYX_THEME[family], family });
}

/** Seeds the stored appearance once per tab, so a later reload reads what the app saved instead. */
async function storeAppearance(page: Page, appearance: Record<string, unknown>): Promise<void> {
  await page.addInitScript(([key, value]) => {
    if (sessionStorage.getItem('theme-spec-seeded') === '1') return;
    sessionStorage.setItem('theme-spec-seeded', '1');
    localStorage.setItem(key, value);
  }, [STORAGE_KEY, JSON.stringify(appearance)] as const);
}

/** Holds the app entry module so the page can be inspected in its pre-mount state. */
async function holdAppModule(page: Page): Promise<() => void> {
  let release = (): void => undefined;
  const released = new Promise<void>((resolve) => { release = resolve; });
  await page.route(/\/assets\/index-[^/]+\.js$/, async (route) => {
    await released;
    await route.continue();
  });
  return release;
}

async function preMountState(page: Page): Promise<{ theme: string | null; astryxTheme: string | null; family: string | null; preference: string | null; canvas: string; colorScheme: string; mounted: number }> {
  // Wait for the stylesheet (a theme-independent :root token), not for a theme to apply.
  await page.waitForFunction(() => getComputedStyle(document.documentElement).getPropertyValue('--mx-space-1').trim() !== '');
  return page.evaluate(() => ({
    theme: document.documentElement.getAttribute('data-theme'),
    astryxTheme: document.documentElement.getAttribute('data-astryx-theme'),
    family: document.documentElement.getAttribute('data-theme-family'),
    preference: document.documentElement.getAttribute('data-color-scheme'),
    canvas: getComputedStyle(document.documentElement).backgroundColor,
    colorScheme: getComputedStyle(document.documentElement).colorScheme,
    mounted: document.getElementById('root')?.childElementCount ?? -1,
  }));
}

async function emulate(page: Page, features: { name: string; value: string }[]): Promise<void> {
  const client = await page.context().newCDPSession(page);
  await client.send('Emulation.setEmulatedMedia', { features });
}

/** Boots the gallery from a blank page, so the stored appearance is re-read by a real load. */
async function bootFresh(page: Page, name: string, appearance: Record<string, unknown>): Promise<void> {
  await page.goto('about:blank');
  await bootGallery(page, { name, width: 1440, height: 900, tab: 'controls', appearance });
}

/** Resolves color tokens through a probe element, so minified spellings (#fff, #rrggbbaa) and light-dark() compare equal. */
async function resolvedColors(page: Page, names: string[], parent = 'body'): Promise<Record<string, string>> {
  return page.evaluate(([keys, selector]) => {
    const probe = document.createElement('div');
    (document.querySelector(selector) ?? document.body).append(probe);
    const result = Object.fromEntries(keys.map((key) => {
      probe.style.backgroundColor = `var(${key})`;
      return [key, getComputedStyle(probe).backgroundColor];
    }));
    probe.remove();
    return result;
  }, [names, parent] as const);
}

/** Reads the fill the first rendered button of a tone paints, which a later hard-coded rule can hide from the token. */
async function buttonFill(page: Page, selector: string): Promise<{ color: string; image: string }> {
  return page.locator(selector).first().evaluate((element) => ({ color: getComputedStyle(element).backgroundColor, image: getComputedStyle(element).backgroundImage }));
}

function milliseconds(value: string): number {
  return value.endsWith('ms') ? Number.parseFloat(value) : Number.parseFloat(value) * 1000;
}

async function rootValues(page: Page, names: string[]): Promise<Record<string, string>> {
  return page.evaluate((keys) => Object.fromEntries(keys.map((key) => [key, getComputedStyle(document.documentElement).getPropertyValue(key).trim()])), names);
}

test.describe('theme system', () => {
  const cases: { stored: Record<string, unknown>; host: 'light' | 'dark'; id: string; preference: string }[] = [
    { stored: { themeFamily: 'mlxcel', colorScheme: 'dark' }, host: 'light', id: 'mlxcel-dark', preference: 'dark' },
    { stored: { themeFamily: 'glass', colorScheme: 'light' }, host: 'dark', id: 'glass-light', preference: 'light' },
    { stored: { themeFamily: 'glass', colorScheme: 'system' }, host: 'dark', id: 'glass-dark', preference: 'system' },
    { stored: { theme: 'dark' }, host: 'light', id: 'mlxcel-dark', preference: 'dark' },
    { stored: {}, host: 'light', id: 'mlxcel-light', preference: 'system' },
  ];
  for (const { stored, host, id, preference } of cases) {
    test(`applies ${id} before the app mounts (stored ${JSON.stringify(stored)}, ${host} host)`, async ({ page }) => {
      await page.emulateMedia({ colorScheme: host });
      await storeAppearance(page, stored);
      const release = await holdAppModule(page);
      await page.goto('/#models', { waitUntil: 'commit' });
      const early = await preMountState(page);
      const [family, scheme] = id.split('-');
      expect(early).toEqual({ theme: scheme, astryxTheme: ASTRYX_THEME[family], family, preference, canvas: CANVAS[id], colorScheme: scheme, mounted: 0 });
      release();
      await expect(page.getByTestId('app-title')).toBeVisible();
      // The mounted root <Theme> writes the same two attributes the bootstrap did.
      await expectRootTheme(page, family, scheme);
      expect(await page.evaluate(() => getComputedStyle(document.documentElement).backgroundColor)).toBe(CANVAS[id]);
    });
  }

  test('follows a system preference live and leaves a pinned scheme alone', async ({ page }) => {
    await page.emulateMedia({ colorScheme: 'light' });
    await storeAppearance(page, { themeFamily: 'glass', colorScheme: 'system' });
    await page.goto('/#gallery');
    await expect(page.getByTestId('gallery-title')).toBeVisible();
    await expectRootTheme(page, 'glass', 'light');
    await page.emulateMedia({ colorScheme: 'dark' });
    await expectRootTheme(page, 'glass', 'dark');
    await expect(page.locator('html')).toHaveAttribute('data-color-scheme', 'system');
    await page.evaluate((key) => localStorage.setItem(key, JSON.stringify({ themeFamily: 'glass', colorScheme: 'light' })), STORAGE_KEY);
    await page.reload();
    await expect(page.getByTestId('gallery-title')).toBeVisible();
    await expectRootTheme(page, 'glass', 'light');
    await page.emulateMedia({ colorScheme: 'light' });
    await page.emulateMedia({ colorScheme: 'dark' });
    await expectRootTheme(page, 'glass', 'light');
  });

  test('switches family and scheme from Settings, persists them, and restores them before mount', async ({ page }) => {
    await installMockApi(page, 'happy');
    await page.emulateMedia({ colorScheme: 'light' });
    await page.setViewportSize({ width: 1440, height: 900 });
    await page.goto('/#settings');
    await expect(page.getByTestId('settings-title')).toBeVisible();
    await expectRootTheme(page, 'mlxcel', 'light');
    await page.getByRole('combobox', { name: 'Theme' }).click();
    await page.getByRole('option', { name: 'Glass' }).click();
    await expectRootTheme(page, 'glass', 'light');
    // The family switch only moved the attribute: the glass canvas is painted.
    expect(await page.evaluate(() => getComputedStyle(document.documentElement).backgroundColor)).toBe(CANVAS['glass-light']);
    await page.getByRole('combobox', { name: 'Color scheme' }).click();
    await page.getByRole('option', { name: 'Dark' }).click();
    await expectRootTheme(page, 'glass', 'dark');
    await expect(page.locator('html')).toHaveAttribute('data-color-scheme', 'dark');
    const stored = await page.evaluate((key) => JSON.parse(localStorage.getItem(key) ?? '{}') as Record<string, unknown>, STORAGE_KEY);
    expect(stored).toMatchObject({ themeFamily: 'glass', colorScheme: 'dark' });
    expect(stored).not.toHaveProperty('theme');
    await expectAxeClean(page);
    await expectSafeLayout(page);
    const release = await holdAppModule(page);
    await page.reload({ waitUntil: 'commit' });
    expect(await preMountState(page)).toMatchObject({ theme: 'dark', astryxTheme: 'mlxcel-glass', family: 'glass', canvas: CANVAS['glass-dark'] });
    release();
    await expect(page.getByTestId('settings-title')).toBeVisible();
  });

  test('keeps the mlxcel baseline values unchanged, now as the Astryx tokens Astryx components read', async ({ page }) => {
    for (const [scheme, expected] of [
      // The pre-#1903 :root values (--color-bg, --color-text, --color-focus, --color-bg-elevated, --color-separator, --material-content-bg).
      ['light', { '--color-background-body': 'rgb(243, 244, 246)', '--color-text-primary': 'rgb(21, 26, 36)', '--color-text-accent': 'rgb(36, 91, 216)', '--color-accent': 'rgb(36, 91, 216)', '--mx-button-primary-fill': 'rgb(36, 91, 216)', '--mx-button-danger-fill': 'rgb(198, 53, 43)', '--color-background-surface': 'rgb(255, 255, 255)', '--color-border': 'rgba(33, 42, 61, 0.12)', '--mx-material-content-bg': 'rgba(255, 255, 255, 0.82)' }],
      ['dark', { '--color-background-body': 'rgb(14, 17, 23)', '--color-text-primary': 'rgb(245, 247, 251)', '--color-text-accent': 'rgb(157, 189, 255)', '--color-accent': 'rgb(36, 91, 216)', '--mx-button-primary-fill': 'rgb(36, 91, 216)', '--mx-button-danger-fill': 'rgb(198, 53, 43)', '--color-background-surface': 'rgb(24, 29, 39)', '--color-border': 'rgba(255, 255, 255, 0.12)', '--mx-material-content-bg': 'rgba(25, 29, 38, 0.84)' }],
    ] as const) {
      await bootFresh(page, `baseline-${scheme}`, { colorScheme: scheme, material: 'glass', locale: 'en', glassIntensity: 35, highContrast: 'off' });
      expect(await resolvedColors(page, Object.keys(expected))).toEqual(expected);
      // Inside the root <Theme> wrapper too, where Astryx re-declares its tokens.
      expect(await resolvedColors(page, Object.keys(expected), '[data-astryx-theme]:not(html)')).toEqual(expected);
      expect(await buttonFill(page, '.ds-button-primary')).toEqual({ color: 'rgb(36, 91, 216)', image: 'none' });
      expect(await buttonFill(page, '.ds-button-danger')).toEqual({ color: 'rgb(198, 53, 43)', image: 'none' });
    }
  });

  test('glass paints its highlighted fill on the rendered filled buttons', async ({ page }) => {
    for (const [colorScheme, primary] of [['light', 'rgb(0, 88, 185)'], ['dark', 'rgb(10, 91, 192)']] as const) {
      await bootFresh(page, `glass-fill-${colorScheme}`, { themeFamily: 'glass', colorScheme, material: 'glass', locale: 'en', glassIntensity: 35, highContrast: 'off' });
      const primaryFill = await buttonFill(page, '.ds-button-primary');
      const dangerFill = await buttonFill(page, '.ds-button-danger');
      expect(primaryFill.image).toContain('linear-gradient');
      expect(primaryFill.color).toBe(primary);
      expect(dangerFill.image).toContain('linear-gradient');
      expect(dangerFill.color).toBe('rgb(179, 38, 30)');
    }
  });

  test('glass chrome follows the intensity scale in Chromium', async ({ page }) => {
    const blurAt = async (glassIntensity: number): Promise<{ filter: string; image: string }> => {
      await bootFresh(page, `glass-${glassIntensity}`, { themeFamily: 'glass', colorScheme: 'light', material: 'glass', locale: 'en', glassIntensity, highContrast: 'off' });
      return page.locator('.material-glass').first().evaluate((element) => ({ filter: getComputedStyle(element).backdropFilter, image: getComputedStyle(element).backgroundImage }));
    };
    const low = await blurAt(0);
    const high = await blurAt(100);
    // A standard backdrop-filter must survive minification (the prefixed form alone is ignored here).
    expect(low.filter).toBe('blur(12px) saturate(1.12)');
    expect(high.filter).toBe('blur(36px) saturate(1.48)');
    expect(low.image).toContain('linear-gradient');
  });

  const opaqueCases: { label: string; appearance: Record<string, unknown>; media: { name: string; value: string }[] }[] = [
    { label: 'reduce transparency setting', appearance: { reduceTransparency: true }, media: [] },
    { label: 'opaque material', appearance: { material: 'opaque' }, media: [] },
    { label: 'high contrast setting', appearance: { highContrast: 'on' }, media: [] },
    { label: 'prefers-contrast: more', appearance: {}, media: [{ name: 'prefers-contrast', value: 'more' }] },
    { label: 'prefers-reduced-transparency: reduce', appearance: {}, media: [{ name: 'prefers-reduced-transparency', value: 'reduce' }] },
    { label: 'both OS preferences at full intensity', appearance: {}, media: [{ name: 'prefers-contrast', value: 'more' }, { name: 'prefers-reduced-transparency', value: 'reduce' }] },
  ];
  for (const { label, appearance, media } of opaqueCases) {
    for (const colorScheme of ['light', 'dark'] as const) {
      test(`glass-${colorScheme} turns opaque under ${label}, whatever the intensity`, async ({ page }) => {
        if (media.length) await emulate(page, media);
        await bootFresh(page, label, { themeFamily: 'glass', colorScheme, material: 'glass', locale: 'en', glassIntensity: 100, highContrast: 'system', ...appearance });
        const chrome = await page.locator('.material-glass').first().evaluate((element) => ({ filter: getComputedStyle(element).backdropFilter, image: getComputedStyle(element).backgroundImage, color: getComputedStyle(element).backgroundColor }));
        expect(chrome.filter).toMatch(/^(none|blur\(0px\)( saturate\(1\))?)$/);
        expect(chrome.image).toBe('none');
        expect(chrome.color).not.toMatch(/rgba\(.*, 0\.\d+\)$/);
        const values = await rootValues(page, ['--mx-button-primary-fill', '--mx-material-content-bg']);
        expect(values['--mx-button-primary-fill']).not.toContain('gradient');
        expect((await buttonFill(page, '.ds-button-primary')).image).toBe('none');
        const surfaces = await resolvedColors(page, ['--mx-material-content-bg', '--color-background-surface']);
        expect(surfaces['--mx-material-content-bg']).toBe(surfaces['--color-background-surface']);
        await expectAxeClean(page);
      });
    }
  }

  test('drops the press transform under reduced motion', async ({ page }) => {
    const transform = async (reduceMotion: boolean): Promise<{ press: string; fast: number }> => {
      await bootFresh(page, 'glass-motion', { themeFamily: 'glass', colorScheme: 'light', material: 'glass', locale: 'en', glassIntensity: 35, reduceMotion });
      const values = await rootValues(page, ['--mx-button-press-transform', '--mx-motion-fast']);
      return { press: values['--mx-button-press-transform'], fast: milliseconds(values['--mx-motion-fast']) };
    };
    const moving = await transform(false);
    expect(moving.press).toMatch(/^scale\(0?\.985\)$/);
    expect(moving.fast).toBe(120);
    expect(await transform(true)).toEqual({ press: 'none', fast: 1 });
    await page.emulateMedia({ reducedMotion: 'reduce' });
    expect(await transform(false)).toEqual({ press: 'none', fast: 1 });
  });
});
