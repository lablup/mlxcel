// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import { expect, test, type Page } from '@playwright/test';
import { expectAxeClean, expectSafeLayout } from './browser-assertions';
import { bootGallery, bootProduct, installMockApi, loginWithMockApi, readyCatalog, selectGalleryTab, type CatalogPage } from './browser-fixtures';
import { fixtureString, type FixtureLocale } from './strings-fixture';

// The layout floor: a 390 px window at native 200 percent zoom is a 195 CSS px viewport.
// WCAG 2.2 SC 1.4.10 asks for 320; 320 stays as the regression guard.
const FLOOR_WIDTHS = [195, 320] as const;
const LOCALES: FixtureLocale[] = ['en', 'ko'];
const appearance = (locale: FixtureLocale): Record<string, unknown> => ({ theme: 'light', material: 'opaque', locale, glassIntensity: 0, highContrast: 'system', reduceMotion: true });

interface ShellGeometry { readonly overflow: number; readonly collisions: string[] }

async function shellGeometry(page: Page): Promise<ShellGeometry> {
  return page.evaluate(() => {
    const visible = (element: Element): boolean => {
      const rect = element.getBoundingClientRect();
      const style = window.getComputedStyle(element);
      return rect.width > 0 && rect.height > 0 && style.visibility !== 'hidden' && style.display !== 'none';
    };
    const title = document.querySelector('.app-toolbar .toolbar-title');
    const box = title?.getBoundingClientRect();
    const collisions = box === undefined ? ['missing toolbar title'] : Array.from(document.querySelectorAll('.app-toolbar .toolbar-actions .ds-icon-button, .app-toolbar .mobile-menu-button, .app-toolbar .toolbar-loaded'))
      .filter(visible)
      .filter((element) => {
        const other = element.getBoundingClientRect();
        return other.left < box.right - 0.5 && other.right > box.left + 0.5 && other.top < box.bottom - 0.5 && other.bottom > box.top + 0.5;
      })
      .map((element) => element.getAttribute('aria-label') ?? element.className);
    return { overflow: document.documentElement.scrollWidth - document.documentElement.clientWidth, collisions };
  });
}

async function expectFloor(page: Page, label: string): Promise<void> {
  const geometry = await shellGeometry(page);
  // Reported for the hand-back record; the assertions are what gate.
  console.log(`SHELL_FLOOR ${label} viewport=${page.viewportSize()?.width} overflow=${geometry.overflow}`);
  expect(geometry.overflow, label).toBeLessThanOrEqual(1);
  expect(geometry.collisions, label).toEqual([]);
}

for (const width of FLOOR_WIDTHS) {
  for (const locale of LOCALES) {
    test(`${width} gallery tabs and drawer hold without horizontal scroll (${locale})`, async ({ page }) => {
      await bootGallery(page, { name: 'floor', width, height: 422, tab: 'controls', appearance: appearance(locale) });
      await expectFloor(page, `${width}-${locale}-gallery-controls`);
      for (const tab of ['states', 'data'] as const) {
        await selectGalleryTab(page, tab);
        await expectFloor(page, `${width}-${locale}-gallery-${tab}`);
      }
      await page.getByTestId('toolbar-menu').click();
      const drawer = page.getByTestId('mobile-nav-sheet');
      await expect(drawer).toBeVisible();
      await expectFloor(page, `${width}-${locale}-gallery-drawer`);
      expect(await drawer.evaluate((element) => element.scrollWidth - element.clientWidth)).toBeLessThanOrEqual(1);
    });

    test(`${width} product routes hold without horizontal scroll, signed out and signed in with a loaded model (${locale})`, async ({ page }) => {
      await installMockApi(page, 'happy', { catalog: readyCatalog });
      await bootProduct(page, { name: 'floor', width, height: 422, signedIn: false, appearance: appearance(locale) });
      await expect(page.getByTestId('auth-login')).toBeVisible();
      await expectFloor(page, `${width}-${locale}-login`);
      await loginWithMockApi(page);
      await expect(page.getByTestId('toolbar-loaded-chip')).toHaveCount(1);
      await expectFloor(page, `${width}-${locale}-models`);
      for (const route of ['chat', 'activity', 'settings']) {
        await page.evaluate((next) => { window.location.hash = next; }, route);
        await expect(page.getByTestId(`${route}-title`)).toBeVisible();
        await expect(page.getByTestId('toolbar-loaded-chip')).toBeVisible();
        await expectFloor(page, `${width}-${locale}-${route}`);
      }
      // The sidebar footer's details disclosure fits the drawer when opened.
      await page.getByTestId('toolbar-menu').click();
      const drawer = page.getByTestId('mobile-nav-sheet');
      await drawer.getByText(fixtureString(locale, 'connection.footer.details'), { exact: true }).click();
      await expect(drawer.getByTestId('connection-footer-instance')).toBeVisible();
      expect(await drawer.evaluate((element) => element.scrollWidth - element.clientWidth)).toBeLessThanOrEqual(1);
      await expectFloor(page, `${width}-${locale}-drawer-details`);
    });
  }
}

test('the toolbar names the server-loaded model and its chip opens the inspector without any model action', async ({ page }) => {
  const mock = await installMockApi(page, 'happy', { catalog: readyCatalog });
  await bootProduct(page, { name: 'chip', width: 1440, height: 900, signedIn: false, appearance: appearance('en') });
  await expect(page.getByTestId('toolbar-loaded')).toHaveText('No model loaded');
  await loginWithMockApi(page);
  const region = page.getByRole('group', { name: 'Loaded models' });
  await expect(region.getByTestId('toolbar-loaded-count')).toHaveText('1 loaded');
  const chip = region.getByRole('button', { name: 'alpha Ready', exact: true });
  await expect(chip).toBeVisible();
  expect((await chip.boundingBox())?.height ?? 0).toBeGreaterThanOrEqual(32);
  await expectAxeClean(page);
  await expectSafeLayout(page);
  await page.getByRole('link', { name: 'Settings' }).first().click();
  await chip.click();
  await expect(page).toHaveURL(/#models$/);
  await expect(page.getByRole('complementary', { name: 'Model details' })).toContainText('alpha');
  // The footer keeps identity and hides the internal counters in a collapsed disclosure.
  const footer = page.locator('.desktop-sidebar footer');
  await expect(footer.getByTestId('connection-ready')).toHaveText(/^model_free · \S+ · v0\.7\.0$/);
  await expect(footer.getByTestId('connection-footer-instance')).toBeHidden();
  await footer.getByText('Connection details', { exact: true }).click();
  await expect(footer.getByTestId('connection-footer-instance')).toHaveText('Server instance: srv_20260912_a');
  expect(mock.calls.filter((call) => call.method !== 'GET')).toEqual([]);
});

test('more than three loaded models collapse into a +n chip that opens the palette on the loaded list', async ({ page }) => {
  const names = ['delta', 'alpha', 'charlie', 'bravo'];
  const fourReady = (catalog: CatalogPage): CatalogPage => {
    const [base] = readyCatalog(catalog).items;
    catalog.items = names.map((name, index) => ({ ...structuredClone(base), identity: { ...base.identity, id: `mdl_${String(index).padStart(43, '0')}`, display_name: name, inference_id: name } }));
    catalog.pagination = { ...catalog.pagination, total_known: names.length };
    return catalog;
  };
  await installMockApi(page, 'happy', { catalog: fourReady });
  await bootProduct(page, { name: 'more', width: 1440, height: 900, signedIn: false, appearance: appearance('en') });
  await page.getByLabel('Session key').fill('good-key');
  await page.getByRole('button', { name: 'Connect', exact: true }).click();
  await expect(page.getByTestId('toolbar-loaded-count')).toHaveText('4 loaded');
  await expect(page.getByTestId('toolbar-loaded-chip')).toHaveText(['alphaReady', 'bravoReady', 'charlieReady']);
  const more = page.getByRole('button', { name: '+1, show all loaded models', exact: true });
  await expect(more).toHaveText('+1');
  await more.click();
  const dialog = page.getByTestId('command-dialog');
  await expect(dialog).toBeVisible();
  await expect(dialog.getByTestId('command-search')).toHaveValue('');
  await expect(dialog.getByTestId('command-model')).toHaveText(['alpha · Ready', 'bravo · Ready', 'charlie · Ready', 'delta · Ready']);
  await expectAxeClean(page);
});

test('the command palette finds a model by name substring and opens its inspector without a model action', async ({ page }) => {
  const mock = await installMockApi(page, 'happy');
  await bootProduct(page, { name: 'palette', width: 1024, height: 768, signedIn: false, appearance: appearance('en') });
  await loginWithMockApi(page);
  await page.evaluate(() => { window.location.hash = 'activity'; });
  await expect(page.getByTestId('activity-title')).toBeVisible();
  await page.keyboard.press('Control+K');
  const dialog = page.getByTestId('command-dialog');
  await expect(dialog).toBeVisible();
  await expect(dialog.getByRole('button', { name: 'Gallery' })).toHaveCount(0);
  await expect(dialog.getByTestId('command-new-chat')).toBeVisible();
  await dialog.getByTestId('command-search').fill('LPH');
  await expect(dialog.getByTestId('command-model')).toHaveText(['alpha · Unloaded']);
  await expectAxeClean(page);
  await dialog.getByTestId('command-model').click();
  await expect(dialog).toBeHidden();
  await expect(page).toHaveURL(/#models$/);
  await expect(page.getByRole('complementary', { name: 'Model details' })).toContainText('alpha');
  expect(mock.calls.map((call) => call.url).join('\n')).not.toContain('/ui-api/v1/model-actions');
  expect(mock.calls.filter((call) => call.method !== 'GET')).toEqual([]);
});
