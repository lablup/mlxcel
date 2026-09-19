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

/** The fixture's ready model repeated under each name, in the given order; ids stay distinct. */
function readyModels(names: ReadonlyArray<string>): (catalog: CatalogPage) => CatalogPage {
  return (catalog) => {
    const [base] = readyCatalog(catalog).items;
    catalog.items = names.map((name, index) => ({ ...structuredClone(base), identity: { ...base.identity, id: `mdl_${String(index).padStart(43, '0')}`, display_name: name, inference_id: name } }));
    catalog.pagination = { ...catalog.pagination, total_known: names.length };
    return catalog;
  };
}

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
  // Signed out, the shell has not read the server, so it claims nothing about it.
  await expect(page.getByTestId('toolbar-loaded')).toHaveText('Loaded models unknown');
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
  await installMockApi(page, 'happy', { catalog: readyModels(['delta', 'alpha', 'charlie', 'bravo']) });
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

test('a tight desktop toolbar keeps naming every loaded model and shrinks the badge to its dot first', async ({ page }) => {
  // 1024 px is a reference viewport; three long names plus "+1" leave each chip about 100 px.
  const names = ['Meta-Llama-3.1-8B-Instruct-4bit', 'Mixtral-8x7B-Instruct-v0.1-4bit', 'Qwen2.5-7B-Instruct-4bit', 'gemma-3-4b-it-4bit'];
  await installMockApi(page, 'happy', { catalog: readyModels(names) });
  await bootProduct(page, { name: 'tight', width: 1024, height: 768, signedIn: false, appearance: appearance('en') });
  await page.getByLabel('Session key').fill('good-key');
  await page.getByRole('button', { name: 'Connect', exact: true }).click();
  await expect(page.getByTestId('toolbar-loaded-count')).toHaveText('4 loaded');
  await expect(page.getByTestId('toolbar-loaded-chip')).toHaveCount(3);
  const geometry = await page.evaluate(() => {
    const width = (element: Element | null | undefined): number => element?.getBoundingClientRect().width ?? 0;
    const chips = Array.from(document.querySelectorAll('.app-toolbar [data-testid="toolbar-loaded-chip"]')).map((chip) => ({ name: width(chip.querySelector('.truncate')), badge: width(chip.querySelector('.ds-status-wrap')) }));
    const region = document.querySelector('.app-toolbar .toolbar-loaded')?.getBoundingClientRect();
    const actions = document.querySelector('.app-toolbar .toolbar-actions')?.getBoundingClientRect();
    return { chips, regionRight: region?.right ?? Infinity, actionsLeft: actions?.left ?? 0 };
  });
  for (const chip of geometry.chips) {
    expect(chip.name, JSON.stringify(geometry)).toBeGreaterThanOrEqual(40);
    expect(chip.badge, JSON.stringify(geometry)).toBeGreaterThanOrEqual(24);
  }
  expect(geometry.regionRight).toBeLessThanOrEqual(geometry.actionsLeft);
  // Clipping the badge label is visual only: the accessible name still carries the state.
  await expect(page.getByRole('button', { name: `${names[0]} Ready`, exact: true })).toBeVisible();
});

// Below 960 px the chip row is its own toolbar row. It scrolls sideways inside itself rather
// than wrapping, so the sticky toolbar keeps two rows (title, chips) however many models are
// loaded. Playwright's headless Chromium hides scrollbars; a classic scrollbar elsewhere adds
// only its own thickness to that second row.
const LONG_NAMES = ['DeepSeek-R1-Distill-Qwen-32B-4bit', 'Meta-Llama-3.1-8B-Instruct-4bit', 'Mixtral-8x7B-Instruct-v0.1-4bit', 'Qwen2.5-7B-Instruct-4bit'];
for (const [width, height] of [[400, 844], [700, 900]] as const) {
  test(`at ${width} px long-named chips scroll in one row and the toolbar stays two rows high`, async ({ page }) => {
    let names: ReadonlyArray<string> = LONG_NAMES.slice(0, 1);
    await installMockApi(page, 'happy', { catalog: (catalog) => readyModels(names)(catalog) });
    await bootProduct(page, { name: 'compact-chips', width, height, signedIn: false, appearance: appearance('en') });
    await loginWithMockApi(page);
    await expect(page.getByTestId('toolbar-loaded-count')).toHaveText('1 loaded');
    const toolbar = page.locator('.app-toolbar');
    const oneChip = (await toolbar.boundingBox())?.height ?? Infinity;
    names = LONG_NAMES;
    await page.getByRole('button', { name: 'Refresh server state', exact: true }).click();
    await expect(page.getByTestId('toolbar-loaded-count')).toHaveText('4 loaded');
    const chips = page.getByTestId('toolbar-loaded-chip');
    await expect(chips).toHaveCount(3);
    const fourChips = (await toolbar.boundingBox())?.height ?? Infinity;
    console.log(`SHELL_COMPACT_CHIPS viewport=${width} one_chip_toolbar=${oneChip} four_chip_toolbar=${fourChips}`);
    expect(fourChips).toBeLessThanOrEqual(oneChip + 2);
    expect(await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)).toBeLessThanOrEqual(1);
    await expectAxeClean(page);
    // The third chip ends past the row's visible edge; tabbing onto it scrolls it into view.
    const region = page.getByTestId('toolbar-loaded');
    const rowRight = async (): Promise<number> => region.evaluate((element) => element.getBoundingClientRect().right);
    const before = await chips.nth(2).boundingBox();
    expect((before?.x ?? 0) + (before?.width ?? 0)).toBeGreaterThan((await rowRight()) + 1);
    await chips.nth(1).focus();
    await page.keyboard.press('Tab');
    await expect(chips.nth(2)).toBeFocused();
    const third = await chips.nth(2).boundingBox();
    if (third === null) throw new Error('The third chip has no box.');
    expect(third.x).toBeGreaterThanOrEqual(0);
    expect(third.x + third.width).toBeLessThanOrEqual(Math.min(width, await rowRight()) + 1);
    expect(await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)).toBeLessThanOrEqual(1);
  });
}

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
  // Below 1100 px the inspector is a drawer, opened on the palette's request (#1918).
  await expect(page.getByRole('dialog', { name: 'Model details' })).toContainText('alpha');
  await expect(page.getByRole('complementary', { name: 'Model details' })).toHaveCount(0);
  expect(mock.calls.map((call) => call.url).join('\n')).not.toContain('/ui-api/v1/model-actions');
  expect(mock.calls.filter((call) => call.method !== 'GET')).toEqual([]);
});
