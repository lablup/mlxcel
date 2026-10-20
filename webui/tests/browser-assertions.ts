import { AxeBuilder } from '@axe-core/playwright';
import { expect, type Locator, type Page } from '@playwright/test';

// Asserts Chromium's actual accessibility tree for the gallery empty state's heading:
// a level-two heading named by its visible title.
export async function expectEmptyStateHeading(page: Page): Promise<void> {
  const selector = '[data-testid="gallery-empty"] .astryx-empty-state-title';
  const heading = page.locator(selector);
  await expect(heading).toBeVisible();
  await expect(heading).toHaveJSProperty('tagName', 'H2');
  const name = await heading.innerText();
  const session = await page.context().newCDPSession(page);
  try {
    const { root } = await session.send('DOM.getDocument');
    const { nodeId } = await session.send('DOM.querySelector', { nodeId: root.nodeId, selector });
    expect(nodeId).toBeGreaterThan(0);
    const { nodes } = await session.send('Accessibility.getPartialAXTree', { nodeId, fetchRelatives: false });
    expect(nodes).toHaveLength(1);
    expect(nodes[0].ignored).toBe(false);
    expect(nodes[0].role?.value).toBe('heading');
    expect(nodes[0].name?.value).toBe(name);
    expect(nodes[0].properties?.find((property) => property.name === 'level')?.value.value).toBe(2);
  } finally {
    await session.detach();
  }
}

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

export async function expectSafeLayout(page: Page): Promise<void> {
  const result = await page.evaluate(() => ({
    overflow: document.documentElement.scrollWidth - document.documentElement.clientWidth,
    unexpectedStyles: Array.from(document.querySelectorAll<HTMLElement>('[style]')).filter((element) => {
      // Screenshot caret cleanup can leave style="". Only trim-empty raw text is inert;
      // do not accept nonempty invalid CSS merely because CSSOM has no properties.
      if ((element.getAttribute('style') ?? '').trim() === '') return false;
      // Library-owned CSSOM only, each kind scoped to the Astryx or ui-common element that writes
      // it; not evidence that production CSP permits it (playwright.csp.config.ts checks that).
      const properties = Array.from(element.style);
      const onlyCustom = (): boolean => properties.some((key) => !key.startsWith('--'));
      // ProgressBar fill: its measured percentage.
      if (element.matches('.ds-progress .astryx-progress-bar-fill')) return properties.some((key) => key !== 'width') || !/^\d+(?:\.\d+)?%$/.test(element.style.width);
      // Popover layers (Selector list, Tooltip, TabList keyboard hint): CSS anchor positioning.
      if (element.matches('[popover]')) return properties.some((key) => !(key.startsWith('--') || key.startsWith('position-') || ['margin-block-start', 'margin-block-end', 'margin-inline-start', 'margin-inline-end', 'justify-self', 'align-self'].includes(key)));
      // A popover's anchor: the Selector trigger box, or a Button carrying a tooltip.
      if (element.matches('.astryx-selector, .astryx-button')) return properties.join(',') !== 'anchor-name' || !/^--/.test(element.style.getPropertyValue('anchor-name'));
      // StyleX dynamic values (field and skeleton sizes, the Drawer's sheet width) travel as custom properties.
      if (element.matches('.astryx-field, .astryx-skeleton, dialog.astryx-drawer.ds-drawer')) return onlyCustom();
      // A layer's anchor marker: a decorative box named for CSS anchor positioning.
      if (element.matches('span[aria-hidden="true"]:empty')) return properties.slice().sort().join(',') !== 'anchor-name,left,top' || !element.style.getPropertyValue('anchor-name').startsWith('--astryx-layer-');
      // Button loading spinner: its box size, through Astryx's own custom property.
      if (element.matches('.astryx-spinner')) return properties.slice().sort().join(',') !== 'height,width' || ['width', 'height'].some((key) => !/^(?:\d+(?:\.\d+)?px|var\(--_spinner-box-size, \d+px\))$/.test(element.style.getPropertyValue(key)));
      // A table with a fixed column: Astryx's floor, the sum of the fixed widths.
      if (element.matches('table.ds-common-table')) return properties.join(',') !== 'min-width' || !/^\d+px$/.test(element.style.minWidth);
      // Table header cells: a fixed column's width from Astryx's pixel() column width.
      if (element.matches('.ds-common-table th')) return properties.slice().sort().join(',') !== 'min-width,width' || element.style.width !== element.style.minWidth || !/^\d+px$/.test(element.style.width);
      // The modal Drawer's scroll lock on <body>, only while a modal dialog is open.
      if (element === document.body) return !document.querySelector('dialog[open][aria-modal="true"]') || properties.some((key) => !['overflow', 'overflow-x', 'overflow-y', 'position', 'top', 'left', 'right', 'padding-right'].includes(key));
      // Shared SmoothHeight: the measured content height while active. Scoped to the product
      // adapter's class, so an unadapted SmoothHeight use elsewhere cannot pass by class name alone.
      if (element.matches('.uic-smooth-height.ds-smooth-height')) return properties.join(',') !== 'height' || !/^\d+px$/.test(element.style.height);
      return true;
    }).map((element) => `${element.outerHTML.slice(0, 120)} [style: ${element.getAttribute('style')}]`),
    smallTargets: Array.from(document.querySelectorAll<HTMLElement>('button, a[href], input, select, textarea')).filter((element) => {
      const rect = element.getBoundingClientRect();
      const style = window.getComputedStyle(element);
      if ((rect.width === 0 && rect.height === 0) || style.visibility === 'hidden' || style.display === 'none') return false;
      return rect.width < 24 || rect.height < 24;
    }).map((element) => element.outerHTML.slice(0, 80)),
  }));
  expect(result.overflow).toBeLessThanOrEqual(1);
  expect(result.unexpectedStyles).toEqual([]);
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
    '.uic-page-layout',
    '.toolbar-title',
    '.toolbar-loaded',
    '.screen-stack',
    '.uic-page-header',
    '.uic-page-header__title',
    '.uic-page-header__description',
    '.uic-page-header__actions',
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
    page.getByText('공통 선택 콤보박스', { exact: true }),
    page.getByRole('combobox', { name: /공통 선택 콤보박스/i }),
  ];
  for (const locator of reachable) await expectLocatorWithinViewportX(locator);
}

export async function reportFontDiagnostics(page: Page, label: string): Promise<void> {
  const selectors = ['.brand-mark', '.toolbar-title p', '.toolbar-title span', '.uic-page-header__title', '.uic-page-header__description', '.app-nav a', '.ds-tabs [role="tab"]', '.ds-button', '.ds-field > span'];
  const client = await page.context().newCDPSession(page);
  try {
    await client.send('DOM.enable');
    await client.send('CSS.enable');
    const { root } = await client.send('DOM.getDocument');
    const diagnostics: Array<{ selector: string; text: string; computed: { fontFamily: string; fontSize: string; fontWeight: string }; platformFonts: unknown[] }> = [];
    for (const selector of selectors) {
      const { nodeId } = await client.send('DOM.querySelector', { nodeId: root.nodeId, selector });
      if (!nodeId) continue;
      const computed = await page.locator(selector).first().evaluate((element) => {
        const style = window.getComputedStyle(element);
        return { fontFamily: style.fontFamily, fontSize: style.fontSize, fontWeight: style.fontWeight };
      });
      const text = await page.locator(selector).first().evaluate((element) => element.textContent?.trim().slice(0, 80) ?? '');
      const { fonts } = await client.send('CSS.getPlatformFontsForNode', { nodeId });
      diagnostics.push({ selector, text, computed, platformFonts: fonts as unknown[] });
    }
    const browser = await page.evaluate(() => ({ userAgent: navigator.userAgent, language: navigator.language, devicePixelRatio: window.devicePixelRatio }));
    console.log(`WEBUI_FONT_DIAGNOSTICS ${JSON.stringify({ label, browser, diagnostics })}`);
  } finally {
    await client.detach();
  }
}
