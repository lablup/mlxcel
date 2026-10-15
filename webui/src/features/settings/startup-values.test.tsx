// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
// Settings > Server prints the server's read-only startup values as user copy, so no value may be
// Rust Debug output or a raw JSON object (#2329). The fixture is the server's own output for a
// configured server; src/server/runtime_settings_values_tests.rs asserts it stays current.
import React, { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';
import bootstrap from '../../../../tests/fixtures/webui/examples/bootstrap.model-free.json';
import catalog from '../../../../tests/fixtures/webui/examples/catalog.page.json';
import startupValues from '../../../../tests/fixtures/settings/startup-values.json';
import { initialSnapshot } from '../../state/reducer';
import type { BootstrapResponse, CatalogEntry, WebUiSnapshot } from '../../api/types';
import type { SettingKind, SettingsResponse, SettingSpec } from '../../api/settings';
import { t, type Locale } from '../../i18n/catalog';
import { ServerSettings } from './server-settings';

const mocks = vi.hoisted(() => ({ snapshot: null as WebUiSnapshot | null, actions: { selectModel: vi.fn(), loadModel: vi.fn(), unloadModel: vi.fn(), getModelProps: vi.fn(), getSettings: vi.fn() } }));
vi.mock('../../state', () => ({ useWebUi: () => mocks.snapshot, useWebUiActions: () => mocks.actions }));

/** `Some(`, `None`, braces of a struct or JSON object, or a bare CamelCase enum variant such as `LongestFirst`. */
const DEBUG_SHAPE = /Some\(|\bNone\b|[{}]|^[A-Z][a-z]+(?:[A-Z][a-z0-9]*)*$/;

const specs: SettingSpec[] = startupValues.values.map((entry) => ({ name: entry.name, type: entry.type as SettingKind, default: entry.default, mutable: false, allowed: null, help: 'Startup server configuration.', reason: 'restart required' }));

async function renderServerTab(locale: Locale, response: SettingsResponse): Promise<{ host: HTMLDivElement; unmount: () => void }> {
  Object.defineProperty(HTMLElement.prototype, 'scrollIntoView', { configurable: true, value: vi.fn() });
  const entry = structuredClone(catalog.items[0]) as unknown as CatalogEntry;
  const ready = { ...entry, lifecycle: { ...entry.lifecycle, state: 'ready' as const, worker_exit_observed: false } };
  const settingsBootstrap = { ...bootstrap, features: [...bootstrap.features, 'settings'], server: { ...bootstrap.server, mode: 'router_pool' as const } };
  mocks.snapshot = { ...initialSnapshot(), auth: { status: 'authenticated', tokenPresent: true }, connection: 'ready', bootstrap: settingsBootstrap as unknown as BootstrapResponse, catalog: [ready], selectedModelId: ready.identity.id };
  mocks.actions.getSettings.mockResolvedValue(response);
  mocks.actions.getModelProps.mockResolvedValue({ nCtx: 2048, totalSlots: 1, kvCacheMode: 'fp16', geometry: null });
  const host = document.createElement('div'); document.body.append(host); const root = createRoot(host);
  await act(async () => root.render(<ServerSettings locale={locale} />));
  return { host, unmount: () => { act(() => root.unmount()); host.remove(); } };
}

function startupCells(host: HTMLElement): HTMLElement[] {
  const list = host.querySelector('[data-testid="settings-startup-values"]');
  if (list === null) throw new Error('Missing startup values');
  return Array.from(list.querySelectorAll<HTMLElement>('dd'));
}

describe('Settings > Server startup values', () => {
  let cleanup: (() => void) | null = null;
  afterEach(() => { cleanup?.(); cleanup = null; });

  it('keeps a list of structured items as sub-rows and joins a list of plain values', async () => {
    const nested: SettingSpec = { name: 'nested_probe', type: 'object', default: { peers: [{ host: 'a.test', port: 9100 }], tags: ['x', 'y'] }, mutable: false, allowed: null, help: 'Startup server configuration.', reason: 'restart required' };
    const rendered = await renderServerTab('en', { schema: [nested], current: {}, fingerprint: 'a'.repeat(64) });
    cleanup = rendered.unmount;
    const texts = startupCells(rendered.host).map((cell) => cell.textContent ?? '');
    expect(texts.filter((text) => DEBUG_SHAPE.test(text))).toEqual([]);
    expect(texts).toContain('x, y');
    expect(texts).toContain('a.test');
  });

  for (const locale of ['en', 'ko'] as const) {
    it(`prints no Debug or JSON shape in any value cell (${locale})`, async () => {
      const rendered = await renderServerTab(locale, { schema: specs, current: {}, fingerprint: 'a'.repeat(64) });
      cleanup = rendered.unmount;
      const cells = startupCells(rendered.host);
      const offending = cells.map((cell) => cell.textContent ?? '').filter((text) => DEBUG_SHAPE.test(text));
      expect(offending).toEqual([]);
      expect(cells.length).toBeGreaterThan(specs.length);
      // Structured values become labelled sub-rows.
      const tensorParallel = Array.from(rendered.host.querySelectorAll('.settings-value-group')).find((row) => row.querySelector(':scope > dt')?.textContent === 'tensor_parallel');
      expect(tensorParallel?.querySelector('.settings-subvalues')?.textContent).toContain('tp_size2');
    });

    it(`prints the localized unset text for an absent value (${locale})`, async () => {
      const current = Object.fromEntries(specs.filter((spec) => spec.type.endsWith('_or_null')).map((spec) => [spec.name, null]));
      const rendered = await renderServerTab(locale, { schema: specs, current, fingerprint: 'a'.repeat(64) });
      cleanup = rendered.unmount;
      const unset = t(locale, 'settings.value.unset');
      const topLevel = Array.from(rendered.host.querySelectorAll('[data-testid="settings-startup-values"] > div > dd')).map((cell) => cell.textContent ?? '');
      expect(topLevel.filter((text) => text === unset)).toHaveLength(Object.keys(current).length);
      expect(startupCells(rendered.host).map((cell) => cell.textContent ?? '').filter((text) => DEBUG_SHAPE.test(text))).toEqual([]);
    });
  }
});
